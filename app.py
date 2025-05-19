# VTCam – Versión con control de alertas repetidas (por similitud configurable)

import os
import cv2
import torch
import requests
import time
import hashlib
from datetime import datetime
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv
import re
from difflib import SequenceMatcher

load_dotenv()

SOURCE_TYPE = os.getenv("SOURCE_TYPE", "webcam")
RTSP_URL = os.getenv("RTSP_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ALERT_IMAGE_DIR = os.getenv("ALERT_IMAGE_DIR", "alerts")

ENABLE_LLM_ANALYSIS = os.getenv("ENABLE_LLM_ANALYSIS", "true").lower() == "true"
FRAGMENT_INTERVAL = int(os.getenv("FRAGMENT_INTERVAL", "60"))
FRAGMENT_LINES = int(os.getenv("FRAGMENT_LINES", "10"))
LLM_ANALYSIS_PROMPT_MODE = os.getenv("LLM_ANALYSIS_PROMPT_MODE", "simple")
LLM_ALERT_KEYWORDS = os.getenv("LLM_ALERT_KEYWORDS", "sospechoso,intruso,persona desconocida,comportamiento extraño").lower().split(",")
LLM_IGNORE_PHRASES = os.getenv("LLM_IGNORE_PHRASES", "no se detecta,nada sospechoso,sin riesgo,no hay,normal").lower().split(",")
SEND_ALL_LLM_RESULTS = os.getenv("SEND_ALL_LLM_RESULTS", "false").lower() == "true"
SEND_VIDEO_CLIP = os.getenv("SEND_VIDEO_CLIP", "false").lower() == "true"
VIDEO_DURATION_SECONDS = int(os.getenv("VIDEO_DURATION_SECONDS", "5"))
ALERT_SIMILARITY_THRESHOLD = float(os.getenv("ALERT_SIMILARITY_THRESHOLD", "0.9"))
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "300"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Dispositivo: {DEVICE}")
print(f"🛰️ Fuente: {'Webcam local' if SOURCE_TYPE == 'webcam' else RTSP_URL}")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

cap = cv2.VideoCapture(0) if SOURCE_TYPE == "webcam" else cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    print("🚫 Error al abrir la fuente de video")
    exit(1)

os.makedirs(ALERT_IMAGE_DIR, exist_ok=True)
LOG_FILE = "VTcam_log.jsonl"
ANALYSIS_FILE = "VTcam_analysis.jsonl"

last_caption = ""
fragment_start = time.time()
current_frame = None
frame_count = 0
FRAME_SKIP = 3

alert_cache = []  # lista de últimos análisis recientes (texto, timestamp)

def escape_markdown(text):
    return re.sub(r'([_*`\[\]()~>#+=|{}.!-])', r'\\\1', text)

def get_alert_hash(text):
    return hashlib.md5(text.strip().lower().encode()).hexdigest()

def is_similar_alert(new_text):
    global alert_cache
    now = time.time()
    alert_cache = [(txt, ts) for txt, ts in alert_cache if now - ts < ALERT_COOLDOWN_SECONDS]
    for past_text, _ in alert_cache:
        similarity = SequenceMatcher(None, new_text.lower(), past_text.lower()).ratio()
        if similarity >= ALERT_SIMILARITY_THRESHOLD:
            return True
    return False

def log_caption(caption):
    timestamp = datetime.utcnow().isoformat()
    entry = {"timestamp": timestamp, "caption": caption}
    with open(LOG_FILE, "a") as f:
        f.write(f"{entry}\n")
    return f"[{timestamp}] {caption}"

def build_prompt(fragment_lines):
    base = "\n".join(fragment_lines)
    if LLM_ANALYSIS_PROMPT_MODE == "detailed":
        return f"""
Eres un sistema de vigilancia analítica avanzada. Analiza el siguiente fragmento.
Busca comportamientos atípicos, riesgo, múltiples personas, acciones sospechosas o inusuales.

Fragmento:
{base}

Resumen estructurado:
- Elementos detectados
- Cambios inusuales
- Riesgo potencial
- Recomendación de acción
"""
    else:
        return f"Analiza este fragmento generado por una cámara. ¿Hay algo raro o sospechoso?\n\nFragmento:\n{base}"

def send_telegram_video(caption_text, cap):
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(ALERT_IMAGE_DIR, f"alert_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))
        start_time = time.time()

        while time.time() - start_time < VIDEO_DURATION_SECONDS:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            out.write(frame)

        out.release()

        if not os.path.exists(video_path):
            print("❌ Video no generado correctamente")
            return

        caption_text = escape_markdown(caption_text[:1000])
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo"
        with open(video_path, "rb") as video_file:
            files = {"video": video_file}
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption_text,
                "parse_mode": "MarkdownV2"
            }
            r = requests.post(url, data=data, files=files)
            r.raise_for_status()
            print("🎥 Video enviado por Telegram.")

    except Exception as e:
        print(f"📵 Error al enviar video por Telegram: {e}")

def should_alert_llm_response(result):
    return any(k.strip() in result.lower() for k in LLM_ALERT_KEYWORDS if k.strip())

def llm_analysis_is_useful(result):
    result_l = result.lower()
    return not any(p.strip() in result_l for p in LLM_IGNORE_PHRASES if p.strip())

def get_last_log_lines(n=10):
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        return [line.strip() for line in lines[-n:]]
    except Exception as e:
        print(f"❌ Error leyendo log: {e}")
        return []

def analyze_fragment():
    lines = get_last_log_lines(FRAGMENT_LINES)
    if not lines:
        return

    prompt = build_prompt(lines)
    payload = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        result = r.json()["choices"][0]["message"]["content"]

        timestamp = datetime.utcnow().isoformat()
        entry = {
            "timestamp": timestamp,
            "fragment": lines,
            "analysis": result
        }
        with open(ANALYSIS_FILE, "a") as f:
            f.write(f"{entry}\n")

        print("🧠 Resultado del análisis LLM:")
        print("─" * 60)
        print(result)
        print("─" * 60)

        if is_similar_alert(result):
            print("⏱️ Alerta similar ya enviada recientemente, se omite.")
            return

        alert_cache.append((result, time.time()))

        if SEND_ALL_LLM_RESULTS and cap is not None:
            send_telegram_video(f"📝 Análisis del LLM:\n\n{result}", cap)
        elif should_alert_llm_response(result) and llm_analysis_is_useful(result) and cap is not None:
            send_telegram_video(f"⚠️ ALERTA DE ANÁLISIS LLM:\n\n{result}", cap)
        else:
            print("✅ Análisis sin alertas, no se envía nada.")

    except Exception as e:
        print(f"❌ Error al analizar con LLM: {e}")

print("🎬 VTCam ejecutándose. Presiona Ctrl+C para salir.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No se pudo capturar imagen")
            continue

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (640, 480))
        current_frame = frame.copy()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

        print(f"📷 Caption detectado: {caption}")
        if caption != last_caption:
            log_caption(caption)
            last_caption = caption

        if time.time() - fragment_start >= FRAGMENT_INTERVAL:
            print(f"⏳ Ejecutando análisis LLM con las últimas {FRAGMENT_LINES} líneas...")
            if ENABLE_LLM_ANALYSIS:
                analyze_fragment()
            fragment_start = time.time()

except KeyboardInterrupt:
    print("🛑 VTCam detenido")
    cap.release()
    cv2.destroyAllWindows()
