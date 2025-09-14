#!/usr/bin/env python3
import os, cv2, time, requests
from datetime import datetime, timezone
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

# ── ENV ───────────────────────────────────────────────────────────
RTSP_URL  = os.getenv("RTSP_URL", "").strip()
if not RTSP_URL:
    raise SystemExit("❌ Falta RTSP_URL en .env")

EVENT_API_URL   = os.getenv("EVENT_API_URL", "http://localhost:8000/event")
CAMERA_NAME     = os.getenv("CAMERA_NAME", "rtsp_cam")

FRAME_W = int(os.getenv("FRAME_W", "640"))
FRAME_H = int(os.getenv("FRAME_H", "480"))

TIME_CAPTURE_IMAGE   = float(os.getenv("TIME_CAPTURE_IMAGE", "10"))    # cada cuánto analiza un frame (segundos)
TIME_ANALYZE_EVENTS  = float(os.getenv("TIME_ANALYZE_EVENTS", "1200")) # ventana de reinicio de análisis (segundos)
MIN_SECONDS_BETWEEN  = float(os.getenv("MIN_SECONDS_BETWEEN", "5"))    # mínimo entre eventos (segundos)

CONF_THRES    = float(os.getenv("CONF_THRES", "0.35"))
SSIM_THRESHOLD= float(os.getenv("SSIM_THRESHOLD", "0.985"))

# ── Modelo YOLO oficial (L) ───────────────────────────────────────
model = YOLO(os.getenv("YOLO_MODEL", "yolov8l.pt"))

# ── Utilidades ────────────────────────────────────────────────────
def now_ts():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def send_event(event_type, desc=""):
    """Envía evento al server con formato compatible"""
    ev = {
        "source": CAMERA_NAME,
        "description": f"{event_type}: {desc}",
        "value": None,
        "timestamp": now_ts(),
    }
    try:
        r = requests.post(EVENT_API_URL, json=ev, timeout=5)
        r.raise_for_status()
        print(f"✅ Evento enviado: {ev}")
    except Exception as e:
        print(f"⚠️ Error enviando {event_type}: {e}")

def detect(frame, gray, last_gray):
    """Evalúa frame y devuelve lista de objetos detectados"""
    events = []
    if last_gray is not None:
        score = ssim(gray, last_gray)
        if score >= SSIM_THRESHOLD:
            return events  # nada nuevo

    res = model.predict(frame, conf=CONF_THRES, verbose=False)
    if res and len(res[0].boxes) > 0:
        for cls_id in res[0].boxes.cls.tolist():
            label = res[0].names[int(cls_id)]
            events.append(label)
    return events

# ── Loop principal ────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    last_gray, last_event_time, last_reset = None, 0, time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ RTSP desconectado, reintentando…")
            time.sleep(3)
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        now = time.time()
        if now - last_reset > TIME_ANALYZE_EVENTS:
            last_gray, last_reset = None, now

        if now - last_event_time < TIME_CAPTURE_IMAGE:
            continue

        events = detect(frame, gray, last_gray)
        last_gray = gray

        if events:
            unique_events = set(events)
            for evt in unique_events:
                send_event(evt, f"Detección de {evt}")
            last_event_time = now

if __name__ == "__main__":
    main()