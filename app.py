import os, cv2, time, json, torch
import numpy as np
from PIL import Image
from datetime import datetime, timezone
from difflib import SequenceMatcher
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from dotenv import load_dotenv
from skimage.metrics import structural_similarity as ssim

load_dotenv()

# ── ENV (solo RTSP) ───────────────────────────────────────────────────────────
RTSP_URL = os.getenv("RTSP_URL", "").strip()
if not RTSP_URL:
    raise SystemExit("❌ Falta RTSP_URL en .env")

EVENT_LOG_FILE = os.getenv("EVENT_LOG_FILE", "events.log")
CAMERA_NAME = os.getenv("CAMERA_NAME", "rtsp_cam")

FRAME_W = int(os.getenv("FRAME_W", "640"))
FRAME_H = int(os.getenv("FRAME_H", "480"))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))                       # procesa 1 de cada N
MIN_SECONDS_BETWEEN = float(os.getenv("MIN_SECONDS_BETWEEN", "5"))   # min entre eventos
CAPTION_SIMILARITY = float(os.getenv("CAPTION_SIMILARITY", "0.90"))  # anti-spam
SSIM_THRESHOLD = float(os.getenv("SSIM_THRESHOLD", "0.985"))         # > = casi idéntico
CONF_THRES = float(os.getenv("CONF_THRES", "0.35"))                  # YOLO conf

OBJECT_WHITELIST = set(os.getenv("OBJECT_WHITELIST",
    "person,car,truck,bus,motorcycle,bicycle,dog,cat,bird,knife,gun,backpack"
).replace(" ", "").split(","))

# Opciones de captura FFmpeg para RTSP (TCP + reintentos)
# (OpenCV las toma si usas backend FFMPEG)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|max_delay;5000000|stimeout;5000000|rw_timeout;5000000")

RECONNECT_DELAY = float(os.getenv("RECONNECT_DELAY", "3"))  # segundos
MAX_READ_FAILS = int(os.getenv("MAX_READ_FAILS", "30"))     # reintenta tras N fallos seguidos

# ── Modelos ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Device: {DEVICE}")

blip_name = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
blip_processor = BlipProcessor.from_pretrained(blip_name)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_name).to(DEVICE)

yolo_name = os.getenv("YOLO_MODEL", "yolov8n.pt")  # n = rápido
yolo = YOLO(yolo_name)

# ── Utils ─────────────────────────────────────────────────────────────────────
def now_ts():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def luminance(frame_bgr: np.ndarray) -> float:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    y = 0.2126*rgb[:,:,0] + 0.7152*rgb[:,:,1] + 0.0722*rgb[:,:,2]
    return float(np.mean(y))

def caption_blip(pil_img: Image.Image, max_new_tokens=28) -> str:
    with torch.no_grad():
        inputs = blip_processor(images=pil_img, return_tensors="pt").to(DEVICE)
        out = blip_model.generate(**inputs, max_new_tokens=max_new_tokens)
        cap = blip_processor.decode(out[0], skip_special_tokens=True)
        return cap.strip()

def detect_yolo(frame_bgr):
    res = yolo.predict(frame_bgr, imgsz=max(FRAME_W, FRAME_H), conf=CONF_THRES, verbose=False)
    counts = {}
    if not res:
        return counts
    for r in res:
        for cls_id in r.boxes.cls.tolist():
            name = r.names[int(cls_id)]
            if name in OBJECT_WHITELIST:
                counts[name] = counts.get(name, 0) + 1
    return counts

def fuse_caption(base: str, counts: dict, is_dark: bool) -> str:
    if counts:
        parts = []
        for label in sorted(counts.keys()):
            n = counts[label]
            lbl = label
            if n > 1 and not lbl.endswith("s"):
                lbl += "s"
            parts.append(f"{n} {lbl}")
        obj_text = ", ".join(parts[:-1]) + (" y " + parts[-1] if len(parts) > 1 else parts[0])
    else:
        obj_text = ""

    useful = any(k in base.lower() for k in ["person", "people", "dog", "car", "truck", "cat"])
    if counts and not useful:
        fused = f"{base}. Objetos: {obj_text}."
    elif counts:
        fused = f"{base} ({obj_text})."
    else:
        fused = base

    if is_dark:
        fused = f"{fused} (escena oscura)"
    return fused.strip()

def write_event(text: str):
    ev = {"timestamp": now_ts(), "type": "camera", "value": text, "source": CAMERA_NAME}
    with open(EVENT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    print(f"📝 {ev}")

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def open_rtsp():
    print(f"🔌 Conectando RTSP: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("🚫 No se pudo abrir RTSP; reintentará…")
    return cap

# ── Loop principal (con reconexión) ───────────────────────────────────────────
print("▶️ vtcam_rtsp_to_events: RTSP → captions BLIP + objetos YOLO → events.log (Ctrl+C para salir)")
cap = open_rtsp()

last_caption = ""
last_write = 0.0
last_gray = None
frame_count = 0
read_fails = 0

try:
    while True:
        if cap is None or not cap.isOpened():
            time.sleep(RECONNECT_DELAY)
            cap = open_rtsp()
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            read_fails += 1
            if read_fails >= MAX_READ_FAILS:
                print("🔁 Demasiados fallos de lectura; reconectando…")
                cap.release()
                cap = open_rtsp()
                read_fails = 0
            else:
                time.sleep(0.1)
            continue
        read_fails = 0

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # SSIM anti-frames idénticos
        if last_gray is not None:
            score = ssim(gray, last_gray)
            if score >= SSIM_THRESHOLD:
                continue
        last_gray = gray

        is_dark = luminance(frame) < 0.20

        # BLIP caption
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        base_caption = caption_blip(pil_img)

        # YOLO objetos + fusión
        counts = detect_yolo(frame)
        final_caption = fuse_caption(base_caption, counts, is_dark)

        # anti-spam: cooldown + similaridad
        now = time.time()
        if now - last_write < MIN_SECONDS_BETWEEN:
            continue
        if last_caption and similar(final_caption, last_caption) >= CAPTION_SIMILARITY:
            continue

        write_event(final_caption)
        last_caption = final_caption
        last_write = now

except KeyboardInterrupt:
    print("🛑 Detenido por usuario")
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
