#!/bin/bash
# ============================================================
# Vast.ai startup script — Jersey Tracker
# Run this once after renting a new instance:
#   chmod +x vastai_start.sh && ./vastai_start.sh
# ============================================================

set -e

echo "=== Installing dependencies ==="
pip install -q ultralytics opencv-python-headless easyocr yt-dlp \
    fastapi uvicorn websockets scikit-learn aiofiles \
    pytorch_lightning timm nltk

echo "=== Checking models ==="
mkdir -p models videos cache

# Download pose model if missing
if [ ! -f "models/yolov8n-pose.pt" ]; then
    python -c "from ultralytics import YOLO; m=YOLO('yolov8n-pose.pt'); import shutil; shutil.move('yolov8n-pose.pt','models/')"
fi

# Export TensorRT engine if missing
if [ ! -f "yolov8n-pose.engine" ]; then
    echo "=== Building TensorRT engine (takes 5 min) ==="
    python -c "
from ultralytics import YOLO
YOLO('models/yolov8n-pose.pt').export(format='engine', half=True, imgsz=640, device=0)
import shutil, pathlib
e = pathlib.Path('yolov8n-pose.engine')
if not e.exists():
    e2 = list(pathlib.Path('.').glob('**/*.engine'))
    if e2: shutil.copy(str(e2[0]), 'yolov8n-pose.engine')
"
fi

# Download best.pt if missing
if [ ! -f "models/best.pt" ]; then
    echo "ERROR: models/best.pt not found. Upload it manually."
    echo "Run on your PC:  curl.exe -F 'reqtype=fileupload' -F 'time=72h' -F 'fileToUpload=@E:/claude jerseys/backend/models/best.pt' https://litterbox.catbox.moe/resources/internals/api.php"
    echo "Then on Vast.ai: wget <url> -O models/best.pt"
fi

echo "=== Starting backend ==="
pkill -f uvicorn 2>/dev/null || true
sleep 1
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
sleep 5

echo "=== Connecting Cloudflare tunnel ==="
if [ -z "$CF_TUNNEL_TOKEN" ]; then
    echo "No CF_TUNNEL_TOKEN set — starting quick tunnel (URL changes each session)"
    nohup cloudflared tunnel --url http://localhost:8000 > tunnel.log 2>&1 &
    sleep 5
    TUNNEL_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel.log | head -1)
    echo ""
    echo "======================================"
    echo "  TUNNEL URL: $TUNNEL_URL"
    echo "  Set this in Vercel: VITE_API_URL=$TUNNEL_URL"
    echo "======================================"
else
    nohup cloudflared tunnel run --token "$CF_TUNNEL_TOKEN" > tunnel.log 2>&1 &
    echo "Named tunnel connected (permanent URL)"
fi

echo ""
echo "=== All done! Backend running on port 8000 ==="
tail -5 backend.log
