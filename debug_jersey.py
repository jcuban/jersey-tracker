"""
Quick diagnostic — grab frames from a video and test jersey + team detection.
Run from the backend directory:
    python debug_jersey.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from pathlib import Path

VIDEOS_DIR = Path(__file__).parent / "videos"

# Pick the first .f137.mp4 (video-only stream)
video_files = sorted(VIDEOS_DIR.glob("*.f137.mp4"))
if not video_files:
    print("No video files found!")
    sys.exit(1)

video_path = video_files[0]
print(f"Testing on: {video_path}")

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print("Could not open video!")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps}, Total frames: {total}, Duration: {total/fps:.1f}s")

# Load jersey model
from jersey_model import JerseyModel
jersey_model = JerseyModel()
print("Jersey model loaded")

# Load tracker
from tracker import PlayerTracker
tracker = PlayerTracker()
print("Tracker loaded")

# Test on frames every 5 minutes of game time
test_times = [300, 600, 900, 1200, 1800, 2400, 3000]  # seconds
results = []

for t in test_times:
    frame_idx = int(t * fps)
    if frame_idx >= total:
        break
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        continue

    print(f"\n--- Frame at {t}s ---")
    h, w = frame.shape[:2]
    print(f"  Frame size: {w}x{h}")

    # Run YOLO person detection
    from detector import Detector
    detector = Detector()
    detection = detector.detect(frame)
    print(f"  Players detected: {len(detection.players)}")

    # Test jersey detection on each player
    for i, p in enumerate(detection.players[:5]):  # first 5 players
        x1, y1, x2, y2 = p["bbox"]
        px1 = max(0, int(x1 * w))
        py1 = max(0, int(y1 * h))
        px2 = min(w, int(x2 * w))
        py2 = min(h, int(y2 * h))
        crop = frame[py1:py2, px1:px2]
        if crop.size == 0:
            continue
        num, conf = jersey_model.detect_number(crop)
        print(f"  Player {i}: bbox=({px1},{py1},{px2},{py2}) size={px2-px1}x{py2-py1} jersey={num} conf={conf:.2f}")

cap.release()
print("\nDone.")
