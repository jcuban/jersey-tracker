"""
Runs the EXACT same code path as processor.py on 3 frames and prints everything.
Run: python debug_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
from pathlib import Path
from detector import Detector
from jersey_model import JerseyModel
from tracker import PlayerTracker

VIDEOS_DIR = Path(__file__).parent / "videos"
video_files = sorted(VIDEOS_DIR.glob("*.f137.mp4"))
video_path = video_files[0]
print(f"Video: {video_path}\n")

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

detector = Detector()
jersey_model = JerseyModel()
tracker = PlayerTracker()

print("Models loaded. Running 5 test frames...\n")

test_frames = [300, 600, 900, 1200, 1800]  # frame indices (not seconds)

for frame_idx in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {frame_idx}: could not read")
        continue

    # --- Detection (same as processor) ---
    detection = detector.detect(frame)
    print(f"Frame {frame_idx}: {len(detection.players)} players, ball={detection.ball is not None}, rim={detection.rim is not None}")

    # --- Tracking (same as processor) ---
    tracked = tracker.track_frame(frame, detection, jersey_model)
    print(f"  Tracked: {len(tracked)} players")
    for p in tracked[:5]:
        print(f"    track_id={p['track_id']} jersey={p['jersey_num']} team={p['team']}")

    print()

cap.release()

# Print final jersey map
print("=== Final jersey_map ===")
print(tracker.jersey_map)
print("\n=== Final team_map ===")
print(tracker.team_map)
