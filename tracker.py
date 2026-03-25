"""
tracker.py — Player tracking using YOLOv8n-pose + ByteTrack.

Single YOLO call per frame (replaces the old double-YOLO architecture):
  - yolov8n-pose detects persons AND provides 17 keypoints in one forward pass
  - ByteTrack (persist=True) assigns stable track IDs
  - Shoulder/hip keypoints are used to crop the jersey region precisely
  - K-means on torso colour separates home vs away teams (light/dark)
"""

import cv2
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from sklearn.cluster import KMeans

# COCO keypoint indices
_KP_L_SHOULDER = 5
_KP_R_SHOULDER = 6
_KP_L_HIP      = 11
_KP_R_HIP      = 12


class PlayerTracker:
    """
    Maintains consistent player identities across frames using ByteTrack.

    Uses yolov8n-pose for combined detection + pose in a single YOLO call.
    Jersey crops are extracted from shoulder→hip keypoints when available.
    """

    JERSEY_LOCK_CONFIDENCE = 0.45
    JERSEY_LOCK_VOTES      = 5
    COLOR_SAMPLES_NEEDED   = 15

    def __init__(self, base_model_path: str = "yolov8n-pose.engine"):
        self._model = None
        self._model_path = base_model_path

        # track_id → locked jersey number
        self.jersey_map: Dict[str, str] = {}
        self._jersey_votes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._jersey_conf: Dict[str, float] = defaultdict(float)
        self._jersey_attempts: Dict[str, int] = defaultdict(int)
        self._JERSEY_MAX_ATTEMPTS = 150

        # track_id → colour samples (RGB)
        self._colour_samples: Dict[str, List[np.ndarray]] = defaultdict(list)
        # track_id → "light" | "dark" | "unknown"
        self.team_map: Dict[str, str] = {}
        self._team_centers: Optional[np.ndarray] = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            print(f"[tracker] Pose model loaded: {self._model_path}")

    # ------------------------------------------------------------------
    # Main tracking call
    # ------------------------------------------------------------------

    def track_frame(
        self,
        frame: np.ndarray,
        detection_result,   # DetectionResult — used for ball/rim only
        jersey_model,
    ) -> List[Dict[str, Any]]:
        """
        Single YOLO-pose forward pass: detect persons + keypoints + ByteTrack IDs.

        Returns list of tracked player dicts:
            {track_id, jersey_num, team, bbox, center, crop}
        """
        self._load()
        h, w = frame.shape[:2]

        results = self._model.track(
            source=frame,
            classes=[0],              # persons only
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            conf=0.3,
            iou=0.5,
            half=True,                # FP16 — safe for person/pose
            device=0,
        )

        tracked: List[Dict[str, Any]] = []

        if not results or results[0].boxes is None:
            return tracked

        boxes = results[0].boxes
        if boxes.id is None:
            return tracked

        # Keypoints tensor: shape (N, 17, 2) in pixel coords
        kpts_xy = None
        if results[0].keypoints is not None:
            try:
                kpts_xy = results[0].keypoints.xy.cpu().numpy()
            except Exception:
                kpts_xy = None

        # --- First pass: collect all crops and metadata ---
        player_data = []
        batch_crops = []
        batch_player_indices = []

        for i in range(len(boxes)):
            track_id = str(int(boxes.id[i].item()))
            x1, y1, x2, y2 = [v.item() for v in boxes.xyxy[i]]
            nx1, ny1, nx2, ny2 = x1 / w, y1 / h, x2 / w, y2 / h
            cx, cy = (nx1 + nx2) / 2, (ny1 + ny2) / 2

            crop = self._pose_jersey_crop(frame, kpts_xy, i, x1, y1, x2, y2, h, w)

            px1, py1 = max(0, int(x1)), max(0, int(y1))
            px2, py2 = min(w, int(x2)), min(h, int(y2))
            full_crop = frame[py1:py2, px1:px2].copy() if px2 > px1 and py2 > py1 else None

            jersey_num = self.jersey_map.get(track_id, "?")
            needs_ocr = (
                jersey_num == "?" and
                crop is not None and
                self._jersey_attempts[track_id] < self._JERSEY_MAX_ATTEMPTS
            )
            if needs_ocr:
                self._jersey_attempts[track_id] += 1
                batch_crops.append(crop)
                batch_player_indices.append(len(player_data))

            player_data.append({
                "track_id": track_id,
                "jersey_num": jersey_num,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "nx1": nx1, "ny1": ny1, "nx2": nx2, "ny2": ny2,
                "cx": cx, "cy": cy,
                "crop": crop,
                "full_crop": full_crop,
            })

        # --- Batch OCR: one GPU pass for all unlocked players ---
        if batch_crops:
            batch_results = jersey_model.detect_number_batch(batch_crops)
            for bi, pi in enumerate(batch_player_indices):
                detected_num, conf = batch_results[bi]
                track_id = player_data[pi]["track_id"]
                if detected_num != "?":
                    self._jersey_votes[track_id][detected_num] += 1
                    self._jersey_conf[track_id] = max(self._jersey_conf[track_id], conf)
                    votes = self._jersey_votes[track_id][detected_num]
                    if votes >= self.JERSEY_LOCK_VOTES and conf >= self.JERSEY_LOCK_CONFIDENCE:
                        self.jersey_map[track_id] = detected_num
                        player_data[pi]["jersey_num"] = detected_num

        # --- Second pass: build tracked list and colour samples ---
        for pd in player_data:
            track_id = pd["track_id"]
            full_crop = pd["full_crop"]

            if full_crop is not None and len(self._colour_samples[track_id]) < 60:
                colour = self._dominant_colour(full_crop)
                if colour is not None:
                    self._colour_samples[track_id].append(colour)

            tracked.append({
                "track_id": track_id,
                "jersey_num": self.jersey_map.get(track_id, pd["jersey_num"]),
                "team": self.team_map.get(track_id, "unknown"),
                "bbox": (pd["nx1"], pd["ny1"], pd["nx2"], pd["ny2"]),
                "center": (pd["cx"], pd["cy"]),
                "crop": full_crop,
            })

        self._maybe_cluster_teams()

        for p in tracked:
            p["team"] = self.team_map.get(p["track_id"], "unknown")

        return tracked

    # ------------------------------------------------------------------
    # Pose-guided jersey crop
    # ------------------------------------------------------------------

    def _pose_jersey_crop(
        self,
        frame: np.ndarray,
        kpts_xy,          # numpy (N, 17, 2) or None
        person_idx: int,
        x1: float, y1: float, x2: float, y2: float,
        h: int, w: int,
    ) -> Optional[np.ndarray]:
        """
        Return the jersey region crop.
        If keypoints are available, crop shoulder→hip; otherwise fall back to
        the upper-torso portion of the bounding box.
        """
        if kpts_xy is not None and person_idx < len(kpts_xy):
            kpts = kpts_xy[person_idx]  # (17, 2)
            ls = kpts[_KP_L_SHOULDER]
            rs = kpts[_KP_R_SHOULDER]
            lh = kpts[_KP_L_HIP]
            rh = kpts[_KP_R_HIP]

            # Only use keypoints if all four are detected (non-zero)
            if not (ls[0] == 0 and ls[1] == 0) and not (rs[0] == 0 and rs[1] == 0) \
                    and not (lh[0] == 0 and lh[1] == 0) and not (rh[0] == 0 and rh[1] == 0):
                pad_x, pad_y = 12, 6
                cx1 = max(0, int(min(ls[0], rs[0])) - pad_x)
                cy1 = max(0, int(min(ls[1], rs[1])) - pad_y)
                cx2 = min(w, int(max(ls[0], rs[0])) + pad_x)
                cy2 = min(h, int(max(lh[1], rh[1])) + pad_y)
                if cx2 > cx1 and cy2 > cy1:
                    return frame[cy1:cy2, cx1:cx2].copy()

        # Fallback: upper 20%-60% of the bounding box
        bx1 = max(0, int(x1))
        bx2 = min(w, int(x2))
        by1 = max(0, int(y1 + (y2 - y1) * 0.20))
        by2 = min(h, int(y1 + (y2 - y1) * 0.60))
        if bx2 > bx1 and by2 > by1:
            return frame[by1:by2, bx1:bx2].copy()
        return None

    # ------------------------------------------------------------------
    # K-means team colour clustering
    # ------------------------------------------------------------------

    def _dominant_colour(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        # Centre 50% vertically, 60% horizontally — avoid arms/background
        torso = crop[int(h * 0.2):int(h * 0.7), int(w * 0.2):int(w * 0.8)]
        if torso.size == 0:
            return None
        # Convert to L*a*b* and cluster on a*/b* only (lighting-invariant)
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        ab = lab[:, :, 1:].reshape(-1, 2).astype(np.float32)
        if len(ab) < 10:
            return None
        return np.mean(ab, axis=0)  # (a*, b*)

    def _maybe_cluster_teams(self):
        eligible = {
            tid: samples
            for tid, samples in self._colour_samples.items()
            if len(samples) >= self.COLOR_SAMPLES_NEEDED
        }
        if len(eligible) < 4:
            return

        tids = list(eligible.keys())
        rep_colours = np.array([np.mean(eligible[t], axis=0) for t in tids])

        # Filter out refs/coaches (near-black or near-white in a*b* space)
        # In L*a*b* a*/b* space both are near (128,128) — we use L* brightness
        # We stored a*b* only, so filter by distance from neutral (128,128)
        neutrality = np.linalg.norm(rep_colours - 128.0, axis=1)
        # Keep players with some colour character; refs in black ≈ (128,128)
        valid_mask = neutrality > 2.0
        if valid_mask.sum() < 4:
            valid_mask = np.ones(len(tids), dtype=bool)

        v_tids = [t for t, v in zip(tids, valid_mask) if v]
        v_colours = rep_colours[valid_mask]

        # Top 20 most-observed tracks
        sample_counts = np.array([len(eligible[t]) for t in v_tids])
        if len(v_tids) > 20:
            top_idx = np.argsort(sample_counts)[-20:]
            v_tids = [v_tids[i] for i in top_idx]
            v_colours = v_colours[top_idx]

        if len(v_tids) < 4:
            return

        km = KMeans(n_clusters=2, n_init=5, max_iter=100, random_state=42)
        labels = km.fit_predict(v_colours)
        self._team_centers = km.cluster_centers_

        # "light" = higher L* brightness; since we only have a*b*, use cluster
        # membership size as tiebreak — but brightness correlates with a*b* distance
        # from neutral. Lighter jerseys (white) are near (128,128); darker are further.
        neutrality_centers = np.linalg.norm(km.cluster_centers_ - 128.0, axis=1)
        light_label = int(np.argmin(neutrality_centers))  # closer to neutral = lighter

        for tid, label in zip(v_tids, labels):
            self.team_map[tid] = "light" if label == light_label else "dark"

        # Assign remaining tracks via nearest cluster
        for tid in tids:
            if tid not in self.team_map and tid in eligible:
                colour = np.mean(eligible[tid], axis=0)
                dists = [np.linalg.norm(colour - c) for c in km.cluster_centers_]
                closest = int(np.argmin(dists))
                self.team_map[tid] = "light" if closest == light_label else "dark"
