"""
detector.py — Ball and rim detection only.

Person detection has moved to tracker.py (yolov8n-pose + ByteTrack).
This module is now lightweight: one YOLO call filtered to class 32 (sports ball)
plus the HSV/Hough rim detector (cached every 30 frames).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

_yolo = None


def _get_yolo(model_path: str = "yolov8n.pt"):
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(model_path)
    return _yolo


_BALL_CLASS = 32  # COCO sports ball


class DetectionResult:
    """Single-frame detection output."""

    def __init__(self):
        # Each player: {bbox, conf, center} — populated by tracker now, kept for cache compat
        self.players: List[Dict[str, Any]] = []
        self.ball: Optional[Tuple[float, float]] = None
        self.rim: Optional[Tuple[float, float, float, float]] = None

    def to_dict(self):
        return {
            "players": self.players,
            "ball": self.ball,
            "rim": self.rim,
        }


class Detector:
    RIM_DETECT_INTERVAL = 30

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.15):
        self.conf = conf
        self.model_path = model_path
        self._model = None
        self._rim_history: List[Tuple[float, float, float, float]] = []
        self._cached_rim: Optional[Tuple[float, float, float, float]] = None
        self._rim_call_count: int = 0

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Detect ball and rim only. Person detection is handled by tracker."""
        self._load()
        h, w = frame.shape[:2]
        result = DetectionResult()

        # Ball detection — class 32 only, cheap
        preds = self._model.predict(
            source=frame,
            classes=[_BALL_CLASS],
            conf=self.conf,
            verbose=False,
            half=False,  # FP32 for ball — small model, accuracy matters
        )

        if preds and preds[0].boxes is not None:
            boxes = preds[0].boxes
            best_conf = -1.0
            for i in range(len(boxes)):
                conf = float(boxes.conf[i].item())
                if conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = [v.item() for v in boxes.xyxy[i]]
                    cx = ((x1 / w) + (x2 / w)) / 2
                    cy = ((y1 / h) + (y2 / h)) / 2
                    result.ball = (cx, cy)

        # Rim detection (cached)
        self._rim_call_count += 1
        if self._rim_call_count % self.RIM_DETECT_INTERVAL == 1:
            rim = self._detect_rim(frame, h, w)
            if rim is not None:
                self._rim_history.append(rim)
                if len(self._rim_history) > 10:
                    self._rim_history.pop(0)
                arr = np.array(self._rim_history)
                self._cached_rim = tuple(arr.mean(axis=0).tolist())  # type: ignore

        result.rim = self._cached_rim
        return result

    def _detect_rim(
        self, frame: np.ndarray, h: int, w: int
    ) -> Optional[Tuple[float, float, float, float]]:
        roi_bottom = int(h * 0.65)
        roi = frame[:roi_bottom, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lo = np.array([5, 150, 100])
        hi = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        circles = cv2.HoughCircles(
            mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=15, minRadius=8, maxRadius=60,
        )
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            best = sorted(circles, key=lambda c: c[2], reverse=True)[0]
            cx_px, cy_px, r = best
            return (cx_px / w, cy_px / h, (2 * r) / w, (2 * r) / h)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if 200 < area < 8000:
                x, y, cw, ch = cv2.boundingRect(largest)
                return ((x + cw / 2) / w, (y + ch / 2) / h, cw / w, ch / h)
        return None

    def crop_player(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0] * w))
        y1 = max(0, int(bbox[1] * h))
        x2 = min(w, int(bbox[2] * w))
        y2 = min(h, int(bbox[3] * h))
        if x2 <= x1 or y2 <= y1:
            return frame[0:1, 0:1]
        return frame[y1:y2, x1:x2].copy()
