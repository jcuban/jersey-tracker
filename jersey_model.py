"""
jersey_model.py — Jersey number detection pipeline.

Two-stage approach:
  Stage 1 — YOLOv8n-pose (in tracker.py) provides shoulder/hip keypoints
             so the crop arriving here is already the jersey region.
             The old best.pt region detector is kept as optional fallback.

  Stage 2 — PaddleOCR (primary, ~3x faster than EasyOCR, GPU-accelerated)
             or EasyOCR (fallback if PaddleOCR not installed).
"""

import os
import re
import cv2

_DIGIT_RE = re.compile(r"\D")  # compiled once at module level
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

MODELS_DIR = Path(__file__).parent / "models"
BEST_PT = MODELS_DIR / "best.pt"

DATASET_DIR = MODELS_DIR / "jersey_dataset" / "Jersey Text Detection.v4i.yolov8"
DATASET_YAML = DATASET_DIR / "data.yaml"


def _run_training():
    print("[jersey_model] best.pt not found — starting fine-tuning …")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATASET_YAML),
        epochs=50,
        imgsz=640,
        project=str(MODELS_DIR),
        name="jersey_run",
        exist_ok=True,
        verbose=True,
    )
    trained_best = MODELS_DIR / "jersey_run" / "weights" / "best.pt"
    if trained_best.exists():
        import shutil
        shutil.copy(str(trained_best), str(BEST_PT))
        print(f"[jersey_model] Fine-tuned model saved → {BEST_PT}")
    else:
        raise FileNotFoundError(f"Training finished but best.pt not found at {trained_best}")


class JerseyModel:
    """
    Jersey number reader.
    Stage 1: optional YOLO region detector (best.pt).
    Stage 2: PaddleOCR or EasyOCR for digit reading.
    """

    CONFIDENCE_THRESHOLD = 0.50

    def __init__(self):
        self._detector = None
        self._ocr = None
        self._ocr_engine = None   # "paddle" | "easyocr"
        self._ensure_model()

    def _ensure_model(self):
        if not BEST_PT.exists():
            if DATASET_YAML.exists():
                _run_training()
            else:
                print(
                    "[jersey_model] WARNING: dataset not found — "
                    "jersey region YOLO disabled (will use keypoint crop directly)."
                )
                return
        self._load_detector()

    def _load_detector(self):
        if self._detector is not None or not BEST_PT.exists():
            return
        from ultralytics import YOLO
        self._detector = YOLO(str(BEST_PT))
        print(f"[jersey_model] Stage-1 detector loaded from {BEST_PT}")

    def _load_ocr(self):
        if self._ocr is not None:
            return

        # PARSeq — CVPR 2024, best-in-class for sports jersey numbers
        try:
            import torch
            from torchvision import transforms
            _model = torch.hub.load(
                "baudm/parseq", "parseq", pretrained=True, verbose=False
            ).eval().cuda().half()  # FP16: ~1.5x faster on RTX 2080 Ti
            self._ocr = _model
            print("[jersey_model] PARSeq ready (GPU, FP16)")
            self._ocr_transform = transforms.Compose([
                transforms.Resize((32, 128)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ])
            self._ocr_engine = "parseq"
            return
        except Exception as e:
            print(f"[jersey_model] PARSeq unavailable ({e}), falling back to EasyOCR…")

        # Fallback: EasyOCR
        try:
            import easyocr
            try:
                self._ocr = easyocr.Reader(["en"], gpu=True, verbose=False)
                self._ocr_engine = "easyocr"
                print("[jersey_model] EasyOCR ready (GPU)")
            except Exception:
                self._ocr = easyocr.Reader(["en"], gpu=False, verbose=False)
                self._ocr_engine = "easyocr"
                print("[jersey_model] EasyOCR ready (CPU)")
        except ImportError:
            print("[jersey_model] WARNING: no OCR engine available.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_number_batch(self, crops: list) -> list:
        """
        Batch version of detect_number — processes all crops in ONE GPU forward pass.
        Returns list of ("number_str", confidence) tuples.
        Much faster than calling detect_number() in a loop when using PARSeq.
        """
        self._load_ocr()
        if not crops:
            return []

        if self._ocr_engine != "parseq" or self._ocr is None:
            # Fallback: call single detect for non-PARSeq engines
            return [self.detect_number(c) for c in crops]

        import torch
        from PIL import Image

        results = [("?", 0.0)] * len(crops)
        tensors_normal = []
        tensors_inverted = []
        valid_indices = []

        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            try:
                resized = cv2.resize(crop, (128, 256))
                h, w = resized.shape[:2]
                scale = max(2, 64 // max(h, 1))
                enlarged = cv2.resize(resized, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                pil_n = Image.fromarray(cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB))
                pil_i = Image.fromarray(cv2.cvtColor(cv2.bitwise_not(enlarged), cv2.COLOR_BGR2RGB))
                tensors_normal.append(self._ocr_transform(pil_n))
                tensors_inverted.append(self._ocr_transform(pil_i))
                valid_indices.append(i)
            except Exception:
                continue

        if not valid_indices:
            return results

        try:
            with torch.no_grad():
                # Pass 1: normal images
                batch = torch.stack(tensors_normal).cuda().half()
                logits = self._ocr(batch)
                probs = logits.softmax(-1)
                preds, confs = self._ocr.tokenizer.decode(probs)
                needs_invert = []
                for j, idx in enumerate(valid_indices):
                    text = preds[j] if j < len(preds) else ""
                    conf = float(confs[j].mean()) if j < len(confs) else 0.0
                    digits = _DIGIT_RE.sub("", text)
                    if digits and 1 <= len(digits) <= 2 and conf >= 0.3:
                        results[idx] = (digits, conf)
                    else:
                        needs_invert.append(j)  # low confidence — try inverted

                # Pass 2: inverted only for low-confidence results
                if needs_invert:
                    inv_tensors = [tensors_inverted[j] for j in needs_invert]
                    inv_indices = [valid_indices[j] for j in needs_invert]
                    batch_inv = torch.stack(inv_tensors).cuda().half()
                    logits_inv = self._ocr(batch_inv)
                    probs_inv = logits_inv.softmax(-1)
                    preds_inv, confs_inv = self._ocr.tokenizer.decode(probs_inv)
                    for k, idx in enumerate(inv_indices):
                        text = preds_inv[k] if k < len(preds_inv) else ""
                        conf = float(confs_inv[k].mean()) if k < len(confs_inv) else 0.0
                        digits = _DIGIT_RE.sub("", text)
                        if digits and 1 <= len(digits) <= 2 and conf >= 0.3:
                            cur_text, cur_conf = results[idx]
                            if conf > cur_conf:
                                results[idx] = (digits, conf)
        except Exception as e:
            # Fallback to single if batch fails
            for i, idx in enumerate(valid_indices):
                results[idx] = self.detect_number(crops[idx])

        return results

    def detect_number(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Read jersey number from a pre-cropped region (shoulder→hip keypoint crop
        from tracker, or full player bbox as fallback).

        Returns ("number_str", confidence) or ("?", 0.0).
        """
        if crop is None or crop.size == 0:
            return "?", 0.0

        try:
            resized = cv2.resize(crop, (128, 256))
        except Exception:
            return "?", 0.0

        number_region = None
        conf = 0.5

        # Stage 1: optional fine-tuned YOLO region detector
        if self._detector is not None:
            try:
                preds = self._detector.predict(source=resized, verbose=False, conf=0.25)
                if preds and preds[0].boxes is not None and len(preds[0].boxes) > 0:
                    boxes = preds[0].boxes
                    best_idx = int(boxes.conf.argmax().item())
                    best_conf = float(boxes.conf[best_idx].item())
                    if best_conf >= self.CONFIDENCE_THRESHOLD:
                        h, w = resized.shape[:2]
                        x1, y1, x2, y2 = [int(v.item()) for v in boxes.xyxy[best_idx]]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        if x2 > x1 and y2 > y1:
                            number_region = resized[y1:y2, x1:x2]
                            conf = best_conf
            except Exception:
                pass

        if number_region is None:
            # Already a pose-guided crop — use as-is (don't re-slice the torso)
            number_region = resized

        number_str = self._ocr_region(number_region)
        return (number_str, conf) if number_str else ("?", 0.0)

    # ------------------------------------------------------------------
    # OCR dispatch
    # ------------------------------------------------------------------

    def _ocr_region(self, region: np.ndarray) -> str:
        self._load_ocr()
        if self._ocr is None:
            return ""

        if self._ocr_engine == "parseq":
            return self._parseq_read(region)
        if self._ocr_engine == "paddle":
            return self._paddle_read(region)
        return self._easyocr_read(region)

    def _parseq_read(self, region: np.ndarray) -> str:
        """PARSeq scene-text recognition — CVPR 2024, optimized for sports numbers."""
        import torch
        import re
        from PIL import Image

        h, w = region.shape[:2]
        scale = max(2, 64 // max(h, 1))
        enlarged = cv2.resize(region, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        best_text = ""
        best_conf = 0.0

        for img_arr in [enlarged, cv2.bitwise_not(enlarged)]:
            try:
                pil = Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
                tensor = self._ocr_transform(pil).unsqueeze(0).cuda().half()  # FP16
                with torch.no_grad():
                    logits = self._ocr(tensor)
                    probs = logits.softmax(-1)
                    preds, confs = self._ocr.tokenizer.decode(probs)
                text = preds[0] if preds else ""
                conf = float(confs[0].mean()) if confs else 0.0
                digits = re.sub(r"\D", "", text)
                if digits and 1 <= len(digits) <= 2 and conf > best_conf:
                    best_conf = conf
                    best_text = digits
            except Exception:
                pass

        return best_text if best_conf >= 0.3 else ""

    def _paddle_read(self, region: np.ndarray) -> str:
        """PaddleOCR v3.4 recognition using predict(), digit filter in post-process."""
        h, w = region.shape[:2]
        scale = max(2, 96 // max(h, 1))
        enlarged = cv2.resize(region, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        best_text = ""
        best_conf = 0.0
        for img in [enlarged, cv2.bitwise_not(enlarged)]:
            try:
                results = self._ocr.predict(img)
                # v3.4 returns a list of result dicts per image
                for res in (results or []):
                    if not isinstance(res, dict):
                        continue
                    for text, score in zip(
                        res.get("rec_text", []),
                        res.get("rec_score", []),
                    ):
                        digits = re.sub(r"\D", "", str(text))
                        if digits and 1 <= len(digits) <= 2 and float(score) > best_conf:
                            best_conf = float(score)
                            best_text = digits
            except Exception:
                pass

        return best_text if best_conf >= 0.3 else ""

    def _easyocr_read(self, region: np.ndarray) -> str:
        """EasyOCR recognition with preprocessing."""
        h, w = region.shape[:2]
        scale = max(2, 96 // max(h, 1))
        enlarged = cv2.resize(region, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

        results_combined = []
        for img_variant in [gray, cv2.bitwise_not(gray)]:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(img_variant)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            try:
                res = self._ocr.readtext(
                    sharpened,
                    allowlist="0123456789",
                    detail=1,
                    paragraph=False,
                    min_size=5,
                    text_threshold=0.4,
                    low_text=0.3,
                    link_threshold=0.2,
                )
                results_combined.extend(res)
            except Exception:
                pass

        if not results_combined:
            return ""

        results_combined.sort(key=lambda r: r[2], reverse=True)
        for _, text, conf in results_combined:
            digits = re.sub(r"\D", "", text)
            if digits and 1 <= len(digits) <= 2 and conf >= 0.35:
                return digits
        return ""
