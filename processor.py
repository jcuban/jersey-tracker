"""
processor.py — End-to-end video processing pipeline.

1. Downloads the YouTube video with yt-dlp.
2. Samples every N frames using grab()/retrieve() (fast — skips decode for dropped frames).
3. Single YOLO-pose call per frame (detection + tracking + keypoints).
4. Ball/rim detection via lightweight separate Detector.
5. PaddleOCR (or EasyOCR fallback) reads jersey numbers from pose-guided crops.
6. Streams progress + stat updates via WebSocket.
"""

import asyncio
import os
import queue
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from detector import Detector
from jersey_model import JerseyModel
from tracker import PlayerTracker
from stats_engine import StatsEngine

VIDEOS_DIR = Path(__file__).parent / "videos"
CACHE_DIR = Path(__file__).parent / "cache"
VIDEOS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


class VideoProcessor:
    def __init__(self, frame_sample_rate: int = 6):
        self.frame_sample_rate = frame_sample_rate
        self.detector = Detector()
        self.jersey_model = JerseyModel()
        self._url_video_cache: dict = {}  # url → existing video Path

    # ------------------------------------------------------------------
    # Public entry point (called from background task)
    # ------------------------------------------------------------------

    async def process(
        self,
        job_id: str,
        url: str,
        mode: str,
        options: Dict[str, Any],
        ws_manager,
        jobs: dict,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        """Full pipeline coroutine — runs in an asyncio executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._process_sync,
            job_id,
            url,
            mode,
            options,
            ws_manager,
            jobs,
            loop,
            start_time,
            end_time,
        )

    def _process_sync(
        self,
        job_id: str,
        url: str,
        mode: str,
        options: Dict[str, Any],
        ws_manager,
        jobs: dict,
        loop,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        """Blocking pipeline — runs in a thread pool."""
        try:
            jobs[job_id]["status"] = "downloading"
            self._send(ws_manager, job_id, loop, {
                "type": "progress",
                "data": {"stage": "Downloading video…", "percentage": 0},
            })

            video_path = self._download_video(url, job_id)
            jobs[job_id]["video_path"] = str(video_path)

            self._send(ws_manager, job_id, loop, {
                "type": "progress",
                "data": {"stage": "Opening video…", "percentage": 5},
            })

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            jobs[job_id]["fps"] = fps
            jobs[job_id]["total_frames"] = total_frames

            stats_engine = StatsEngine()
            tracker = PlayerTracker()

            jobs[job_id]["status"] = "processing"

            # ---- Clamp start/end to video bounds ----
            start_frame = int((start_time or 0) * fps)
            end_frame = int((end_time * fps)) if end_time else total_frames
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))

            jobs[job_id]["start_frame"] = start_frame
            jobs[job_id]["end_frame"] = end_frame

            # ---- Seek to start ----
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)

            # ---- Producer-consumer: CPU decodes frames, GPU runs inference ----
            # CPU thread fills frame_queue; main thread drains it on GPU.
            QUEUE_SIZE = 8  # buffer up to 8 decoded frames
            frame_queue: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
            _SENTINEL = object()

            sampled_frames = max(1, (end_frame - start_frame) // self.frame_sample_rate)
            _log_path = Path(__file__).parent / "debug_run.log"

            # ---- Smart frame skipping constants ----
            # Frames are skipped when the scene hasn't changed (timeouts, dead balls, huddles).
            # Motion score = mean absolute pixel difference between consecutive sampled frames.
            # Tuned for 1080p basketball: <4.0 = dead ball, >4.0 = active play.
            MOTION_THRESHOLD = 4.0          # below this → skip (dead ball / timeout)
            FORCE_EVERY_N = 90              # always process at least 1 frame every N sampled frames
                                            # (catches slow-moving plays that look static)

            def _reader_thread():
                """Reads + decodes frames on CPU.
                Applies smart motion-based skipping before pushing to GPU queue.
                Skips frames where the scene hasn't changed significantly.
                """
                idx = start_frame
                prev_gray = None
                frames_since_forced = 0

                while idx < end_frame:
                    grabbed = cap.grab()
                    if not grabbed:
                        break

                    if (idx - start_frame) % self.frame_sample_rate == 0:
                        ret, frame = cap.retrieve()
                        if ret:
                            # Compute motion score vs previous sampled frame
                            small = cv2.resize(frame, (160, 90))
                            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(float)

                            if prev_gray is None:
                                motion = 999.0  # always process first frame
                            else:
                                motion = float(cv2.absdiff(
                                    gray.astype("uint8"),
                                    prev_gray.astype("uint8")
                                ).mean())

                            frames_since_forced += 1
                            force = frames_since_forced >= FORCE_EVERY_N

                            if motion >= MOTION_THRESHOLD or force:
                                frame_queue.put((idx, frame, motion))
                                prev_gray = gray
                                if force:
                                    frames_since_forced = 0
                            # else: dead ball — silently skip
                    idx += 1
                frame_queue.put(_SENTINEL)

            reader = threading.Thread(target=_reader_thread, daemon=True)
            reader.start()

            processed = 0
            last_ws_update = 0
            _jersey_hits = 0
            _jersey_misses = 0

            while True:
                item = frame_queue.get()
                if item is _SENTINEL:
                    break
                frame_idx, frame, motion_score = item

                # --- Ball + rim detection (lightweight) ---
                detection = self.detector.detect(frame)

                # --- Tracking: single YOLO-pose call (persons + keypoints + ByteTrack) ---
                tracked_players = tracker.track_frame(frame, detection, self.jersey_model)

                # --- Jersey detection logging (first 500 sampled frames) ---
                if processed < 500:
                    for p in tracked_players:
                        if p["jersey_num"] != "?":
                            _jersey_hits += 1
                        else:
                            _jersey_misses += 1
                if processed == 500:
                    with open(_log_path, "w") as f:
                        f.write(f"Jersey hits: {_jersey_hits}\n")
                        f.write(f"Jersey misses: {_jersey_misses}\n")
                        f.write(f"Jersey map: {tracker.jersey_map}\n")
                        f.write(f"Team map sample: {dict(list(tracker.team_map.items())[:10])}\n")

                # --- Stats engine ---
                stats_engine.update(
                    frame_idx=frame_idx,
                    tracked_players=tracked_players,
                    ball_pos=detection.ball,
                    rim_pos=detection.rim,
                    fps=fps,
                )

                processed += 1
                pct = min(99, 5 + int(94 * processed / max(1, sampled_frames)))

                new_events = stats_engine.get_events()
                if processed - last_ws_update >= 30 or len(new_events) > 0:
                    last_ws_update = processed
                    snapshot = stats_engine.get_stats_snapshot()
                    self._send(ws_manager, job_id, loop, {
                        "type": "stats_update",
                        "data": {
                            "players": snapshot,
                            "events": new_events[-20:],
                        },
                    })
                    self._send(ws_manager, job_id, loop, {
                        "type": "progress",
                        "data": {
                            "stage": "Processing frames…",
                            "percentage": pct,
                            "frame": frame_idx,
                            "total_frames": total_frames,
                            "processed": processed,
                            "total_samples": sampled_frames,
                        },
                    })

            reader.join()

            cap.release()

            # Final result
            final_stats = stats_engine.get_player_stats_by_mode(
                mode,
                jersey=options.get("jersey", ""),
                team=options.get("team", ""),
            )
            all_events = stats_engine.get_events()

            jobs[job_id]["status"] = "complete"
            jobs[job_id]["stats"] = final_stats
            jobs[job_id]["events"] = all_events
            jobs[job_id]["all_stats"] = stats_engine.get_stats_snapshot()

            self._send(ws_manager, job_id, loop, {
                "type": "complete",
                "data": {
                    "players": final_stats,
                    "all_players": stats_engine.get_stats_snapshot(),
                    "events": all_events,
                },
            })

        except Exception as exc:
            tb = traceback.format_exc()
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(exc)
            self._send(ws_manager, job_id, loop, {
                "type": "error",
                "data": {"message": str(exc), "traceback": tb},
            })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _download_video(self, url: str, job_id: str) -> Path:
        output_path = VIDEOS_DIR / f"{job_id}.mp4"
        if output_path.exists():
            return output_path

        # Reuse already-downloaded video for the same URL
        if url in self._url_video_cache and self._url_video_cache[url].exists():
            return self._url_video_cache[url]

        # Check if any existing video file in the dir can be reused (f137 streams)
        try:
            existing = list(VIDEOS_DIR.glob("*.f137.mp4"))
            if existing:
                best = max(existing, key=lambda p: p.stat().st_size)
                self._url_video_cache[url] = best
                return best
        except Exception:
            pass

        import subprocess, sys, shutil
        # Locate yt-dlp executable (prefer our venv's copy)
        venv_ytdlp = Path(sys.executable).parent / "yt-dlp.exe"
        ytdlp_exe = str(venv_ytdlp) if venv_ytdlp.exists() else shutil.which("yt-dlp") or "yt-dlp"

        node_exe = r"C:\Program Files\nodejs\node.exe"
        cmd = [
            ytdlp_exe,
            "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--output", str(output_path),
            "--no-playlist",
            "--js-runtimes", f"node:{node_exe}",
            "--remote-components", "ejs:github",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

        if not output_path.exists():
            # yt-dlp may have added suffix — find the file
            candidates = list(VIDEOS_DIR.glob(f"{job_id}*"))
            if candidates:
                return candidates[0]
            raise FileNotFoundError(f"Download failed: {url}")

        return output_path

    @staticmethod
    def _send(ws_manager, job_id: str, loop, message: dict):
        """Thread-safe WebSocket send."""
        try:
            future = asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast(job_id, message), loop
            )
            future.result(timeout=2)
        except Exception:
            pass  # Don't crash processing if WS send fails
