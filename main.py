"""
main.py — FastAPI server for the Basketball Stat Tracker.

Endpoints:
  POST /api/process          → start a processing job
  GET  /api/jobs/{job_id}    → poll job status
  WS   /ws/{job_id}          → live updates (progress + stats)
  GET  /api/video/{job_id}   → serve downloaded video (with Range support)
  GET  /api/export/{job_id}  → download CSV of final boxscore
  GET  /api/events/{job_id}  → get full event log as JSON
"""

import asyncio
import csv
import io
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

import aiofiles
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from processor import VideoProcessor, VIDEOS_DIR

# ---------------------------------------------------------------------------
app = FastAPI(title="Basketball Stat Tracker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory job registry
# ---------------------------------------------------------------------------
jobs: Dict[str, Dict[str, Any]] = {}

# Singleton processor (loads YOLO models once)
processor = VideoProcessor(frame_sample_rate=6)


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self._connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, job_id: str, ws: WebSocket):
        await ws.accept()
        self._connections.setdefault(job_id, []).append(ws)

    def disconnect(self, job_id: str, ws: WebSocket):
        conns = self._connections.get(job_id, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, job_id: str, message: dict):
        conns = self._connections.get(job_id, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            conns.remove(ws)


ws_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ProcessRequest(BaseModel):
    url: str
    mode: str = "all"            # "single" | "team" | "all"
    jersey: Optional[str] = ""  # for mode=single
    team: Optional[str] = ""    # for mode=team ("home" | "away")
    frame_sample_rate: int = 6
    start_time: Optional[float] = None  # seconds into video to start processing
    end_time: Optional[float] = None    # seconds into video to stop processing


class JobStatus(BaseModel):
    job_id: str
    status: str
    percentage: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/process")
async def start_processing(req: ProcessRequest):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "url": req.url,
        "mode": req.mode,
        "options": {"jersey": req.jersey, "team": req.team},
        "stats": None,
        "events": None,
        "video_path": None,
        "fps": 30.0,
        "error": None,
    }

    # Adjust frame sample rate if requested
    processor.frame_sample_rate = max(1, req.frame_sample_rate)

    # Launch as background task so the HTTP response returns immediately
    asyncio.create_task(
        processor.process(
            job_id=job_id,
            url=req.url,
            mode=req.mode,
            options={"jersey": req.jersey or "", "team": req.team or ""},
            ws_manager=ws_manager,
            jobs=jobs,
            start_time=req.start_time,
            end_time=req.end_time,
        )
    )

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "error": job.get("error"),
        "stats": job.get("stats"),
        "has_video": job.get("video_path") is not None,
    }


@app.get("/api/events/{job_id}")
async def get_events(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"events": jobs[job_id].get("events", [])}


@app.get("/api/export/{job_id}")
async def export_csv(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    stats = job.get("stats") or {}

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Jersey", "Team", "PTS", "REB", "AST", "BLK", "STL"])

    for track_id, p in stats.items():
        writer.writerow([
            p.get("jersey_number", "?"),
            p.get("team", ""),
            p.get("points", 0),
            p.get("rebounds", 0),
            p.get("assists", 0),
            p.get("blocks", 0),
            p.get("steals", 0),
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=boxscore_{job_id[:8]}.csv"},
    )


# ---------------------------------------------------------------------------
# Video streaming with HTTP Range support (needed for seekable HTML5 player)
# ---------------------------------------------------------------------------

@app.get("/api/video/{job_id}")
async def stream_video(request: Request, job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    video_path = jobs[job_id].get("video_path")
    if not video_path or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video not available yet")

    file_size = Path(video_path).stat().st_size
    range_header = request.headers.get("Range")

    start = 0
    end = file_size - 1
    status_code = 200

    if range_header:
        range_match = range_header.strip().replace("bytes=", "")
        parts = range_match.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        status_code = 206

    chunk_size = 1024 * 1024  # 1 MB chunks

    async def file_generator():
        async with aiofiles.open(video_path, "rb") as f:
            await f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                data = await f.read(read_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Content-Type": "video/mp4",
    }

    return StreamingResponse(
        file_generator(),
        status_code=status_code,
        headers=headers,
        media_type="video/mp4",
    )


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await ws_manager.connect(job_id, websocket)
    try:
        # Send current job state immediately on connect
        if job_id in jobs:
            job = jobs[job_id]
            await websocket.send_json({
                "type": "connected",
                "data": {
                    "job_id": job_id,
                    "status": job["status"],
                    "stats": job.get("stats"),
                },
            })
            # If job already completed, send final result
            if job["status"] == "complete":
                await websocket.send_json({
                    "type": "complete",
                    "data": {
                        "players": job.get("stats", {}),
                        "all_players": job.get("all_stats", {}),
                        "events": job.get("events", []),
                    },
                })

        # Keep connection alive
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        ws_manager.disconnect(job_id, websocket)
    except Exception:
        ws_manager.disconnect(job_id, websocket)
