"""
Statistics engine — tracks points, rebounds, assists, blocks, steals
using heuristic state-machine analysis of detection results.
"""

import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any


@dataclass
class PlayerStats:
    jersey_number: str
    team: str  # 'light' or 'dark'
    points: int = 0
    rebounds: int = 0
    assists: int = 0
    blocks: int = 0
    steals: int = 0
    events: List[dict] = field(default_factory=list)

    def to_dict(self):
        return {
            "jersey_number": self.jersey_number,
            "team": self.team,
            "points": self.points,
            "rebounds": self.rebounds,
            "assists": self.assists,
            "blocks": self.blocks,
            "steals": self.steals,
        }


class StatsEngine:
    """
    Heuristic stat-tracking engine.

    Coordinates are normalised (0–1) relative to frame dimensions.
    The y-axis runs top→bottom (y=0 is top of frame).
    """

    # Tuning constants
    POSSESSION_RADIUS = 0.12       # fraction of frame — ball within this = possessed
    SHOT_UPWARD_THRESHOLD = -0.018 # ball dy per sampled frame to count as rising
    SHOT_TIMEOUT_FRAMES = 120      # ~4 s at 30 fps; abort shot if no resolution
    THREE_POINT_DIST = 0.30        # normalised distance from rim to count as 3PT
    REBOUND_WINDOW_FRAMES = 90     # frames after missed shot to award a rebound
    BLOCK_RIM_RADIUS = 0.18        # ball must be this close to rim for a block
    STEAL_MIN_FRAMES = 8           # consecutive frames defender holds ball to confirm steal

    def __init__(self):
        # track_id → PlayerStats
        self.players: Dict[str, PlayerStats] = {}
        # jersey_num → track_id  (first confident mapping wins)
        self.jersey_to_track: Dict[str, str] = {}

        # Ball position ring-buffer: (cx, cy, frame_idx)
        self.ball_history: deque = deque(maxlen=60)

        # Smoothed rim bounding box (cx, cy, w, h) normalised
        self.rim_position: Optional[Tuple[float, float, float, float]] = None

        # Possession
        self.possession: Optional[str] = None           # current track_id
        self.possession_frames: int = 0
        self.last_passer: Optional[str] = None          # for assist tracking
        self.pre_shot_passer: Optional[str] = None

        # Shot state
        self.shot_in_flight: bool = False
        self.shot_start_frame: int = 0
        self.shot_player: Optional[str] = None
        self.shot_start_pos: Optional[Tuple[float, float]] = None

        # Rebound window
        self.rebound_window: bool = False
        self.rebound_start_frame: int = 0
        self.missed_shot_team: Optional[str] = None

        # Block candidate (defensive player near rim during shot)
        self.block_candidate: Optional[str] = None

        # Global event log (ordered by frame)
        self.events: List[dict] = []
        self.fps: float = 30.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame_idx: int,
        tracked_players: List[Dict[str, Any]],
        ball_pos: Optional[Tuple[float, float]],
        rim_pos: Optional[Tuple[float, float, float, float]],
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Process one sampled frame and return the current stats snapshot.

        tracked_players items:
            {track_id, jersey_num, team, bbox:(x1,y1,x2,y2), center:(cx,cy)}
        ball_pos: (cx, cy) normalised or None
        rim_pos:  (cx, cy, w, h) normalised or None
        """
        self.fps = fps

        # Register / update players
        for p in tracked_players:
            self._update_player_registry(p)

        # Smooth rim position
        if rim_pos is not None:
            self._smooth_rim(rim_pos)

        # Ball tracking
        if ball_pos:
            self.ball_history.append((ball_pos[0], ball_pos[1], frame_idx))

        # Possession & steal detection
        if ball_pos and tracked_players:
            self._update_possession(frame_idx, tracked_players, ball_pos)

        # Shot / made / missed / block
        if ball_pos and self.rim_position:
            self._shot_state_machine(frame_idx, tracked_players, ball_pos)

        return self.get_stats_snapshot()

    def get_stats_snapshot(self) -> Dict[str, Any]:
        return {tid: p.to_dict() for tid, p in self.players.items()}

    def get_events(self) -> List[dict]:
        return self.events

    def get_player_stats_by_mode(self, mode: str, jersey: str = "", team: str = "") -> Dict[str, Any]:
        """Filter stats by the user-selected mode."""
        snap = self.get_stats_snapshot()
        if mode == "single":
            return {
                tid: s for tid, s in snap.items()
                if s["jersey_number"] == jersey
                and (not team or s["team"] == team)
            }
        elif mode == "team":
            return {tid: s for tid, s in snap.items() if s["team"] == team}
        return snap  # all

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_player_registry(self, p: dict):
        tid = p["track_id"]
        jersey = p.get("jersey_num", "?")
        team = p.get("team", "unknown")

        if tid not in self.players:
            self.players[tid] = PlayerStats(jersey_number=jersey, team=team)
        else:
            # Lock in jersey number once confidently detected
            existing = self.players[tid]
            if existing.jersey_number == "?" and jersey != "?":
                existing.jersey_number = jersey
            if jersey != "?":
                self.jersey_to_track[jersey] = tid

    def _smooth_rim(self, rim_pos: Tuple[float, float, float, float]):
        if self.rim_position is None:
            self.rim_position = rim_pos
        else:
            alpha = 0.08
            self.rim_position = tuple(
                alpha * n + (1 - alpha) * o
                for n, o in zip(rim_pos, self.rim_position)
            )

    def _update_possession(
        self,
        frame_idx: int,
        tracked_players: List[dict],
        ball_pos: Tuple[float, float],
    ):
        bx, by = ball_pos
        min_dist = float("inf")
        closest = None

        for p in tracked_players:
            cx, cy = p["center"]
            d = math.hypot(bx - cx, by - cy)
            if d < self.POSSESSION_RADIUS and d < min_dist:
                min_dist = d
                closest = p

        if closest is None:
            # Ball is loose
            if self.rebound_window and self.possession:
                # Someone just lost possession during rebound window → rebound
                pass
            self.possession_frames = 0
            return

        new_tid = closest["track_id"]

        if new_tid != self.possession:
            old_tid = self.possession
            # Steal check: possession changed, not during rebound, not off a shot
            if (
                old_tid
                and not self.rebound_window
                and not self.shot_in_flight
                and old_tid in self.players
                and new_tid in self.players
            ):
                old_team = self.players[old_tid].team
                new_team = self.players[new_tid].team
                if old_team and new_team and old_team != new_team:
                    self._record_steal(frame_idx, new_tid)

            # Rebound: gaining possession during rebound window
            if self.rebound_window and new_tid in self.players:
                self._record_rebound(frame_idx, new_tid)
                self.rebound_window = False

            # Track last passer for assist purposes
            if old_tid and not self.shot_in_flight:
                self.last_passer = old_tid

            self.possession = new_tid
            self.possession_frames = 1
        else:
            self.possession_frames += 1

    def _shot_state_machine(
        self,
        frame_idx: int,
        tracked_players: List[dict],
        ball_pos: Tuple[float, float],
    ):
        bx, by = ball_pos
        rim_cx, rim_cy, rim_w, rim_h = self.rim_position  # type: ignore

        history = list(self.ball_history)

        if not self.shot_in_flight:
            # --- Detect shot start ---
            if len(history) >= 5:
                recent = history[-5:]
                dy = recent[-1][1] - recent[0][1]  # negative = rising
                dx = abs(recent[-1][0] - recent[0][0])

                # Ball rising, has horizontal movement, shooter has possession
                if dy < self.SHOT_UPWARD_THRESHOLD and by < 0.75 and dx > 0.005:
                    if self.possession and self.possession in self.players:
                        self.shot_in_flight = True
                        self.shot_start_frame = frame_idx
                        self.shot_player = self.possession
                        self.shot_start_pos = (bx, by)
                        self.pre_shot_passer = self.last_passer

                        # Identify block candidate: closest opposing player to rim
                        self.block_candidate = self._find_block_candidate(
                            tracked_players, rim_cx, rim_cy
                        )
        else:
            # --- Track shot in flight ---
            dist_to_rim = math.hypot(bx - rim_cx, by - rim_cy)
            rim_radius = max(rim_w, rim_h)

            # Check for resolution near the rim
            if dist_to_rim < rim_radius * 2.0:
                if len(history) >= 3:
                    recent = history[-3:]
                    dy = recent[-1][1] - recent[0][1]  # positive = ball dropping

                # Ball is falling through rim region
                if by >= rim_cy and dist_to_rim < rim_radius * 1.2:
                    # Check for block: defender touched ball near rim
                    if self.block_candidate:
                        bc_pos = self._get_player_center(tracked_players, self.block_candidate)
                        if bc_pos:
                            bd = math.hypot(bc_pos[0] - rim_cx, bc_pos[1] - rim_cy)
                            if bd < self.BLOCK_RIM_RADIUS:
                                self._record_block(frame_idx, self.block_candidate)
                                self._reset_shot()
                                return

                    self._record_made_shot(frame_idx)
                    self._reset_shot()

                # Ball bouncing away from rim = missed
                elif dist_to_rim > rim_radius * 1.5 and by < rim_cy + 0.05:
                    self._record_missed_shot(frame_idx)
                    self._reset_shot()

            # Shot timeout
            elif frame_idx - self.shot_start_frame > self.SHOT_TIMEOUT_FRAMES:
                self._reset_shot()

        # Expire rebound window
        if self.rebound_window and frame_idx - self.rebound_start_frame > self.REBOUND_WINDOW_FRAMES:
            self.rebound_window = False

    def _reset_shot(self):
        self.shot_in_flight = False
        self.shot_player = None
        self.shot_start_pos = None
        self.block_candidate = None

    # ------------------------------------------------------------------
    # Event recorders
    # ------------------------------------------------------------------

    def _record_made_shot(self, frame_idx: int):
        if not self.shot_player or self.shot_player not in self.players:
            return
        player = self.players[self.shot_player]
        is_three = self._is_three_point(self.shot_start_pos)
        points = 3 if is_three else 2
        player.points += points
        ts = frame_idx / self.fps
        ev = {
            "type": "made_shot",
            "player_track": self.shot_player,
            "player_jersey": player.jersey_number,
            "team": player.team,
            "points": points,
            "is_three": is_three,
            "frame": frame_idx,
            "timestamp": round(ts, 2),
        }
        player.events.append(ev)
        self.events.append(ev)

        # Assist: last passer to scorer, same team
        if (
            self.pre_shot_passer
            and self.pre_shot_passer != self.shot_player
            and self.pre_shot_passer in self.players
        ):
            ap = self.players[self.pre_shot_passer]
            if ap.team == player.team:
                self._record_assist(frame_idx, self.pre_shot_passer)

    def _record_missed_shot(self, frame_idx: int):
        ts = frame_idx / self.fps
        player_jersey = "?"
        player_team = "unknown"
        if self.shot_player and self.shot_player in self.players:
            player_jersey = self.players[self.shot_player].jersey_number
            player_team = self.players[self.shot_player].team
        ev = {
            "type": "missed_shot",
            "player_track": self.shot_player,
            "player_jersey": player_jersey,
            "team": player_team,
            "frame": frame_idx,
            "timestamp": round(ts, 2),
        }
        self.events.append(ev)
        self.rebound_window = True
        self.rebound_start_frame = frame_idx
        self.missed_shot_team = player_team

    def _record_rebound(self, frame_idx: int, track_id: str):
        player = self.players[track_id]
        player.rebounds += 1
        ts = frame_idx / self.fps
        ev = {
            "type": "rebound",
            "player_track": track_id,
            "player_jersey": player.jersey_number,
            "team": player.team,
            "frame": frame_idx,
            "timestamp": round(ts, 2),
        }
        player.events.append(ev)
        self.events.append(ev)

    def _record_assist(self, frame_idx: int, track_id: str):
        player = self.players[track_id]
        player.assists += 1
        ts = frame_idx / self.fps
        ev = {
            "type": "assist",
            "player_track": track_id,
            "player_jersey": player.jersey_number,
            "team": player.team,
            "frame": frame_idx,
            "timestamp": round(ts, 2),
        }
        player.events.append(ev)
        self.events.append(ev)

    def _record_steal(self, frame_idx: int, track_id: str):
        player = self.players[track_id]
        player.steals += 1
        ts = frame_idx / self.fps
        ev = {
            "type": "steal",
            "player_track": track_id,
            "player_jersey": player.jersey_number,
            "team": player.team,
            "frame": frame_idx,
            "timestamp": round(ts, 2),
        }
        player.events.append(ev)
        self.events.append(ev)

    def _record_block(self, frame_idx: int, track_id: str):
        player = self.players[track_id]
        player.blocks += 1
        ts = frame_idx / self.fps
        ev = {
            "type": "block",
            "player_track": track_id,
            "player_jersey": player.jersey_number,
            "team": player.team,
            "frame": frame_idx,
            "timestamp": round(ts, 2),
        }
        player.events.append(ev)
        self.events.append(ev)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _is_three_point(self, pos: Optional[Tuple[float, float]]) -> bool:
        if not pos or not self.rim_position:
            return False
        px, py = pos
        rim_cx, rim_cy = self.rim_position[0], self.rim_position[1]
        return math.hypot(px - rim_cx, py - rim_cy) > self.THREE_POINT_DIST

    def _find_block_candidate(
        self,
        tracked_players: List[dict],
        rim_cx: float,
        rim_cy: float,
    ) -> Optional[str]:
        """Return the closest opposing-team player to the rim."""
        if not self.shot_player or self.shot_player not in self.players:
            return None
        shooter_team = self.players[self.shot_player].team
        best_dist = float("inf")
        best_tid = None
        for p in tracked_players:
            if p["track_id"] == self.shot_player:
                continue
            if p.get("team") == shooter_team:
                continue
            cx, cy = p["center"]
            d = math.hypot(cx - rim_cx, cy - rim_cy)
            if d < best_dist:
                best_dist = d
                best_tid = p["track_id"]
        return best_tid

    def _get_player_center(
        self, tracked_players: List[dict], track_id: str
    ) -> Optional[Tuple[float, float]]:
        for p in tracked_players:
            if p["track_id"] == track_id:
                return p["center"]
        return None
