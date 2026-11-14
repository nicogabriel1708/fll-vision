"""
Thread-safe Match Timer & Livestream State Manager for FLL-Vision.
Faithfully implements the exact timer phases and rules of `fll-robot-game-livestream`:
  - TeamColor.RED (left) and TeamColor.BLUE (right)
  - TimerPhase.PRE (10s pre-countdown) -> TimerPhase.MAIN (150s match countdown)
  - Sound effect triggers:
      * PRE_COUNT (low.mp3)  : PRE phase, totalSeconds in {3, 2, 1}
      * MATCH_START (high.mp3): MAIN phase, totalSeconds == duration (150)
      * MAIN_COUNT (beep.mp3): MAIN phase, totalSeconds in {5, 4, 3, 2, 1}
      * MATCH_END (buzz.mp3) : MAIN phase, totalSeconds == 0 (or match aborted)
"""
import math
import threading
import time
from typing import Dict, Any, Optional, List


class MatchStateManager:
    def __init__(
        self,
        default_duration_sec: int = 150,
        pre_count_enabled: bool = True,
        pre_count_duration_sec: int = 10,
    ):
        self._lock = threading.Lock()
        self.round_name: str = "Robot Game Match"
        self.red_team_name: str = "BIOGLOW Builders"
        self.blue_team_name: str = "RoboSpark Prime"
        self.team_left: Dict[str, str] = {
            "number": "101",
            "name": self.red_team_name,
            "organization": "HTL Hollabrunn",
        }
        self.team_right: Dict[str, str] = {
            "number": "102",
            "name": self.blue_team_name,
            "organization": "Gymnasium Hollabrunn",
        }
        self.uploaded_team_list: List[str] = []

        self.duration_sec: int = default_duration_sec
        self.pre_count_enabled: bool = pre_count_enabled
        self.pre_count_duration_sec: int = pre_count_duration_sec

        self.has_timer_started: bool = False
        self.timer_phase: str = "pre" if pre_count_enabled else "main"
        # Status values: 'idle' (Waiting screen), 'ready' (Match Stream shown, timer not started),
        #                'running', 'paused', 'finished', 'aborted'
        self.status: str = "idle"

        self._phase_duration: float = float(pre_count_duration_sec if pre_count_enabled else default_duration_sec)
        self._accumulated_elapsed: float = 0.0
        self._started_monotonic: Optional[float] = None

        self.swap_streams: bool = False
        self.audio_enabled: bool = True
        self.last_action: str = "init"
        self.updated_at: float = time.time()

    def _reset_timer_vars_unlocked(self):
        self.has_timer_started = False
        self.timer_phase = "pre" if self.pre_count_enabled else "main"
        self._phase_duration = float(
            self.pre_count_duration_sec if self.pre_count_enabled else self.duration_sec
        )
        self._accumulated_elapsed = 0.0
        self._started_monotonic = None

    def _advance_clock_unlocked(self) -> float:
        if self.status == "running" and self._started_monotonic is not None:
            now_mono = time.monotonic()
            elapsed = self._accumulated_elapsed + (now_mono - self._started_monotonic)
        else:
            elapsed = self._accumulated_elapsed

        rem = self._phase_duration - elapsed

        # Handle automatic transition from PRE (10s) -> MAIN (150s)
        if rem <= 0.0 and self.status == "running":
            if self.timer_phase == "pre":
                overflow = max(0.0, -rem)
                self.timer_phase = "main"
                self._phase_duration = float(self.duration_sec)
                self._accumulated_elapsed = overflow
                self._started_monotonic = time.monotonic()
                rem = max(0.0, self._phase_duration - overflow)
                self.last_action = "transition_to_main"
                self.updated_at = time.time()
            else:
                rem = 0.0
                self.status = "finished"
                self._accumulated_elapsed = self._phase_duration
                self._started_monotonic = None
                self.last_action = "timer_finished"
                self.updated_at = time.time()

        return max(0.0, rem)

    def select_match(
        self,
        round_name: str = "",
        team_left: Optional[Dict[str, str]] = None,
        team_right: Optional[Dict[str, str]] = None,
        red_team_name: Optional[str] = None,
        blue_team_name: Optional[str] = None,
        duration_sec: Optional[int] = None,
        pre_count_enabled: Optional[bool] = None,
        pre_count_duration_sec: Optional[int] = None,
        switch_to_ready: bool = True,
    ) -> Dict[str, Any]:
        with self._lock:
            if round_name:
                self.round_name = round_name.strip()
            if red_team_name is not None:
                self.red_team_name = red_team_name.strip()
                self.team_left["name"] = self.red_team_name
            if blue_team_name is not None:
                self.blue_team_name = blue_team_name.strip()
                self.team_right["name"] = self.blue_team_name

            if team_left:
                name_l = str(team_left.get("name", self.red_team_name)).strip()
                self.red_team_name = name_l
                self.team_left = {
                    "number": str(team_left.get("number", "")).strip(),
                    "name": name_l,
                    "organization": str(team_left.get("organization", "")).strip(),
                }
            if team_right:
                name_r = str(team_right.get("name", self.blue_team_name)).strip()
                self.blue_team_name = name_r
                self.team_right = {
                    "number": str(team_right.get("number", "")).strip(),
                    "name": name_r,
                    "organization": str(team_right.get("organization", "")).strip(),
                }

            if duration_sec is not None and duration_sec > 0:
                self.duration_sec = int(duration_sec)
            if pre_count_enabled is not None:
                self.pre_count_enabled = bool(pre_count_enabled)
            if pre_count_duration_sec is not None and pre_count_duration_sec >= 0:
                self.pre_count_duration_sec = int(pre_count_duration_sec)

            if switch_to_ready and self.status in ("idle", "finished", "aborted", "ready"):
                self.status = "ready"
                self._reset_timer_vars_unlocked()

            self.last_action = "select_match"
            self.updated_at = time.time()
            return self._snapshot_unlocked()

    def set_uploaded_team_list(self, team_names: List[str]) -> List[str]:
        with self._lock:
            cleaned = [t.strip() for t in team_names if t and t.strip()]
            self.uploaded_team_list = cleaned
            self.updated_at = time.time()
            return list(self.uploaded_team_list)

    def start_match(self) -> Dict[str, Any]:
        with self._lock:
            if self.status in ("idle", "ready", "finished", "aborted"):
                self._reset_timer_vars_unlocked()
            if self.status != "running":
                self.has_timer_started = True
                self._started_monotonic = time.monotonic()
                self.status = "running"
                self.last_action = "start"
                self.updated_at = time.time()
            return self._snapshot_unlocked()

    def pause_match(self) -> Dict[str, Any]:
        with self._lock:
            if self.status == "running" and self._started_monotonic is not None:
                self._advance_clock_unlocked()
                if self.status == "running" and self._started_monotonic is not None:
                    self._accumulated_elapsed += time.monotonic() - self._started_monotonic
                    self._started_monotonic = None
                    self.status = "paused"
                    self.last_action = "pause"
                    self.updated_at = time.time()
            return self._snapshot_unlocked()

    def toggle_start_pause(self) -> Dict[str, Any]:
        with self._lock:
            is_running = self.status == "running"
        if is_running:
            return self.pause_match()
        return self.start_match()

    def restart_match(self) -> Dict[str, Any]:
        with self._lock:
            self._reset_timer_vars_unlocked()
            self.status = "ready"
            self.last_action = "restart"
            self.updated_at = time.time()
            return self._snapshot_unlocked()

    def abort_match(self) -> Dict[str, Any]:
        with self._lock:
            if self.status == "running" and self._started_monotonic is not None:
                self._accumulated_elapsed += time.monotonic() - self._started_monotonic
            self._started_monotonic = None
            self.status = "aborted"
            self.last_action = "abort"
            self.updated_at = time.time()
            return self._snapshot_unlocked()

    def set_idle(self, idle: bool = True) -> Dict[str, Any]:
        with self._lock:
            if idle:
                self._reset_timer_vars_unlocked()
                self.status = "idle"
                self.last_action = "show_idle"
            else:
                if self.status == "idle":
                    self.status = "ready"
                    self.last_action = "hide_idle"
            self.updated_at = time.time()
            return self._snapshot_unlocked()

    def adjust_time(self, delta_sec: int) -> Dict[str, Any]:
        with self._lock:
            if not self.has_timer_started or self.status in ("ready", "idle"):
                self.duration_sec = max(10, min(900, self.duration_sec + int(delta_sec)))
                if not self.pre_count_enabled:
                    self._phase_duration = float(self.duration_sec)
            else:
                self._accumulated_elapsed = max(
                    0.0,
                    min(self._phase_duration, self._accumulated_elapsed - float(delta_sec)),
                )
            self.last_action = "adjust_time"
            self.updated_at = time.time()
            return self._snapshot_unlocked()

    def set_duration(
        self,
        duration_sec: int,
        pre_count_enabled: Optional[bool] = None,
        pre_count_duration_sec: Optional[int] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            self.duration_sec = max(5, min(1800, int(duration_sec)))
            if pre_count_enabled is not None:
                self.pre_count_enabled = bool(pre_count_enabled)
            if pre_count_duration_sec is not None:
                self.pre_count_duration_sec = max(0, min(60, int(pre_count_duration_sec)))
            if self.status in ("ready", "idle", "finished", "aborted"):
                self._reset_timer_vars_unlocked()
            self.last_action = "set_duration"
            self.updated_at = time.time()
            return self._snapshot_unlocked()

    def toggle_swap_streams(self) -> Dict[str, Any]:
        with self._lock:
            self.swap_streams = not self.swap_streams
            self.last_action = "swap_streams"
            self.updated_at = time.time()
            return self._snapshot_unlocked()

    def toggle_audio(self) -> Dict[str, Any]:
        with self._lock:
            self.audio_enabled = not self.audio_enabled
            self.last_action = "toggle_audio"
            self.updated_at = time.time()
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> Dict[str, Any]:
        rem = self._advance_clock_unlocked()
        total_seconds = int(math.ceil(rem - 1e-4)) if self.status == "running" else int(round(rem))
        minutes = total_seconds // 60
        seconds = total_seconds % 60

        # Exact display logic from Timer.jsx:
        # Before timer starts (!hasTimerStarted), display [Math.floor(duration / 60), duration % 60]
        if self.has_timer_started:
            display_minutes = minutes
            display_seconds = seconds
        else:
            display_minutes = self.duration_sec // 60
            display_seconds = self.duration_sec % 60

        is_timer_red = bool(
            self.has_timer_started
            and (
                self.timer_phase == "pre"
                or (self.timer_phase == "main" and minutes == 0 and seconds <= 10)
                or self.status == "aborted"
            )
        )

        return {
            "round_name": self.round_name,
            "teamNames": {
                "red": self.red_team_name,
                "blue": self.blue_team_name,
            },
            "team_left": self.team_left,
            "team_right": self.team_right,
            "uploaded_team_list": self.uploaded_team_list,
            "duration_sec": self.duration_sec,
            "pre_count_enabled": self.pre_count_enabled,
            "pre_count_duration_sec": self.pre_count_duration_sec,
            "has_timer_started": self.has_timer_started,
            "timer_phase": self.timer_phase,
            "total_seconds": total_seconds,
            "minutes": minutes,
            "seconds": seconds,
            "display_minutes": display_minutes,
            "display_seconds": display_seconds,
            "is_timer_red": is_timer_red,
            "remaining_sec": round(rem, 2),
            "formatted_time": f"{display_minutes}:{display_seconds:02d}",
            "status": self.status,
            "swap_streams": self.swap_streams,
            "audio_enabled": self.audio_enabled,
            "last_action": self.last_action,
            "updated_at": self.updated_at,
        }

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()
