"""
utils/alert.py
==============
Alert Level Determination and Event Logging

WHERE IT IS USED:
  - Called at the end of every frame's processing pipeline in app.py
  - Receives the ANN fatigue score and individual metric values
  - Returns current alert level (0–3) and a message for the frontend

HOW IT WORKS:
  Alert Level is determined by a combination of:
  1. ANN fatigue score (primary signal — weighted combination of all features)
  2. Individual metric overrides (for edge cases the ANN might miss)

  Level thresholds:
    0 — Normal    : score < 25  AND  EAR OK  AND  PERCLOS < 20%
    1 — Mild      : score 25–44  OR  PERCLOS 20–35%
    2 — Moderate  : score 45–69  OR  PERCLOS 35–55%  OR  consecutive closed > 20 frames
    3 — Critical  : score >= 70  OR  PERCLOS > 55%   OR  consecutive closed > 45 frames
"""

import time
from collections import deque


ALERT_LEVELS = {
    0: {
        "name":   "Normal",
        "color":  "#00C68A",
        "desc":   "Driver is alert — all metrics normal",
        "sound":  None,
    },
    1: {
        "name":   "Mild Fatigue",
        "color":  "#FFAA00",
        "desc":   "Early signs of fatigue — please stay alert",
        "sound":  "mild",
    },
    2: {
        "name":   "Moderate Drowsiness",
        "color":  "#FF7800",
        "desc":   "Significant drowsiness detected — consider pulling over",
        "sound":  "moderate",
    },
    3: {
        "name":   "CRITICAL — Microsleep Risk",
        "color":  "#FF4D4D",
        "desc":   "SEVERE DROWSINESS — Pull over immediately!",
        "sound":  "critical",
    },
}


class AlertManager:
    """
    Manages alert level transitions with hysteresis to prevent
    rapid flickering between levels.

    Hysteresis: Level only increases immediately, but must stay
    lower for N consecutive frames before decreasing.
    This prevents a single good frame from clearing a drowsy alert.
    """

    def __init__(self, hysteresis_frames=15):
        """
        Args:
            hysteresis_frames: How many consecutive lower-level frames
                               needed before alert level decreases
        """
        self.hysteresis_frames    = hysteresis_frames
        self.current_level        = 0
        self._lower_frame_count   = 0
        self._event_log           = deque(maxlen=200)
        self.total_alerts         = 0
        self._last_alert_time     = 0
        self._alert_cooldown_sec  = 3.0   # Min seconds between alert triggers

    def update(self, fatigue_score: float, ear: float,
               perclos: float, consecutive_closed: int) -> dict:
        """
        Compute alert level for current frame.

        Args:
            fatigue_score:       ANN output 0–100
            ear:                 current Eye Aspect Ratio
            perclos:             rolling PERCLOS %
            consecutive_closed:  frames eye has been continuously closed

        Returns:
            dict with level (int), info dict, and whether it's a new alert
        """
        # ── Compute raw level from multiple signals ──────────────────────
        raw_level = 0

        # Primary: ANN fatigue score
        if   fatigue_score >= 70: raw_level = 3
        elif fatigue_score >= 45: raw_level = 2
        elif fatigue_score >= 25: raw_level = 1

        # Override 1: PERCLOS threshold
        if   perclos > 55: raw_level = max(raw_level, 3)
        elif perclos > 35: raw_level = max(raw_level, 2)
        elif perclos > 20: raw_level = max(raw_level, 1)

        # Override 2: Prolonged eye closure (microsleep)
        if   consecutive_closed > 45: raw_level = max(raw_level, 3)
        elif consecutive_closed > 20: raw_level = max(raw_level, 2)

        # Override 3: Very low EAR sustained
        if ear < 0.18 and ear > 0:    raw_level = max(raw_level, 2)

        # ── Hysteresis: only decrease level after N consecutive lower frames ──
        is_new_alert = False

        if raw_level >= self.current_level:
            # Increase immediately
            if raw_level > self.current_level:
                is_new_alert = True
                self.total_alerts += 1
                self._log_event(raw_level, fatigue_score, perclos, ear)
            self.current_level      = raw_level
            self._lower_frame_count = 0
        else:
            # Need hysteresis_frames consecutive lower readings to decrease
            self._lower_frame_count += 1
            if self._lower_frame_count >= self.hysteresis_frames:
                self.current_level      = raw_level
                self._lower_frame_count = 0

        return {
            "level":        self.current_level,
            "info":         ALERT_LEVELS[self.current_level],
            "is_new_alert": is_new_alert,
            "raw_level":    raw_level,
        }

    def _log_event(self, level: int, fatigue: float, perclos: float, ear: float):
        """Internal event logger."""
        self._event_log.append({
            "timestamp": time.time(),
            "time_str":  time.strftime("%H:%M:%S"),
            "level":     level,
            "level_name": ALERT_LEVELS[level]["name"],
            "fatigue":   round(fatigue, 1),
            "perclos":   round(perclos, 1),
            "ear":       round(ear, 3),
        })

    def get_event_log(self) -> list:
        """Return list of logged alert events (newest first)."""
        return list(reversed(self._event_log))

    def reset(self):
        """Reset alert state for a new session."""
        self.current_level        = 0
        self._lower_frame_count   = 0
        self.total_alerts         = 0
        self._event_log.clear()
