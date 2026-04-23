"""
utils/perclos.py
================
PERCLOS = PERcentage of eye CLOSure

WHERE IT IS USED:
  - Updated every frame in app.py after EAR is computed
  - PERCLOS value is one of the 6 inputs to the ANN model
  - Also used independently in alert.py for threshold-based triggering

HOW IT WORKS:
  1. Every frame we record whether the eyes were closed (EAR < threshold)
  2. We keep a rolling window of the last WINDOW_SEC seconds of records
  3. PERCLOS = (frames where eyes closed) / (total frames in window) × 100

  Scientific basis:
    - Wierwille & Ellsworth (1994) defined PERCLOS as the gold standard
    - Normal drivers: PERCLOS 0–10%
    - Mildly drowsy:  PERCLOS 10–30%
    - Drowsy:         PERCLOS > 35%  ← alert threshold used here
    - Severely drowsy: PERCLOS > 50%
"""

import time
from collections import deque


class PERCLOSTracker:
    """
    Maintains a rolling-window PERCLOS calculator.

    Usage:
        tracker = PERCLOSTracker(window_sec=60)
        tracker.update(ear_value)
        score = tracker.get_perclos()   # returns 0.0 – 100.0
    """

    def __init__(self, window_sec=60, ear_threshold=0.25):
        """
        Args:
            window_sec:     How many seconds of history to keep (default 60s)
            ear_threshold:  EAR value below which eye is considered closed
        """
        self.window_sec    = window_sec
        self.ear_threshold = ear_threshold

        # Each entry: (timestamp_float, is_closed_bool)
        self._history = deque()

        self._total_frames  = 0
        self._closed_frames = 0

    def update(self, ear: float) -> None:
        """
        Record one frame's eye state.

        Args:
            ear: current Eye Aspect Ratio for this frame
        """
        now = time.time()
        is_closed = ear < self.ear_threshold

        # Add current frame
        self._history.append((now, is_closed))

        # Remove entries older than the window
        cutoff = now - self.window_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def get_perclos(self) -> float:
        """
        Compute PERCLOS from current window.

        Returns:
            float: 0.0–100.0  (percentage of time eyes were closed)
        """
        if not self._history:
            return 0.0

        closed = sum(1 for _, c in self._history if c)
        total  = len(self._history)
        return round((closed / total) * 100.0, 2)

    def get_window_size(self) -> int:
        """Returns number of frames currently in the window."""
        return len(self._history)

    def reset(self) -> None:
        """Clear all history."""
        self._history.clear()

    def set_threshold(self, threshold: float) -> None:
        """Update EAR threshold and recompute existing history."""
        self.ear_threshold = threshold


class BlinkTracker:
    """
    Detects individual blinks and computes blink rate per minute.

    A blink is defined as:
      - Eye closes (EAR drops below threshold)
      - Eye reopens within 400ms (not a prolonged closure = microsleep)

    Blink rate > 25/min can indicate fatigue.
    Normal resting blink rate is 12-20/min.
    """

    def __init__(self, ear_threshold=0.25, max_blink_frames=12):
        """
        Args:
            ear_threshold:    EAR below which eye is 'closed'
            max_blink_frames: Max consecutive closed frames for a blink
                              (beyond this = microsleep, not blink)
        """
        self.ear_threshold    = ear_threshold
        self.max_blink_frames = max_blink_frames

        self._eye_closed      = False
        self._closed_frames   = 0
        self._blink_times     = deque()   # timestamps of confirmed blinks
        self.total_blinks     = 0

    def update(self, ear: float) -> bool:
        """
        Process one frame. Returns True if a blink was just completed.

        Args:
            ear: current EAR value

        Returns:
            bool: True if this frame completed a blink event
        """
        is_closed = ear < self.ear_threshold
        blink_detected = False

        if is_closed:
            self._closed_frames += 1
            self._eye_closed = True
        else:
            # Eye just opened
            if self._eye_closed and 1 <= self._closed_frames <= self.max_blink_frames:
                # Valid blink — not too short, not too long (microsleep)
                blink_detected = True
                self.total_blinks += 1
                self._blink_times.append(time.time())
                # Remove blinks older than 60s
                cutoff = time.time() - 60.0
                while self._blink_times and self._blink_times[0] < cutoff:
                    self._blink_times.popleft()
            self._closed_frames = 0
            self._eye_closed = False

        return blink_detected

    def get_blink_rate(self) -> int:
        """
        Returns blink rate per minute based on last 60 seconds.

        Returns:
            int: blinks per minute
        """
        cutoff = time.time() - 60.0
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        return len(self._blink_times)

    def get_consecutive_closed_frames(self) -> int:
        """Returns how many frames the eye has been continuously closed."""
        return self._closed_frames

    def reset(self) -> None:
        self._eye_closed = False
        self._closed_frames = 0
        self._blink_times.clear()
        self.total_blinks = 0
