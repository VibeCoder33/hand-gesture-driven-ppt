import time
from collections import deque

SLIDE_GESTURES = ("NEXT", "PREVIOUS")
TOGGLE_GESTURE = "ARM"


class GestureFilter:
    """Turns noisy per-frame predictions into reliable, one-shot actions.

    A gesture fires only if ALL of these hold:
      1. confidence >= threshold (otherwise the frame counts as NEUTRAL)
      2. it wins a majority vote over the last `vote_window` seconds
      3. it has been stable for its hold time (slide_hold_time / toggle_hold_time)
      4. its cooldown is over
      5. it hasn't already fired during this same hold (latch) - release it first
      6. for slide gestures: the system is ARMED
    """

    def __init__(self, confidence_threshold=0.75,
                 slide_hold_time=0.5, toggle_hold_time=0.8,
                 slide_cooldown=1.0, toggle_cooldown=2.0,
                 vote_window=0.4, vote_ratio=0.7, min_votes=3):
        self.confidence_threshold = confidence_threshold
        self.slide_hold_time = slide_hold_time
        self.toggle_hold_time = toggle_hold_time
        self.slide_cooldown = slide_cooldown
        self.toggle_cooldown = toggle_cooldown
        self.vote_window = vote_window
        self.vote_ratio = vote_ratio
        self.min_votes = min_votes

        self.is_armed = False
        self.reset()
        self.last_slide_time = -1e9
        self.last_toggle_time = -1e9

    # ---- state helpers -------------------------------------------------
    def reset(self):
        """Call whenever the hand disappears or a frame fails.
        Clears the vote history, hold timer and latch (not the armed state)."""
        self.history = deque()
        self.current_gesture = None
        self.gesture_start_time = 0.0
        self.latched = None

    def slide_cooldown_left(self, now=None):
        now = time.monotonic() if now is None else now
        return max(0.0, self.slide_cooldown - (now - self.last_slide_time))

    def toggle_cooldown_left(self, now=None):
        now = time.monotonic() if now is None else now
        return max(0.0, self.toggle_cooldown - (now - self.last_toggle_time))

    def hold_progress(self, now=None):
        """0..1 progress of the current hold, for the on-screen bar."""
        if self.current_gesture is None:
            return 0.0
        now = time.monotonic() if now is None else now
        need = self._hold_time(self.current_gesture)
        return min(1.0, (now - self.gesture_start_time) / need)

    def _hold_time(self, gesture):
        return self.toggle_hold_time if gesture == TOGGLE_GESTURE else self.slide_hold_time

    def _stable_gesture(self, now):
        while self.history and now - self.history[0][0] > self.vote_window:
            self.history.popleft()
        if len(self.history) < self.min_votes:
            return None
        labels = [label for _, label in self.history]
        top = max(set(labels), key=labels.count)
        return top if labels.count(top) / len(labels) >= self.vote_ratio else None

    # ---- main entry point ---------------------------------------------
    def update(self, predicted_gesture, confidence, now=None):
        """Feed one frame's prediction. Returns 'NEXT'/'PREVIOUS', a
        'SYSTEM ARMED'/'SYSTEM DISARMED' message, or None."""
        now = time.monotonic() if now is None else now

        known = predicted_gesture in SLIDE_GESTURES or predicted_gesture == TOGGLE_GESTURE
        label = predicted_gesture if (known and confidence >= self.confidence_threshold) else "NEUTRAL"
        self.history.append((now, label))
        stable = self._stable_gesture(now)

        if stable is None:                 # ambiguous votes: restart the hold, keep the latch
            self.current_gesture = None
            return None
        if stable == "NEUTRAL":            # hand released: clear hold and latch
            self.current_gesture = None
            self.latched = None
            return None
        if stable == self.latched:         # still holding a gesture that already fired
            return None
        self.latched = None

        # cooldown: ignore the gesture entirely until it is over (hold starts afterwards)
        cooling = (self.toggle_cooldown_left(now) if stable == TOGGLE_GESTURE
                   else self.slide_cooldown_left(now))
        if cooling > 0:
            self.current_gesture = None
            return None

        if stable != self.current_gesture:  # a new gesture: start its hold timer
            self.current_gesture = stable
            self.gesture_start_time = now
            return None

        if now - self.gesture_start_time < self._hold_time(stable):
            return None

        # ---- held long enough: fire ----
        self.current_gesture = None
        self.latched = stable

        if stable == TOGGLE_GESTURE:
            self.is_armed = not self.is_armed
            self.last_toggle_time = now
            self.last_slide_time = now      # no slide right after (dis)arming
            return f"SYSTEM {'ARMED' if self.is_armed else 'DISARMED'}"

        if not self.is_armed:               # slide gestures are ignored while disarmed
            return None
        self.last_slide_time = now
        return stable
