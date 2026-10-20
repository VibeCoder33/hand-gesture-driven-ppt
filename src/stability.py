import time
from collections import deque

SLIDE_GESTURES = ("NEXT", "PREVIOUS")
ARM_GESTURE = "ARM"
POINTER_GESTURE = "POINTER"
# Backwards compatibility alias
TOGGLE_GESTURE = ARM_GESTURE


class GestureFilter:
    """Turns noisy per-frame predictions into reliable, one-shot actions and mode switches.

    Mode state machine:
      DISARMED --(ARM hold)--> ARMED --(POINTER hold)--> POINTER MODE --(POINTER hold)--> ARMED --(ARM hold)--> DISARMED

    Rules:
      1. While DISARMED: only ARM (L sign) is listened to. Slides and POINTER are ignored.
      2. While ARMED (slides mode): ARM toggles DISARMED, POINTER toggles POINTER mode, NEXT/PREVIOUS trigger slides.
      3. While POINTER MODE: only POINTER (Shaka) is listened to. ARM and slides are ignored.
         (Prevents pointing hand with loose thumb from locking/disarming the system).
      4. Safety timeout: if hand disappears for >= pointer_timeout (3s), pointer mode auto-disengages back to ARMED.
      5. Latch: every gesture fires once per continuous hold and must be released before firing again.
    """

    def __init__(self, confidence_threshold=0.75,
                 slide_hold_time=0.5, toggle_hold_time=0.8, pointer_hold_time=0.35,
                 slide_cooldown=1.0, toggle_cooldown=2.0, pointer_cooldown=0.8,
                 pointer_timeout=3.0, vote_window=0.25, vote_ratio=0.7, min_votes=3):
        self.confidence_threshold = confidence_threshold
        self.slide_hold_time = slide_hold_time
        self.toggle_hold_time = toggle_hold_time
        self.pointer_hold_time = pointer_hold_time
        self.slide_cooldown = slide_cooldown
        self.toggle_cooldown = toggle_cooldown
        self.pointer_cooldown = pointer_cooldown
        self.pointer_timeout = pointer_timeout
        self.vote_window = vote_window
        self.vote_ratio = vote_ratio
        self.min_votes = min_votes

        self.is_armed = False
        self.is_pointer_mode = False
        self.last_slide_time = -1e9
        self.last_toggle_time = -1e9
        self.last_pointer_time = -1e9
        self.last_hand_seen_time = -1e9
        self.reset()

    # ---- state helpers -------------------------------------------------
    def reset(self):
        """Call to reset the vote history, hold timer and latch. Keeps armed and pointer mode states."""
        self.history = deque()
        self.current_gesture = None
        self.gesture_start_time = 0.0
        self.latched = None

    def no_hand(self, now=None):
        """Called when no hand is detected in a frame.
        Resets transient filters and enforces the 3.0s safety timeout in pointer mode."""
        now = time.monotonic() if now is None else now
        self.reset()
        if self.is_pointer_mode and (now - self.last_hand_seen_time >= self.pointer_timeout):
            self.is_pointer_mode = False
            return "POINTER TIMEOUT (OFF)"
        return None

    def slide_cooldown_left(self, now=None):
        now = time.monotonic() if now is None else now
        return max(0.0, self.slide_cooldown - (now - self.last_slide_time))

    def toggle_cooldown_left(self, now=None):
        now = time.monotonic() if now is None else now
        return max(0.0, self.toggle_cooldown - (now - self.last_toggle_time))

    def pointer_cooldown_left(self, now=None):
        now = time.monotonic() if now is None else now
        return max(0.0, self.pointer_cooldown - (now - self.last_pointer_time))

    def pointer_timeout_left(self, now=None):
        """Seconds left before pointer mode automatically times out when hand is absent."""
        if not self.is_pointer_mode or self.last_hand_seen_time <= 0:
            return 0.0
        now = time.monotonic() if now is None else now
        elapsed = now - self.last_hand_seen_time
        return max(0.0, self.pointer_timeout - elapsed)

    def hold_progress(self, now=None):
        """0..1 progress of the current hold, for the on-screen bar."""
        if self.current_gesture is None:
            return 0.0
        now = time.monotonic() if now is None else now
        need = self._hold_time(self.current_gesture)
        return min(1.0, (now - self.gesture_start_time) / need)

    def _hold_time(self, gesture):
        if gesture == ARM_GESTURE:
            return self.toggle_hold_time
        elif gesture == POINTER_GESTURE:
            return self.pointer_hold_time
        return self.slide_hold_time

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
        """Feed one frame's prediction while hand is visible.
        Returns:
          - 'NEXT' or 'PREVIOUS' (slide command)
          - 'SYSTEM ARMED' or 'SYSTEM DISARMED'
          - 'POINTER MODE ON' or 'POINTER MODE OFF'
          - None
        """
        now = time.monotonic() if now is None else now
        self.last_hand_seen_time = now

        # Determine if predicted gesture is recognized in current mode
        is_valid = False
        if not self.is_armed:
            # DISARMED: only ARM (L sign) is recognized
            is_valid = (predicted_gesture == ARM_GESTURE)
        elif self.is_pointer_mode:
            # POINTER MODE: ONLY POINTER (Shaka) is recognized.
            # ARM and slides are strictly ignored to prevent lockouts while pointing.
            is_valid = (predicted_gesture == POINTER_GESTURE)
        else:
            # ARMED (slides mode): ARM, POINTER, and slides are recognized
            is_valid = (predicted_gesture == ARM_GESTURE or
                        predicted_gesture == POINTER_GESTURE or
                        predicted_gesture in SLIDE_GESTURES)

        label = predicted_gesture if (is_valid and confidence >= self.confidence_threshold) else "NEUTRAL"
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

        # Check cooldown
        if stable == ARM_GESTURE and self.toggle_cooldown_left(now) > 0:
            self.current_gesture = None
            return None
        elif stable == POINTER_GESTURE and self.pointer_cooldown_left(now) > 0:
            self.current_gesture = None
            return None
        elif stable in SLIDE_GESTURES and self.slide_cooldown_left(now) > 0:
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

        if stable == ARM_GESTURE:
            self.is_armed = not self.is_armed
            self.last_toggle_time = now
            self.last_slide_time = now      # no slide right after (dis)arming
            return f"SYSTEM {'ARMED' if self.is_armed else 'DISARMED'}"

        if stable == POINTER_GESTURE:
            self.is_pointer_mode = not self.is_pointer_mode
            self.last_pointer_time = now
            self.last_slide_time = now
            return f"POINTER MODE {'ON' if self.is_pointer_mode else 'OFF'}"

        if not self.is_armed or self.is_pointer_mode:
            return None

        self.last_slide_time = now
        return stable
