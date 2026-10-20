"""Geometric pointer controller for PowerPoint presentations.

Maps the index fingertip position within an expanded camera region-of-interest (ROI)
to the primary display, using absolute coordinate mapping, velocity-adaptive smoothing,
and a low-latency dead zone. The cursor is only updated while the hand is in an index-pointing shape.
"""
import math
import sys
import cv2
import pyautogui

from src.rules import is_pointing_pose

# Fast Win32 cursor positioning on Windows for 0ms latency
if sys.platform == "win32":
    import ctypes
    _set_cursor_pos = ctypes.windll.user32.SetCursorPos
else:
    _set_cursor_pos = None

# PyAutoGUI safety configuration
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


class PointerController:
    """Controls cursor movement using MediaPipe index fingertip geometry with low latency."""

    def __init__(self, roi=(0.08, 0.08, 0.92, 0.92), alpha=0.55, dead_zone=2.0,
                 safe_margin=5, screen_size=None):
        """
        roi: (x_min, y_min, x_max, y_max) normalized region in camera image.
             Default (0.08, 0.08, 0.92, 0.92) provides a wide, spacious 84% x 84% pointing area.
        alpha: base exponential smoothing weight (0 < alpha <= 1). Default 0.55 for crisp response.
        dead_zone: minimum movement in screen pixels before cursor moves. Default 2.0 px.
        safe_margin: minimum distance in pixels from screen edges to avoid (0,0) failsafe.
        screen_size: optional (width, height) tuple for testing/headless environments.
        """
        self.x_min, self.y_min, self.x_max, self.y_max = roi
        self.alpha = float(alpha)
        self.dead_zone = float(dead_zone)
        self.safe_margin = int(safe_margin)

        if screen_size is not None:
            self.screen_width, self.screen_height = screen_size
        else:
            try:
                self.screen_width, self.screen_height = pyautogui.size()
            except Exception:
                self.screen_width, self.screen_height = (1920, 1080)

        self.reset()

    def reset(self):
        """Reset internal filter state (e.g. when entering pointer mode or losing hand)."""
        self.smooth_x = None
        self.smooth_y = None
        self.screen_x = None
        self.screen_y = None
        self.is_pointing = False

    def freeze(self):
        """Freeze cursor position without wiping coordinates."""
        self.is_pointing = False

    def compute_screen_coords(self, norm_x, norm_y):
        """Maps normalized camera coordinates (mirrored) to screen pixels."""
        # Normalize relative to ROI
        rel_x = (norm_x - self.x_min) / (self.x_max - self.x_min + 1e-9)
        rel_y = (norm_y - self.y_min) / (self.y_max - self.y_min + 1e-9)

        # Clamp to [0.0, 1.0]
        rel_x = max(0.0, min(1.0, rel_x))
        rel_y = max(0.0, min(1.0, rel_y))

        # Map to screen dimensions
        target_x = rel_x * (self.screen_width - 1)
        target_y = rel_y * (self.screen_height - 1)

        # Edge clamping: prevent hitting (0, 0) top-left corner which triggers PyAutoGUI failsafe
        target_x = max(self.safe_margin, min(self.screen_width - self.safe_margin, target_x))
        target_y = max(self.safe_margin, min(self.screen_height - self.safe_margin, target_y))

        return target_x, target_y

    def update(self, hand_landmarks, dry_run=False):
        """Process one frame's landmarks.
        Returns: (is_pointing, (cursor_x, cursor_y) or None)
        """
        if not is_pointing_pose(hand_landmarks):
            self.freeze()
            return False, (self.screen_x, self.screen_y) if self.screen_x is not None else None

        self.is_pointing = True
        index_tip = hand_landmarks.landmark[8]
        target_x, target_y = self.compute_screen_coords(index_tip.x, index_tip.y)

        if self.smooth_x is None:
            self.smooth_x = target_x
            self.smooth_y = target_y
        else:
            # Distance moved from current smoothed position
            dist = math.hypot(target_x - self.smooth_x, target_y - self.smooth_y)
            if dist >= self.dead_zone:
                # Velocity-adaptive smoothing:
                # Fast sweeps get high alpha (up to 0.85) for instant zero-lag response.
                # Slow movements get base alpha (0.50-0.55) to suppress hand tremor.
                speed_factor = min(1.0, max(0.0, (dist - self.dead_zone) / 50.0))
                effective_alpha = self.alpha + (0.85 - self.alpha) * speed_factor

                self.smooth_x = effective_alpha * target_x + (1.0 - effective_alpha) * self.smooth_x
                self.smooth_y = effective_alpha * target_y + (1.0 - effective_alpha) * self.smooth_y

        self.screen_x = int(round(self.smooth_x))
        self.screen_y = int(round(self.smooth_y))

        if not dry_run:
            try:
                if _set_cursor_pos is not None:
                    _set_cursor_pos(self.screen_x, self.screen_y)
                else:
                    pyautogui.moveTo(self.screen_x, self.screen_y)
            except Exception as e:
                print(f"Pointer moveTo error: {e}")

        return True, (self.screen_x, self.screen_y)

    def draw_roi(self, img):
        """Draw the active mapping rectangle and state info on the video frame."""
        h, w = img.shape[:2]
        x1, y1 = int(self.x_min * w), int(self.y_min * h)
        x2, y2 = int(self.x_max * w), int(self.y_max * h)

        # Draw ROI boundary (subtle cyan/gold)
        box_color = (255, 220, 0) if self.is_pointing else (160, 160, 160)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(img, "EXPANDED POINTING REGION", (x1 + 6, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # Draw cursor target indicator if pointing
        if self.is_pointing and self.screen_x is not None:
            status_text = f"Cursor: ({self.screen_x}, {self.screen_y}) [ACTIVE]"
            cv2.putText(img, status_text, (x1 + 6, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Cursor: FROZEN (not pointing)", (x1 + 6, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
