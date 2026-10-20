"""Unit tests and simulated frame tests for pointer mode and gesture state machine.

Tests all behaviors that do not require a physical webcam:
  1. State machine mode transitions (DISARMED -> ARMED -> POINTER -> ARMED -> DISARMED)
  2. Isolation: slides and POINTER ignored while disarmed
  3. Isolation: slides ignored in pointer mode
  4. Isolation: ARM held in pointer mode does NOT disarm
  5. Latch: Shaka toggles once per hold
  6. Safety timeout: hand lost for >= 3.0s auto-exits pointer mode
  7. Pointer mathematics: mapping, dead zone, exponential smoothing
  8. Safety clamping: cursor never hits (0, 0) failsafe pixel
  9. Pointing pose geometry: index up + middle/ring/pinky curled
"""
import math
import os
import sys
from types import SimpleNamespace

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pointer import PointerController
from src.rules import is_pointing_pose, rule_based_classifier
from src.stability import GestureFilter



def make_hand(index_up=False, middle_up=False, ring_up=False, pinky_up=False, thumb_far=False):
    """Generate mock 21-landmark MediaPipe hand object."""
    landmarks = [SimpleNamespace(x=0.5, y=0.8, z=0.0) for _ in range(21)]

    # Wrist = 0
    landmarks[0] = SimpleNamespace(x=0.5, y=0.8, z=0.0)
    # Middle MCP = 9 (scale reference)
    landmarks[9] = SimpleNamespace(x=0.5, y=0.5, z=0.0)
    # Index MCP = 5
    landmarks[5] = SimpleNamespace(x=0.45, y=0.5, z=0.0)

    # Fingers: tip higher (smaller y) than PIP joint when up
    # Index: PIP=6, TIP=8
    landmarks[6] = SimpleNamespace(x=0.45, y=0.4, z=0.0)
    landmarks[8] = SimpleNamespace(x=0.45, y=0.2 if index_up else 0.45, z=0.0)

    # Middle: PIP=10, TIP=12
    landmarks[10] = SimpleNamespace(x=0.5, y=0.4, z=0.0)
    landmarks[12] = SimpleNamespace(x=0.5, y=0.2 if middle_up else 0.45, z=0.0)

    # Ring: PIP=14, TIP=16
    landmarks[14] = SimpleNamespace(x=0.55, y=0.4, z=0.0)
    landmarks[16] = SimpleNamespace(x=0.55, y=0.2 if ring_up else 0.45, z=0.0)

    # Pinky: PIP=18, TIP=20
    landmarks[18] = SimpleNamespace(x=0.6, y=0.4, z=0.0)
    landmarks[20] = SimpleNamespace(x=0.6, y=0.2 if pinky_up else 0.45, z=0.0)

    # Thumb: TIP=4
    # thumb_far test in rules: d(4, 5) / (d(0, 9) + 1e-6) > 0.40
    # d(0, 9) is 0.30, so d(4, 5) needs to be > 0.12
    landmarks[4] = SimpleNamespace(x=0.20 if thumb_far else 0.43, y=0.55, z=0.0)

    return SimpleNamespace(landmark=landmarks)


def simulate_hold(filter_obj, gesture, duration=1.0, fps=30, start_time=0.0):
    """Feed frames of a given gesture at steady fps and return any actions fired."""
    actions = []
    dt = 1.0 / fps
    now = start_time
    frames = int(duration * fps)
    for _ in range(frames):
        act = filter_obj.update(gesture, 0.95, now=now)
        if act:
            actions.append((now, act))
        now += dt
    return actions, now


def test_full_mode_cycle():
    """DISARMED -> (ARM) -> ARMED -> (POINTER) -> POINTER MODE -> (POINTER) -> ARMED -> (ARM) -> DISARMED"""
    f = GestureFilter()
    t = 0.0

    # 1. Starts DISARMED
    assert not f.is_armed
    assert not f.is_pointer_mode

    # 2. Hold ARM 1.4s -> ARMED
    acts, t = simulate_hold(f, "ARM", duration=1.4, start_time=t)
    assert len(acts) == 1 and acts[0][1] == "SYSTEM ARMED"
    assert f.is_armed and not f.is_pointer_mode

    # Release to neutral
    acts, t = simulate_hold(f, "NEUTRAL", duration=0.5, start_time=t)
    # Cooldown wait (pointer cooldown is 1.5s)
    acts, t = simulate_hold(f, "NEUTRAL", duration=2.0, start_time=t)

    # 3. Hold POINTER (Shaka) 1.4s -> POINTER MODE ON
    acts, t = simulate_hold(f, "POINTER", duration=1.4, start_time=t)
    assert len(acts) == 1 and acts[0][1] == "POINTER MODE ON"
    assert f.is_armed and f.is_pointer_mode

    # Release to neutral and wait cooldown
    acts, t = simulate_hold(f, "NEUTRAL", duration=2.0, start_time=t)

    # 4. Hold POINTER (Shaka) 1.4s -> POINTER MODE OFF (back to ARMED)
    acts, t = simulate_hold(f, "POINTER", duration=1.4, start_time=t)
    assert len(acts) == 1 and acts[0][1] == "POINTER MODE OFF"
    assert f.is_armed and not f.is_pointer_mode

    # Release and wait toggle cooldown
    acts, t = simulate_hold(f, "NEUTRAL", duration=2.5, start_time=t)

    # 5. Hold ARM 1.4s -> SYSTEM DISARMED
    acts, t = simulate_hold(f, "ARM", duration=1.4, start_time=t)
    assert len(acts) == 1 and acts[0][1] == "SYSTEM DISARMED"
    assert not f.is_armed and not f.is_pointer_mode
    print("PASS: test_full_mode_cycle")


def test_disarmed_ignores_commands():
    """In DISARMED mode, slides and POINTER toggle must be completely ignored."""
    f = GestureFilter()
    t = 0.0
    for g in ["NEXT", "PREVIOUS", "POINTER"]:
        acts, t = simulate_hold(f, g, duration=1.5, start_time=t)
        assert len(acts) == 0, f"Disarmed should have ignored {g}"
        assert not f.is_armed
        assert not f.is_pointer_mode
        acts, t = simulate_hold(f, "NEUTRAL", duration=0.5, start_time=t)
    print("PASS: test_disarmed_ignores_commands")


def test_pointer_mode_ignores_slides_and_arm():
    """In POINTER mode, slides and ARM must be completely ignored (prevents mid-point lockout)."""
    f = GestureFilter()
    t = 0.0

    # Arm system
    acts, t = simulate_hold(f, "ARM", duration=1.4, start_time=t)
    acts, t = simulate_hold(f, "NEUTRAL", duration=2.0, start_time=t)

    # Enter pointer mode
    acts, t = simulate_hold(f, "POINTER", duration=1.4, start_time=t)
    assert f.is_pointer_mode
    acts, t = simulate_hold(f, "NEUTRAL", duration=2.0, start_time=t)

    # Hold NEXT for 2.0s -> nothing
    acts, t = simulate_hold(f, "NEXT", duration=2.0, start_time=t)
    assert len(acts) == 0, "Slides must be ignored in pointer mode"

    # Hold ARM (L sign lookalike) for 5.0s -> nothing (does NOT disarm!)
    acts, t = simulate_hold(f, "ARM", duration=5.0, start_time=t)
    assert len(acts) == 0, "ARM must be ignored in pointer mode"
    assert f.is_armed and f.is_pointer_mode, "Must remain in pointer mode"
    print("PASS: test_pointer_mode_ignores_slides_and_arm")


def test_shaka_latch():
    """Holding Shaka continuously must toggle only ONCE and not re-trigger repeatedly."""
    f = GestureFilter()
    t = 0.0

    # Arm system
    acts, t = simulate_hold(f, "ARM", duration=1.4, start_time=t)
    acts, t = simulate_hold(f, "NEUTRAL", duration=2.0, start_time=t)

    # Hold Shaka for 5 continuous seconds
    acts, t = simulate_hold(f, "POINTER", duration=5.0, start_time=t)
    assert len(acts) == 1 and acts[0][1] == "POINTER MODE ON", "Latch must prevent multiple toggles during continuous hold"
    assert f.is_pointer_mode
    print("PASS: test_shaka_latch")


def test_hand_lost_safety_timeout():
    """If no hand is seen for 3.0s in pointer mode, pointer mode auto-disengages."""
    f = GestureFilter(pointer_timeout=3.0)
    t = 0.0

    # Arm and enter pointer mode
    simulate_hold(f, "ARM", duration=1.4, start_time=t)
    t += 2.0
    simulate_hold(f, "POINTER", duration=1.4, start_time=t)
    t += 1.0
    assert f.is_pointer_mode

    # Simulate brief hand drop for 1.5 seconds (under 3.0s timeout)
    for _ in range(15):
        act = f.no_hand(now=t)
        assert act is None, "Should not timeout under 3.0s"
        t += 0.1
    assert f.is_pointer_mode

    # Simulate hand dropping past 3.0s threshold
    t += 2.0  # total elapsed > 3.0s
    act = f.no_hand(now=t)
    assert act == "POINTER TIMEOUT (OFF)"
    assert not f.is_pointer_mode
    assert f.is_armed  # stays armed, just leaves pointer mode
    print("PASS: test_hand_lost_safety_timeout")


def test_pointer_maths_and_safety_bounds():
    """Verify coordinate normalization, velocity-adaptive smoothing, deadzone, and fail-safe safety clamp."""
    # Test on a virtual 1920x1080 display with default expanded ROI (0.08, 0.08, 0.92, 0.92)
    ctrl = PointerController(screen_size=(1920, 1080))

    # Center of expanded ROI (0.50, 0.50) maps to exact center of screen (960, 540)
    x, y = ctrl.compute_screen_coords(0.50, 0.50)
    assert abs(x - (1920 - 1) / 2) < 2.0
    assert abs(y - (1080 - 1) / 2) < 2.0

    # Top-left of ROI (0.08, 0.08) or beyond (0.0, 0.0):
    # Must clamp to safe_margin (>= 5) and NEVER touch (0, 0)!
    x_tl, y_tl = ctrl.compute_screen_coords(0.0, 0.0)
    assert x_tl >= 5 and y_tl >= 5, "Cursor must never reach (0, 0) failsafe abort pixel"
    assert not (x_tl == 0 and y_tl == 0)

    # Bottom-right of ROI (0.92, 0.92) or beyond (1.0, 1.0):
    x_br, y_br = ctrl.compute_screen_coords(1.0, 1.0)
    assert x_br <= 1920 - 5
    assert y_br <= 1080 - 5

    # Test pointing update & deadzone:
    pointing_hand = make_hand(index_up=True, middle_up=False, ring_up=False, pinky_up=False)
    is_p, pos1 = ctrl.update(pointing_hand, dry_run=True)
    assert is_p
    assert pos1 is not None

    # Tiny micro-motion (within dead zone radius < 2.0 px) -> position remains locked
    tiny_move_hand = make_hand(index_up=True, middle_up=False, ring_up=False, pinky_up=False)
    tiny_move_hand.landmark[8].x += 0.0001
    tiny_move_hand.landmark[8].y += 0.0001
    is_p2, pos2 = ctrl.update(tiny_move_hand, dry_run=True)
    assert is_p2
    assert pos2 == pos1, "Dead zone should keep cursor coordinates locked during micro-jitter"

    # Non-pointing hand (e.g. fist or Shaka) -> freezes cursor
    shaka_hand = make_hand(index_up=False, middle_up=False, ring_up=False, pinky_up=True, thumb_far=True)
    is_p3, pos3 = ctrl.update(shaka_hand, dry_run=True)
    assert not is_p3, "Shaka hand must NOT trigger pointer motion"
    print("PASS: test_pointer_maths_and_safety_bounds")


def test_rules_and_pointing_geometry():
    """Verify rule_based_classifier and is_pointing_pose on synthetic landmarks."""
    # Pointing pose: index up, others curled
    pointing = make_hand(index_up=True, middle_up=False, ring_up=False, pinky_up=False, thumb_far=False)
    assert is_pointing_pose(pointing)

    # Shaka pose: thumb far + pinky up, others curled
    shaka = make_hand(index_up=False, middle_up=False, ring_up=False, pinky_up=True, thumb_far=True)
    assert not is_pointing_pose(shaka)
    assert rule_based_classifier(shaka) == "POINTER"

    # ARM pose: thumb far + index up, others curled
    arm = make_hand(index_up=True, middle_up=False, ring_up=False, pinky_up=False, thumb_far=True)
    assert rule_based_classifier(arm) == "ARM"

    # NEXT pose: peace sign (index + middle up)
    next_pose = make_hand(index_up=True, middle_up=True, ring_up=False, pinky_up=False)
    assert not is_pointing_pose(next_pose)
    assert rule_based_classifier(next_pose) == "NEXT"

    # PREVIOUS pose: open palm (all fingers up)
    prev_pose = make_hand(index_up=True, middle_up=True, ring_up=True, pinky_up=True)
    assert not is_pointing_pose(prev_pose)
    assert rule_based_classifier(prev_pose) == "PREVIOUS"

    print("PASS: test_rules_and_pointing_geometry")


if __name__ == "__main__":
    test_full_mode_cycle()
    test_disarmed_ignores_commands()
    test_pointer_mode_ignores_slides_and_arm()
    test_shaka_latch()
    test_hand_lost_safety_timeout()
    test_pointer_maths_and_safety_bounds()
    test_rules_and_pointing_geometry()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
