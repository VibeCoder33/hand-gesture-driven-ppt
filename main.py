"""Live gesture-controlled slides with low-latency pointer mode.

Usage:
  python main.py             # normal live use
  python main.py --dry-run   # simulation: counts triggers, tracks cursor without moving mouse or pressing keys
  python main.py --ppt-laser # enables automatic PowerPoint laser pointer toggle (Ctrl+L / Ctrl+A)

State machine:
  DISARMED --(L sign hold 0.8s)--> ARMED --(Shaka hold 0.35s)--> POINTER MODE --(Shaka hold 0.35s)--> ARMED --(L sign hold 0.8s)--> DISARMED
"""
import argparse
import sys
import time

import cv2
import joblib
import mediapipe as mp
import sklearn

from src.actions import set_ppt_laser, trigger_action
from src.features import FEATURE_VERSION, normalize_landmarks
from src.pointer import PointerController
from src.rules import is_pointing_pose
from src.stability import GestureFilter

parser = argparse.ArgumentParser(description="Live hand gesture presenter & high-speed pointer")
parser.add_argument("--dry-run", action="store_true", help="print actions and track cursor without moving mouse/pressing keys")
parser.add_argument("--camera", type=int, default=0, help="webcam device index")
parser.add_argument("--ppt-laser", action="store_true", help="automatically toggle PPT laser pointer mode (Ctrl+L/Ctrl+A)")
parser.add_argument("--alpha", type=float, default=0.55, help="cursor base exponential smoothing factor (0.1..0.9)")
parser.add_argument("--dead-zone", type=float, default=2.0, help="dead zone radius in pixels to prevent jitter")
parser.add_argument("--roi", type=float, nargs=4, default=[0.08, 0.08, 0.92, 0.92],
                    metavar=("XMIN", "YMIN", "XMAX", "YMAX"), help="camera tracking region for pointer mapping (expanded 84% default)")
parser.add_argument("--pointer-hold", type=float, default=0.35, help="seconds to hold Shaka to toggle pointer mode (default 0.35s)")
args = parser.parse_args()

print("Loading gesture model...")
try:
    bundle = joblib.load("models/gesture_model.pkl")
    if bundle.get("feature_version") != FEATURE_VERSION:
        sys.exit("Model was trained with different features. Run: python train.py")
    if bundle.get("sklearn_version") != sklearn.__version__:
        print(f"Note: model trained with scikit-learn {bundle.get('sklearn_version')}, "
              f"you have {sklearn.__version__}. Retrain if you see errors.")
    model = bundle["model"]
    print(f"Loaded {bundle.get('name', 'model')} successfully.")
except Exception as e:
    sys.exit(f"Could not load model: {e}. Please run python train.py first.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

gesture_filter = GestureFilter(
    confidence_threshold=0.75,
    slide_hold_time=0.5,
    toggle_hold_time=0.8,
    pointer_hold_time=args.pointer_hold,
    slide_cooldown=1.0,
    toggle_cooldown=2.0,
    pointer_cooldown=0.8,
    pointer_timeout=3.0,
    vote_window=0.25,
)

pointer_controller = PointerController(
    roi=tuple(args.roi),
    alpha=args.alpha,
    dead_zone=args.dead_zone,
)

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    sys.exit(f"Could not open webcam on index {args.camera}.")

prev_time = time.monotonic()
last_error = None
trigger_count = 0

print("System active (starts DISARMED).")
print("  - Hold 'L' sign (0.8s) to ARM.")
print(f"  - While ARMED, hold 'Shaka' ({args.pointer_hold:.2f}s) to enter POINTER MODE.")
print(f"  - While in POINTER MODE, point index finger to move cursor, hold 'Shaka' ({args.pointer_hold:.2f}s) to exit.")
print("  - Press 'q' in video window to quit.")

try:
    while cap.isOpened():
        ok, img = cap.read()
        if not ok:
            continue

        # Horizontal flip so image acts like an intuitive mirror
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        now = time.monotonic()

        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        prediction, confidence = "None", 0.0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # High-performance fast path:
            # While in pointer mode and actively pointing, bypass the slow model prediction (saves ~35ms)
            # so the cursor moves at the full 30+ FPS camera frame rate with zero lag!
            pointing_now = is_pointing_pose(hand_landmarks)

            if gesture_filter.is_pointer_mode and pointing_now:
                # Active pointing: instant cursor update
                is_pointing, pos = pointer_controller.update(hand_landmarks, dry_run=args.dry_run)
                tip = hand_landmarks.landmark[8]
                cx, cy = int(tip.x * w), int(tip.y * h)
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)
                prediction, confidence = "POINTING", 1.0
                gesture_filter.last_hand_seen_time = now
            else:
                # Not pointing (or in slide mode): run ML classification to listen for gestures
                try:
                    probs = model.predict_proba([normalize_landmarks(hand_landmarks)])[0]
                    best = probs.argmax()
                    prediction, confidence = model.classes_[best], float(probs[best])

                    action = gesture_filter.update(prediction, confidence, now=now)
                    if action in ("NEXT", "PREVIOUS"):
                        if trigger_action(action, dry_run=args.dry_run):
                            trigger_count += 1
                    elif action:
                        print(f"STATE CHANGE: {action}")
                        if action == "POINTER MODE ON":
                            pointer_controller.reset()
                            if args.ppt_laser:
                                set_ppt_laser(True, dry_run=args.dry_run)
                        elif action in ("POINTER MODE OFF", "POINTER TIMEOUT (OFF)"):
                            pointer_controller.reset()
                            if args.ppt_laser:
                                set_ppt_laser(False, dry_run=args.dry_run)
                except Exception as e:
                    gesture_filter.reset()
                    if str(e) != last_error:
                        print(f"Prediction error: {e}")
                        last_error = str(e)

                # If in pointer mode but hand stopped pointing, freeze cursor
                if gesture_filter.is_pointer_mode:
                    pointer_controller.freeze()
        else:
            # Hand missing: reset transient state and check safety timeout
            timeout_action = gesture_filter.no_hand(now=now)
            if timeout_action:
                print(f"STATE CHANGE: {timeout_action}")
                pointer_controller.reset()
                if args.ppt_laser:
                    set_ppt_laser(False, dry_run=args.dry_run)
            pointer_controller.freeze()

        # FPS calculation
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now

        # ---- On-screen display (HUD) ----
        # 1. FPS and raw classification
        cv2.putText(img, f"FPS: {int(fps)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        conf_color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
        cv2.putText(img, f"Raw: {prediction} ({confidence:.2f})", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, conf_color, 2)

        # 2. State machine mode badge
        if not gesture_filter.is_armed:
            mode_text, mode_color = "DISARMED (Locked)", (0, 0, 255)
        elif gesture_filter.is_pointer_mode:
            mode_text, mode_color = "POINTER MODE (Laser/Cursor)", (255, 255, 0)
        else:
            mode_text, mode_color = "ARMED (Slide Mode)", (0, 255, 0)
        cv2.putText(img, mode_text, (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.75, mode_color, 2)

        # 3. Pointer ROI and timeout display
        if gesture_filter.is_pointer_mode:
            pointer_controller.draw_roi(img)
            t_left = gesture_filter.pointer_timeout_left(now=now)
            if not results.multi_hand_landmarks and t_left > 0:
                cv2.putText(img, f"Hand absent: auto-exit in {t_left:.1f}s", (10, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

        # 4. Hold-time progress bar
        prog = gesture_filter.hold_progress(now=now)
        cv2.rectangle(img, (10, 102), (210, 117), (80, 80, 80), 1)
        if prog > 0:
            active_g = gesture_filter.current_gesture or ""
            cv2.rectangle(img, (10, 102), (10 + int(200 * prog), 117), (0, 255, 255), -1)
            cv2.putText(img, f"Holding {active_g}...", (220, 114),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 5. Cooldown status
        if gesture_filter.slide_cooldown_left(now=now) > 0:
            cv2.putText(img, "SLIDE COOLDOWN", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        elif gesture_filter.toggle_cooldown_left(now=now) > 0:
            cv2.putText(img, "ARM COOLDOWN", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        elif gesture_filter.pointer_cooldown_left(now=now) > 0:
            cv2.putText(img, "POINTER COOLDOWN", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

        # 6. Dry run indicator
        if args.dry_run:
            cv2.putText(img, f"DRY RUN - triggers: {trigger_count}", (10, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        cv2.imshow("Gesture Presenter & Pointer", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    if args.ppt_laser and not args.dry_run and gesture_filter.is_pointer_mode:
        set_ppt_laser(False, dry_run=False)
    if args.dry_run:
        print(f"Dry run finished. Actions that would have fired: {trigger_count}")
