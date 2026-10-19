"""Live gesture-controlled slides.

  python main.py             # normal use
  python main.py --dry-run   # never presses keys; counts triggers (for the false-trigger test)

Click the slideshow window so it has focus - key presses go to the focused window.
"""
import argparse
import sys
import time

import cv2
import joblib
import mediapipe as mp
import sklearn

from src.actions import trigger_action
from src.features import FEATURE_VERSION, normalize_landmarks
from src.stability import GestureFilter

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="print actions instead of pressing keys")
parser.add_argument("--camera", type=int, default=0)
args = parser.parse_args()

print("Loading model...")
bundle = joblib.load("models/gesture_model.pkl")
if bundle.get("feature_version") != FEATURE_VERSION:
    sys.exit("Model was trained with different features. Run: python train.py")
if bundle.get("sklearn_version") != sklearn.__version__:
    print(f"Note: model trained with scikit-learn {bundle.get('sklearn_version')}, "
          f"you have {sklearn.__version__}. Retrain if you see errors.")
model = bundle["model"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
gesture_filter = GestureFilter(confidence_threshold=0.75, slide_hold_time=0.5, toggle_hold_time=0.8,
                               slide_cooldown=1.0, toggle_cooldown=2.0)

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    sys.exit("Could not open the webcam.")

prev_time = time.monotonic()
last_error, trigger_count = None, 0
print("System active (starts DISARMED). Hold the L sign to arm. Press 'q' to quit.")

while cap.isOpened():
    ok, img = cap.read()
    if not ok:
        continue

    img = cv2.flip(img, 1)                                   # same mirroring as collect.py
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    prediction, confidence = "None", 0.0
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        try:
            probs = model.predict_proba([normalize_landmarks(hand_landmarks)])[0]
            best = probs.argmax()
            prediction, confidence = model.classes_[best], float(probs[best])

            action = gesture_filter.update(prediction, confidence)
            if action in ("NEXT", "PREVIOUS"):
                if trigger_action(action, dry_run=args.dry_run):
                    trigger_count += 1
            elif action:
                print(f"STATE CHANGE: {action}")
        except Exception as e:                               # report each new error once
            gesture_filter.reset()
            if str(e) != last_error:
                print(f"Prediction error: {e}")
                last_error = str(e)
    else:
        gesture_filter.reset()                               # hand left the frame: drop stale hold timer

    now = time.monotonic()
    fps = 1 / (now - prev_time) if now > prev_time else 0
    prev_time = now

    # ---- overlay ----
    cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
    cv2.putText(img, f"Raw: {prediction} ({confidence:.2f})", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    armed = gesture_filter.is_armed
    cv2.putText(img, "ARMED (Live)" if armed else "DISARMED (Locked)", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if armed else (0, 0, 255), 2)

    prog = gesture_filter.hold_progress()                    # hold-time bar
    cv2.rectangle(img, (10, 120), (210, 135), (80, 80, 80), 1)
    cv2.rectangle(img, (10, 120), (10 + int(200 * prog), 135), (0, 255, 255), -1)

    if gesture_filter.slide_cooldown_left() > 0:
        cv2.putText(img, "SLIDE COOLDOWN", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif gesture_filter.toggle_cooldown_left() > 0:
        cv2.putText(img, "TOGGLE COOLDOWN", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    if args.dry_run:
        cv2.putText(img, f"DRY RUN - triggers: {trigger_count}", (10, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    cv2.imshow("Gesture Presenter", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if args.dry_run:
    print(f"Dry run finished. Actions that would have fired: {trigger_count}")
