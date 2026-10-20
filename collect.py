"""Record landmark data. Run once per person:  python collect.py --person alice"""
import argparse
import csv
import os
import sys
import time

import cv2
import mediapipe as mp

from src.features import COORD_COLUMNS

GESTURES = {
    0: "NEXT",       # peace sign: index + middle up, ring + pinky down
    1: "PREVIOUS",   # open palm: all five fingers out
    2: "ARM",        # "L" sign: index up, thumb stuck out ~90 degrees, other fingers curled
    3: "NEUTRAL",    # hard negatives & near-misses: thumb only out, pinky only out,
                     # pointing hand (index up, thumb curled or loose), loose fist, relaxed hand
    4: "POINTER",    # Shaka: thumb and pinky extended, middle three fingers curled
}
HEADERS = COORD_COLUMNS + ["label", "person_id", "session"]

parser = argparse.ArgumentParser()
parser.add_argument("--person", required=True, help="name/id of the person being recorded")
parser.add_argument("--out", default="data/landmarks.csv")
parser.add_argument("--save-every", type=int, default=2, help="save every Nth frame (removes near-duplicates)")
parser.add_argument("--countdown", type=float, default=2.0, help="seconds to get into pose after pressing a key")
args = parser.parse_args()

# ---- CSV setup -------------------------------------------------------------
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
if os.path.exists(args.out):
    with open(args.out, newline="") as f:
        first_line = f.readline().strip().split(",")
    if first_line != HEADERS:
        sys.exit(f"'{args.out}' has an old header (no person_id/session columns). "
                 f"Rename it (e.g. to data/landmarks_old.csv) and run again.")
    csv_file = open(args.out, mode="a", newline="")
    writer = csv.writer(csv_file)
else:
    csv_file = open(args.out, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(HEADERS)

session = time.strftime("%Y%m%d_%H%M%S")

# ---- MediaPipe + camera ----------------------------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Could not open the webcam.")

current_label = None
recording = False
record_start = 0.0
frame_idx = 0
counts = {name: 0 for name in GESTURES.values()}

print(f"Recording for person: {args.person}   session: {session}")
print("--- Controls ---")
for key, name in GESTURES.items():
    print(f"Press '{key}' to record '{name}'")
print("Press 's' to stop, 'q' to quit")
print("Tips: move/tilt your hand slowly, vary distance and lighting, ~1-2 minutes per gesture.")

try:
    while cap.isOpened():
        ok, img = cap.read()
        if not ok:
            continue
        img = cv2.flip(img, 1)                      # mirror view (main.py does the same)
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        frame_idx += 1

        live = recording and time.time() >= record_start      # countdown finished
        if results.multi_hand_landmarks:
            hl = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)
            if live and current_label and frame_idx % args.save_every == 0:
                row = []
                for lm in hl.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                writer.writerow(row + [current_label, args.person, session])
                counts[current_label] += 1
                if sum(counts.values()) % 50 == 0:
                    csv_file.flush()

        # ---- overlay ----
        if recording and not live:
            status, color = f"GET READY: {current_label} ...", (0, 165, 255)
        elif live:
            status, color = f"RECORDING: {current_label} ({counts[current_label]})", (0, 0, 255)
        else:
            status, color = "PAUSED", (0, 255, 0)
        cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, " | ".join(f"{k}:{v}" for k, v in counts.items()), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Data Collection", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            recording = False
        elif key != 255 and chr(key).isdigit() and int(chr(key)) in GESTURES:
            current_label = GESTURES[int(chr(key))]
            recording = True
            record_start = time.time() + args.countdown
finally:
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Saved:", counts)
