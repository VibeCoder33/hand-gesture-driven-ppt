import cv2
import mediapipe as mp
import csv
import os

# Configuration
DATA_FILE = 'data/landmarks.csv'
GESTURES = {
    0: "NEXT",      # Peace Sign
    1: "PREVIOUS",  # Open Palm
    2: "ARM",       # Shaka (Thumb & Pinky out)
    3: "NEUTRAL"    # Relaxed, pointing, half-formed gestures (Hard Negatives)
}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize CSV with 3D headers (x0, y0, z0, x1, y1, z1... x20, y20, z20, label)
if not os.path.exists('data'):
    os.makedirs('data')
    
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, mode='w', newline='') as f:
        headers = []
        for i in range(21):
            headers.extend([f'x{i}', f'y{i}', f'z{i}'])
        headers.append('label')
        csv.writer(f).writerow(headers)

cap = cv2.VideoCapture(0)
current_label = None
recording = False
samples_collected = 0

print("--- Controls ---")
for key, name in GESTURES.items():
    print(f"Press '{key}' to start recording '{name}'")
print("Press 's' to stop recording")
print("Press 'q' to quit")

while cap.isOpened():
    success, img = cap.read()
    if not success: continue

    # Flip the image horizontally for a natural mirror effect
    img = cv2.flip(img, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Save the raw 3D coordinates (x, y, z)
        if recording and current_label:
            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            row.append(current_label)
            
            with open(DATA_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow(row)
            samples_collected += 1

    # UI Overlay
    status = f"RECORDING: {current_label} ({samples_collected})" if recording else "PAUSED"
    color = (0, 0, 255) if recording else (0, 255, 0)
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Data Collection (3D Features)", img)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        recording = False
    elif chr(key).isdigit() and int(chr(key)) in GESTURES:
        current_label = GESTURES[int(chr(key))]
        recording = True
        samples_collected = 0

cap.release()
cv2.destroyAllWindows()