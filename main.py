import cv2
import mediapipe as mp
import joblib
import time
from src.features import normalize_landmarks
from src.stability import GestureFilter
from src.actions import trigger_action

print("Loading model...")
model = joblib.load('models/gesture_model.pkl')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the stability filter
gesture_filter = GestureFilter(confidence_threshold=0.8, hold_time=0.5, cooldown_time=1.0)

cap = cv2.VideoCapture(0)
prev_time = 0

print("System active. Make sure PowerPoint or a PDF is open and in focus!")

while cap.isOpened():
    success, img = cap.read()
    if not success: continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    prediction = "None"
    confidence = 0.0

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        try:
            normalized_vector = normalize_landmarks(hand_landmarks)
            probabilities = model.predict_proba([normalized_vector])[0]
            max_prob_index = probabilities.argmax()
            
            prediction = model.classes_[max_prob_index]
            confidence = probabilities[max_prob_index]
            
            # Run the raw prediction through the filter
            confirmed_action = gesture_filter.update(prediction, confidence)
            
            # If the filter validates it, trigger the keyboard press
            if confirmed_action:
                trigger_action(confirmed_action)
                
        except Exception as e:
            print(f"Prediction error: {e}")

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # UI Overlay
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
    cv2.putText(img, f'Raw: {prediction}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Show cooldown status
    time_since_action = curr_time - gesture_filter.last_action_time
    if time_since_action < gesture_filter.cooldown_time:
        cv2.putText(img, 'COOLDOWN ACTIVE', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Day 6: Live Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()