import time

class GestureFilter:
    def __init__(self, confidence_threshold=0.8, hold_time=0.5, cooldown_time=1.0):
        self.confidence_threshold = confidence_threshold
        self.hold_time = hold_time
        self.cooldown_time = cooldown_time
        
        self.current_gesture = None
        self.gesture_start_time = 0
        self.last_action_time = 0
        
    def update(self, predicted_gesture, confidence):
        current_time = time.time()
        
        # 1. Enforce Cooldown
        if current_time - self.last_action_time < self.cooldown_time:
            return None 
            
        # 2. Ignore low confidence or NEUTRAL/ARM states for now
        # (We will add the ARM/DISARM logic on Day 7)
        if confidence < self.confidence_threshold or predicted_gesture in ["NEUTRAL", "None", "ARM"]:
            self.current_gesture = None
            return None
            
        # 3. If the gesture changes, start a new timer
        if predicted_gesture != self.current_gesture:
            self.current_gesture = predicted_gesture
            self.gesture_start_time = current_time
            return None
            
        # 4. If the gesture is held long enough, return it as a confirmed action
        if current_time - self.gesture_start_time >= self.hold_time:
            self.last_action_time = current_time
            self.current_gesture = None # Reset after firing
            return predicted_gesture
            
        # Still waiting for hold time to complete
        return None