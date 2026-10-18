import time

class GestureFilter:
    def __init__(self, confidence_threshold=0.75, slide_hold_time=0.5, toggle_hold_time=0.3, slide_cooldown=1.0, toggle_cooldown=2.0):
        self.confidence_threshold = confidence_threshold
        self.slide_hold_time = slide_hold_time
        self.toggle_hold_time = toggle_hold_time
        self.slide_cooldown = slide_cooldown
        self.toggle_cooldown = toggle_cooldown # Long timeout to prevent rapid flickering
        
        self.current_gesture = None
        self.gesture_start_time = 0
        
        self.last_slide_time = 0
        self.last_toggle_time = 0
        self.is_armed = False  
        
    def update(self, predicted_gesture, confidence):
        current_time = time.time()
        
        # 1. Enforce Toggle Cooldown (Stops rapid flickering of the ARM state)
        if predicted_gesture == "ARM":
            if current_time - self.last_toggle_time < self.toggle_cooldown:
                return None
        else:
            # 2. Enforce Slide Cooldown (Only applies to NEXT/PREVIOUS)
            if current_time - self.last_slide_time < self.slide_cooldown:
                return None 
            
        # 3. Ignore low confidence or NEUTRAL states
        if confidence < self.confidence_threshold or predicted_gesture in ["NEUTRAL", "None"]:
            self.current_gesture = None
            return None
            
        # 4. Start a new timer if the gesture changes
        if predicted_gesture != self.current_gesture:
            self.current_gesture = predicted_gesture
            self.gesture_start_time = current_time
            return None
            
        # 5. Determine which hold time to use
        required_time = self.toggle_hold_time if self.current_gesture == "ARM" else self.slide_hold_time
            
        # 6. Gesture held long enough to fire
        if current_time - self.gesture_start_time >= required_time:
            action_to_fire = self.current_gesture
            self.current_gesture = None # Reset the holding state
            
            if action_to_fire == "ARM":
                self.last_toggle_time = current_time # Start the 2-second toggle lockout
                self.is_armed = not self.is_armed
                return f"SYSTEM {'ARMED' if self.is_armed else 'DISARMED'}"
            
            # Only allow slide controls if the system is armed
            if self.is_armed and action_to_fire in ["NEXT", "PREVIOUS"]:
                self.last_slide_time = current_time # Start the 1-second slide lockout
                return action_to_fire
                
        return None