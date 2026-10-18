import pyautogui

# Failsafe: moving your mouse to a corner of the screen aborts pyautogui
pyautogui.FAILSAFE = True 

def trigger_action(gesture):
    if gesture == "NEXT":
        pyautogui.press('right')
        print("ACTION: Pressed RIGHT ARROW (Next Slide)")
        
    elif gesture == "PREVIOUS":
        pyautogui.press('left')
        print("ACTION: Pressed LEFT ARROW (Previous Slide)")