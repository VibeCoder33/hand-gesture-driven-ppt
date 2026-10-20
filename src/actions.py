import pyautogui

pyautogui.FAILSAFE = True   # slam the mouse into a screen corner to abort
pyautogui.PAUSE = 0.0

KEYMAP = {"NEXT": "right", "PREVIOUS": "left"}


def trigger_action(gesture, dry_run=False):
    """Presses the key for a slide gesture. The slideshow window must have focus.
    dry_run=True only prints (use it for the false-trigger test)."""
    key = KEYMAP.get(gesture)
    if key is None:
        return False
    if not dry_run:
        pyautogui.press(key)
    print(f"ACTION: {gesture} -> '{key}'" + ("  [dry run]" if dry_run else ""))
    return True


def set_ppt_laser(enable=True, dry_run=False):
    """Activates or deactivates PowerPoint's built-in laser pointer mode.
    Ctrl+L turns on the laser pointer; Ctrl+A restores the standard arrow cursor."""
    if dry_run:
        print(f"ACTION: PPT Laser {'ON (Ctrl+L)' if enable else 'OFF (Ctrl+A)'}  [dry run]")
        return
    try:
        if enable:
            pyautogui.hotkey("ctrl", "l")
        else:
            pyautogui.hotkey("ctrl", "a")
    except Exception as e:
        print(f"Error toggling PPT laser: {e}")

