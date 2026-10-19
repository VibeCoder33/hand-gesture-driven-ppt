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
