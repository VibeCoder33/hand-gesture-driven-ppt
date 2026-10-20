import math


def rule_based_classifier(landmarks):
    """Hand-coded baseline. `landmarks` needs a .landmark list with .x/.y per point.

    MediaPipe indices: tips 4/8/12/16/20, PIP joints 6/10/14/18, index base 5, middle base 9.
    A finger is 'up' when its tip is higher (smaller y) than its PIP joint.
    """
    p = landmarks.landmark
    d = lambda a, b: math.hypot(p[a].x - p[b].x, p[a].y - p[b].y)

    index_up = p[8].y < p[6].y
    middle_up = p[12].y < p[10].y
    ring_up = p[16].y < p[14].y
    pinky_up = p[20].y < p[18].y

    # Orientation-independent thumb test: thumb tip far from the index base,
    # measured relative to hand size. (0.40 is a starting value - retune on new data.)
    thumb_far = d(4, 5) / (d(0, 9) + 1e-6) > 0.40

    # ARM / DISARM: "L" = index up + thumb out, other fingers curled
    if index_up and not (middle_up or ring_up or pinky_up) and thumb_far:
        return "ARM"
    # POINTER / SHAKA: thumb out + pinky up, index/middle/ring curled
    if thumb_far and pinky_up and not (index_up or middle_up or ring_up):
        return "POINTER"
    # PREVIOUS: open palm = four fingers up (no thumb test; it was unreliable)
    if index_up and middle_up and ring_up and pinky_up:
        return "PREVIOUS"
    # NEXT: peace sign = index + middle up, ring + pinky down
    if index_up and middle_up and not ring_up and not pinky_up:
        return "NEXT"
    return "NEUTRAL"


def is_pointing_pose(landmarks):
    """Checks if hand is in a pointing shape: index finger extended, middle/ring/pinky curled.
    The thumb may be relaxed or tucked. Used for geometric cursor tracking in pointer mode."""
    p = landmarks.landmark
    index_up = p[8].y < p[6].y
    middle_up = p[12].y < p[10].y
    ring_up = p[16].y < p[14].y
    pinky_up = p[20].y < p[18].y
    return index_up and not (middle_up or ring_up or pinky_up)

