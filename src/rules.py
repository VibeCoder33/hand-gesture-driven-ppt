def rule_based_classifier(landmarks):
    """
    A simple baseline classifier using hand-coded logic.
    landmarks: The raw 21 MediaPipe landmark objects
    """
    # MediaPipe Landmark Indices:
    # Thumb tip: 4, Index tip: 8, Middle tip: 12, Ring tip: 16, Pinky tip: 20
    # Lower joints (PIP joints): 3, 6, 10, 14, 18
    
    pts = landmarks.landmark
    
    # A finger is "extended" if its tip is higher (smaller Y) than its lower joint
    index_up = pts[8].y < pts[6].y
    middle_up = pts[12].y < pts[10].y
    ring_up = pts[16].y < pts[14].y
    pinky_up = pts[20].y < pts[18].y
    
    # Thumb is tricky because it moves sideways, so we check X distance
    # Assuming Right Hand for the baseline: tip should be further left (smaller X)
    thumb_out = pts[4].x < pts[3].x 

    # --- Gesture Logic ---
    
    # ARM/DISARM (Shaka): Thumb out, Pinky up, others down
    if thumb_out and pinky_up and not index_up and not middle_up and not ring_up:
        return "ARM"
        
    # PREVIOUS (Open Palm): All fingers extended
    elif thumb_out and index_up and middle_up and ring_up and pinky_up:
        return "PREVIOUS"
        
    # NEXT (Peace Sign): Index and middle up, others down
    elif index_up and middle_up and not ring_up and not pinky_up:
        return "NEXT"
        
    else:
        return "NEUTRAL"