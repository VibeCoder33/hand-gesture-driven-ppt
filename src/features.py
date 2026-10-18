import numpy as np

def normalize_landmarks(landmarks):
    """
    Takes a list of 21 MediaPipe landmark objects and returns a 
    flat list of 42 normalized [x, y] coordinates.
    """
    # 1. Extract raw x, y
    raw_points = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
    
    # 2. Translation: Subtract wrist (point 0) from all points
    wrist = raw_points[0]
    translated = raw_points - wrist
    
    # 3. Scaling: Find distance from wrist to middle finger base (point 9)
    # Add a tiny number (1e-6) to prevent divide-by-zero errors
    scale = np.linalg.norm(translated[9]) + 1e-6
    normalized = translated / scale
    
    # 4. Flatten back to a 1D list of 42 numbers
    return normalized.flatten().tolist()