import numpy as np

def normalize_landmarks(landmarks):
    # Extract x, y, and z
    raw_points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
    # Translate based on wrist (x, y, z)
    wrist = raw_points[0]
    translated = raw_points - wrist
    
    # Scale based on 3D distance to middle finger base
    scale = np.linalg.norm(translated[9]) + 1e-6
    normalized = translated / scale
    
    # Flatten to a 63-number 1D list
    return normalized.flatten().tolist()