"""Single source of truth for feature extraction and CSV loading.

Training, baseline testing and the live app all go through
features_from_array(), so they can never disagree about the features.
"""
import numpy as np
import pandas as pd

FEATURE_VERSION = 2          # bump if you change the features -> forces retraining
N_LANDMARKS = 21

COORD_COLUMNS = [f"{a}{i}" for i in range(N_LANDMARKS) for a in "xyz"]
EXTRA_NAMES = ["thumb_index_angle", "d_4_5", "d_4_8",
               "d_4_0", "d_8_0", "d_12_0", "d_16_0", "d_20_0"]


def features_from_array(P):
    """P: raw MediaPipe coords, shape (n, 21, 3) or (21, 3).
    Returns an (n, 71) matrix: 63 normalized coords + 8 hand-shape features."""
    P = np.asarray(P, dtype=float)
    if P.ndim == 2:
        P = P[None]

    # 1. translate so the wrist is the origin, 2. scale by hand size
    T = P - P[:, 0:1, :]
    scale = np.linalg.norm(T[:, 9, :], axis=1) + 1e-6
    N = T / scale[:, None, None]
    base = N.reshape(len(P), -1)

    def dist(a, b):
        return np.linalg.norm(N[:, a, :2] - N[:, b, :2], axis=1)

    # angle between thumb direction (2->4) and index direction (5->8)
    u = N[:, 4, :2] - N[:, 2, :2]
    v = N[:, 8, :2] - N[:, 5, :2]
    cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    angle = np.arccos(np.clip(cos, -1, 1)) / np.pi        # 0..1

    extras = np.column_stack([angle, dist(4, 5), dist(4, 8),
                              dist(4, 0), dist(8, 0), dist(12, 0), dist(16, 0), dist(20, 0)])
    return np.hstack([base, extras])


def normalize_landmarks(landmarks):
    """Live use: takes one MediaPipe hand_landmarks object, returns a list of 71 floats."""
    P = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    return features_from_array(P)[0].tolist()


def load_csv(path):
    """Loads a collect.py CSV. Drops NaN rows and exact duplicates.
    Returns (P, y, groups, df); groups is None if the file has no person_id column."""
    df = pd.read_csv(path)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df["label"] = df["label"].astype(str).str.strip()
    P = df[COORD_COLUMNS].values.reshape(-1, N_LANDMARKS, 3)
    y = df["label"].values
    groups = df["person_id"].astype(str).values if "person_id" in df.columns else None
    return P, y, groups, df
