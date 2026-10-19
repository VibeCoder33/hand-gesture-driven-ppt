"""Train and compare models with person-held-out evaluation, then save the best one.

  python train.py                      # uses data/landmarks.csv
  python train.py --data other.csv
"""
import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features import FEATURE_VERSION, features_from_array, load_csv

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/landmarks.csv")
args = parser.parse_args()

MODELS = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                   random_state=42, n_jobs=-1),
    "SVM (RBF)":    lambda: make_pipeline(StandardScaler(),
                                          SVC(C=10, probability=True, class_weight="balanced",
                                              random_state=42)),
    "kNN (k=5)":    lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "MLP (64,32)":  lambda: make_pipeline(StandardScaler(),
                                          MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                                        random_state=42)),
}


def block_groups(y, n_blocks=5):
    """Fallback when there is no person_id: split every contiguous recording run into
    n_blocks consecutive chunks and use the chunk number as the group."""
    g = np.zeros(len(y), dtype=int)
    start = 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[start]:
            idx = np.arange(start, i)
            g[idx] = np.minimum((idx - start) * n_blocks // len(idx), n_blocks - 1)
            start = i
    return g.astype(str)


print(f"Loading {args.data} ...")
P, y, persons, df = load_csv(args.data)
X = features_from_array(P)
print(f"{len(y)} rows, {X.shape[1]} features")
print(pd.Series(y).value_counts().to_string(), "\n")

if persons is not None and len(set(persons)) >= 2:
    groups, mode = persons, f"leave-one-person-out ({len(set(persons))} people)"
    print("Rows per person:\n" + pd.Series(persons).value_counts().to_string(), "\n")
else:
    groups, mode = block_groups(y), "contiguous-block CV (NO person_id -> optimistic, single-person data)"
    print("WARNING: no person_id / only one person. Falling back to block CV.")
    print("         Record several people with collect.py for a realistic number.\n")
print("Evaluation:", mode, "\n")

logo = LeaveOneGroupOut()
labels = sorted(set(y))
rows, pooled = [], {}
for name, make in MODELS.items():
    t0 = time.time()
    fold_acc, y_true, y_pred = [], [], []
    for tr, te in logo.split(X, y, groups):
        m = make().fit(X[tr], y[tr])
        p = m.predict(X[te])
        fold_acc.append(accuracy_score(y[te], p))
        y_true.extend(y[te]); y_pred.extend(p)
    pooled[name] = (np.array(y_true), np.array(y_pred))
    rows.append({"model": name,
                 "accuracy_mean": np.mean(fold_acc), "accuracy_std": np.std(fold_acc),
                 "macro_f1": f1_score(y_true, y_pred, average="macro"),
                 "seconds": time.time() - t0})
    print(f"{name:14s} acc {np.mean(fold_acc)*100:5.1f}% (+/- {np.std(fold_acc)*100:.1f})  "
          f"macro-F1 {rows[-1]['macro_f1']:.3f}")

res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
os.makedirs("results", exist_ok=True)
res.to_csv("results/model_comparison.csv", index=False)
best = res.iloc[0]["model"]
print(f"\nBest model by macro-F1: {best}\n")

yt, yp = pooled[best]
print("Classification report (pooled over held-out folds):")
print(classification_report(yt, yp, zero_division=0))
cm = pd.DataFrame(confusion_matrix(yt, yp, labels=labels), index=labels, columns=labels)
print("Confusion matrix (rows = true, columns = predicted):")
print(cm, "\n")
cm.to_csv("results/confusion_matrix.csv")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45); ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm.values[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{best}\n{mode.split(' (')[0]}")
    fig.tight_layout(); fig.savefig("results/confusion_matrix.png", dpi=150)
except ImportError:
    pass

# Final model: refit the winner on ALL data and save it with metadata
final = MODELS[best]().fit(X, y)
os.makedirs("models", exist_ok=True)
joblib.dump({"model": final, "name": best, "feature_version": FEATURE_VERSION,
             "n_features": X.shape[1], "sklearn_version": sklearn.__version__},
            "models/gesture_model.pkl")
print("Saved models/gesture_model.pkl  |  results/ has the comparison table and confusion matrix")
