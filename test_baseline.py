"""Evaluate the rule-based baseline on the recorded data (no training involved)."""
import argparse
from types import SimpleNamespace

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.features import load_csv
from src.rules import rule_based_classifier

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/landmarks.csv")
args = parser.parse_args()

P, y, persons, _ = load_csv(args.data)
pred = []
for hand in P:
    lm = SimpleNamespace(landmark=[SimpleNamespace(x=pt[0], y=pt[1], z=pt[2]) for pt in hand])
    pred.append(rule_based_classifier(lm))
pred = pd.Series(pred).values

print(f"Overall accuracy: {accuracy_score(y, pred) * 100:.2f}%\n")
print(classification_report(y, pred, zero_division=0))
labels = sorted(set(y))
print(pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels))

if persons is not None:
    print("\nAccuracy per person:")
    for person in sorted(set(persons)):
        m = persons == person
        print(f"  {person:12s} {accuracy_score(y[m], pred[m]) * 100:.1f}%  ({m.sum()} rows)")
