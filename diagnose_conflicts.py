"""Diagnostic and conflict resolution tool for gesture landmark datasets.

Analyzes data/landmarks.csv for label ambiguity between NEUTRAL and active gestures.
Optionally generates a cleaned dataset by removing ambiguous transition frames.

Usage:
  python diagnose_conflicts.py                 # View conflict diagnosis
  python diagnose_conflicts.py --clean         # Export cleaned data to data/landmarks_clean.csv
  python diagnose_conflicts.py --clean --apply # Overwrite data/landmarks.csv with cleaned data (creates backup first)
"""
import argparse
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.features import COORD_COLUMNS, load_csv
from src.rules import rule_based_classifier


def analyze_dataset(csv_path):
    print(f"Loading dataset: {csv_path} ...")
    P, y, persons, df = load_csv(csv_path)
    total_rows = len(df)
    print(f"Total valid rows: {total_rows}")
    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    # Evaluate rule-based predictions
    rule_preds = []
    for hand in P:
        lm = SimpleNamespace(landmark=[SimpleNamespace(x=pt[0], y=pt[1], z=pt[2]) for pt in hand])
        rule_preds.append(rule_based_classifier(lm))
    df["rule_pred"] = rule_preds

    # Conflicts where true label is NEUTRAL but hand geometry matches an active gesture
    neutral_mask = (df["label"] == "NEUTRAL")
    neutral_conflicts = df[neutral_mask & (df["rule_pred"] != "NEUTRAL")]
    
    # Active gesture frames that matched NEUTRAL rule (under-formed gestures)
    active_mask = (df["label"] != "NEUTRAL")
    active_as_neutral = df[active_mask & (df["rule_pred"] == "NEUTRAL")]

    print("\n" + "=" * 60)
    print("NEUTRAL CONFLICT ANALYSIS")
    print("=" * 60)
    print(f"Frames labeled NEUTRAL that match an active gesture shape: {len(neutral_conflicts)} ({len(neutral_conflicts)/total_rows*100:.1f}%)")
    print(neutral_conflicts["rule_pred"].value_counts().to_string())

    if "person_id" in df.columns:
        print("\nConflicts by person:")
        print(neutral_conflicts["person_id"].value_counts().to_string())

    print("\nActive gesture frames that match NEUTRAL (weak/under-formed gestures):")
    print(active_as_neutral["label"].value_counts().to_string())

    return df, neutral_conflicts, active_as_neutral


def clean_dataset(df, out_path):
    """Removes contradictory frames where a frame labeled NEUTRAL has the exact shape of an active gesture."""
    initial_count = len(df)
    # Remove blatant contradictions: labeled NEUTRAL, but geometric rules recognize PREVIOUS/ARM/NEXT/POINTER with high fidelity
    clean_mask = ~((df["label"] == "NEUTRAL") & (df["rule_pred"] != "NEUTRAL"))
    cleaned_df = df[clean_mask].copy()

    # Drop the temporary diagnostic column
    if "rule_pred" in cleaned_df.columns:
        cleaned_df = cleaned_df.drop(columns=["rule_pred"])

    cleaned_df.to_csv(out_path, index=False)
    pruned_count = initial_count - len(cleaned_df)
    print(f"\n[CLEANING COMPLETE] Pruned {pruned_count} conflicting frames ({pruned_count/initial_count*100:.1f}% of data).")
    print(f"Cleaned dataset written to: {out_path} ({len(cleaned_df)} rows remaining).")
    print("New distribution:")
    print(cleaned_df["label"].value_counts().to_string())
    return cleaned_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose and resolve NEUTRAL gesture conflicts")
    parser.add_argument("--data", default="data/landmarks.csv", help="path to landmark CSV")
    parser.add_argument("--clean", action="store_true", help="export cleaned CSV without blatant NEUTRAL contradictions")
    parser.add_argument("--apply", action="store_true", help="backup and overwrite existing dataset with cleaned data")
    args = parser.parse_args()

    df, n_conflicts, a_neutral = analyze_dataset(args.data)

    if args.clean or args.apply:
        out_target = "data/landmarks_clean.csv" if not args.apply else args.data
        if args.apply:
            backup_path = args.data.replace(".csv", "_backup.csv")
            shutil.copyfile(args.data, backup_path)
            print(f"Created backup at: {backup_path}")
        clean_dataset(df, out_target)
