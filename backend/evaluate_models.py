"""
Evaluate all trained models on the test set and print Accuracy, F1, ROC-AUC.
Uses same split as train.py (random_state=42, test_size=0.2, stratify=y).
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from app.config import MODELS_DIR, SCALER_PATH, AVAILABLE_MODELS, get_model_path
from app.features import RAW_FEATURES, add_derived_features, ALL_FEATURES

DATA_PATH = BASE / "data" / "water_potability.csv"


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    for col in RAW_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    df = add_derived_features(df)
    X = df[ALL_FEATURES]
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = joblib.load(SCALER_PATH)
    X_test_s = scaler.transform(X_test)

    results = []
    for model_id, name in AVAILABLE_MODELS.items():
        path = get_model_path(model_id)
        if not path.exists():
            continue
        try:
            model = joblib.load(path)
            pred = model.predict(X_test_s)
            proba = model.predict_proba(X_test_s)[:, 1]
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average="weighted")
            auc = roc_auc_score(y_test, proba)
            results.append((name, acc, f1, auc))
        except Exception as e:
            results.append((name, float("nan"), float("nan"), float("nan")))

    print("\n" + "=" * 80)
    print("MODEL ACCURACY REPORT (test set, 20% holdout)")
    print("=" * 80)
    print(f"{'Model':<30} {'Accuracy':>12} {'F1 (weighted)':>14} {'ROC-AUC':>10}")
    print("-" * 80)
    for name, acc, f1, auc in sorted(results, key=lambda x: -x[1]):
        print(f"{name:<30} {acc:>11.2%} {f1:>14.4f} {auc:>10.4f}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
