"""
Generate and save confusion matrices and model comparison figures.
Run from backend/: python save_model_figures.py
Saves to project root images/ folder.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

# Run from backend/
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))
from app.config import AVAILABLE_MODELS, get_model_path, BEST_MODEL_JSON
from app.features import RAW_FEATURES, add_derived_features, ALL_FEATURES


def main():
    # Load data
    df = pd.read_csv(BACKEND_DIR / "data" / "water_potability.csv")
    for col in RAW_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    df = add_derived_features(df)
    X = df[ALL_FEATURES]
    y = df["Potability"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = joblib.load(BACKEND_DIR / "models" / "scaler.pkl")
    X_test_s = scaler.transform(X_test)

    # Load models
    models_dir = BACKEND_DIR / "models"
    model_ids = [
        k for k in AVAILABLE_MODELS
        if (models_dir / f"model_{k}.pkl").exists()
    ]
    models = {}
    for mid in model_ids:
        models[mid] = joblib.load(get_model_path(mid))

    best_id = "ensemble"
    if BEST_MODEL_JSON.exists():
        try:
            data = json.loads(BEST_MODEL_JSON.read_text())
            best_id = data.get("best_model_id", best_id)
        except Exception:
            pass

    # 1. Confusion matrix grid (all models)
    n = len(model_ids)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    for idx, mid in enumerate(model_ids):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        pred = models[mid].predict(X_test_s)
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["N", "P"], yticklabels=["N", "P"])
        ax.set_title(AVAILABLE_MODELS.get(mid, mid))
        ax.set_xlabel("Pred")
        ax.set_ylabel("Actual")
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")
    plt.tight_layout()
    plt.suptitle("Confusion Matrices - All Models (N=Not Potable, P=Potable)", y=1.02, fontsize=12)
    fig.savefig(IMAGES_DIR / "confusion_matrix_grid.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: confusion_matrix_grid.png")

    # 2. Model comparison (Accuracy, F1, ROC-AUC)
    metrics = []
    for mid in model_ids:
        pred = models[mid].predict(X_test_s)
        proba = models[mid].predict_proba(X_test_s)[:, 1]
        metrics.append({
            "model": AVAILABLE_MODELS.get(mid, mid),
            "id": mid,
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred, average="weighted"),
            "roc_auc": roc_auc_score(y_test, proba),
        })
    df_metrics = pd.DataFrame(metrics).sort_values("accuracy", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = range(len(df_metrics))
    colors = ["#2ecc71" if r["id"] == best_id else "#3498db" for _, r in df_metrics.iterrows()]
    axes[0].barh(x, df_metrics["accuracy"], color=colors)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(df_metrics["model"], fontsize=9)
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("Accuracy by Model")
    axes[0].set_xlim(0, 1)
    axes[1].barh(x, df_metrics["f1"], color=colors)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(df_metrics["model"], fontsize=9)
    axes[1].set_xlabel("F1 (weighted)")
    axes[1].set_title("F1 Score by Model")
    axes[1].set_xlim(0, 1)
    axes[2].barh(x, df_metrics["roc_auc"], color=colors)
    axes[2].set_yticks(x)
    axes[2].set_yticklabels(df_metrics["model"], fontsize=9)
    axes[2].set_xlabel("ROC-AUC")
    axes[2].set_title("ROC-AUC by Model")
    axes[2].set_xlim(0, 1)
    plt.tight_layout()
    fig.suptitle("Model Comparison (green = best model)", y=1.02, fontsize=12)
    fig.savefig(IMAGES_DIR / "model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: model_comparison.png")

    # 3. Individual confusion matrices (per model)
    cm_dir = IMAGES_DIR / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    for mid in model_ids:
        pred = models[mid].predict(X_test_s)
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Not Potable", "Potable"],
                    yticklabels=["Not Potable", "Potable"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {AVAILABLE_MODELS.get(mid, mid)}")
        plt.tight_layout()
        fig.savefig(cm_dir / f"confusion_matrix_{mid}.png", dpi=120, bbox_inches="tight")
        plt.close()
    print(f"Saved: confusion_matrices/confusion_matrix_*.png ({len(model_ids)} files)")

    # 4. ROC curves comparison (all models in one figure)
    fig, ax = plt.subplots(figsize=(8, 6))
    for mid in model_ids:
        proba = models[mid].predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        label = f"{AVAILABLE_MODELS.get(mid, mid)} (AUC={auc:.3f})"
        ax.plot(fpr, tpr, lw=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Models")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "roc_curves_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: roc_curves_comparison.png")

    print(f"\nAll figures saved to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
