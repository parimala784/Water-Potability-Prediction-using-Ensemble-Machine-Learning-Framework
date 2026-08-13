import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Default model (legacy path for backward compatibility)
MODEL_PATH = MODELS_DIR / "water_quality_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

FEATURE_NAMES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

# Available models: id -> display name
AVAILABLE_MODELS = {
    "logistic": "Logistic Regression",
    "rf": "Random Forest",
    "brf": "Balanced Random Forest",
    "svm": "SVM (RBF)",
    "extratrees": "Extra Trees",
    "mlp": "MLP Neural Network",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "ensemble": "Ensemble (Voting)",
    "stacking": "Stacking (Meta-Learner)",
}

DEFAULT_MODEL_ID = "ensemble"

BEST_MODEL_JSON = MODELS_DIR / "best_model.json"

# Decision threshold for potable (probability_potable >= this -> potable).
# 0.5: standard boundary; clearly unsafe inputs (e.g. ph 4.5, high THMs) stay Not Potable.
PREDICTION_THRESHOLD = 0.5


def get_default_model_id() -> str:
    """Return best model ID from training, or fallback to DEFAULT_MODEL_ID."""
    if BEST_MODEL_JSON.exists():
        try:
            data = json.loads(BEST_MODEL_JSON.read_text())
            mid = data.get("best_model_id")
            if mid and mid in AVAILABLE_MODELS:
                return mid
        except Exception:
            pass
    return DEFAULT_MODEL_ID


def get_model_path(model_id: str) -> Path:
    """Return path to .pkl file for given model id."""
    return MODELS_DIR / f"model_{model_id}.pkl"
