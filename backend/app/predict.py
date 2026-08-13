import joblib
import numpy as np

# Ensure StackingPredictor is importable for joblib unpickling
from . import stacking  # noqa: F401

from .config import SCALER_PATH, FEATURE_NAMES, AVAILABLE_MODELS, get_model_path, get_default_model_id, PREDICTION_THRESHOLD
from .features import prepare_features_from_dict

# WHO/EPA-style limits: water is clearly unsafe if any of these are violated.
# Used to override model when it would wrongly say "Safe to Drink".
UNSAFE_LIMITS = {
    "ph": (5.0, 10.0),           # outside 5-10 is clearly unsafe (strict: WHO 6.5-8.5)
    "Trihalomethanes": 80.0,     # EPA MCL 80 µg/L
    "Turbidity": 5.0,             # very high (>5 NTU) clearly unsafe
    "Sulfate": 500.0,             # very high sulfate
}


def _is_clearly_unsafe(features: dict) -> bool:
    """Return True if input violates known unsafe limits (override to Not Potable)."""
    ph = features.get("ph")
    if ph is not None:
        try:
            p = float(ph)
            if p < UNSAFE_LIMITS["ph"][0] or p > UNSAFE_LIMITS["ph"][1]:
                return True
        except (TypeError, ValueError):
            pass
    thm = features.get("Trihalomethanes")
    if thm is not None:
        try:
            if float(thm) > UNSAFE_LIMITS["Trihalomethanes"]:
                return True
        except (TypeError, ValueError):
            pass
    turb = features.get("Turbidity")
    if turb is not None:
        try:
            if float(turb) > UNSAFE_LIMITS["Turbidity"]:
                return True
        except (TypeError, ValueError):
            pass
    sulf = features.get("Sulfate")
    if sulf is not None:
        try:
            if float(sulf) > UNSAFE_LIMITS["Sulfate"]:
                return True
        except (TypeError, ValueError):
            pass
    return False


def load_model(model_id: str):
    path = get_model_path(model_id)
    if not path.exists():
        # Fallback to legacy default model if specific model missing
        from .config import MODELS_DIR, MODEL_PATH
        fallback = MODEL_PATH if MODEL_PATH.exists() else None
        if fallback and model_id in AVAILABLE_MODELS:
            return joblib.load(fallback)
        raise FileNotFoundError(f"Model '{model_id}' not found at {path}. Run training first.")
    return joblib.load(path)


def load_scaler():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Run training first.")
    return joblib.load(SCALER_PATH)


def predict(features: dict, model_id: str | None = None) -> dict:
    mid = model_id or get_default_model_id()
    model = load_model(mid)
    scaler = load_scaler()
    X = prepare_features_from_dict(features)
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0]
    prob_potable = float(proba[1])
    prob_not_potable = float(proba[0])

    # Override: clearly unsafe water (WHO/EPA limits) -> always Not Potable
    if _is_clearly_unsafe(features):
        prob_potable = min(prob_potable, 0.49)
        prob_not_potable = 1.0 - prob_potable
        pred = 0
    else:
        pred = 1 if prob_potable >= PREDICTION_THRESHOLD else 0

    return {
        "prediction": pred,
        "potable": bool(pred),
        "probability_potable": prob_potable,
        "probability_not_potable": prob_not_potable,
        "feature_values": features,
        "model_used": mid,
    }
