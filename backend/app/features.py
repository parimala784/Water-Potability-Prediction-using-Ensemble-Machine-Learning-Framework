"""
Feature engineering for water potability prediction.
Same transforms applied at train and predict time.
"""
import numpy as np
import pandas as pd

RAW_FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

DERIVED_FEATURES = [
    "ph_safe",           # 1 if 6.5 <= ph <= 8.5 else 0
    "turbidity_safe",    # 1 if turbidity < 1 else 0
    "solids_log",        # log(solids + 1)
    "conductivity_ratio",  # conductivity / (solids + 1)
    "ph_turbidity",      # ph * turbidity interaction
]

ALL_FEATURES = RAW_FEATURES + DERIVED_FEATURES


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Input df must have RAW_FEATURES columns."""
    out = df.copy()
    out["ph_safe"] = ((out["ph"] >= 6.5) & (out["ph"] <= 8.5)).astype(float)
    out["turbidity_safe"] = (out["Turbidity"] < 1.0).astype(float)
    out["solids_log"] = np.log1p(out["Solids"])
    out["conductivity_ratio"] = out["Conductivity"] / (out["Solids"] + 1)
    out["ph_turbidity"] = out["ph"] * out["Turbidity"]
    return out


def prepare_features_from_dict(features: dict) -> np.ndarray:
    """Convert raw feature dict to array with derived features for prediction."""
    raw = np.array([[float(features.get(f, 0)) for f in RAW_FEATURES]])
    df = pd.DataFrame(raw, columns=RAW_FEATURES)
    df = add_derived_features(df)
    return df[ALL_FEATURES].values
