from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .predict import predict as run_predict
from .config import FEATURE_NAMES, AVAILABLE_MODELS, get_default_model_id

app = FastAPI(
    title="Water Quality Prediction API",
    description="ML API for water potability prediction and visualization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionInput(BaseModel):
    ph: Optional[float] = None
    Hardness: Optional[float] = None
    Solids: Optional[float] = None
    Chloramines: Optional[float] = None
    Sulfate: Optional[float] = None
    Conductivity: Optional[float] = None
    Organic_carbon: Optional[float] = None
    Trihalomethanes: Optional[float] = None
    Turbidity: Optional[float] = None

    class Config:
        extra = "forbid"


@app.get("/")
def root():
    return {"message": "Water Quality Prediction API", "docs": "/docs"}


@app.get("/api/features")
def get_features():
    return {"features": FEATURE_NAMES}


@app.get("/api/models")
def get_models():
    default_id = get_default_model_id()
    return {
        "models": [{"id": k, "name": v} for k, v in AVAILABLE_MODELS.items()],
        "default_model_id": default_id,
    }


@app.post("/api/predict")
def predict_endpoint(input_data: PredictionInput, model_id: Optional[str] = None):
    features = input_data.model_dump()
    if all(v is None for v in features.values()):
        raise HTTPException(status_code=400, detail="At least one feature value required")
    # Fill None with 0 for missing optional fields
    for k in features:
        if features[k] is None:
            features[k] = 0.0
    try:
        result = run_predict(features, model_id=model_id)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/stats")
def get_dataset_stats():
    import pandas as pd
    from .config import DATA_DIR, FEATURE_NAMES
    path = DATA_DIR / "water_potability.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = pd.read_csv(path)
    # Basic stats
    stats = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "potability_counts": df["Potability"].value_counts().to_dict(),
        "missing_per_column": df.isnull().sum().to_dict(),
    }
    # Per-feature means and standard deviations for numeric features used by the model
    try:
        feature_means = df[FEATURE_NAMES].mean().to_dict()
        feature_stds = df[FEATURE_NAMES].std().to_dict()
        # Ensure plain Python floats for JSON serialization
        stats["feature_means"] = {k: float(v) for k, v in feature_means.items()}
        stats["feature_stds"] = {k: float(v) for k, v in feature_stds.items()}
    except Exception:
        # Fallback gracefully if columns are missing
        stats["feature_means"] = {}
        stats["feature_stds"] = {}
    return stats
