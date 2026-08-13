"""
Stacking predictor: meta-learner combining base model probabilities.
Moved here so joblib can deserialize it when loading from predict.py.
"""
import numpy as np


class StackingPredictor:
    """Meta-learner that combines base model probabilities for final prediction."""

    def __init__(self, base_models: dict, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        """No-op; already fitted externally."""
        return self

    def predict_proba(self, X):
        probas = [
            m.predict_proba(X)[:, 1]
            for m in self.base_models.values()
        ]
        X_meta = np.column_stack(probas)
        p = self.meta_model.predict_proba(X_meta)
        return p

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
