"""
Train water potability classifiers with feature engineering, SMOTE, ensemble, and hyperparameter tuning.
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from app.config import (
    DATA_DIR,
    MODELS_DIR,
    SCALER_PATH,
    MODEL_PATH,
    AVAILABLE_MODELS,
    get_model_path,
)
from app.features import RAW_FEATURES, add_derived_features, ALL_FEATURES
from app.stacking import StackingPredictor

DATA_PATH = DATA_DIR / "water_potability.csv"

# Use SMOTE to balance classes (helps minority "Potable" class)
USE_SMOTE = True
# Synthetic data: add jittered copies to expand training set
USE_SYNTHETIC_JITTER = True
SYNTHETIC_COPIES = 1       # extra copies per sample (1 = 2x total; 2 = 3x)
JITTER_NOISE = 0.025       # std of Gaussian noise (relative to scaled features)
# Tuning metric: "accuracy" prioritizes overall correctness
TUNING_SCORING = "accuracy"
N_ITER = 16  # Search iterations (increase for better params, slower training)


def add_synthetic_jitter(X: np.ndarray, y: np.ndarray, n_copies: int = 2, noise: float = 0.025, seed: int = 42) -> tuple:
    """Expand dataset by adding copies with small Gaussian noise (jitter augmentation)."""
    rng = np.random.default_rng(seed)
    X_list, y_list = [X], [y]
    for _ in range(n_copies):
        X_jitter = X + noise * rng.standard_normal(X.shape)
        X_list.append(X_jitter)
        y_list.append(y)
    return np.vstack(X_list), np.concatenate(y_list)


def train_and_evaluate(model, model_id: str, X_train, X_test, y_train, y_test):
    """Train model and return (model, accuracy, f1, auc)."""
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    auc = roc_auc_score(y_test, proba)
    print(f"\n--- {AVAILABLE_MODELS.get(model_id, model_id)} ---")
    print("Accuracy:", acc)
    print("F1 (weighted):", f1)
    print("ROC-AUC:", auc)
    return model, acc, f1, auc


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Imputation
    for col in RAW_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Feature engineering
    print("Adding derived features...")
    df = add_derived_features(df)

    X = df[ALL_FEATURES]
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_train_orig, y_train_orig = X_train_s.copy(), y_train.copy()

    if USE_SMOTE:
        print("Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_s, y_train = smote.fit_resample(X_train_s, y_train)
        print(f"  Resampled train size: {len(y_train)}")

    if USE_SYNTHETIC_JITTER:
        print("Adding synthetic jittered samples...")
        X_train_s, y_train = add_synthetic_jitter(
            X_train_s, y_train,
            n_copies=SYNTHETIC_COPIES,
            noise=JITTER_NOISE,
            seed=42,
        )
        print(f"  Augmented train size: {len(y_train)}")

    joblib.dump(scaler, SCALER_PATH)
    results = {}

    # 1. Logistic Regression (tuned)
    print("\nTraining Logistic Regression (with hyperparameter search)...")
    lr_base = LogisticRegression(max_iter=5000, random_state=42, n_jobs=1)
    lr_params = {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["lbfgs"],  # lbfgs converges faster; saga can hit max_iter
        "class_weight": [None, "balanced"],
    }
    lr_search = RandomizedSearchCV(
        lr_base, lr_params, n_iter=N_ITER, cv=5, scoring=TUNING_SCORING,
        random_state=42, n_jobs=1, verbose=0
    )
    lr_search.fit(X_train_s, y_train)
    lr = lr_search.best_estimator_
    m, acc, f1, auc = train_and_evaluate(lr, "logistic", X_train_s, X_test_s, y_train, y_test)
    results["logistic"] = (m, acc, f1)
    joblib.dump(m, get_model_path("logistic"))

    # 2. Random Forest (tuned)
    print("\nTraining Random Forest (with hyperparameter search)...")
    rf_base = RandomForestClassifier(random_state=42, n_jobs=1)
    param_dist = {
        "n_estimators": [150, 200, 300, 400],
        "max_depth": [10, 14, 18, 22, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, "log2"],
    }
    rf_search = RandomizedSearchCV(
        rf_base, param_dist, n_iter=N_ITER, cv=5, scoring=TUNING_SCORING,
        random_state=42, n_jobs=1, verbose=0
    )
    rf_search.fit(X_train_s, y_train)
    rf = rf_search.best_estimator_
    m, acc, f1, auc = train_and_evaluate(rf, "rf", X_train_s, X_test_s, y_train, y_test)
    results["rf"] = (m, acc, f1)
    joblib.dump(m, get_model_path("rf"))

    # 3. Balanced Random Forest (tuned)
    print("\nTraining Balanced Random Forest (with hyperparameter search)...")
    brf_base = BalancedRandomForestClassifier(random_state=42, n_jobs=1)
    brf_params = {
        "n_estimators": [150, 200, 300],
        "max_depth": [8, 12, 16, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5],
    }
    brf_search = RandomizedSearchCV(
        brf_base, brf_params, n_iter=N_ITER, cv=5, scoring=TUNING_SCORING,
        random_state=42, n_jobs=1, verbose=0
    )
    brf_search.fit(X_train_s, y_train)
    brf = brf_search.best_estimator_
    m, acc, f1, auc = train_and_evaluate(brf, "brf", X_train_s, X_test_s, y_train, y_test)
    results["brf"] = (m, acc, f1)
    joblib.dump(m, get_model_path("brf"))

    # 4. XGBoost (tuned)
    try:
        import xgboost as xgb
        print("\nTraining XGBoost (with hyperparameter search)...")
        xgb_base = xgb.XGBClassifier(random_state=42, n_jobs=1)
        xgb_params = {
            "n_estimators": [150, 200, 300],
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.05, 0.1, 0.15],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.8, 1.0],
        }
        xgb_search = RandomizedSearchCV(
            xgb_base, xgb_params, n_iter=N_ITER, cv=5, scoring=TUNING_SCORING,
            random_state=42, n_jobs=1, verbose=0
        )
        xgb_search.fit(X_train_s, y_train)
        xgb_model = xgb_search.best_estimator_
        m, acc, f1, auc = train_and_evaluate(xgb_model, "xgboost", X_train_s, X_test_s, y_train, y_test)
        results["xgboost"] = (m, acc, f1)
        joblib.dump(m, get_model_path("xgboost"))
    except ImportError:
        pass

    # 5. CatBoost (tuned)
    try:
        from catboost import CatBoostClassifier
        print("\nTraining CatBoost (with hyperparameter search)...")
        cat_base = CatBoostClassifier(random_state=42, verbose=0, thread_count=1)
        cat_params = {
            "iterations": [150, 200, 300],
            "depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.15],
            "l2_leaf_reg": [1, 3, 5],
        }
        cat_search = RandomizedSearchCV(
            cat_base, cat_params, n_iter=N_ITER, cv=5, scoring=TUNING_SCORING,
            random_state=42, n_jobs=1, verbose=0
        )
        cat_search.fit(X_train_s, y_train)
        cat_model = cat_search.best_estimator_
        m, acc, f1, auc = train_and_evaluate(cat_model, "catboost", X_train_s, X_test_s, y_train, y_test)
        results["catboost"] = (m, acc, f1)
        joblib.dump(m, get_model_path("catboost"))
    except ImportError:
        pass

    # 6. LightGBM (tuned)
    try:
        import lightgbm as lgb
        print("\nTraining LightGBM (with hyperparameter search)...")
        lgb_base = lgb.LGBMClassifier(random_state=42, n_jobs=1, verbose=-1)
        lgb_params = {
            "n_estimators": [150, 200, 300],
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.05, 0.1, 0.15],
            "num_leaves": [31, 63, 127],
            "min_child_samples": [10, 20, 30],
        }
        lgb_search = RandomizedSearchCV(
            lgb_base, lgb_params, n_iter=N_ITER, cv=5, scoring=TUNING_SCORING,
            random_state=42, n_jobs=1, verbose=0
        )
        lgb_search.fit(X_train_s, y_train)
        lgb_model = lgb_search.best_estimator_
        m, acc, f1, auc = train_and_evaluate(lgb_model, "lightgbm", X_train_s, X_test_s, y_train, y_test)
        results["lightgbm"] = (m, acc, f1)
        joblib.dump(m, get_model_path("lightgbm"))
    except ImportError:
        pass

    # 7. SVM with RBF kernel (tuned)
    print("\nTraining SVM (RBF kernel, hyperparameter search)...")
    svm_base = SVC(kernel="rbf", probability=True, random_state=42)
    svm_params = {
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 0.01, 0.001],
        "class_weight": [None, "balanced"],
    }
    svm_search = RandomizedSearchCV(
        svm_base, svm_params, n_iter=min(N_ITER, 12), cv=4, scoring=TUNING_SCORING,
        random_state=42, n_jobs=1, verbose=0
    )
    svm_search.fit(X_train_s, y_train)
    svm = svm_search.best_estimator_
    m, acc, f1, auc = train_and_evaluate(svm, "svm", X_train_s, X_test_s, y_train, y_test)
    results["svm"] = (m, acc, f1)
    joblib.dump(m, get_model_path("svm"))

    # 8. Extra Trees (tuned)
    print("\nTraining Extra Trees (with hyperparameter search)...")
    et_base = ExtraTreesClassifier(random_state=42, n_jobs=1)
    et_params = {
        "n_estimators": [150, 200, 300, 400],
        "max_depth": [10, 14, 18, 22, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.5, "log2"],
    }
    et_search = RandomizedSearchCV(
        et_base, et_params, n_iter=N_ITER, cv=5, scoring=TUNING_SCORING,
        random_state=42, n_jobs=1, verbose=0
    )
    et_search.fit(X_train_s, y_train)
    et = et_search.best_estimator_
    m, acc, f1, auc = train_and_evaluate(et, "extratrees", X_train_s, X_test_s, y_train, y_test)
    results["extratrees"] = (m, acc, f1)
    joblib.dump(m, get_model_path("extratrees"))

    # 9. MLP Neural Network (tuned)
    print("\nTraining MLP Neural Network (with hyperparameter search)...")
    mlp_base = MLPClassifier(random_state=42, max_iter=1000, early_stopping=True, validation_fraction=0.1)
    mlp_params = {
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
        "activation": ["relu", "tanh"],
    }
    mlp_search = RandomizedSearchCV(
        mlp_base, mlp_params, n_iter=min(N_ITER, 12), cv=4, scoring=TUNING_SCORING,
        random_state=42, n_jobs=1, verbose=0
    )
    mlp_search.fit(X_train_s, y_train)
    mlp = mlp_search.best_estimator_
    m, acc, f1, auc = train_and_evaluate(mlp, "mlp", X_train_s, X_test_s, y_train, y_test)
    results["mlp"] = (m, acc, f1)
    joblib.dump(m, get_model_path("mlp"))

    # 10. Ensemble (soft voting)
    print("\nTraining Ensemble (soft voting)...")
    estimators = [(k, results[k][0]) for k in ["logistic", "rf", "brf", "svm", "extratrees", "mlp"] if k in results]
    if "xgboost" in results:
        estimators.append(("xgboost", results["xgboost"][0]))
    if "catboost" in results:
        estimators.append(("catboost", results["catboost"][0]))
    if "lightgbm" in results:
        estimators.append(("lightgbm", results["lightgbm"][0]))
    if len(estimators) < 2:
        estimators = [(k, results[k][0]) for k in results]
    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
    m, acc, f1, auc = train_and_evaluate(ensemble, "ensemble", X_train_s, X_test_s, y_train, y_test)
    results["ensemble"] = (m, acc, f1)
    joblib.dump(m, get_model_path("ensemble"))

    # 11. Stacking (meta-learner on base model probabilities)
    print("\nTraining Stacking ensemble...")
    base_names = [k for k in ["logistic", "rf", "brf", "svm", "extratrees", "mlp", "xgboost", "catboost", "lightgbm"] if k in results]
    if len(base_names) >= 2:
        base_dict = {k: results[k][0] for k in base_names}
        probas = [base_dict[k].predict_proba(X_train_orig)[:, 1] for k in base_names]
        X_meta = np.column_stack(probas)
        meta = LogisticRegression(max_iter=1000, random_state=42, C=0.5)
        meta.fit(X_meta, y_train_orig)
        stacking = StackingPredictor(base_dict, meta)
        m, acc, f1, auc = train_and_evaluate(stacking, "stacking", X_train_orig, X_test_s, y_train_orig, y_test)
        results["stacking"] = (m, acc, f1)
        joblib.dump(m, get_model_path("stacking"))

    # Best model by accuracy (then F1)
    best_id = max(results.keys(), key=lambda k: (results[k][1], results[k][2]))
    joblib.dump(results[best_id][0], MODEL_PATH)
    best_acc = results[best_id][1]
    import json
    best_info = {"best_model_id": best_id, "accuracy": float(best_acc)}
    (MODELS_DIR / "best_model.json").write_text(json.dumps(best_info, indent=2))
    print(f"\nDefault model set to: {AVAILABLE_MODELS.get(best_id, best_id)} (Accuracy: {best_acc:.2%})")
    print(f"All models saved to {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
