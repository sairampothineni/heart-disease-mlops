import os
import time
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ------------------------------------------------------------------
# Progress helper
# ------------------------------------------------------------------
def log_step(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(
    BASE_DIR, "notebooks", "data", "processed", "heart_disease_cleaned.csv"
)
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_pipeline.pkl")
TARGET_COL = "target"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


# ------------------------------------------------------------------
# build_pipeline (Random Forest) – USED BY PYTEST
# ------------------------------------------------------------------
def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_cols = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = RandomForestClassifier(
        random_state=42,
        n_jobs=1,  # CI & Windows safe
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


# ------------------------------------------------------------------
# Logistic Regression pipeline
# ------------------------------------------------------------------
def build_logistic_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_cols = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = LogisticRegression(
        solver="saga",
        max_iter=1000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
def main() -> None:

    log_step("Starting model training and evaluation")

    # ------------------------------
    # Load data
    # ------------------------------
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # ------------------------------
    # Train / test split
    # ------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # ======================================================
    # 1️⃣ LOGISTIC REGRESSION – BASELINE MODEL
    # ======================================================
    log_step("Evaluating Logistic Regression with cross-validation")

    log_reg_pipeline = build_logistic_pipeline(X_train)

    log_reg_scores = cross_val_score(
        log_reg_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
    )

    log_reg_mean = log_reg_scores.mean()
    log_reg_std = log_reg_scores.std()

    log_step(
        f"Logistic Regression CV ROC-AUC: "
        f"{log_reg_mean:.4f} ± {log_reg_std:.4f}"
    )

    # ======================================================
    # 2️⃣ RANDOM FOREST – GRID SEARCH
    # ======================================================
    rf_pipeline = build_pipeline(X_train)

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
        verbose=2,
    )

    log_step("Running GridSearchCV for Random Forest")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # ------------------------------
    # Cross-validation results
    # ------------------------------
    cv_results = pd.DataFrame(grid_search.cv_results_)
    best_index = grid_search.best_index_

    rf_mean = cv_results.loc[best_index, "mean_test_score"]
    rf_std = cv_results.loc[best_index, "std_test_score"]

    log_step(
        f"Random Forest CV ROC-AUC: "
        f"{rf_mean:.4f} ± {rf_std:.4f}"
    )

    # ======================================================
    # 3️⃣ MODEL COMPARISON (REQUIRED)
    # ======================================================
    comparison_df = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest"],
            "CV ROC-AUC Mean": [log_reg_mean, rf_mean],
            "CV ROC-AUC Std": [log_reg_std, rf_std],
        }
    )

    log_step("Model comparison:")
    log_step(comparison_df.to_string(index=False))

    # ======================================================
    # 4️⃣ TEST SET EVALUATION (FINAL MODEL)
    # ======================================================
    log_step("Evaluating best Random Forest on test set")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    for key, value in test_metrics.items():
        log_step(f"{key}: {value:.4f}")

    # ======================================================
    # Save final model
    # ======================================================
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    log_step(f"Final model saved at {MODEL_PATH}")
    log_step("Training completed successfully")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
