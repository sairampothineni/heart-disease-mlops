import os
import time
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ------------------------------------------------------------------
# Progress helper
# ------------------------------------------------------------------


def log_step(message):
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
# ✅ build_pipeline (USED BY PYTEST)
# ------------------------------------------------------------------


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Build and return a Random Forest pipeline.
    This function is SAFE to import in tests.
    """

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=5,
        random_state=42,
        n_jobs=1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


# ------------------------------------------------------------------
# Main execution (NOT run during tests)
# ------------------------------------------------------------------


def main():

    log_step("Starting Random Forest training with preprocessing pipeline")

    log_step("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    log_step(f"Dataset loaded with shape {df.shape}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    log_step("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(X_train)

    log_step("Training model...")
    pipeline.fit(X_train, y_train)

    log_step("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    for key, value in metrics.items():
        log_step(f"{key}: {value:.4f}")

    log_step("Saving model...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    log_step(f"Model saved at {MODEL_PATH}")
    log_step("Training completed successfully")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    main()
