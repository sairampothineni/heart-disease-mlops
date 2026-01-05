"""
Task 2: Model Training, Hyperparameter Tuning, and Evaluation
Heart Disease Dataset
"""

# ===============================
# 1. IMPORTS
# ===============================

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ===============================
# 2. LOAD DATA (CORRECT PATH)
# ===============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "notebooks" / "data" / "processed" / "heart_disease_cleaned.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]

# ===============================
# 3. TRAIN-TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 4. PREPROCESSING
# ===============================

numeric_features = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ]
)

# ===============================
# 5. MODELS
# ===============================

log_reg_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(
            solver="saga",
            max_iter=1000,
            random_state=42
        ))
    ]
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=42
        ))
    ]
)

# ===============================
# 6. HYPERPARAMETERS (NO PENALTY!)
# ===============================

log_reg_param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__l1_ratio": [0.25, 0.5, 0.75]
}

rf_param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5]
}

# ===============================
# 7. CROSS-VALIDATION
# ===============================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ===============================
# 8. GRID SEARCH (WINDOWS SAFE)
# ===============================

log_reg_grid = GridSearchCV(
    log_reg_pipeline,
    log_reg_param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=1
)

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=1
)

log_reg_grid.fit(X_train, y_train)
rf_grid.fit(X_train, y_train)

# ===============================
# 9. EVALUATION FUNCTION
# ===============================


def evaluate(model):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }


# ===============================
# 10. RESULTS
# ===============================

results = pd.DataFrame(
    [
        evaluate(log_reg_grid.best_estimator_),
        evaluate(rf_grid.best_estimator_)
    ],
    index=[
        "Logistic Regression (ElasticNet)",
        "Random Forest"
    ]
)

print("\nModel Performance Comparison:")
print(results)

print("\nBest Logistic Regression Parameters:")
print(log_reg_grid.best_params_)

print("\nBest Random Forest Parameters:")
print(rf_grid.best_params_)
