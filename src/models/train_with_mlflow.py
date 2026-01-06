import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

# ---------------------------------------------------
# Paths & MLflow configuration (SAFE)
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "notebooks",
    "data",
    "processed",
    "heart_disease_cleaned.csv",
)

MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR}")
mlflow.set_experiment("Heart Disease Classification")


# ---------------------------------------------------
# Main function
# ---------------------------------------------------
def main():

    # ---------------------------------------------------
    # Load dataset
    # ---------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    # ---------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # ---------------------------------------------------
    # Train-test split
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ---------------------------------------------------
    # Models
    # ---------------------------------------------------
    models = {
        "Logistic Regression (ElasticNet)": LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.5,
            C=0.1,
            max_iter=5000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_split=5,
            random_state=42,
            n_jobs=1,  # CI & Windows safe
        ),
    }

    # ---------------------------------------------------
    # Training + MLflow logging
    # ---------------------------------------------------
    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model),
                ]
            )

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            # -----------------------------
            # Metrics
            # -----------------------------
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }

            mlflow.log_metrics(metrics)
            mlflow.log_params(model.get_params())

            # -----------------------------
            # Confusion Matrix Plot
            # -----------------------------
            cm = confusion_matrix(y_test, y_pred)
            disp_cm = ConfusionMatrixDisplay(cm)

            fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
            disp_cm.plot(ax=ax_cm)
            plt.title(f"{model_name} – Confusion Matrix")

            cm_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
            plt.savefig(cm_path, bbox_inches="tight")
            mlflow.log_artifact(cm_path)
            plt.close(fig_cm)

            # -----------------------------
            # ROC Curve Plot (NEW)
            # -----------------------------
            fig_roc, ax_roc = plt.subplots(figsize=(5, 5))
            RocCurveDisplay.from_predictions(
                y_test,
                y_proba,
                ax=ax_roc,
            )
            plt.title(f"{model_name} – ROC Curve")

            roc_path = f"roc_curve_{model_name.replace(' ', '_')}.png"
            plt.savefig(roc_path, bbox_inches="tight")
            mlflow.log_artifact(roc_path)
            plt.close(fig_roc)

            # -----------------------------
            # Log Model
            # -----------------------------
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
            )

            print(f"\n{model_name} logged successfully")
            print("Metrics:", metrics)


# ---------------------------------------------------
# Entry point
# ---------------------------------------------------
if __name__ == "__main__":
    main()
