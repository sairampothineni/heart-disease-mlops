import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


TARGET_COL = "target"


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if TARGET_COL not in df.columns:
        raise ValueError("Target column missing")
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def build_preprocessor(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_cols = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def build_model_pipeline(preprocessor):
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf),
        ]
    )
