import pandas as pd
from sklearn.pipeline import Pipeline
from src.models.model_utils import load_data
from src.models.train_rf_with_pipeline import build_pipeline

def test_training_pipeline_runs(tmp_path):
    """Ensure model pipeline trains without errors"""
    df = pd.DataFrame({
        "age": [60, 55, 50, 45],
        "chol": [240, 260, 230, 220],
        "target": [1, 0, 1, 0]
    })

    file_path = tmp_path / "train.csv"
    df.to_csv(file_path, index=False)

    data = load_data(file_path)
    X = data.drop(columns=["target"])
    y = data["target"]

    pipeline = build_pipeline(X)
    pipeline.fit(X, y)

    assert isinstance(pipeline, Pipeline)
