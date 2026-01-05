import pandas as pd                          # Pandas for creating training DataFrames
from sklearn.pipeline import Pipeline        # Scikit-learn Pipeline class
from src.models.model_utils import load_data # Function to load and validate dataset
from src.models.train_rf_with_pipeline import build_pipeline  # Function to build ML pipeline

def test_training_pipeline_runs(tmp_path):
    """Ensure model pipeline trains without errors"""  # Test description
    df = pd.DataFrame({                     # Create a small synthetic training dataset
        "age": [60, 55, 50, 45],             # Sample age values
        "chol": [240, 260, 230, 220],        # Sample cholesterol values
        "target": [1, 0, 1, 0]              # Binary target labels
    })

    file_path = tmp_path / "train.csv"       # Temporary file path for training data
    df.to_csv(file_path, index=False)        # Save dataset to CSV file

    data = load_data(file_path)              # Load data using project utility function
    X = data.drop(columns=["target"])        # Separate features (X)
    y = data["target"]                       # Separate target labels (y)

    pipeline = build_pipeline(X)             # Build preprocessing + model pipeline
    pipeline.fit(X, y)                       # Train the pipeline on the dataset

    assert isinstance(pipeline, Pipeline)    # Ensure returned object is a sklearn Pipeline
