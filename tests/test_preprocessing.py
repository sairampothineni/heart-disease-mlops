import pandas as pd                          # Pandas for creating sample DataFrames
from src.models.model_utils import build_preprocessor  # Import function that builds the preprocessing pipeline

def test_preprocessor_creation():
    df = pd.DataFrame({                     # Create a sample DataFrame to simulate input features
        "age": [50, 60],                    # Numerical feature: age
        "chol": [200, 240],                 # Numerical feature: cholesterol
        "sex": ["M", "F"]                   # Categorical feature: sex
    })

    preprocessor = build_preprocessor(df)   # Build preprocessing pipeline using the sample DataFrame

    assert preprocessor is not None          # Ensure the preprocessor object is created
    assert hasattr(preprocessor, "fit")     # Verify preprocessor has a fit method (sklearn-style)
