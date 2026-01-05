import pandas as pd
from src.models.model_utils import build_preprocessor

def test_preprocessor_creation():
    df = pd.DataFrame({
        "age": [50, 60],
        "chol": [200, 240],
        "sex": ["M", "F"]
    })

    preprocessor = build_preprocessor(df)

    assert preprocessor is not None
    assert hasattr(preprocessor, "fit")
