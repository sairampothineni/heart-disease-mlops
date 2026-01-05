import pandas as pd
import pytest
from src.models.model_utils import load_data

def test_load_data_success(tmp_path):
    """Test loading data with target column present"""
    df = pd.DataFrame({
        "age": [60, 55],
        "chol": [240, 260],
        "target": [1, 0]
    })
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)

    loaded_df = load_data(file_path)

    assert not loaded_df.empty
    assert "target" in loaded_df.columns


def test_load_data_missing_target(tmp_path):
    """Test error raised when target column is missing"""
    df = pd.DataFrame({"age": [60, 55]})
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError):
        load_data(file_path)
