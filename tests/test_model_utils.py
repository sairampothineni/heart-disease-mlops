import pandas as pd                          # Pandas for creating test DataFrames
import pytest                                # Pytest framework for testing and assertions
from src.models.model_utils import load_data # Import the function under test


def test_load_data_success(tmp_path):
    """Test loading data with target column present"""  # Test description for successful case
    df = pd.DataFrame({                     # Create a sample DataFrame with required columns
        "age": [60, 55],                    # Example age values
        "chol": [240, 260],                 # Example cholesterol values
        "target": [1, 0]                   # Target column required by the model
    })
    file_path = tmp_path / "data.csv"       # Create a temporary file path using pytest fixture
    df.to_csv(file_path, index=False)       # Save DataFrame to CSV without index column

    loaded_df = load_data(file_path)        # Call load_data function with CSV file path

    assert not loaded_df.empty              # Ensure returned DataFrame is not empty
    assert "target" in loaded_df.columns   # Verify target column exists in loaded data


def test_load_data_missing_target(tmp_path):
    """Test error raised when target column is missing"""  # Test description for failure case
    df = pd.DataFrame({"age": [60, 55]})    # Create DataFrame without target column
    file_path = tmp_path / "data.csv"       # Create temporary CSV file path
    df.to_csv(file_path, index=False)       # Save invalid DataFrame to CSV

    with pytest.raises(ValueError):          # Expect ValueError due to missing target column
        load_data(file_path)                # Call function which should raise the error
