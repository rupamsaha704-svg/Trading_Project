import pandas as pd

from src.data_validation import validate_dataset


def test_validate_dataset_reports_expected_structure():
    df = pd.DataFrame(
        {
            "time": [pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:05:00")],
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.2],
        }
    )

    report = validate_dataset(df, "test.csv")

    assert report.row_count == 2
    assert report.column_names == ["time", "open", "high", "low", "close"]
    assert report.summary["INFO"] >= 1
