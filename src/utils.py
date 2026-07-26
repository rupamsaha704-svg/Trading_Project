import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_data(file_name):
    data_path = Path(file_name)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    df = pd.read_csv(data_path)
    df["time"] = pd.to_datetime(df["time"])
    return df


def save_data(df, file_name):
    output_path = Path(file_name)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    df.to_csv(output_path, index=False)