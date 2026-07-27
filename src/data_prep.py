"""
Automatic data preparation from MT5 ZIP export.

Finds the XAUUSD ZIP in the project root, extracts it, detects tab-separated
MT5 format, combines <DATE>+<TIME>, renames columns, and saves as
XAUUSD_M5_12M.csv.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Known ZIP filename pattern
ZIP_PATTERN = "XAUUSD_M5_*.zip"
OUTPUT_CSV = "XAUUSD_M5_12M.csv"


def prepare_data(
    project_root: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Extract and convert MT5 ZIP to standard CSV.

    Steps:
    1. Find XAUUSD_M5_*.zip in project root
    2. Extract the contained TSV file
    3. Detect tab-separated format with angle-bracket column names
    4. Combine <DATE> and <TIME> into a single 'time' column
    5. Rename columns to: time, open, high, low, close, tick_volume
    6. Save as XAUUSD_M5_12M.csv

    Args:
        project_root: Root directory (defaults to module ROOT).
        force: If True, regenerate even if CSV already exists.

    Returns:
        Path to the output CSV file.

    Raises:
        FileNotFoundError: If no ZIP file is found.
        ValueError: If the extracted file has unexpected format.
    """
    root = project_root or ROOT
    output_path = root / OUTPUT_CSV

    # Skip if already exists (unless forced)
    if output_path.exists() and not force:
        return output_path

    # Find ZIP
    zip_files = list(root.glob(ZIP_PATTERN))
    if not zip_files:
        raise FileNotFoundError(
            f"No ZIP file matching '{ZIP_PATTERN}' found in {root}. "
            "Please ensure the XAUUSD data ZIP is in the project root."
        )

    zip_path = zip_files[0]  # Use the first match

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        # Find the data file (usually .csv or .txt)
        data_files = [m for m in members if not m.startswith("__MACOSX")]
        if not data_files:
            raise ValueError(f"ZIP {zip_path.name} contains no data files")

        data_filename = data_files[0]
        zf.extract(data_filename, path=root)
        extracted_path = root / data_filename

    # Read and detect format
    try:
        df = pd.read_csv(extracted_path, sep="\t", nrows=5)
        if "<DATE>" in df.columns or "<OPEN>" in df.columns:
            # MT5 tab-separated format detected
            df = pd.read_csv(extracted_path, sep="\t")
        else:
            # Try comma-separated
            df = pd.read_csv(extracted_path)
    except Exception:
        df = pd.read_csv(extracted_path)

    # Normalize column names (strip angle brackets and lowercase)
    original_cols = list(df.columns)
    col_map = {col: col.strip("<>").lower() for col in original_cols}
    df = df.rename(columns=col_map)

    # Combine date + time if separate columns exist
    if "date" in df.columns and "time" not in df.columns:
        # Look for a time-like column
        time_col = None
        for col in df.columns:
            if col in ("time", "time_col"):
                time_col = col
                break
        # If there's a column that looks like HH:MM:SS
        remaining = [c for c in df.columns if c not in ("date",) and df[c].dtype == object]
        for col in remaining:
            sample = str(df[col].iloc[0])
            if ":" in sample and len(sample) <= 8:
                time_col = col
                break

        if time_col:
            df["time"] = pd.to_datetime(df["date"].astype(str) + " " + df[time_col].astype(str))
            df = df.drop(columns=["date", time_col])
        else:
            df["time"] = pd.to_datetime(df["date"])
            df = df.drop(columns=["date"])

    elif "date" in df.columns and "time" in df.columns:
        # Both exist — combine them
        df["time"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
        df = df.drop(columns=["date"])

    # Rename tick volume variants
    if "tickvol" in df.columns and "tick_volume" not in df.columns:
        df = df.rename(columns={"tickvol": "tick_volume"})

    # Ensure required columns exist
    required = ["time", "open", "high", "low", "close", "tick_volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"After processing, missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Select and order columns
    extra_cols = [c for c in df.columns if c not in required]
    df = df[required + extra_cols]

    # Save
    df.to_csv(output_path, index=False)

    # Cleanup extracted raw file (keep only the converted CSV)
    if extracted_path.exists() and extracted_path != output_path:
        extracted_path.unlink(missing_ok=True)

    return output_path


def ensure_data_ready(project_root: Optional[Path] = None) -> Path:
    """Ensure XAUUSD_M5_12M.csv exists, preparing it from ZIP if needed.

    This is the main entry point for other modules to call.
    """
    return prepare_data(project_root=project_root, force=False)
