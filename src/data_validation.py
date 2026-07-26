from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


@dataclass
class ValidationFinding:
    severity: str
    code: str
    message: str
    details: Dict[str, Any]


@dataclass
class DataQualityReport:
    file_path: str
    row_count: int
    column_names: List[str]
    findings: List[ValidationFinding]
    summary: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "row_count": self.row_count,
            "column_names": self.column_names,
            "findings": [asdict(f) for f in self.findings],
            "summary": self.summary,
        }


REQUIRED_COLUMNS = ["time", "open", "high", "low", "close"]


def _severity_count(findings: List[ValidationFinding]) -> Dict[str, int]:
    return {
        "INFO": sum(1 for f in findings if f.severity == "INFO"),
        "WARNING": sum(1 for f in findings if f.severity == "WARNING"),
        "ERROR": sum(1 for f in findings if f.severity == "ERROR"),
    }


def validate_dataset(df: pd.DataFrame, file_path: str) -> DataQualityReport:
    findings: List[ValidationFinding] = []

    if df is None:
        findings.append(
            ValidationFinding(
                severity="ERROR",
                code="DATAFRAME_MISSING",
                message="Input dataframe is missing.",
                details={},
            )
        )
        return DataQualityReport(
            file_path=file_path,
            row_count=0,
            column_names=[],
            findings=findings,
            summary=_severity_count(findings),
        )

    findings.append(
        ValidationFinding(
            severity="INFO",
            code="DATASET_LOADED",
            message="Dataset loaded successfully.",
            details={"row_count": int(len(df))},
        )
    )

    column_names = list(df.columns)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in column_names]
    if missing_columns:
        findings.append(
            ValidationFinding(
                severity="ERROR",
                code="MISSING_COLUMNS",
                message="Required OHLC columns are missing.",
                details={"missing_columns": missing_columns},
            )
        )
    else:
        findings.append(
            ValidationFinding(
                severity="INFO",
                code="REQUIRED_COLUMNS_PRESENT",
                message="Required OHLC columns are present.",
                details={"columns": REQUIRED_COLUMNS},
            )
        )

    if "time" in df.columns:
        if df["time"].isna().sum() > 0:
            findings.append(
                ValidationFinding(
                    severity="ERROR",
                    code="MISSING_TIME_VALUES",
                    message="Missing values were found in the time column.",
                    details={"missing_count": int(df["time"].isna().sum())},
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    severity="INFO",
                    code="TIME_VALUES_PRESENT",
                    message="The time column has no missing values.",
                    details={},
                )
            )

    for column in ["open", "high", "low", "close"]:
        if column in df.columns:
            missing_count = int(df[column].isna().sum())
            if missing_count > 0:
                findings.append(
                    ValidationFinding(
                        severity="ERROR",
                        code="MISSING_OHLC_VALUES",
                        message=f"Missing values found in {column}.",
                        details={"column": column, "missing_count": missing_count},
                    )
                )
            else:
                findings.append(
                    ValidationFinding(
                        severity="INFO",
                        code="OHLC_VALUES_PRESENT",
                        message=f"No missing values found in {column}.",
                        details={"column": column},
                    )
                )

    if "time" in df.columns:
        duplicated_timestamps = int(df["time"].duplicated().sum())
        if duplicated_timestamps > 0:
            findings.append(
                ValidationFinding(
                    severity="ERROR",
                    code="DUPLICATE_TIMESTAMPS",
                    message="Duplicate timestamps were found.",
                    details={"duplicate_count": duplicated_timestamps},
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    severity="INFO",
                    code="NO_DUPLICATE_TIMESTAMPS",
                    message="No duplicate timestamps were found.",
                    details={},
                )
            )

    if "time" in df.columns:
        is_sorted = bool(df["time"].is_monotonic_increasing)
        if not is_sorted:
            findings.append(
                ValidationFinding(
                    severity="WARNING",
                    code="UNSORTED_TIMESTAMPS",
                    message="Timestamps are not sorted in ascending order.",
                    details={},
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    severity="INFO",
                    code="TIMESTAMPS_SORTED",
                    message="Timestamps are sorted in ascending order.",
                    details={},
                )
            )

    if all(col in df.columns for col in ["open", "high", "low", "close"]):
        invalid_rows = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )
        invalid_count = int(invalid_rows.sum())
        if invalid_count > 0:
            findings.append(
                ValidationFinding(
                    severity="ERROR",
                    code="INVALID_OHLC",
                    message="Some candles have invalid OHLC relationships.",
                    details={"invalid_count": invalid_count},
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    severity="INFO",
                    code="VALID_OHLC",
                    message="OHLC relationships appear valid.",
                    details={},
                )
            )

    if "time" in df.columns and len(df) > 1:
        diffs = pd.Series(df["time"].diff().dropna())
        if diffs.empty:
            findings.append(
                ValidationFinding(
                    severity="WARNING",
                    code="EMPTY_TIME_DIFFERENCES",
                    message="No time differences could be computed.",
                    details={},
                )
            )
        else:
            expected = pd.Timedelta(minutes=5)
            irregular = diffs[diffs != expected]
            if len(irregular) > 0:
                findings.append(
                    ValidationFinding(
                        severity="WARNING",
                        code="INCONSISTENT_TIMEFRAME",
                        message="The time series contains intervals that differ from the expected 5-minute cadence.",
                        details={"irregular_interval_count": int(len(irregular))},
                    )
                )
            else:
                findings.append(
                    ValidationFinding(
                        severity="INFO",
                        code="CONSISTENT_TIMEFRAME",
                        message="The time series follows the expected 5-minute cadence.",
                        details={},
                    )
                )

    if "time" in df.columns and len(df) > 1:
        expected_index = pd.date_range(start=df["time"].min(), end=df["time"].max(), freq="5min")
        actual_index = pd.DatetimeIndex(df["time"])
        missing_candles = len(expected_index.difference(actual_index))
        if missing_candles > 0:
            findings.append(
                ValidationFinding(
                    severity="WARNING",
                    code="MISSING_CANDLES",
                    message="The dataset appears to contain missing expected candles.",
                    details={"missing_candle_count": missing_candles},
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    severity="INFO",
                    code="NO_MISSING_CANDLES",
                    message="No missing candles were detected for the observed time range.",
                    details={},
                )
            )

    return DataQualityReport(
        file_path=file_path,
        row_count=int(len(df)),
        column_names=column_names,
        findings=findings,
        summary=_severity_count(findings),
    )


def format_report(report: DataQualityReport) -> str:
    lines: List[str] = []
    lines.append("Data Quality Report")
    lines.append("===================")
    lines.append(f"File: {report.file_path}")
    lines.append(f"Rows: {report.row_count}")
    lines.append(f"Columns: {', '.join(report.column_names)}")
    lines.append("Summary:")
    lines.append(f"  INFO: {report.summary.get('INFO', 0)}")
    lines.append(f"  WARNING: {report.summary.get('WARNING', 0)}")
    lines.append(f"  ERROR: {report.summary.get('ERROR', 0)}")
    lines.append("")
    lines.append("Findings:")
    for finding in report.findings:
        lines.append(f"- [{finding.severity}] {finding.code}: {finding.message}")
        if finding.details:
            details_text = ", ".join(f"{key}={value}" for key, value in finding.details.items())
            lines.append(f"  Details: {details_text}")
    return "\n".join(lines)


def save_report(report: DataQualityReport, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_report(report), encoding="utf-8")
    return output
