from __future__ import annotations

from pathlib import Path

import pandas as pd

COHORT_REQUIRED_COLUMNS = ["patient_id", "study_id", "site", "venous_image"]
GENOMICS_REQUIRED_COLUMNS = ["patient_id", "signature_vector_path"]

PATH_LIKE_COLUMNS = [
    "venous_image",
    "arterial_image",
    "venous_tumor_mask",
    "arterial_tumor_mask",
    "tumor_mask",
    "venous_pancreas_mask",
    "arterial_pancreas_mask",
    "pancreas_mask",
    "detector_json",
    "venous_encoder_features",
    "arterial_encoder_features",
    "signature_vector_path",
]


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _missing_columns(frame: pd.DataFrame, required: list[str]) -> list[str]:
    return [column for column in required if column not in frame.columns]


def validate_manifest(path: Path, required_columns: list[str]) -> list[str]:
    frame = load_csv(path)
    issues: list[str] = []
    missing = _missing_columns(frame, required_columns)
    if missing:
        issues.append(f"{path}: missing required columns: {', '.join(missing)}")

    for column in [col for col in PATH_LIKE_COLUMNS if col in frame.columns]:
        empty_mask = frame[column].fillna("").astype(str).str.strip().eq("")
        if column in required_columns and empty_mask.any():
            issues.append(f"{path}: required path column '{column}' contains empty values")

        for value in frame.loc[~empty_mask, column].astype(str):
            expanded = Path(value).expanduser()
            if not expanded.exists():
                issues.append(f"{path}: path does not exist for column '{column}': {value}")

    return issues


def merge_manifests(
    cohort_path: Path,
    genomics_path: Path,
    output_path: Path,
    join_key: str = "patient_id",
) -> pd.DataFrame:
    cohort = load_csv(cohort_path)
    genomics = load_csv(genomics_path)
    merged = cohort.merge(genomics, on=join_key, how="inner", validate="one_to_one")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged
