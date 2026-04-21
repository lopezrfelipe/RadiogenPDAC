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
    "venous_artery_mask",
    "arterial_artery_mask",
    "artery_mask",
    "venous_vein_mask",
    "arterial_vein_mask",
    "vein_mask",
    "venous_duct_mask",
    "arterial_duct_mask",
    "duct_mask",
    "venous_cbd_mask",
    "arterial_cbd_mask",
    "cbd_mask",
    "venous_cyst_mask",
    "arterial_cyst_mask",
    "cyst_mask",
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


def _is_missing(value: object) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""


def _resample_mask_to_reference(mask_image, reference_image):
    import SimpleITK as sitk

    if (
        mask_image.GetSize() == reference_image.GetSize()
        and mask_image.GetSpacing() == reference_image.GetSpacing()
        and mask_image.GetOrigin() == reference_image.GetOrigin()
        and mask_image.GetDirection() == reference_image.GetDirection()
    ):
        return mask_image
    return sitk.Resample(
        mask_image,
        reference_image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def _union_mask_paths(
    primary_mask_path: Path | None,
    secondary_mask_path: Path | None,
    destination_path: Path,
) -> Path | None:
    import SimpleITK as sitk
    import numpy as np

    if primary_mask_path is None and secondary_mask_path is None:
        return None
    if secondary_mask_path is None:
        return primary_mask_path
    if primary_mask_path is None:
        return secondary_mask_path

    primary_image = sitk.ReadImage(str(primary_mask_path))
    secondary_image = _resample_mask_to_reference(sitk.ReadImage(str(secondary_mask_path)), primary_image)

    primary_array = sitk.GetArrayFromImage(primary_image) > 0
    secondary_array = sitk.GetArrayFromImage(secondary_image) > 0
    union_array = np.logical_or(primary_array, secondary_array).astype(np.uint8)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    union_image = sitk.GetImageFromArray(union_array)
    union_image.CopyInformation(primary_image)
    sitk.WriteImage(union_image, str(destination_path), useCompression=True)
    return destination_path


def build_hybrid_structure_manifest(
    base_manifest_path: Path,
    override_manifest_path: Path,
    output_manifest_path: Path,
    output_mask_dir: Path,
    structures: list[str],
    join_keys: list[str] | None = None,
) -> pd.DataFrame:
    join_keys = join_keys or ["patient_id", "phase"]
    base = load_csv(base_manifest_path)
    override = load_csv(override_manifest_path)

    merged = base.merge(
        override,
        on=join_keys,
        how="left",
        suffixes=("", "__override"),
        validate="one_to_one",
    )

    for structure in structures:
        column = f"{structure}_mask"
        override_column = f"{column}__override"
        if column not in merged.columns:
            continue
        if override_column not in merged.columns:
            continue

        hybrid_paths: list[str | None] = []
        for _, row in merged.iterrows():
            primary_value = row.get(column)
            override_value = row.get(override_column)

            primary_path = None if _is_missing(primary_value) else Path(str(primary_value)).expanduser().resolve()
            secondary_path = None if _is_missing(override_value) else Path(str(override_value)).expanduser().resolve()

            if primary_path is not None and not primary_path.exists():
                raise FileNotFoundError(primary_path)
            if secondary_path is not None and not secondary_path.exists():
                raise FileNotFoundError(secondary_path)

            if primary_path is not None and secondary_path is not None:
                identifier = "__".join(str(row[key]).replace("/", "_") for key in join_keys)
                hybrid_path = output_mask_dir / structure / f"{identifier}_{structure}.nii.gz"
                resolved = _union_mask_paths(primary_path, secondary_path, hybrid_path)
            else:
                resolved = primary_path or secondary_path
            hybrid_paths.append(None if resolved is None else str(resolved))

        merged[column] = hybrid_paths
        merged.drop(columns=[override_column], inplace=True)

    leftover_override_columns = [column for column in merged.columns if column.endswith("__override")]
    if leftover_override_columns:
        merged.drop(columns=leftover_override_columns, inplace=True)

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_manifest_path, index=False)
    return merged
