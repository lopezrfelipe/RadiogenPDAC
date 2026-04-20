from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from radiogenpdac.pdac_encoder import _bootstrap_pdac_detection


DEFAULT_STRUCTURE_PATTERNS: dict[str, list[str]] = {
    "tumor": ["mask_pancreatic_tumor"],
    "pancreas": ["mask_pancreas"],
    "duct": ["mask_pancreatic_duct"],
    "cbd": ["mask_cbd"],
    "artery": ["mask_celiac_aa", "mask_arteries"],
    "vein": ["mask_veins"],
    "cyst": ["mask_pancreatic_cyst"],
}

DEFAULT_LABEL_MAP: dict[str, int] = {
    "pancreas": 1,
    "tumor": 2,
    "cbd": 3,
    "artery": 4,
    "vein": 5,
    "duct": 6,
    "cyst": 7,
}


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""


def _parse_json_map(raw_value: str | None, default: dict[str, Any]) -> dict[str, Any]:
    if raw_value is None or str(raw_value).strip() == "":
        return default
    parsed = json.loads(raw_value)
    return {str(key): value for key, value in parsed.items()}


def _normalize_filename(filename: str) -> str:
    name = filename.lower()
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".npy", ".npz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _tokenize_case_identifier(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", _normalize_filename(value)) if token]


def _extract_case_keys(value: str) -> dict[str, Any]:
    normalized = _normalize_filename(value)
    tokens = _tokenize_case_identifier(value)

    marker_index = None
    for marker in ("studydate", "sd"):
        if marker in tokens:
            marker_index = tokens.index(marker)
            break

    patient_tokens = tokens[:marker_index] if marker_index is not None and marker_index > 0 else tokens[:1]
    date_token = None
    if marker_index is not None and marker_index + 1 < len(tokens):
        date_token = tokens[marker_index + 1]
    if date_token is None:
        date_token = next((token for token in tokens if token.isdigit() and len(token) >= 6), None)

    phase_token = next((token for token in tokens if "venous" in token or "arterial" in token), None)
    return {
        "normalized": normalized,
        "tokens": tokens,
        "patient_key": "_".join(patient_tokens),
        "date_key": date_token,
        "phase_token": phase_token,
    }


def _should_ignore_discovery_file(path: Path) -> bool:
    name = path.name
    lowered = name.lower()
    return (
        name.startswith(".")
        or name.startswith("._")
        or lowered == ".ds_store"
    )


def _is_volume_file(path: Path) -> bool:
    if _should_ignore_discovery_file(path):
        return False
    suffixes = (
        "".join(path.suffixes[-2:]).lower()
        if len(path.suffixes) >= 2
        else path.suffix.lower()
    )
    return suffixes in {".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"}


def _find_structure_mask(segmentation_dir: Path, keywords: list[str]) -> str | None:
    files = sorted(
        path
        for path in segmentation_dir.iterdir()
        if path.is_file() and not _should_ignore_discovery_file(path)
    )
    keyword_set = [keyword.lower() for keyword in keywords]
    for file_path in files:
        normalized = _normalize_filename(file_path.name)
        if any(keyword in normalized for keyword in keyword_set):
            return str(file_path.resolve())
    return None


def _resolve_segmentation_dir_for_volume(
    segmentation_phase_dir: Path,
    image_path: Path,
    phase: str,
) -> Path:
    default_dir = segmentation_phase_dir / _normalize_filename(image_path.name)
    if not segmentation_phase_dir.exists():
        return default_dir

    image_keys = _extract_case_keys(image_path.name)
    phase_lower = phase.strip().lower()
    candidates: list[tuple[int, str, Path]] = []

    for candidate_dir in segmentation_phase_dir.iterdir():
        if not candidate_dir.is_dir() or _should_ignore_discovery_file(candidate_dir):
            continue

        candidate_keys = _extract_case_keys(candidate_dir.name)
        score = 0

        if candidate_keys["normalized"] == image_keys["normalized"]:
            score += 1000

        if image_keys["patient_key"] and candidate_keys["patient_key"] == image_keys["patient_key"]:
            score += 100
        elif image_keys["patient_key"]:
            continue

        if image_keys["date_key"] and candidate_keys["date_key"] == image_keys["date_key"]:
            score += 50
        elif image_keys["date_key"] and candidate_keys["date_key"] is not None:
            continue

        if any(phase_lower in token for token in candidate_keys["tokens"]):
            score += 10

        if score > 0:
            candidates.append((score, candidate_keys["normalized"], candidate_dir))

    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2].resolve()

    return default_dir


def build_phase_ingestion_manifest(
    input_csv: str | Path,
    output_csv: str | Path,
    structure_patterns: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(input_csv)
    patterns = structure_patterns or DEFAULT_STRUCTURE_PATTERNS

    required = ["patient_id", "phase", "image_path", "segmentation_dir"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in ingestion CSV: {', '.join(missing)}")

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        segmentation_dir = Path(str(row["segmentation_dir"])).expanduser().resolve()
        if not segmentation_dir.exists():
            raise FileNotFoundError(segmentation_dir)

        resolved = row.to_dict()
        resolved["image_path"] = str(Path(str(row["image_path"])).expanduser().resolve())
        resolved["segmentation_dir"] = str(segmentation_dir)
        for structure, keywords in patterns.items():
            resolved[f"{structure}_mask"] = _find_structure_mask(segmentation_dir, keywords)
        rows.append(resolved)

    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def scan_cluster_complete_cases(
    framework_root: str | Path,
    output_dir: str | Path,
    data_root: str | Path | None = None,
    phases: list[str] | None = None,
    structure_patterns: dict[str, list[str]] | None = None,
    required_structures: list[str] | None = None,
) -> dict[str, Path]:
    phases = phases or ["venous", "arterial"]
    patterns = structure_patterns or DEFAULT_STRUCTURE_PATTERNS
    required_structures = required_structures or ["tumor", "pancreas", "artery", "vein"]

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    discovered_csv = output_root / "cluster_phase_manifest.csv"
    discovered = discover_cluster_phase_manifest(
        framework_root=framework_root,
        output_csv=discovered_csv,
        data_root=data_root,
        phases=phases,
    )

    rows: list[dict[str, Any]] = []
    for _, row in discovered.iterrows():
        record = row.to_dict()
        segmentation_dir = Path(str(record["segmentation_dir"])).expanduser().resolve()
        missing_items: list[str] = []

        record["segmentation_dir_exists"] = int(segmentation_dir.exists())
        for structure, keywords in patterns.items():
            mask_path = None
            if segmentation_dir.exists():
                mask_path = _find_structure_mask(segmentation_dir, keywords)
            record[f"{structure}_mask"] = mask_path

        if not segmentation_dir.exists():
            missing_items.append("segmentation_dir")
        for structure in required_structures:
            if _is_missing(record.get(f"{structure}_mask")):
                missing_items.append(structure)

        record["required_structures_json"] = json.dumps(required_structures)
        record["missing_items_json"] = json.dumps(missing_items)
        record["is_complete"] = int(len(missing_items) == 0)
        rows.append(record)

    inventory = pd.DataFrame(rows).sort_values(["phase", "patient_id"]).reset_index(drop=True)
    inventory_csv = output_root / "cluster_case_inventory.csv"
    inventory.to_csv(inventory_csv, index=False)

    output_paths: dict[str, Path] = {
        "discovered": discovered_csv,
        "inventory": inventory_csv,
    }
    for phase in phases:
        phase_frame = inventory.loc[
            (inventory["phase"].astype(str).str.lower() == phase.lower()) & (inventory["is_complete"] == 1)
        ].reset_index(drop=True)
        phase_csv = output_root / f"{phase.lower()}_training_manifest.csv"
        phase_frame.to_csv(phase_csv, index=False)
        output_paths[phase.lower()] = phase_csv
    return output_paths


def discover_cluster_phase_manifest(
    framework_root: str | Path,
    output_csv: str | Path,
    data_root: str | Path | None = None,
    phases: list[str] | None = None,
) -> pd.DataFrame:
    framework_path = Path(framework_root).expanduser().resolve()
    data_path = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else framework_path.parent / "data"
    )
    phases = phases or ["venous", "arterial"]

    volumes_root = data_path / "volumes"
    segmentations_root = data_path / "segmentations"
    if not volumes_root.exists():
        raise FileNotFoundError(volumes_root)
    if not segmentations_root.exists():
        raise FileNotFoundError(segmentations_root)

    rows: list[dict[str, Any]] = []
    for phase in phases:
        volume_dir = volumes_root / phase
        segmentation_phase_dir = segmentations_root / phase
        if not volume_dir.exists():
            continue
        for image_path in sorted(path for path in volume_dir.iterdir() if path.is_file() and _is_volume_file(path)):
            patient_id = _normalize_filename(image_path.name)
            segmentation_dir = _resolve_segmentation_dir_for_volume(
                segmentation_phase_dir=segmentation_phase_dir,
                image_path=image_path,
                phase=phase,
            )
            rows.append(
                {
                    "patient_id": patient_id,
                    "phase": phase,
                    "image_path": str(image_path.resolve()),
                    "segmentation_dir": str(segmentation_dir.resolve()),
                    "study_id": patient_id,
                    "site": "cluster_site",
                    "split_group": patient_id,
                }
            )

    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values(["patient_id", "phase"]).reset_index(drop=True)
    result.to_csv(output_path, index=False)
    return result


def build_wide_cohort_manifest_from_phase_table(
    phase_manifest_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    phase_frame = pd.read_csv(phase_manifest_csv)
    rows: dict[str, dict[str, Any]] = {}

    carry_columns = {
        "study_id",
        "site",
        "split_group",
        "split",
        "age",
        "sex",
        "ca19_9",
        "stage",
    }
    structure_columns = [column for column in phase_frame.columns if column.endswith("_mask")]

    for _, row in phase_frame.iterrows():
        patient_id = str(row["patient_id"])
        phase = str(row["phase"]).strip().lower()
        target = rows.setdefault(
            patient_id,
            {
                "patient_id": patient_id,
                "study_id": row.get("study_id", patient_id),
                "site": row.get("site", "unknown"),
            },
        )
        for column in carry_columns:
            value = row.get(column)
            if column not in target or _is_missing(target.get(column)):
                target[column] = value

        target[f"{phase}_image"] = row["image_path"]
        for structure_column in structure_columns:
            mask_value = row.get(structure_column)
            if not _is_missing(mask_value):
                target[f"{phase}_{structure_column}"] = mask_value
                if structure_column == "tumor_mask" and _is_missing(target.get("tumor_mask")):
                    target["tumor_mask"] = mask_value
                if structure_column == "pancreas_mask" and _is_missing(target.get("pancreas_mask")):
                    target["pancreas_mask"] = mask_value

    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows.values()).sort_values("patient_id").reset_index(drop=True)
    result.to_csv(output_path, index=False)
    return result


def _load_image(path: str | Path):
    import SimpleITK as sitk

    return sitk.ReadImage(str(path))


def _write_float_image(source_path: str | Path, destination: Path) -> None:
    import SimpleITK as sitk

    image = sitk.Cast(_load_image(source_path), sitk.sitkFloat32)
    sitk.WriteImage(image, str(destination), useCompression=True)


def _resample_to_reference(mask_image, reference_image):
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


def _build_label_volume(
    reference_image,
    mask_paths: dict[str, str],
    label_map: dict[str, int],
    structure_priority: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    import SimpleITK as sitk

    label_array = np.zeros(tuple(reversed(reference_image.GetSize())), dtype=np.uint8)
    voxel_counts: dict[str, int] = {}

    for structure in structure_priority:
        mask_path = mask_paths.get(structure)
        if _is_missing(mask_path):
            voxel_counts[structure] = 0
            continue
        mask_image = _resample_to_reference(_load_image(mask_path), reference_image)
        mask_array = sitk.GetArrayFromImage(mask_image) > 0
        label_array[mask_array] = np.uint8(label_map[structure])
        voxel_counts[structure] = int(mask_array.sum())
    return label_array, voxel_counts


def _crop_image_and_label_volume(
    reference_image,
    label_volume: np.ndarray,
    crop_mode: str,
    crop_margin_mm: list[float],
    crop_structure_label: int | None,
    crop_fallback_label: int | None,
):
    import SimpleITK as sitk

    if crop_mode == "none":
        return reference_image, label_volume, None

    selected_mask = None
    if crop_structure_label is not None:
        selected_mask = label_volume == crop_structure_label
    if (selected_mask is None or not np.any(selected_mask)) and crop_fallback_label is not None:
        selected_mask = label_volume == crop_fallback_label
    if selected_mask is None or not np.any(selected_mask):
        return reference_image, label_volume, {"mode": "none", "reason": "missing_crop_structure"}

    coordinates = np.argwhere(selected_mask)
    min_zyx = coordinates.min(axis=0)
    max_zyx = coordinates.max(axis=0)
    spacing_xyz = reference_image.GetSpacing()
    margin_xyz = [max(0, int(round(crop_margin_mm[idx] / spacing_xyz[idx]))) for idx in range(3)]

    x_start = max(0, int(min_zyx[2]) - margin_xyz[0])
    x_stop = min(reference_image.GetSize()[0], int(max_zyx[2]) + margin_xyz[0] + 1)
    y_start = max(0, int(min_zyx[1]) - margin_xyz[1])
    y_stop = min(reference_image.GetSize()[1], int(max_zyx[1]) + margin_xyz[1] + 1)
    z_start = max(0, int(min_zyx[0]) - margin_xyz[2])
    z_stop = min(reference_image.GetSize()[2], int(max_zyx[0]) + margin_xyz[2] + 1)

    cropped_image = reference_image[x_start:x_stop, y_start:y_stop, z_start:z_stop]
    cropped_label_volume = label_volume[z_start:z_stop, y_start:y_stop, x_start:x_stop]
    metadata = {
        "mode": crop_mode,
        "bbox_xyz": {
            "x_start": int(x_start),
            "x_stop": int(x_stop),
            "y_start": int(y_start),
            "y_stop": int(y_stop),
            "z_start": int(z_start),
            "z_stop": int(z_stop),
        },
    }
    return cropped_image, cropped_label_volume, metadata


def prepare_phase_finetune_dataset_from_ingestion(
    phase_manifest_csv: str | Path,
    phase: str,
    dataset_id: int,
    dataset_name: str,
    pdac_root: str | Path,
    nnunet_raw_dir: str | Path,
    output_index_csv: str | Path,
    task_mode: str = "multiclass",
    structure_priority: list[str] | None = None,
    label_map: dict[str, int] | None = None,
    crop_mode: str = "pancreas_roi",
    crop_margin_mm: list[float] | None = None,
) -> pd.DataFrame:
    _bootstrap_pdac_detection(pdac_root=pdac_root, nnunet_raw_dir=nnunet_raw_dir)

    phase_frame = pd.read_csv(phase_manifest_csv)
    phase = phase.strip().lower()
    phase_frame = phase_frame.loc[phase_frame["phase"].astype(str).str.lower() == phase].reset_index(drop=True)

    label_values = label_map or DEFAULT_LABEL_MAP
    if task_mode not in {"tumor_only", "multiclass"}:
        raise ValueError("task_mode must be 'tumor_only' or 'multiclass'")

    if task_mode == "tumor_only":
        structure_priority = ["tumor"]
        dataset_labels = {"background": 0, "tumor": 1}
        effective_label_map = {"tumor": 1}
    else:
        structure_priority = structure_priority or ["pancreas", "artery", "vein", "cyst", "tumor"]
        dataset_labels = {"background": 0}
        for structure in structure_priority:
            if structure in label_values:
                dataset_labels[structure] = int(label_values[structure])
        effective_label_map = {structure: dataset_labels[structure] for structure in dataset_labels if structure != "background"}

    dataset_dir = Path(nnunet_raw_dir).expanduser().resolve() / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    crop_margin_mm = crop_margin_mm or [80.0, 80.0, 30.0]
    crop_structure_label = dataset_labels.get("pancreas") if crop_mode == "pancreas_roi" else dataset_labels.get("tumor")
    crop_fallback_label = dataset_labels.get("tumor")

    index_rows: list[dict[str, Any]] = []
    for case_index, row in phase_frame.iterrows():
        patient_id = str(row["patient_id"])
        case_id = f"{patient_id}_{phase}".replace(" ", "_")
        image_dest = images_dir / f"{case_id}_0000.nii.gz"
        label_dest = labels_dir / f"{case_id}.nii.gz"

        image_path = row["image_path"]
        mask_paths = {
            structure: row.get(f"{structure}_mask")
            for structure in effective_label_map.keys()
        }
        if _is_missing(mask_paths.get("tumor")):
            continue

        reference = _load_image(image_path)
        label_volume, voxel_counts = _build_label_volume(
            reference_image=reference,
            mask_paths=mask_paths,
            label_map=effective_label_map,
            structure_priority=structure_priority,
        )
        cropped_image, cropped_label_volume, crop_metadata = _crop_image_and_label_volume(
            reference_image=reference,
            label_volume=label_volume,
            crop_mode=crop_mode,
            crop_margin_mm=crop_margin_mm,
            crop_structure_label=crop_structure_label,
            crop_fallback_label=crop_fallback_label,
        )

        import SimpleITK as sitk

        sitk.WriteImage(sitk.Cast(cropped_image, sitk.sitkFloat32), str(image_dest), useCompression=True)
        label_image = sitk.GetImageFromArray(cropped_label_volume.astype(np.uint8))
        label_image.CopyInformation(cropped_image)
        sitk.WriteImage(label_image, str(label_dest), useCompression=True)

        record = row.to_dict()
        record["case_id"] = case_id
        record["phase"] = phase
        record["task_mode"] = task_mode
        record["dataset_id"] = dataset_id
        record["dataset_name"] = dataset_name
        record["prepared_image_path"] = str(image_dest)
        record["prepared_label_path"] = str(label_dest)
        record["tumor_label"] = int(effective_label_map["tumor"])
        record["structure_labels_json"] = json.dumps(dataset_labels)
        record["voxel_counts_json"] = json.dumps(voxel_counts)
        record["crop_mode"] = crop_mode
        record["crop_margin_mm_json"] = json.dumps(crop_margin_mm)
        record["crop_metadata_json"] = json.dumps(crop_metadata)
        index_rows.append(record)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": dataset_labels,
        "numTraining": len(index_rows),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": f"{phase} fine-tuning dataset prepared by RadiogenPDAC ({task_mode}).",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")

    output_path = Path(output_index_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(index_rows)
    result.to_csv(output_path, index=False)
    return result


def write_nnunet_splits(
    prepared_index_csv: str | Path,
    nnunet_preprocessed_dir: str | Path,
    dataset_id: int,
    dataset_name: str,
    output_json: str | Path | None = None,
    split_column: str | None = None,
    n_folds: int = 5,
    seed: int = 12345,
) -> Path:
    frame = pd.read_csv(prepared_index_csv)
    case_ids = frame["case_id"].astype(str).tolist()

    if split_column and split_column in frame.columns:
        train_cases = frame.loc[frame[split_column].astype(str).str.lower() == "train", "case_id"].astype(str).tolist()
        val_cases = frame.loc[frame[split_column].astype(str).str.lower() == "val", "case_id"].astype(str).tolist()
        if not train_cases or not val_cases:
            raise ValueError(f"Split column '{split_column}' must contain both train and val rows")
        splits = [{"train": train_cases, "val": val_cases}]
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        ordered_case_ids = np.array(sorted(case_ids))
        splits = []
        for train_idx, val_idx in splitter.split(ordered_case_ids):
            splits.append(
                {
                    "train": ordered_case_ids[train_idx].tolist(),
                    "val": ordered_case_ids[val_idx].tolist(),
                }
            )

    dataset_folder = Path(nnunet_preprocessed_dir).expanduser().resolve() / f"Dataset{dataset_id:03d}_{dataset_name}"
    dataset_folder.mkdir(parents=True, exist_ok=True)
    destination = (
        Path(output_json).expanduser().resolve()
        if output_json is not None
        else dataset_folder / "splits_final.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return destination


def _load_case_ids_for_split(
    split_json: str | Path,
    fold: int,
) -> list[str]:
    splits = json.loads(Path(split_json).read_text(encoding="utf-8"))
    if fold >= len(splits):
        raise ValueError(f"Requested fold {fold}, but only {len(splits)} split(s) exist")
    return [str(case_id) for case_id in splits[fold]["val"]]


def compute_tumor_metrics_on_folder(
    reference_folder: str | Path,
    prediction_folder: str | Path,
    case_ids: list[str] | None,
    reference_tumor_label: int,
    prediction_tumor_label: int,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    import SimpleITK as sitk

    reference_dir = Path(reference_folder).expanduser().resolve()
    prediction_dir = Path(prediction_folder).expanduser().resolve()
    case_id_set = set(case_ids) if case_ids is not None else None

    per_case: list[dict[str, Any]] = []
    for prediction_path in sorted(prediction_dir.glob("*.nii.gz")):
        case_id = prediction_path.name.replace(".nii.gz", "")
        if case_id_set is not None and case_id not in case_id_set:
            continue
        reference_path = reference_dir / f"{case_id}.nii.gz"
        if not reference_path.exists():
            continue

        reference = sitk.GetArrayFromImage(sitk.ReadImage(str(reference_path)))
        prediction = sitk.GetArrayFromImage(sitk.ReadImage(str(prediction_path)))
        ref_mask = reference == reference_tumor_label
        pred_mask = prediction == prediction_tumor_label
        tp = int(np.logical_and(ref_mask, pred_mask).sum())
        fp = int(np.logical_and(~ref_mask, pred_mask).sum())
        fn = int(np.logical_and(ref_mask, ~pred_mask).sum())
        denominator = 2 * tp + fp + fn
        dice = float((2 * tp) / denominator) if denominator > 0 else float("nan")
        gt_voxels = int(ref_mask.sum())
        tumor_coverage = float(tp / gt_voxels) if gt_voxels > 0 else float("nan")
        predicted_overlap = float(tp / max(int(pred_mask.sum()), 1))
        per_case.append(
            {
                "case_id": case_id,
                "dice": dice,
                "tumor_gt_coverage": tumor_coverage,
                "predicted_tumor_precision_proxy": predicted_overlap,
                "gt_tumor_voxels": gt_voxels,
                "pred_tumor_voxels": int(pred_mask.sum()),
            }
        )

    summary = {
        "num_cases": len(per_case),
        "mean_dice": float(np.nanmean([row["dice"] for row in per_case])) if per_case else float("nan"),
        "mean_tumor_gt_coverage": (
            float(np.nanmean([row["tumor_gt_coverage"] for row in per_case])) if per_case else float("nan")
        ),
        "mean_predicted_tumor_precision_proxy": (
            float(np.nanmean([row["predicted_tumor_precision_proxy"] for row in per_case])) if per_case else float("nan")
        ),
        "per_case": per_case,
    }
    if output_json is not None:
        destination = Path(output_json).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_encoder_model_on_split(
    pdac_root: str | Path,
    nnunet_raw_dir: str | Path,
    nnunet_preprocessed_dir: str | Path,
    nnunet_results_dir: str | Path,
    model_training_output_dir: str | Path,
    images_folder: str | Path,
    reference_folder: str | Path,
    split_json: str | Path | None,
    fold: int,
    output_folder: str | Path,
    reference_tumor_label: int,
    prediction_tumor_label: int,
    checkpoint_name: str = "checkpoint_final.pth",
    device: str = "cuda",
) -> dict[str, Any]:
    _bootstrap_pdac_detection(
        pdac_root=pdac_root,
        nnunet_raw_dir=nnunet_raw_dir,
        nnunet_preprocessed_dir=nnunet_preprocessed_dir,
        nnunet_results_dir=nnunet_results_dir,
    )

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    output_dir = Path(output_folder).expanduser().resolve()
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    predictor = nnUNetPredictor(
        device=torch.device(device),
        perform_everything_on_gpu=device == "cuda",
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(Path(model_training_output_dir).expanduser().resolve()),
        use_folds=(fold,),
        checkpoint_name=checkpoint_name,
    )

    case_ids = _load_case_ids_for_split(split_json, fold) if split_json is not None else None
    if case_ids is None:
        input_lists = sorted([[str(path.resolve())] for path in Path(images_folder).glob("*_0000.nii.gz")])
    else:
        input_lists = [
            [str((Path(images_folder).expanduser().resolve() / f"{case_id}_0000.nii.gz").resolve())]
            for case_id in case_ids
        ]

    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_lists,
        output_folder_or_list_of_truncated_output_files=str(predictions_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    return compute_tumor_metrics_on_folder(
        reference_folder=reference_folder,
        prediction_folder=predictions_dir,
        case_ids=case_ids,
        reference_tumor_label=reference_tumor_label,
        prediction_tumor_label=prediction_tumor_label,
        output_json=output_dir / "tumor_metrics.json",
    )


def _read_tumor_label_from_index(prepared_index_csv: str | Path) -> int:
    frame = pd.read_csv(prepared_index_csv)
    if frame.empty:
        raise ValueError(f"No prepared cases found in {prepared_index_csv}")
    return int(frame["tumor_label"].iloc[0])


def run_phase_finetune_workflow(
    phase_manifest_csv: str | Path,
    phase: str,
    dataset_id: int,
    dataset_name: str,
    pdac_root: str | Path,
    nnunet_raw_dir: str | Path,
    nnunet_preprocessed_dir: str | Path,
    nnunet_results_dir: str | Path,
    pretrained_weights: str | Path,
    original_model_training_output_dir: str | Path,
    workflow_root: str | Path,
    task_mode: str = "multiclass",
    crop_mode: str = "pancreas_roi",
    crop_margin_mm: list[float] | None = None,
    split_column: str | None = "split",
    plans_identifier: str = "nnUNetPlans",
    configuration: str = "3d_fullres",
    trainer_name: str = "nnUNetTrainer_ftce",
    device: str = "cuda",
    num_gpus: int = 1,
    n_folds: int = 5,
    seed: int = 12345,
    structure_priority: list[str] | None = None,
    label_map: dict[str, int] | None = None,
    gpu_memory_target_gb: float = 8.0,
    num_processes: int = 4,
    overwrite_target_spacing: list[float] | None = None,
    checkpoint_name: str = "checkpoint_final.pth",
    fold: int = 0,
) -> dict[str, Any]:
    from radiogenpdac.pdac_encoder import finetune_phase_encoder, plan_and_preprocess_phase_dataset

    workflow_dir = Path(workflow_root).expanduser().resolve()
    workflow_dir.mkdir(parents=True, exist_ok=True)

    prepared_index_csv = workflow_dir / f"{phase}_prepared_index.csv"
    prepared = prepare_phase_finetune_dataset_from_ingestion(
        phase_manifest_csv=phase_manifest_csv,
        phase=phase,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        pdac_root=pdac_root,
        nnunet_raw_dir=nnunet_raw_dir,
        output_index_csv=prepared_index_csv,
        task_mode=task_mode,
        structure_priority=structure_priority,
        label_map=label_map,
        crop_mode=crop_mode,
        crop_margin_mm=crop_margin_mm,
    )
    if prepared.empty:
        raise ValueError(f"No {phase} cases with usable tumor masks were found in {phase_manifest_csv}")

    splits_json = write_nnunet_splits(
        prepared_index_csv=prepared_index_csv,
        nnunet_preprocessed_dir=nnunet_preprocessed_dir,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        output_json=workflow_dir / "splits_final.json",
        split_column=split_column if split_column and split_column in prepared.columns else None,
        n_folds=n_folds,
        seed=seed,
    )

    plan_and_preprocess_phase_dataset(
        dataset_id=dataset_id,
        pdac_root=pdac_root,
        nnunet_raw_dir=nnunet_raw_dir,
        nnunet_preprocessed_dir=nnunet_preprocessed_dir,
        nnunet_results_dir=nnunet_results_dir,
        plans_identifier=plans_identifier,
        configurations=[configuration],
        gpu_memory_target_gb=gpu_memory_target_gb,
        num_processes=num_processes,
        verify_integrity=False,
        overwrite_target_spacing=overwrite_target_spacing,
    )

    tumor_label = _read_tumor_label_from_index(prepared_index_csv)
    dataset_root = Path(nnunet_raw_dir).expanduser().resolve() / f"Dataset{dataset_id:03d}_{dataset_name}"

    baseline_dir = workflow_dir / "baseline_eval"
    baseline_metrics = evaluate_encoder_model_on_split(
        pdac_root=pdac_root,
        nnunet_raw_dir=nnunet_raw_dir,
        nnunet_preprocessed_dir=nnunet_preprocessed_dir,
        nnunet_results_dir=nnunet_results_dir,
        model_training_output_dir=original_model_training_output_dir,
        images_folder=dataset_root / "imagesTr",
        reference_folder=dataset_root / "labelsTr",
        split_json=splits_json,
        fold=fold,
        output_folder=baseline_dir,
        reference_tumor_label=tumor_label,
        prediction_tumor_label=1,
        checkpoint_name=checkpoint_name,
        device=device,
    )

    finetuned_model_dir = finetune_phase_encoder(
        dataset_id=dataset_id,
        pdac_root=pdac_root,
        nnunet_raw_dir=nnunet_raw_dir,
        nnunet_preprocessed_dir=nnunet_preprocessed_dir,
        nnunet_results_dir=nnunet_results_dir,
        configuration=configuration,
        fold=fold,
        trainer_name=trainer_name,
        plans_identifier=plans_identifier,
        pretrained_weights=pretrained_weights,
        device=device,
        num_gpus=num_gpus,
    )

    validation_prediction_dir = finetuned_model_dir / f"fold_{fold}" / "validation"
    gt_segmentations_dir = (
        Path(nnunet_preprocessed_dir).expanduser().resolve()
        / f"Dataset{dataset_id:03d}_{dataset_name}"
        / "gt_segmentations"
    )
    post_eval_json = workflow_dir / "post_finetune_validation_tumor_metrics.json"
    post_metrics = compute_tumor_metrics_on_folder(
        reference_folder=gt_segmentations_dir,
        prediction_folder=validation_prediction_dir,
        case_ids=None,
        reference_tumor_label=tumor_label,
        prediction_tumor_label=tumor_label,
        output_json=post_eval_json,
    )

    summary = {
        "phase": phase,
        "task_mode": task_mode,
        "crop_mode": crop_mode,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "prepared_index_csv": str(prepared_index_csv),
        "splits_json": str(splits_json),
        "baseline_metrics_json": str(baseline_dir / "tumor_metrics.json"),
        "baseline_mean_dice": baseline_metrics["mean_dice"],
        "baseline_mean_tumor_gt_coverage": baseline_metrics["mean_tumor_gt_coverage"],
        "finetuned_model_dir": str(finetuned_model_dir),
        "post_finetune_metrics_json": str(post_eval_json),
        "post_finetune_mean_dice": post_metrics["mean_dice"],
        "post_finetune_mean_tumor_gt_coverage": post_metrics["mean_tumor_gt_coverage"],
        "tumor_label": tumor_label,
    }
    summary_path = workflow_dir / "workflow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["workflow_summary_json"] = str(summary_path)
    return summary
