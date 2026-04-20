from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _is_missing_path(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == ""


def resolve_phase_path(
    row: pd.Series,
    phase: str,
    suffix: str,
    fallback_columns: list[str] | None = None,
) -> str | None:
    columns = [f"{phase}_{suffix}"]
    if fallback_columns:
        columns.extend(fallback_columns)
    for column in columns:
        value = row.get(column)
        if not _is_missing_path(value):
            return str(value)
    return None


def load_array(path: str | Path) -> np.ndarray:
    target_path = Path(path)
    suffixes = target_path.suffixes
    if suffixes and suffixes[-1] == ".npy":
        return np.load(target_path).astype(np.float32)
    if suffixes and suffixes[-1] == ".npz":
        data = np.load(target_path)
        if "image" in data:
            return data["image"].astype(np.float32)
        return data[list(data.keys())[0]].astype(np.float32)
    if "".join(suffixes[-2:]) == ".nii.gz" or (suffixes and suffixes[-1] == ".nii"):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise RuntimeError("nibabel is required to load NIfTI files") from exc
        return np.asarray(nib.load(str(target_path)).get_fdata(), dtype=np.float32)
    raise ValueError(f"Unsupported volume format: {target_path}")


def load_optional_array(path: str | Path | None) -> np.ndarray | None:
    if _is_missing_path(path):
        return None
    return load_array(path)


def parse_detector(path: str | Path | None) -> dict[str, Any]:
    if _is_missing_path(path):
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_center_from_detector(
    detector: dict[str, Any],
    phase: str,
    context: str,
) -> list[int] | None:
    candidates = [
        f"{phase}_{context}_center_zyx",
        f"{phase}_{context}_center",
        f"{context}_center_zyx",
        f"{context}_center",
        f"{phase}_center_zyx",
        f"{phase}_center",
        "center_zyx",
        "center",
    ]
    for key in candidates:
        if key in detector:
            value = detector[key]
            if isinstance(value, dict) and "zyx" in value:
                value = value["zyx"]
            return [int(round(float(x))) for x in value]
    return None


def center_from_mask(mask: np.ndarray | None) -> list[int] | None:
    if mask is None:
        return None
    coordinates = np.argwhere(mask > 0)
    if coordinates.size == 0:
        return None
    return coordinates.mean(axis=0).round().astype(int).tolist()


def default_center(volume: np.ndarray) -> list[int]:
    return [size // 2 for size in volume.shape]


def choose_center(
    phase: str,
    context: str,
    volume: np.ndarray,
    mask: np.ndarray | None,
    detector: dict[str, Any],
    fallback_center: list[int] | None,
) -> tuple[list[int], str]:
    mask_center = center_from_mask(mask)
    if mask_center is not None:
        return mask_center, "mask"
    detector_center = _find_center_from_detector(detector, phase, context)
    if detector_center is not None:
        return detector_center, "detector"
    if fallback_center is not None:
        return fallback_center, "cross_phase_fallback"
    return default_center(volume), "volume_center"


def clip_and_scale(volume: np.ndarray, window: list[float]) -> np.ndarray:
    lower, upper = float(window[0]), float(window[1])
    clipped = np.clip(volume, lower, upper)
    scaled = (clipped - lower) / max(upper - lower, 1e-6)
    return scaled.astype(np.float32)


def crop_centered(volume: np.ndarray, center: list[int], size: list[int]) -> np.ndarray:
    slices = []
    pad_width: list[tuple[int, int]] = []
    for axis, axis_size in enumerate(size):
        half = axis_size // 2
        start = center[axis] - half
        end = start + axis_size
        pad_before = max(0, -start)
        pad_after = max(0, end - volume.shape[axis])
        actual_start = max(0, start)
        actual_end = min(volume.shape[axis], end)
        slices.append(slice(actual_start, actual_end))
        pad_width.append((pad_before, pad_after))
    cropped = volume[tuple(slices)]
    if any(before > 0 or after > 0 for before, after in pad_width):
        cropped = np.pad(cropped, pad_width, mode="constant")
    return cropped.astype(np.float32)


def resize_volume(volume: np.ndarray, output_shape: list[int]) -> np.ndarray:
    tensor = torch.from_numpy(volume[None, None, ...])
    resized = F.interpolate(tensor, size=output_shape, mode="trilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def preprocess_case(
    row: pd.Series,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    imaging_cfg = data_cfg["imaging"]
    phases: list[str] = imaging_cfg["phases"]
    contexts: list[str] = model_cfg["context_streams"]
    output_shape: list[int] = imaging_cfg["output_patch_shape"]
    window: list[float] = imaging_cfg["intensity_window_hu"]
    crop_sizes = imaging_cfg["roi_strategy"]["crop_sizes_voxels"]

    detector = parse_detector(row.get("detector_json"))
    patient_id = str(row["patient_id"])

    tensors: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"patient_id": patient_id, "tokens": {}, "phase_presence": {}}
    centers_by_context: dict[str, list[int]] = {}

    for phase in phases:
        phase_path = resolve_phase_path(row, phase, "image")
        if phase_path is None:
            metadata["phase_presence"][phase] = 0
            for context in contexts:
                key = f"{phase}__{context}"
                tensors[key] = np.zeros(output_shape, dtype=np.float32)
                metadata["tokens"][key] = {"present": 0, "center_source": "missing_phase"}
            continue

        volume = clip_and_scale(load_array(phase_path), window)
        pancreas_mask = load_optional_array(
            resolve_phase_path(row, phase, "pancreas_mask", fallback_columns=["pancreas_mask"])
        )

        metadata["phase_presence"][phase] = 1
        for context in contexts:
            mask = None
            if context == "tumor_roi":
                mask = load_optional_array(
                    resolve_phase_path(row, phase, "tumor_mask", fallback_columns=["tumor_mask"])
                )
            elif context == "pancreas_context":
                mask = pancreas_mask

            fallback_center = centers_by_context.get(context)
            center, center_source = choose_center(
                phase=phase,
                context=context,
                volume=volume,
                mask=mask,
                detector=detector,
                fallback_center=fallback_center,
            )
            centers_by_context.setdefault(context, center)
            crop = crop_centered(volume, center=center, size=crop_sizes[context])
            resized = resize_volume(crop, output_shape)
            key = f"{phase}__{context}"
            tensors[key] = resized
            metadata["tokens"][key] = {
                "present": 1,
                "center": center,
                "center_source": center_source,
            }

    patient_dir = output_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    npz_path = patient_dir / "tokens.npz"
    metadata_path = patient_dir / "metadata.json"
    np.savez_compressed(npz_path, **tensors)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    result = row.to_dict()
    result["preprocessed_npz"] = str(npz_path)
    result["preprocessed_metadata"] = str(metadata_path)
    return result


def preprocess_manifest(
    manifest_path: Path,
    output_dir: Path,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path)
    processed_rows = [
        preprocess_case(row, data_cfg=data_cfg, model_cfg=model_cfg, output_dir=output_dir)
        for _, row in frame.iterrows()
    ]
    return pd.DataFrame(processed_rows)
