#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("nibabel is required to load NIfTI files") from exc
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def _load_preprocessed_case(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    data = np.asarray(payload["data"], dtype=np.float32)
    seg = np.asarray(payload["seg"], dtype=np.float32)
    return data, seg


def _load_validation_case_ids(splits_json: Path, fold: int) -> list[str]:
    with splits_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {splits_json}, got {type(payload).__name__}")
    if fold < 0 or fold >= len(payload):
        raise IndexError(f"Fold {fold} is out of range for {splits_json}")
    split = payload[fold]
    val_cases = split.get("val")
    if not isinstance(val_cases, list) or not val_cases:
        raise ValueError(f"No validation cases found for fold {fold} in {splits_json}")
    return [str(case_id) for case_id in val_cases]


def _find_largest_tumor_slice(mask_zyx: np.ndarray, tumor_label: int) -> int:
    tumor = mask_zyx == float(tumor_label)
    areas = tumor.reshape(tumor.shape[0], -1).sum(axis=1)
    if np.max(areas) <= 0:
        raise ValueError(f"Tumor label {tumor_label} not present in mask")
    return int(np.argmax(areas))


def _center_from_slice(mask_yx: np.ndarray, tumor_label: int) -> tuple[int, int]:
    coords = np.argwhere(mask_yx == float(tumor_label))
    if coords.size == 0:
        raise ValueError(f"Tumor label {tumor_label} not present on selected slice")
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(round((y_min + y_max) / 2.0)), int(round((x_min + x_max) / 2.0))


def _crop_around_center(slice_yx: np.ndarray, center_yx: tuple[int, int], crop_size: int) -> np.ndarray:
    half = crop_size // 2
    center_y, center_x = center_yx
    y_start = center_y - half
    y_stop = y_start + crop_size
    x_start = center_x - half
    x_stop = x_start + crop_size

    pad_top = max(0, -y_start)
    pad_bottom = max(0, y_stop - slice_yx.shape[0])
    pad_left = max(0, -x_start)
    pad_right = max(0, x_stop - slice_yx.shape[1])

    actual_y_start = max(0, y_start)
    actual_y_stop = min(slice_yx.shape[0], y_stop)
    actual_x_start = max(0, x_start)
    actual_x_stop = min(slice_yx.shape[1], x_stop)

    cropped = slice_yx[actual_y_start:actual_y_stop, actual_x_start:actual_x_stop]
    if any(value > 0 for value in (pad_top, pad_bottom, pad_left, pad_right)):
        cropped = np.pad(
            cropped,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=float(slice_yx.min()),
        )
    return cropped.astype(np.float32)


def _window_raw(slice_yx: np.ndarray, lower: float, upper: float) -> np.ndarray:
    clipped = np.clip(slice_yx, lower, upper)
    return ((clipped - lower) / max(upper - lower, 1e-6)).astype(np.float32)


def _display_normalized(slice_yx: np.ndarray, lower: float, upper: float) -> np.ndarray:
    clipped = np.clip(slice_yx, lower, upper)
    return ((clipped - lower) / max(upper - lower, 1e-6)).astype(np.float32)


def _render_png(
    output_png: Path,
    case_id: str,
    raw_crop: np.ndarray,
    normalized_crop: np.ndarray,
    raw_window: tuple[float, float],
    normalized_range: tuple[float, float],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("matplotlib is required to render audit PNGs") from exc

    figure, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=180)
    axes[0].imshow(raw_crop, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Raw windowed\nHU [{raw_window[0]:.0f}, {raw_window[1]:.0f}]")
    axes[1].imshow(normalized_crop, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(
        "nnU-Net normalized\n"
        f"display [{normalized_range[0]:.1f}, {normalized_range[1]:.1f}]"
    )
    for axis in axes:
        axis.axis("off")
    figure.suptitle(case_id, fontsize=11)
    figure.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_png, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a trial PNG for one validation case showing raw-windowed and "
            "nnU-Net-normalized axial crops centered on the tumor."
        )
    )
    parser.add_argument("--nnunet-raw-dir", type=Path, required=True)
    parser.add_argument("--nnunet-preprocessed-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--configuration", type=str, default="3d_fullres")
    parser.add_argument("--tumor-label", type=int, default=1)
    parser.add_argument("--case-id", type=str, default=None)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--raw-window", type=float, nargs=2, default=(-100.0, 240.0))
    parser.add_argument("--normalized-display-range", type=float, nargs=2, default=(-2.5, 2.5))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    dataset_dir = args.nnunet_raw_dir.expanduser().resolve() / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    preprocessed_dataset_dir = (
        args.nnunet_preprocessed_dir.expanduser().resolve() / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    )
    splits_json = preprocessed_dataset_dir / "splits_final.json"
    preprocessed_config_dir = preprocessed_dataset_dir / args.configuration

    if not splits_json.exists():
        raise FileNotFoundError(splits_json)
    if not preprocessed_config_dir.exists():
        raise FileNotFoundError(preprocessed_config_dir)

    validation_case_ids = _load_validation_case_ids(splits_json, args.fold)
    case_id = args.case_id or validation_case_ids[0]
    if case_id not in validation_case_ids:
        raise ValueError(f"Case {case_id} is not in validation fold {args.fold}")

    raw_image_path = dataset_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
    raw_label_path = dataset_dir / "labelsTr" / f"{case_id}.nii.gz"
    preprocessed_case_path = preprocessed_config_dir / f"{case_id}.npz"

    if not raw_image_path.exists():
        raise FileNotFoundError(raw_image_path)
    if not raw_label_path.exists():
        raise FileNotFoundError(raw_label_path)
    if not preprocessed_case_path.exists():
        raise FileNotFoundError(preprocessed_case_path)

    raw_image_xyz = _load_nifti(raw_image_path)
    raw_label_xyz = _load_nifti(raw_label_path)
    raw_image_zyx = np.transpose(raw_image_xyz, (2, 1, 0))
    raw_label_zyx = np.transpose(raw_label_xyz, (2, 1, 0))

    raw_slice_index = _find_largest_tumor_slice(raw_label_zyx, args.tumor_label)
    raw_center = _center_from_slice(raw_label_zyx[raw_slice_index], args.tumor_label)
    raw_crop = _crop_around_center(raw_image_zyx[raw_slice_index], raw_center, args.crop_size)
    raw_display = _window_raw(raw_crop, args.raw_window[0], args.raw_window[1])

    preprocessed_data, preprocessed_seg = _load_preprocessed_case(preprocessed_case_path)
    normalized_image_zyx = preprocessed_data[0]
    normalized_seg_zyx = preprocessed_seg[0]
    normalized_slice_index = _find_largest_tumor_slice(normalized_seg_zyx, args.tumor_label)
    normalized_center = _center_from_slice(normalized_seg_zyx[normalized_slice_index], args.tumor_label)
    normalized_crop = _crop_around_center(
        normalized_image_zyx[normalized_slice_index],
        normalized_center,
        args.crop_size,
    )
    normalized_display = _display_normalized(
        normalized_crop,
        args.normalized_display_range[0],
        args.normalized_display_range[1],
    )

    output_png = args.output_dir.expanduser().resolve() / f"{case_id}.png"
    _render_png(
        output_png=output_png,
        case_id=case_id,
        raw_crop=raw_display,
        normalized_crop=normalized_display,
        raw_window=(float(args.raw_window[0]), float(args.raw_window[1])),
        normalized_range=(
            float(args.normalized_display_range[0]),
            float(args.normalized_display_range[1]),
        ),
    )
    print(output_png)


if __name__ == "__main__":
    main()
