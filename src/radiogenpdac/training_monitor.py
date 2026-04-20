from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_fold_dir(model_training_output_dir: str | Path, fold: int) -> Path:
    base = Path(model_training_output_dir).expanduser().resolve()
    if base.name.startswith("fold_"):
        return base
    fold_dir = base / f"fold_{fold}"
    return fold_dir if fold_dir.exists() else base


def _find_latest_training_log(fold_dir: Path) -> Path | None:
    candidates = sorted(fold_dir.glob("training_log_*.txt"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def _tail_lines(path: Path | None, n: int) -> list[str]:
    if path is None or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return [line.rstrip("\n") for line in handle.readlines()[-n:]]
    except OSError:
        return []


def _safe_mtime_iso(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _load_checkpoint_payload(checkpoint_path: Path | None) -> dict[str, Any] | None:
    if checkpoint_path is None or not checkpoint_path.exists():
        return None
    import torch

    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    except (OSError, RuntimeError, EOFError, ValueError):
        return None
    return checkpoint if isinstance(checkpoint, dict) else None


def _extract_epoch_rows(logging_payload: dict[str, Any]) -> list[dict[str, Any]]:
    train_losses = list(logging_payload.get("train_losses", []))
    val_losses = list(logging_payload.get("val_losses", []))
    mean_fg_dice = list(logging_payload.get("mean_fg_dice", []))
    ema_fg_dice = list(logging_payload.get("ema_fg_dice", []))
    learning_rates = list(logging_payload.get("lrs", []))
    epoch_start = list(logging_payload.get("epoch_start_timestamps", []))
    epoch_end = list(logging_payload.get("epoch_end_timestamps", []))
    dice_per_class = list(logging_payload.get("dice_per_class_or_region", []))

    n_epochs = min(
        len(train_losses),
        len(val_losses),
        len(mean_fg_dice),
        len(ema_fg_dice),
        len(learning_rates),
        len(epoch_start),
        len(epoch_end),
    )
    rows: list[dict[str, Any]] = []
    for epoch in range(n_epochs):
        epoch_seconds = None
        if epoch < len(epoch_start) and epoch < len(epoch_end):
            epoch_seconds = float(epoch_end[epoch] - epoch_start[epoch])
        row = {
            "epoch": epoch,
            "train_loss": float(train_losses[epoch]),
            "val_loss": float(val_losses[epoch]),
            "mean_fg_dice": float(mean_fg_dice[epoch]),
            "ema_fg_dice": float(ema_fg_dice[epoch]),
            "learning_rate": float(learning_rates[epoch]),
            "epoch_seconds": epoch_seconds,
            "dice_per_class_or_region_json": json.dumps(dice_per_class[epoch]) if epoch < len(dice_per_class) else "[]",
        }
        rows.append(row)
    return rows


def _safe_best_epoch(rows: list[dict[str, Any]], key: str) -> int | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get(key, float("-inf")))).get("epoch")


def _read_validation_summary(validation_summary_json: Path | None) -> dict[str, Any] | None:
    if validation_summary_json is None or not validation_summary_json.exists():
        return None
    try:
        with validation_summary_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    foreground_mean = payload.get("foreground_mean", {})
    mean_metrics = payload.get("mean", {})
    return {
        "foreground_mean_dice": foreground_mean.get("Dice"),
        "foreground_mean_iou": foreground_mean.get("IoU"),
        "mean_metrics": mean_metrics,
    }


def summarize_training_output(
    model_training_output_dir: str | Path,
    fold: int = 0,
    tail_lines: int = 20,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_dir = _resolve_fold_dir(model_training_output_dir, fold)
    checkpoint_latest = fold_dir / "checkpoint_latest.pth"
    checkpoint_best = fold_dir / "checkpoint_best.pth"
    checkpoint_final = fold_dir / "checkpoint_final.pth"
    progress_png = fold_dir / "progress.png"
    validation_summary_json = fold_dir / "validation" / "summary.json"
    training_log = _find_latest_training_log(fold_dir)

    status = "not_started"
    active_checkpoint = None
    if checkpoint_final.exists():
        status = "finished"
        active_checkpoint = checkpoint_final
    elif checkpoint_latest.exists():
        status = "running"
        active_checkpoint = checkpoint_latest
    elif checkpoint_best.exists():
        status = "checkpoint_only"
        active_checkpoint = checkpoint_best

    checkpoint_payload = _load_checkpoint_payload(active_checkpoint)
    if checkpoint_payload is None and active_checkpoint == checkpoint_latest and checkpoint_best.exists():
        checkpoint_payload = _load_checkpoint_payload(checkpoint_best)
        if checkpoint_payload is not None:
            active_checkpoint = checkpoint_best
    checkpoint_load_ok = checkpoint_payload is not None or active_checkpoint is None
    logging_payload = checkpoint_payload.get("logging", {}) if checkpoint_payload else {}
    epoch_rows = _extract_epoch_rows(logging_payload)
    epoch_frame = pd.DataFrame(epoch_rows)

    latest_row = epoch_rows[-1] if epoch_rows else {}
    best_ema_epoch = _safe_best_epoch(epoch_rows, "ema_fg_dice")
    best_mean_epoch = _safe_best_epoch(epoch_rows, "mean_fg_dice")
    validation_summary = _read_validation_summary(validation_summary_json)

    summary: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "status": status,
        "model_training_output_dir": str(Path(model_training_output_dir).expanduser().resolve()),
        "fold_dir": str(fold_dir),
        "fold": fold,
        "active_checkpoint": str(active_checkpoint) if active_checkpoint else None,
        "active_checkpoint_mtime_utc": _safe_mtime_iso(active_checkpoint),
        "active_checkpoint_load_ok": checkpoint_load_ok,
        "checkpoint_latest_exists": checkpoint_latest.exists(),
        "checkpoint_best_exists": checkpoint_best.exists(),
        "checkpoint_final_exists": checkpoint_final.exists(),
        "checkpoint_best_path": str(checkpoint_best) if checkpoint_best.exists() else None,
        "checkpoint_final_path": str(checkpoint_final) if checkpoint_final.exists() else None,
        "progress_png_exists": progress_png.exists(),
        "progress_png_path": str(progress_png) if progress_png.exists() else None,
        "validation_summary_exists": validation_summary_json.exists(),
        "validation_summary_json": str(validation_summary_json) if validation_summary_json.exists() else None,
        "training_log_path": str(training_log) if training_log else None,
        "training_log_mtime_utc": _safe_mtime_iso(training_log),
        "last_log_lines": _tail_lines(training_log, tail_lines),
        "num_completed_epochs": len(epoch_rows),
        "last_completed_epoch": int(latest_row["epoch"]) if latest_row else None,
        "next_epoch_from_checkpoint": checkpoint_payload.get("current_epoch") if checkpoint_payload else None,
        "latest_train_loss": latest_row.get("train_loss"),
        "latest_val_loss": latest_row.get("val_loss"),
        "latest_mean_fg_dice": latest_row.get("mean_fg_dice"),
        "latest_ema_fg_dice": latest_row.get("ema_fg_dice"),
        "latest_learning_rate": latest_row.get("learning_rate"),
        "latest_epoch_seconds": latest_row.get("epoch_seconds"),
        "latest_dice_per_class_or_region": json.loads(latest_row["dice_per_class_or_region_json"])
        if latest_row
        else [],
        "best_ema_fg_dice_epoch": best_ema_epoch,
        "best_ema_fg_dice": epoch_rows[best_ema_epoch]["ema_fg_dice"] if best_ema_epoch is not None else None,
        "best_mean_fg_dice_epoch": best_mean_epoch,
        "best_mean_fg_dice": epoch_rows[best_mean_epoch]["mean_fg_dice"] if best_mean_epoch is not None else None,
        "best_ema_from_checkpoint": checkpoint_payload.get("_best_ema") if checkpoint_payload else None,
        "validation_foreground_mean_dice": validation_summary.get("foreground_mean_dice")
        if validation_summary
        else None,
        "validation_foreground_mean_iou": validation_summary.get("foreground_mean_iou")
        if validation_summary
        else None,
    }
    return summary, epoch_frame


def write_training_monitor_outputs(
    model_training_output_dir: str | Path,
    fold: int = 0,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    tail_lines: int = 20,
) -> tuple[dict[str, Any], Path, Path]:
    summary, epoch_frame = summarize_training_output(
        model_training_output_dir=model_training_output_dir,
        fold=fold,
        tail_lines=tail_lines,
    )
    fold_dir = _resolve_fold_dir(model_training_output_dir, fold)
    json_path = Path(output_json).expanduser().resolve() if output_json else fold_dir / "training_monitor.json"
    csv_path = Path(output_csv).expanduser().resolve() if output_csv else fold_dir / "training_monitor.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    epoch_frame.to_csv(csv_path, index=False)
    return summary, json_path, csv_path


def watch_training_monitor(
    model_training_output_dir: str | Path,
    fold: int = 0,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    tail_lines: int = 20,
    poll_interval_sec: int = 30,
    max_polls: int | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    polls = 0
    last_summary: dict[str, Any] | None = None
    last_json: Path | None = None
    last_csv: Path | None = None

    while True:
        last_summary, last_json, last_csv = write_training_monitor_outputs(
            model_training_output_dir=model_training_output_dir,
            fold=fold,
            output_json=output_json,
            output_csv=output_csv,
            tail_lines=tail_lines,
        )
        polls += 1
        if last_summary["status"] == "finished":
            break
        if max_polls is not None and polls >= max_polls:
            break
        if poll_interval_sec <= 0:
            break
        time.sleep(poll_interval_sec)

    assert last_summary is not None and last_json is not None and last_csv is not None
    return last_summary, last_json, last_csv
