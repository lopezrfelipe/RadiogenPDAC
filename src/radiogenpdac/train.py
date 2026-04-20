from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

try:
    from lightning.pytorch.loggers import MLFlowLogger
except Exception:  # pragma: no cover
    MLFlowLogger = None

from radiogenpdac.datamodule import RadiogenomicsDataModule
from radiogenpdac.lightning_module import RadiogenomicsLightningModule


def _build_logger(train_cfg: dict[str, Any], output_dir: Path):
    tracking = train_cfg.get("tracking", {})
    if tracking.get("backend") == "mlflow" and MLFlowLogger is not None:
        return MLFlowLogger(
            experiment_name="radiogenpdac",
            tracking_uri=tracking.get("uri", f"file:{output_dir / 'mlruns'}"),
            log_model=tracking.get("log_model_checkpoints", True),
        )
    return CSVLogger(save_dir=str(output_dir), name="csv_logs")


def run_training(
    manifest_path: str | Path,
    split_path: str | Path,
    fold: int,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seed = train_cfg.get("seed", 2026)
    L.seed_everything(seed, workers=True)

    datamodule = RadiogenomicsDataModule(
        manifest_path=manifest_path,
        split_path=split_path,
        fold=fold,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        target_cfg=target_cfg,
        train_cfg=train_cfg,
    )
    datamodule.setup("fit")
    metadata = datamodule.metadata

    model = RadiogenomicsLightningModule(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        phases=metadata["phases"],
        contexts=metadata["contexts"],
        clinical_dim=metadata["clinical_spec"].output_dim,
        external_feature_dim=metadata["external_feature_spec"].output_dim,
        target_spec=metadata["target_spec"],
    )

    trainer_cfg = train_cfg.get("trainer", {})
    callbacks = [
        ModelCheckpoint(
            dirpath=output_path / "checkpoints",
            save_top_k=1,
            monitor="val/loss",
            mode="min",
            filename=f"fold{fold}" + "-{epoch:03d}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = _build_logger(train_cfg, output_path)

    trainer = L.Trainer(
        accelerator=trainer_cfg.get("accelerator", "gpu"),
        devices=trainer_cfg.get("devices", 1),
        num_nodes=trainer_cfg.get("num_nodes", 1),
        strategy=trainer_cfg.get("strategy", "auto"),
        precision=trainer_cfg.get("precision", "16-mixed"),
        max_epochs=trainer_cfg.get("max_epochs", 100),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 0.0),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        deterministic=trainer_cfg.get("deterministic", False),
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(output_path),
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")
    return Path(callbacks[0].best_model_path)


def export_embeddings(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    split_path: str | Path,
    fold: int,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    output_path: str | Path,
) -> Path:
    datamodule = RadiogenomicsDataModule(
        manifest_path=manifest_path,
        split_path=split_path,
        fold=fold,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        target_cfg=target_cfg,
        train_cfg=train_cfg,
    )
    datamodule.setup("predict")
    metadata = datamodule.metadata

    model = RadiogenomicsLightningModule.load_from_checkpoint(
        str(checkpoint_path),
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        phases=metadata["phases"],
        contexts=metadata["contexts"],
        clinical_dim=metadata["clinical_spec"].output_dim,
        external_feature_dim=metadata["external_feature_spec"].output_dim,
        target_spec=metadata["target_spec"],
    )
    trainer_cfg = train_cfg.get("trainer", {})
    trainer = L.Trainer(
        accelerator=trainer_cfg.get("accelerator", "gpu"),
        devices=1 if trainer_cfg.get("accelerator", "gpu") != "cpu" else trainer_cfg.get("devices", 1),
        logger=False,
        enable_checkpointing=False,
    )
    predictions = trainer.predict(model, datamodule=datamodule)

    rows = []
    for batch in predictions:
        patient_ids = batch["patient_id"]
        embeddings = batch["embedding"].numpy()
        for patient_id, embedding in zip(patient_ids, embeddings):
            rows.append({"patient_id": patient_id, "embedding": embedding.tolist()})

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_json(destination, orient="records", indent=2)
    return destination
