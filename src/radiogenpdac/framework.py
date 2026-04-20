from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def render_framework_summary(
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> dict[str, Any]:
    phases = data_cfg.get("imaging", {}).get("phases", [])
    targets = target_cfg.get("recommendation", {}).get("primary_target_family", [])
    auxiliary = target_cfg.get("recommendation", {}).get("secondary_target_family", [])

    return {
        "project": "pdac_radiogenomics_framework",
        "recommendation": {
            "target_strategy": (
                "Use a hybrid multitask setup: predict low-dimensional latent transcriptomic factors "
                "and pathway scores as primary targets, then add subtype and major drivers as "
                "secondary heads. Do not make unsupervised patient clusters the primary objective."
            ),
            "roi_strategy": (
                "Avoid mandatory arterial-to-venous registration. Build per-phase tumor and context "
                "crops independently, then fuse venous and arterial embeddings with attention. "
                "Treat segmentation as an ROI utility, not the end goal."
            ),
            "segmentation_policy": (
                "Use manual masks wherever available and high-confidence pseudo-labels elsewhere, "
                "with confidence flags carried into training and QC."
            ),
        },
        "inputs": {
            "phases": phases,
            "reference_phase": data_cfg.get("imaging", {}).get("reference_phase"),
            "required_manifest_columns": ["patient_id", "study_id", "site", "venous_image"],
        },
        "model": {
            "name": model_cfg.get("name"),
            "backbone_family": model_cfg.get("backbone_family"),
            "phase_fusion": model_cfg.get("phase_fusion"),
            "heads": model_cfg.get("heads", {}),
        },
        "targets": {
            "primary": targets,
            "secondary": auxiliary,
            "latent_method": target_cfg.get("latent_representation", {}).get("method"),
            "latent_factor_count": target_cfg.get("latent_representation", {}).get("n_factors"),
            "cluster_role": target_cfg.get("auxiliary_clustering", {}).get("role"),
        },
        "training": {
            "framework": train_cfg.get("framework"),
            "trainer": train_cfg.get("trainer", {}),
            "tracking": train_cfg.get("tracking", {}),
        },
    }


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
