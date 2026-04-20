from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from radiogenpdac.targets import TargetSpec, encode_targets


@dataclass
class ClinicalSpec:
    feature_names: list[str]
    numeric_stats: dict[str, tuple[float, float]]
    categorical_maps: dict[str, dict[str, int]]
    output_dim: int


@dataclass
class ExternalFeatureSpec:
    phase_columns: dict[str, str]
    phase_dims: dict[str, int]
    include_missing_flags: bool
    output_dim: int


def build_clinical_spec(train_frame: pd.DataFrame, covariates: list[str]) -> ClinicalSpec:
    feature_names: list[str] = []
    numeric_stats: dict[str, tuple[float, float]] = {}
    categorical_maps: dict[str, dict[str, int]] = {}
    output_dim = 0

    for covariate in covariates:
        if covariate not in train_frame.columns:
            continue
        series = train_frame[covariate]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= max(3, len(series) // 2):
            mean = float(numeric.mean()) if numeric.notna().any() else 0.0
            std = float(numeric.std()) if numeric.notna().any() else 1.0
            numeric_stats[covariate] = (mean, std if std > 0 else 1.0)
            feature_names.append(covariate)
            output_dim += 1
        else:
            values = sorted(value for value in series.dropna().astype(str).unique() if value)
            categorical_maps[covariate] = {value: idx for idx, value in enumerate(values)}
            feature_names.extend([f"{covariate}={value}" for value in values])
            output_dim += len(values)

    return ClinicalSpec(
        feature_names=feature_names,
        numeric_stats=numeric_stats,
        categorical_maps=categorical_maps,
        output_dim=output_dim,
    )


def encode_clinical(row: pd.Series, spec: ClinicalSpec) -> np.ndarray:
    features = np.zeros(spec.output_dim, dtype=np.float32)
    offset = 0
    for name, (mean, std) in spec.numeric_stats.items():
        value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
        if pd.isna(value):
            normalized = 0.0
        else:
            normalized = float((value - mean) / std)
        features[offset] = normalized
        offset += 1
    for name, mapping in spec.categorical_maps.items():
        value = row.get(name)
        if value is not None and not pd.isna(value):
            encoded_index = mapping.get(str(value))
            if encoded_index is not None:
                features[offset + encoded_index] = 1.0
        offset += len(mapping)
    return features


def load_feature_vector(path: str | Path) -> np.ndarray:
    target_path = Path(path)
    if target_path.suffix == ".npy":
        return np.load(target_path).astype(np.float32).reshape(-1)
    if target_path.suffix == ".npz":
        data = np.load(target_path)
        key = "embedding" if "embedding" in data else list(data.keys())[0]
        return np.asarray(data[key], dtype=np.float32).reshape(-1)
    if target_path.suffix == ".json":
        payload = json.loads(target_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "embedding" in payload:
                payload = payload["embedding"]
            elif "features" in payload:
                payload = payload["features"]
        return np.asarray(payload, dtype=np.float32).reshape(-1)
    raise ValueError(f"Unsupported feature format: {target_path}")


def build_external_feature_spec(
    train_frame: pd.DataFrame,
    model_cfg: dict[str, Any],
) -> ExternalFeatureSpec:
    external_cfg = model_cfg.get("external_encoder_features", {})
    if not external_cfg.get("enabled", False):
        return ExternalFeatureSpec({}, {}, False, 0)

    phase_columns = {
        str(phase): str(column)
        for phase, column in external_cfg.get("phase_feature_columns", {}).items()
        if column in train_frame.columns
    }
    include_missing_flags = external_cfg.get("include_missing_flags", True)
    phase_dims: dict[str, int] = {}
    default_feature_dim = int(external_cfg.get("feature_dim", 0))

    for phase, column in phase_columns.items():
        non_empty = train_frame[column].dropna().astype(str)
        non_empty = non_empty.loc[non_empty.str.strip() != ""]
        if not non_empty.empty:
            phase_dims[phase] = int(load_feature_vector(non_empty.iloc[0]).shape[0])
        elif default_feature_dim > 0:
            phase_dims[phase] = default_feature_dim

    output_dim = sum(phase_dims.values())
    if include_missing_flags:
        output_dim += len(phase_dims)
    return ExternalFeatureSpec(phase_columns, phase_dims, include_missing_flags, output_dim)


def encode_external_features(row: pd.Series, spec: ExternalFeatureSpec) -> np.ndarray:
    if spec.output_dim == 0:
        return np.zeros(0, dtype=np.float32)

    segments: list[np.ndarray] = []
    flags: list[float] = []
    for phase, column in spec.phase_columns.items():
        dim = spec.phase_dims.get(phase, 0)
        if dim == 0:
            continue
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            feature = load_feature_vector(str(value))
            if feature.shape[0] != dim:
                raise ValueError(
                    f"Feature dimension mismatch for patient {row.get('patient_id')} column {column}: "
                    f"expected {dim}, got {feature.shape[0]}"
                )
            segments.append(feature.astype(np.float32))
            flags.append(1.0)
        else:
            segments.append(np.zeros(dim, dtype=np.float32))
            flags.append(0.0)
    if spec.include_missing_flags:
        segments.append(np.asarray(flags, dtype=np.float32))
    return np.concatenate(segments).astype(np.float32) if segments else np.zeros(0, dtype=np.float32)


class RadiogenomicsDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        phases: list[str],
        contexts: list[str],
        clinical_spec: ClinicalSpec,
        external_feature_spec: ExternalFeatureSpec,
        target_projector: Any,
        target_spec: TargetSpec,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.phases = phases
        self.contexts = contexts
        self.clinical_spec = clinical_spec
        self.external_feature_spec = external_feature_spec
        self.target_projector = target_projector
        self.target_spec = target_spec

        self.token_keys = [f"{phase}__{context}" for phase in phases for context in contexts]
        self.phase_ids = torch.tensor(
            [phase_index for phase_index, phase in enumerate(phases) for _ in contexts],
            dtype=torch.long,
        )
        self.context_ids = torch.tensor(
            [context_index for _ in phases for context_index, _ in enumerate(contexts)],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        data = np.load(row["preprocessed_npz"])
        images = []
        token_mask = []
        for token_key in self.token_keys:
            token = np.asarray(data[token_key], dtype=np.float32)
            images.append(torch.from_numpy(token).unsqueeze(0))
            token_mask.append(1.0 if np.any(token) else 0.0)

        encoded_targets = encode_targets(row, self.target_projector, self.target_spec)
        clinical = encode_clinical(row, self.clinical_spec)
        external_features = encode_external_features(row, self.external_feature_spec)

        return {
            "patient_id": str(row["patient_id"]),
            "images": torch.stack(images, dim=0),
            "token_mask": torch.tensor(token_mask, dtype=torch.float32),
            "phase_ids": self.phase_ids.clone(),
            "context_ids": self.context_ids.clone(),
            "clinical": torch.from_numpy(clinical),
            "external_features": torch.from_numpy(external_features),
            "latent_signature": torch.from_numpy(encoded_targets["latent_signature"]),
            "latent_signature_mask": torch.from_numpy(encoded_targets["latent_signature_mask"]),
            "pathway_scores": torch.from_numpy(encoded_targets["pathway_scores"]),
            "pathway_scores_mask": torch.from_numpy(encoded_targets["pathway_scores_mask"]),
            "subtype": torch.tensor(encoded_targets["subtype"], dtype=torch.long),
            "subtype_mask": torch.tensor(encoded_targets["subtype_mask"], dtype=torch.float32),
            "mutations": torch.from_numpy(encoded_targets["mutations"]),
            "mutations_mask": torch.from_numpy(encoded_targets["mutations_mask"]),
            "tsr": torch.tensor(encoded_targets["tsr"], dtype=torch.long),
            "tsr_mask": torch.tensor(encoded_targets["tsr_mask"], dtype=torch.float32),
        }
