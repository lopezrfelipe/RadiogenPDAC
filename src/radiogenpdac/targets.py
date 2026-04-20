from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_signature_vector(path: str | Path) -> np.ndarray:
    target_path = Path(path)
    if target_path.suffix == ".npy":
        array = np.load(target_path)
    elif target_path.suffix == ".npz":
        data = np.load(target_path)
        if "signature" in data:
            array = data["signature"]
        else:
            array = data[list(data.keys())[0]]
    else:
        raise ValueError(f"Unsupported signature vector file: {target_path}")
    return np.asarray(array, dtype=np.float32).reshape(-1)


def parse_json_dict(value: Any) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        return json.loads(value)
    return {}


@dataclass
class TargetSpec:
    latent_dim: int
    pathway_names: list[str]
    subtype_to_index: dict[str, int]
    mutation_names: list[str]
    has_tsr: bool


class SignatureProjector:
    def __init__(self, method: str = "pca", n_factors: int = 16) -> None:
        if method != "pca":
            raise ValueError(
                f"Unsupported latent method '{method}'. This implementation currently supports PCA."
            )
        self.method = method
        self.n_factors = n_factors
        self.model = PCA(n_components=n_factors)

    def fit(self, vectors: list[np.ndarray]) -> None:
        matrix = np.stack(vectors, axis=0)
        n_components = min(self.n_factors, matrix.shape[0], matrix.shape[1])
        self.model = PCA(n_components=n_components)
        self.model.fit(matrix)

    def transform(self, vector: np.ndarray) -> np.ndarray:
        return self.model.transform(vector.reshape(1, -1)).astype(np.float32).reshape(-1)


def build_target_spec(
    train_frame: pd.DataFrame,
    model_cfg: dict[str, Any],
    projector: SignatureProjector,
) -> TargetSpec:
    pathway_names: set[str] = set()
    for value in train_frame.get("pathway_scores_json", pd.Series(dtype=object)).tolist():
        pathway_names.update(parse_json_dict(value).keys())

    subtype_values = sorted(
        value
        for value in train_frame.get("subtype_label", pd.Series(dtype=object)).dropna().astype(str).unique()
        if value
    )
    mutation_names = model_cfg.get("heads", {}).get("mutation_classification", [])
    has_tsr = "tsr_label" in train_frame.columns
    latent_dim = getattr(projector.model, "n_components_", projector.n_factors)

    return TargetSpec(
        latent_dim=latent_dim,
        pathway_names=sorted(pathway_names),
        subtype_to_index={label: index for index, label in enumerate(subtype_values)},
        mutation_names=list(mutation_names),
        has_tsr=has_tsr,
    )


def encode_targets(
    row: pd.Series,
    projector: SignatureProjector,
    spec: TargetSpec,
) -> dict[str, np.ndarray | int | float]:
    signature = load_signature_vector(row["signature_vector_path"])
    latent_signature = projector.transform(signature)

    pathway_scores = np.full(len(spec.pathway_names), np.nan, dtype=np.float32)
    parsed_pathways = parse_json_dict(row.get("pathway_scores_json"))
    for index, name in enumerate(spec.pathway_names):
        if name in parsed_pathways:
            pathway_scores[index] = float(parsed_pathways[name])

    subtype_value = row.get("subtype_label")
    subtype_index = -1
    subtype_available = 0
    if subtype_value is not None and not pd.isna(subtype_value) and str(subtype_value) in spec.subtype_to_index:
        subtype_index = spec.subtype_to_index[str(subtype_value)]
        subtype_available = 1

    mutations = np.full(len(spec.mutation_names), np.nan, dtype=np.float32)
    mutation_payload = parse_json_dict(row.get("driver_mutations_json"))
    for index, name in enumerate(spec.mutation_names):
        if name in mutation_payload:
            mutations[index] = float(mutation_payload[name])

    tsr_value = row.get("tsr_label")
    tsr_index = -1
    tsr_available = 0
    if tsr_value is not None and not pd.isna(tsr_value):
        normalized = str(tsr_value).strip().lower()
        if normalized in {"low", "0", "false"}:
            tsr_index = 0
            tsr_available = 1
        elif normalized in {"high", "1", "true"}:
            tsr_index = 1
            tsr_available = 1

    return {
        "latent_signature": latent_signature,
        "latent_signature_mask": np.ones(spec.latent_dim, dtype=np.float32),
        "pathway_scores": pathway_scores,
        "pathway_scores_mask": np.isfinite(pathway_scores).astype(np.float32),
        "subtype": subtype_index,
        "subtype_mask": subtype_available,
        "mutations": mutations,
        "mutations_mask": np.isfinite(mutations).astype(np.float32),
        "tsr": tsr_index,
        "tsr_mask": tsr_available,
    }
