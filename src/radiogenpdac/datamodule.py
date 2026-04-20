from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

from radiogenpdac.dataset import (
    ClinicalSpec,
    ExternalFeatureSpec,
    RadiogenomicsDataset,
    build_clinical_spec,
    build_external_feature_spec,
)
from radiogenpdac.targets import SignatureProjector, TargetSpec, build_target_spec, load_signature_vector


class RadiogenomicsDataModule(L.LightningDataModule):
    def __init__(
        self,
        manifest_path: str | Path,
        split_path: str | Path,
        fold: int,
        data_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        target_cfg: dict[str, Any],
        train_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.split_path = Path(split_path)
        self.fold = fold
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.target_cfg = target_cfg
        self.train_cfg = train_cfg

        self.phases = data_cfg["imaging"]["phases"]
        self.contexts = model_cfg["context_streams"]
        self.clinical_covariates = model_cfg.get("clinical_covariates", [])
        self.batching_cfg = train_cfg.get("batching", {})

        self.frame: pd.DataFrame | None = None
        self.train_frame: pd.DataFrame | None = None
        self.val_frame: pd.DataFrame | None = None
        self.test_frame: pd.DataFrame | None = None

        self.target_projector: SignatureProjector | None = None
        self.target_spec: TargetSpec | None = None
        self.clinical_spec: ClinicalSpec | None = None
        self.external_feature_spec: ExternalFeatureSpec | None = None

    def prepare_data(self) -> None:
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        if not self.split_path.exists():
            raise FileNotFoundError(self.split_path)

    def _prepare_frames(self) -> None:
        manifest = pd.read_csv(self.manifest_path)
        splits = pd.read_csv(self.split_path)
        split_frame = splits.loc[splits["fold"] == self.fold, ["patient_id", "split"]].drop_duplicates()
        frame = manifest.merge(split_frame, on="patient_id", how="inner", validate="one_to_one")
        self.frame = frame
        self.train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
        self.val_frame = frame.loc[frame["split"] == "val"].reset_index(drop=True)
        self.test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)

    def setup(self, stage: str | None = None) -> None:
        if self.frame is None:
            self._prepare_frames()
            assert self.train_frame is not None
            method = self.target_cfg.get("latent_representation", {}).get("method", "pca")
            n_factors = self.target_cfg.get("latent_representation", {}).get("n_factors", 16)
            self.target_projector = SignatureProjector(method=method, n_factors=n_factors)
            train_signatures = [
                load_signature_vector(path) for path in self.train_frame["signature_vector_path"].tolist()
            ]
            self.target_projector.fit(train_signatures)
            self.target_spec = build_target_spec(self.train_frame, self.model_cfg, self.target_projector)
            self.clinical_spec = build_clinical_spec(self.train_frame, self.clinical_covariates)
            self.external_feature_spec = build_external_feature_spec(self.train_frame, self.model_cfg)

        assert self.train_frame is not None
        assert self.val_frame is not None
        assert self.test_frame is not None
        assert self.target_projector is not None
        assert self.target_spec is not None
        assert self.clinical_spec is not None
        assert self.external_feature_spec is not None

        if stage in {None, "fit"}:
            self.train_dataset = RadiogenomicsDataset(
                self.train_frame,
                phases=self.phases,
                contexts=self.contexts,
                clinical_spec=self.clinical_spec,
                external_feature_spec=self.external_feature_spec,
                target_projector=self.target_projector,
                target_spec=self.target_spec,
            )
            self.val_dataset = RadiogenomicsDataset(
                self.val_frame,
                phases=self.phases,
                contexts=self.contexts,
                clinical_spec=self.clinical_spec,
                external_feature_spec=self.external_feature_spec,
                target_projector=self.target_projector,
                target_spec=self.target_spec,
            )
        if stage in {None, "test", "predict"}:
            self.test_dataset = RadiogenomicsDataset(
                self.test_frame,
                phases=self.phases,
                contexts=self.contexts,
                clinical_spec=self.clinical_spec,
                external_feature_spec=self.external_feature_spec,
                target_projector=self.target_projector,
                target_spec=self.target_spec,
            )

    @property
    def metadata(self) -> dict[str, Any]:
        if self.target_spec is None or self.clinical_spec is None:
            raise RuntimeError("Call setup() before accessing metadata")
        return {
            "phases": self.phases,
            "contexts": self.contexts,
            "target_spec": self.target_spec,
            "clinical_spec": self.clinical_spec,
            "external_feature_spec": self.external_feature_spec,
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batching_cfg.get("train_batch_size", 2),
            shuffle=True,
            num_workers=self.batching_cfg.get("num_workers", 4),
            persistent_workers=self.batching_cfg.get("persistent_workers", False),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batching_cfg.get("eval_batch_size", 2),
            shuffle=False,
            num_workers=self.batching_cfg.get("num_workers", 4),
            persistent_workers=self.batching_cfg.get("persistent_workers", False),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batching_cfg.get("eval_batch_size", 2),
            shuffle=False,
            num_workers=self.batching_cfg.get("num_workers", 4),
            persistent_workers=self.batching_cfg.get("persistent_workers", False),
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
