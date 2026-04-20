from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch import nn

from radiogenpdac.losses import masked_bce_with_logits, masked_cross_entropy, masked_mse
from radiogenpdac.model import DualPhaseLateFusionModel
from radiogenpdac.targets import TargetSpec


class RadiogenomicsLightningModule(L.LightningModule):
    def __init__(
        self,
        model_cfg: dict[str, Any],
        train_cfg: dict[str, Any],
        phases: list[str],
        contexts: list[str],
        clinical_dim: int,
        external_feature_dim: int,
        target_spec: TargetSpec,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["target_spec"])
        self.model = DualPhaseLateFusionModel(
            model_cfg=model_cfg,
            phases=phases,
            contexts=contexts,
            clinical_dim=clinical_dim,
            external_feature_dim=external_feature_dim,
            target_spec=target_spec,
        )
        self.train_cfg = train_cfg

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.model(
            images=batch["images"],
            token_mask=batch["token_mask"],
            phase_ids=batch["phase_ids"],
            context_ids=batch["context_ids"],
            clinical=batch["clinical"],
            external_features=batch["external_features"],
        )

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        outputs = self(batch)
        losses: dict[str, torch.Tensor] = {}
        losses["latent_mse"] = masked_mse(
            outputs["latent_signature"],
            batch["latent_signature"],
            batch["latent_signature_mask"],
        )
        if "pathway_scores" in outputs:
            losses["pathway_mse"] = masked_mse(
                outputs["pathway_scores"],
                batch["pathway_scores"],
                batch["pathway_scores_mask"],
            )

        if "subtype_logits" in outputs:
            losses["subtype_ce"] = masked_cross_entropy(
                outputs["subtype_logits"],
                batch["subtype"],
                batch["subtype_mask"],
            )
        if "mutation_logits" in outputs:
            losses["mutation_bce"] = masked_bce_with_logits(
                outputs["mutation_logits"],
                batch["mutations"],
                batch["mutations_mask"],
            )
        if "tsr_logits" in outputs:
            losses["tsr_ce"] = masked_cross_entropy(
                outputs["tsr_logits"],
                batch["tsr"],
                batch["tsr_mask"],
            )

        total_loss = sum(losses.values())
        self.log(f"{stage}/loss", total_loss, prog_bar=True, batch_size=batch["images"].size(0))
        for name, value in losses.items():
            self.log(f"{stage}/{name}", value, batch_size=batch["images"].size(0))
        return total_loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
        outputs = self(batch)
        return {
            "patient_id": batch["patient_id"],
            "embedding": outputs["embedding"].detach().cpu(),
            "latent_signature": outputs["latent_signature"].detach().cpu(),
        }

    def configure_optimizers(self) -> dict[str, Any]:
        optimization_cfg = self.train_cfg.get("optimization", {})
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=optimization_cfg.get("learning_rate", 1e-4),
            weight_decay=optimization_cfg.get("weight_decay", 1e-2),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.trainer.max_epochs),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
