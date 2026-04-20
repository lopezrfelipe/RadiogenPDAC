from __future__ import annotations

from typing import Any

import torch
from torch import nn

from radiogenpdac.targets import TargetSpec


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm3d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class TokenEncoder3D(nn.Module):
    def __init__(self, channels: list[int], in_channels: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for index, channel in enumerate(channels):
            stride = 1 if index == 0 else 2
            layers.append(ResidualBlock3D(current, channel, stride=stride))
            current = channel
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.output_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x


class DualPhaseLateFusionModel(nn.Module):
    def __init__(
        self,
        model_cfg: dict[str, Any],
        phases: list[str],
        contexts: list[str],
        clinical_dim: int,
        external_feature_dim: int,
        target_spec: TargetSpec,
    ) -> None:
        super().__init__()
        self.phases = phases
        self.contexts = contexts
        self.embedding_dim = model_cfg.get("token_embedding_dim", 256)
        channels = model_cfg.get("token_encoder_channels", [32, 64, 128, 256])
        separate_phase_encoders = model_cfg.get("separate_phase_encoders", True)

        if separate_phase_encoders:
            self.phase_encoders = nn.ModuleList([TokenEncoder3D(channels) for _ in phases])
        else:
            shared = TokenEncoder3D(channels)
            self.phase_encoders = nn.ModuleList([shared for _ in phases])

        token_dim = self.phase_encoders[0].output_dim
        self.token_projection = nn.Linear(token_dim, self.embedding_dim)
        self.phase_embedding = nn.Embedding(len(phases), self.embedding_dim)
        self.context_embedding = nn.Embedding(len(contexts), self.embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=model_cfg.get("transformer_heads", 8),
            dim_feedforward=self.embedding_dim * 4,
            dropout=model_cfg.get("dropout", 0.1),
            activation="gelu",
            batch_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_cfg.get("transformer_layers", 2),
        )

        self.clinical_projector = (
            nn.Sequential(
                nn.Linear(clinical_dim, self.embedding_dim),
                nn.GELU(),
                nn.LayerNorm(self.embedding_dim),
            )
            if clinical_dim > 0
            else None
        )

        external_cfg = model_cfg.get("external_encoder_features", {})
        external_projection_dim = int(external_cfg.get("projection_dim", self.embedding_dim // 2))
        self.external_feature_projector = (
            nn.Sequential(
                nn.LayerNorm(external_feature_dim),
                nn.Linear(external_feature_dim, external_projection_dim),
                nn.GELU(),
                nn.LayerNorm(external_projection_dim),
            )
            if external_feature_dim > 0 and external_cfg.get("enabled", False)
            else None
        )

        fused_dim = self.embedding_dim
        if clinical_dim > 0:
            fused_dim += self.embedding_dim
        if self.external_feature_projector is not None:
            fused_dim += external_projection_dim
        heads_cfg = model_cfg.get("heads", {})
        self.latent_head = nn.Linear(fused_dim, target_spec.latent_dim)
        self.pathway_head = (
            nn.Linear(fused_dim, len(target_spec.pathway_names))
            if target_spec.pathway_names
            else None
        )
        self.subtype_head = (
            nn.Linear(fused_dim, len(target_spec.subtype_to_index))
            if target_spec.subtype_to_index
            else None
        )
        self.mutation_head = (
            nn.Linear(fused_dim, len(target_spec.mutation_names))
            if target_spec.mutation_names
            else None
        )
        self.tsr_head = nn.Linear(fused_dim, 2) if target_spec.has_tsr else None

        self.dropout = nn.Dropout(model_cfg.get("dropout", 0.1))
        nn.init.normal_(self.cls_token, std=0.02)

    def _encode_tokens(
        self,
        images: torch.Tensor,
        phase_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, token_count = images.shape[:2]
        token_embeddings = images.new_zeros(batch_size, token_count, self.embedding_dim)

        for phase_index, encoder in enumerate(self.phase_encoders):
            token_selector = phase_ids[0] == phase_index
            if not torch.any(token_selector):
                continue
            phase_tokens = images[:, token_selector]
            encoded = encoder(phase_tokens.reshape(-1, *phase_tokens.shape[2:]))
            encoded = self.token_projection(encoded).reshape(batch_size, -1, self.embedding_dim)
            encoded = encoded + self.phase_embedding(phase_ids[:, token_selector])
            encoded = encoded + self.context_embedding(context_ids[:, token_selector])
            token_embeddings[:, token_selector] = encoded
        return token_embeddings

    def forward(
        self,
        images: torch.Tensor,
        token_mask: torch.Tensor,
        phase_ids: torch.Tensor,
        context_ids: torch.Tensor,
        clinical: torch.Tensor,
        external_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        token_embeddings = self._encode_tokens(images, phase_ids, context_ids)
        cls = self.cls_token.expand(images.size(0), -1, -1)
        sequence = torch.cat([cls, token_embeddings], dim=1)
        padding_mask = torch.cat(
            [torch.zeros(images.size(0), 1, device=images.device), 1.0 - token_mask],
            dim=1,
        ).bool()
        fused = self.fusion(sequence, src_key_padding_mask=padding_mask)
        embedding = self.dropout(fused[:, 0])

        if self.clinical_projector is not None:
            clinical_embedding = self.clinical_projector(clinical)
            embedding = torch.cat([embedding, clinical_embedding], dim=1)
        if self.external_feature_projector is not None:
            external_embedding = self.external_feature_projector(external_features)
            embedding = torch.cat([embedding, external_embedding], dim=1)

        outputs = {
            "embedding": embedding,
            "latent_signature": self.latent_head(embedding),
        }
        if self.pathway_head is not None:
            outputs["pathway_scores"] = self.pathway_head(embedding)
        if self.subtype_head is not None:
            outputs["subtype_logits"] = self.subtype_head(embedding)
        if self.mutation_head is not None:
            outputs["mutation_logits"] = self.mutation_head(embedding)
        if self.tsr_head is not None:
            outputs["tsr_logits"] = self.tsr_head(embedding)
        return outputs
