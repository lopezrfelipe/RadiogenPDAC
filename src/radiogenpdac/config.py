from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_all_configs(
    data_config: str | Path,
    model_config: str | Path,
    target_config: str | Path,
    train_config: str | Path,
) -> dict[str, dict[str, Any]]:
    return {
        "data": load_yaml(data_config),
        "model": load_yaml(model_config),
        "target": load_yaml(target_config),
        "train": load_yaml(train_config),
    }
