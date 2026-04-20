from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedShuffleSplit


@dataclass
class SplitConfig:
    n_folds: int
    group_column: str
    stratify_column: str | None
    test_fraction: float
    seed: int


def _build_group_frame(
    frame: pd.DataFrame,
    group_column: str,
    stratify_column: str | None,
) -> pd.DataFrame:
    group_frame = frame[[group_column]].drop_duplicates().copy()
    if stratify_column:
        labels = frame.groupby(group_column)[stratify_column].nunique()
        inconsistent = labels[labels > 1]
        if not inconsistent.empty:
            raise ValueError(
                f"Stratify column '{stratify_column}' is inconsistent within groups: "
                f"{', '.join(inconsistent.index.astype(str).tolist())}"
            )
        group_labels = frame.groupby(group_column)[stratify_column].first().reset_index()
        group_frame = group_frame.merge(group_labels, on=group_column, how="left")
    return group_frame


def _select_test_groups(group_frame: pd.DataFrame, cfg: SplitConfig) -> set[str]:
    if cfg.test_fraction <= 0:
        return set()

    if cfg.stratify_column and cfg.stratify_column in group_frame.columns:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=cfg.test_fraction,
            random_state=cfg.seed,
        )
        y = group_frame[cfg.stratify_column]
        indices = range(len(group_frame))
        try:
            _, test_idx = next(splitter.split(indices, y))
        except ValueError:
            test_idx = list(group_frame.sample(frac=cfg.test_fraction, random_state=cfg.seed).index)
    else:
        test_idx = list(group_frame.sample(frac=cfg.test_fraction, random_state=cfg.seed).index)

    return set(group_frame.iloc[test_idx][cfg.group_column].astype(str))


def build_split_table(frame: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    if cfg.group_column not in frame.columns:
        raise ValueError(f"Group column '{cfg.group_column}' is missing from manifest")

    group_frame = _build_group_frame(frame, cfg.group_column, cfg.stratify_column)
    test_groups = _select_test_groups(group_frame, cfg)
    development = frame.loc[~frame[cfg.group_column].astype(str).isin(test_groups)].copy()

    split_rows: list[pd.DataFrame] = []

    if cfg.stratify_column and cfg.stratify_column in development.columns:
        splitter = StratifiedGroupKFold(
            n_splits=cfg.n_folds,
            shuffle=True,
            random_state=cfg.seed,
        )
        split_iter = splitter.split(
            development,
            development[cfg.stratify_column],
            groups=development[cfg.group_column],
        )
    else:
        splitter = GroupKFold(n_splits=cfg.n_folds)
        split_iter = splitter.split(development, groups=development[cfg.group_column])

    for fold, (_, val_idx) in enumerate(split_iter):
        fold_frame = frame.copy()
        fold_frame["fold"] = fold
        fold_frame["split"] = "train"
        fold_frame.loc[fold_frame[cfg.group_column].isin(development.iloc[val_idx][cfg.group_column]), "split"] = "val"
        if test_groups:
            fold_frame.loc[fold_frame[cfg.group_column].astype(str).isin(test_groups), "split"] = "test"
        split_rows.append(fold_frame)

    return pd.concat(split_rows, ignore_index=True)
