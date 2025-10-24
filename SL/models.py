"""Data models for the Streamlit clustering app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class RunConfig:
    """Configuration captured when launching a clustering run."""

    dataset_name: str
    filters: Dict[str, str]
    column_mapping: Dict[str, str]
    part_number_col: str
    chosen_attributes: List[str]
    chosen_types: List[str]
    chosen_unit_flags: List[str]
    manual_configuration: bool
    eps_min: float | None
    eps_max: float | None
    eps_step: float | None
    min_samples: int | None
    numeric_weight: float
    metric: str
    manual_eps: float
    min_fill_ratio: int


@dataclass
class RunResult:
    """Aggregated outcome of a clustering run."""

    config: RunConfig
    filtered_df: pd.DataFrame
    result_df: pd.DataFrame
    grouped_df: pd.DataFrame
    cluster_summary: pd.DataFrame
    candidate_df: pd.DataFrame
    eps_selected: float
    roster_parts: Dict[int, List[str]]
    metric: str
    part_number_col: str
    vectors: np.ndarray
    cat_vectors: np.ndarray | None


__all__ = ["RunConfig", "RunResult"]
