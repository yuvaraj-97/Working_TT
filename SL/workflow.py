"""Core clustering workflow logic."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

from clustering import (
    add_likeness_score,
    cluster_and_score,
    enforce_min_cluster_size,
    recommend_eps,
)
from exporters import build_excel_workbook
from feature_engineering import encode_features
from reporting import export_cluster_report_pdf

from .attributes import update_attribute_session_state
from .data_access import apply_filters
from .models import RunConfig, RunResult
from .state import navigate_to, store_run_history
from .ui.loading import LoadingScreen


def perform_clustering(
    consolidated_df: pd.DataFrame, config: RunConfig, loader: LoadingScreen
) -> RunResult:
    """Execute clustering and build a :class:`RunResult`."""

    filtered_df = apply_filters(
        consolidated_df, config.column_mapping, config.filters
    )

    if filtered_df.empty:
        raise ValueError("No rows remain after applying the selected filters.")

    loader.update("Encoding features for clustering…", 10)
    vectors, _, cat_vectors = encode_features(
        filtered_df,
        config.chosen_attributes,
        config.chosen_types,
        config.chosen_unit_flags,
        config.numeric_weight,
    )

    metric = config.metric
    loader.update("Validating metric selection…", 25)
    if metric == "jaccard" and (cat_vectors is None or cat_vectors.size == 0):
        st.warning(
            "Jaccard metric requires at least one categorical attribute. Falling back to Euclidean."
        )
        metric = "euclidean"

    loader.update("Preparing epsilon sweep…", 40)
    if config.manual_configuration:
        if None in (config.eps_min, config.eps_max, config.eps_step):
            raise ValueError("Manual configuration requires eps min, max, and step values.")
        if config.eps_min >= config.eps_max:
            raise ValueError("Eps maximum must be greater than eps minimum.")
        eps_values = np.arange(
            config.eps_min,
            config.eps_max + (config.eps_step or 0) / 2,
            config.eps_step,
        )
        min_samples = config.min_samples if config.min_samples is not None else 2
    else:
        feature_matrix = (
            cat_vectors if metric == "jaccard" and cat_vectors is not None else vectors
        )
        loader.update("Recommending epsilon…", 60)
        eps_values, min_samples = recommend_eps(
            feature_matrix,
            metric,
            config.manual_eps,
        )

    loader.update("Clustering candidate sweeps…", 75)
    candidate_df, roster_parts = cluster_and_score(
        vectors,
        metric,
        eps_values,
        min_samples,
        config.chosen_attributes,
        config.chosen_types,
        config.part_number_col,
        cat_vectors=cat_vectors,
    )

    if candidate_df.empty:
        raise ValueError("No viable clustering configuration was produced.")

    loader.update("Selecting best cluster result…", 85)
    result_df = add_likeness_score(candidate_df.iloc[0]["result_df"])
    grouped_df = candidate_df.iloc[0]["grouped_df"]
    eps_selected = candidate_df.iloc[0]["eps"]
    candidate_df = candidate_df.drop(columns=["result_df", "grouped_df"])

    loader.update("Calculating cluster summary…", 95)
    grouped_df = enforce_min_cluster_size(grouped_df, min_size=2)
    cluster_summary = (
        grouped_df.groupby("cluster")["likeness_score"].agg(
            cluster_size="count", mean_likeness="mean"
        )
    ).reset_index()

    loader.finalize("Clustering complete!")

    return RunResult(
        config=config,
        filtered_df=filtered_df,
        result_df=result_df,
        grouped_df=grouped_df,
        cluster_summary=cluster_summary,
        candidate_df=candidate_df,
        eps_selected=float(eps_selected),
        roster_parts=roster_parts,
        metric=metric,
        part_number_col=config.part_number_col,
        vectors=vectors,
        cat_vectors=cat_vectors,
    )


def execute_pending_run(consolidated_df: pd.DataFrame) -> None:
    """Run any queued clustering request and transition to the results view."""

    pending_config: RunConfig | None = st.session_state.get("pending_run")
    if pending_config is None:
        navigate_to(st.session_state.get("last_screen", "setup"))
        return

    loader = LoadingScreen("Running clustering", "This may take a few seconds")
    try:
        result = perform_clustering(consolidated_df, pending_config, loader)
    except Exception as exc:  # pragma: no cover - surface friendly error
        loader.clear()
        st.error(str(exc))
        navigate_to(st.session_state.get("last_screen", "setup"))
        return

    loader.clear()
    st.session_state.last_result = result
    st.session_state.pending_run = None
    store_run_history(result)
    navigate_to("results")


def prepare_attribute_config(
    filtered_df: pd.DataFrame,
    excluded_columns: Tuple[str, ...],
    signature: Tuple,
) -> pd.DataFrame:
    """Ensure the attribute configuration is ready for rendering."""

    update_attribute_session_state(filtered_df, excluded_columns, signature)
    attribute_config = st.session_state.attribute_config
    if attribute_config is None:
        return attribute_config
    return attribute_config.copy(deep=True)


def build_downloads(result: RunResult) -> Tuple[bytes, bytes]:
    """Create the Excel workbook and PDF report for download buttons."""

    excel_bytes = build_excel_workbook(result.result_df, result.grouped_df)
    pdf_bytes = export_cluster_report_pdf(
        result.result_df,
        candidate_eps_table=result.candidate_df,
        metric=result.metric,
        eps_selected=result.eps_selected,
        attributes=result.config.chosen_attributes,
        types=result.config.chosen_types,
        output=None,
        top_n=10,
    )
    return excel_bytes, pdf_bytes


__all__ = [
    "perform_clustering",
    "execute_pending_run",
    "prepare_attribute_config",
    "build_downloads",
]
