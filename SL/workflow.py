"""Core clustering workflow logic."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from clustering import (
    add_likeness_score,
    cluster_and_score,
    enforce_min_cluster_size,
    recommend_eps,
)
from exporters import build_excel_workbook
from feature_engineering import encode_features
from reporting import export_cluster_report_pdf

from attributes import update_attribute_session_state
from data_access import apply_filters
from models import RunConfig, RunResult
from state import navigate_to, store_run_history
from ui.loading import LoadingScreen


def _auto_eps_sweep(feature_matrix: np.ndarray, metric: str) -> np.ndarray:
    """Derive a reasonable epsilon sweep based on nearest-neighbour distances."""

    if feature_matrix.size == 0 or feature_matrix.shape[0] < 2:
        return np.array([0.5])

    neighbour_count = min(10, feature_matrix.shape[0] - 1)
    if neighbour_count <= 0:
        return np.array([0.5])

    metric_name = metric if metric in {"euclidean", "cosine", "jaccard"} else "euclidean"
    try:
        nn = NearestNeighbors(n_neighbors=neighbour_count + 1, metric=metric_name)
        nn.fit(feature_matrix)
        distances, _ = nn.kneighbors(feature_matrix)
    except Exception:
        nn = NearestNeighbors(n_neighbors=neighbour_count + 1, metric="euclidean")
        nn.fit(feature_matrix)
        distances, _ = nn.kneighbors(feature_matrix)

    distances = distances[:, 1:]
    if distances.size == 0:
        return np.array([0.5])

    kth_distances = distances[:, -1]
    lower, upper = np.percentile(kth_distances, [15, 85])
    min_eps = max(lower, 0.01)
    max_eps = max(upper, min_eps + 0.05)
    step = max((max_eps - min_eps) / 20, 0.01)

    values = np.arange(min_eps, max_eps + step, step)
    values = np.clip(values, 0.01, None)
    return np.unique(values)


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
        if eps_values.size == 0:
            raise ValueError("Manual epsilon sweep produced no values.")
        min_samples = config.min_samples if config.min_samples is not None else 2
    else:
        feature_matrix = (
            cat_vectors if metric == "jaccard" and cat_vectors is not None else vectors
        )
        loader.update("Estimating epsilon sweep…", 55)
        eps_values = _auto_eps_sweep(feature_matrix, metric)
        min_samples = 2

    loader.update("Evaluating candidate clusters…", 70)
    candidate_df = cluster_and_score(
        filtered_df,
        vectors,
        eps_values,
        metric,
        cat_vectors=cat_vectors,
        min_samples=min_samples,
    )

    if candidate_df.empty:
        raise ValueError("No viable clustering configuration was produced.")

    candidate_df = candidate_df.copy()
    candidate_df["score"] = (
        candidate_df["silhouette_score"]
        * (1 - candidate_df["proportion_noise"])
        * np.log(candidate_df["num_clusters"] + 1)
    )
    candidate_df = candidate_df.sort_values("score", ascending=False).reset_index(
        drop=True
    )

    loader.update("Selecting best cluster result…", 85)
    recommended_eps = float(recommend_eps(candidate_df))
    if config.manual_configuration and config.manual_eps is not None:
        eps_selected = float(config.manual_eps)
    else:
        eps_selected = recommended_eps

    candidate_df["recommended"] = np.isclose(
        candidate_df["eps"], eps_selected, rtol=1e-4, atol=1e-4
    )
    candidate_df = candidate_df.rename(
        columns={"score": "Score", "recommended": "Recommended"}
    )

    loader.update("Generating final clusters…", 90)
    db = DBSCAN(eps=eps_selected, min_samples=min_samples, metric=metric)
    if metric == "jaccard" and cat_vectors is not None:
        labels = db.fit_predict(cat_vectors)
    else:
        labels = db.fit_predict(vectors)

    result_df = filtered_df.copy()
    result_df["cluster"] = labels
    result_df = add_likeness_score(result_df, vectors, labels, metric, cat_vectors)
    result_df = enforce_min_cluster_size(result_df, min_size=2)

    grouped_df = (
        result_df.groupby(config.part_number_col)[
            config.chosen_attributes + ["cluster", "likeness_score"]
        ]
        .first()
        .reset_index()
    )

    roster_parts = (
        result_df[result_df["cluster"] != -1]
        .sort_values(["cluster", "likeness_score"], ascending=[True, False])
        .groupby("cluster")[config.part_number_col]
        .apply(lambda series: series.astype(str).head(3).tolist())
        .to_dict()
    )

    loader.update("Calculating cluster summary…", 95)
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
    st.rerun()


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
