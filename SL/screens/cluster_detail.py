"""Cluster detail screen."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from models import RunResult
from state import navigate_to


def render_cluster_detail_screen() -> None:
    """Render detailed metrics for a selected cluster."""

    result: RunResult | None = st.session_state.get("last_result")
    if result is None:
        st.info("Results will appear after running clustering from the setup screen.")
        return

    cluster_id = st.session_state.get("selected_cluster")
    if cluster_id is None:
        st.info("Select a cluster from the results screen to view its details.")
        return

    cluster_summary = result.cluster_summary
    if cluster_summary.empty or cluster_id not in cluster_summary["cluster"].values:
        st.warning("The selected cluster is no longer available. Rerun clustering to refresh.")
        return

    def _go_back_to_results() -> None:
        st.session_state.pending_navigation_target = "Results"
        navigate_to("results")

    st.button(
        "← Back to results",
        on_click=_go_back_to_results,
        type="secondary",
    )

    summary_row = cluster_summary[cluster_summary["cluster"] == cluster_id].iloc[0]
    if int(cluster_id) < 0:
        cluster_heading = "Noise cluster overview"
    else:
        cluster_number = int(cluster_id) + 1
        cluster_heading = f"Cluster {cluster_number} overview"

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='material-header'>{cluster_heading}</div>",
        unsafe_allow_html=True,
    )
    dataset_label = result.config.filters.get(
        "Commodity", result.config.dataset_name
    )
    st.caption(f"Dataset: {dataset_label}")

    overview_columns = st.columns(3)
    with overview_columns[0]:
        st.metric("Parts", int(summary_row["cluster_size"]))
    with overview_columns[1]:
        st.metric("Mean likeness", f"{summary_row['mean_likeness']:.2f}")
    with overview_columns[2]:
        st.metric("Metric", result.metric)

    representatives = result.roster_parts.get(int(cluster_id), [])
    if representatives:
        st.caption("Representative parts: " + ", ".join(representatives))

    st.markdown("</div>", unsafe_allow_html=True)

    cluster_parts = result.grouped_df[result.grouped_df["cluster"] == cluster_id]
    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>Cluster members</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(cluster_parts, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    feature_matrix = (
        result.cat_vectors
        if result.metric == "jaccard" and result.cat_vectors is not None
        else result.vectors
    )
    cluster_mask = result.result_df["cluster"].to_numpy() == cluster_id
    cluster_vectors = feature_matrix[cluster_mask]
    cluster_parts_labels = (
        result.result_df.loc[cluster_mask, result.part_number_col].astype(str)
    )

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>Similarity diagnostics</div>",
        unsafe_allow_html=True,
    )

    if cluster_vectors.shape[0] > 1:
        from sklearn.metrics import pairwise_distances

        distances = pairwise_distances(
            cluster_vectors,
            metric=result.metric
            if result.metric != "jaccard" or result.cat_vectors is not None
            else "euclidean",
        )
        similarity = 1.0 / (1.0 + distances)
        heatmap_df = pd.DataFrame(
            similarity,
            index=cluster_parts_labels,
            columns=cluster_parts_labels,
        ).reset_index(names=result.part_number_col)
        heatmap_long = heatmap_df.melt(
            id_vars=result.part_number_col,
            var_name="Peer",
            value_name="Similarity",
        )
        heatmap_chart = (
            alt.Chart(heatmap_long)
            .mark_rect()
            .encode(
                x=alt.X("Peer:N", title="Peer part"),
                y=alt.Y(f"{result.part_number_col}:N", title="Part"),
                color=alt.Color("Similarity:Q", scale=alt.Scale(scheme="blues")),
                tooltip=[result.part_number_col, "Peer", "Similarity"],
            )
        )
        st.altair_chart(heatmap_chart, use_container_width=True)
    else:
        st.info("Not enough parts in this cluster to produce similarity diagnostics.")

    st.markdown("</div>", unsafe_allow_html=True)
