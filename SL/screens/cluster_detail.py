"""Cluster detail screen."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from models import RunResult
from state import navigate_to
from ui.components import material_card


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

    dataset_label = result.config.filters.get(
        "Commodity", result.config.dataset_name
    )

    with material_card(cluster_heading) as card:
        card.caption(f"Dataset: {dataset_label}")

        overview_columns = card.columns(3)
        overview_columns[0].metric("Parts", int(summary_row["cluster_size"]))
        overview_columns[1].metric("Mean likeness", f"{summary_row['mean_likeness']:.2f}")
        overview_columns[2].metric("Metric", result.metric)

        representatives = result.roster_parts.get(int(cluster_id), [])
        if representatives:
            card.caption("Representative parts: " + ", ".join(representatives))

    cluster_parts = result.grouped_df[result.grouped_df["cluster"] == cluster_id]
    with material_card("Cluster members") as card:
        card.dataframe(cluster_parts, use_container_width=True)

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

    toggle_key = f"show_similarity_{cluster_id}"
    with material_card("Similarity diagnostics") as card:
        show_similarity = card.toggle(
            "Show similarity diagnostics",
            value=False,
            key=toggle_key,
            help="Display the similarity heatmap for the current cluster.",
        )

        if show_similarity:
            if cluster_vectors.shape[0] > 1:
                from sklearn.metrics import pairwise_distances

                distances = pairwise_distances(
                    cluster_vectors,
                    metric=
                    result.metric
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
                        color=alt.Color(
                            "Similarity:Q", scale=alt.Scale(scheme="blues")
                        ),
                        tooltip=[result.part_number_col, "Peer", "Similarity"],
                    )
                )
                card.altair_chart(heatmap_chart, use_container_width=True)
            else:
                card.info(
                    "Not enough parts in this cluster to produce similarity diagnostics."
                )
        else:
            card.caption("Enable the toggle to explore similarity diagnostics.")
