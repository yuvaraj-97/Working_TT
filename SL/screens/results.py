"""Results screen showing clustering summaries."""

from __future__ import annotations

import streamlit as st

from models import RunResult
from state import navigate_to
from workflow import build_downloads


def render_results_screen() -> None:
    """Display the latest clustering results."""

    result: RunResult | None = st.session_state.get("last_result")
    if result is None:
        st.info("Run clustering from the setup screen to see results here.")
        return

    dataset_label = result.config.filters.get(
        "Commodity", result.config.dataset_name
    )
    st.subheader("Run summary")
    st.caption(f"Dataset: {dataset_label}")
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Clusters", int(result.cluster_summary.shape[0]))
    with summary_cols[1]:
        st.metric("Metric", result.metric)
    with summary_cols[2]:
        st.metric("Eps selected", f"{result.eps_selected:.3f}")

    if result.cluster_summary.empty:
        st.warning("No clusters were identified. Adjust the configuration and try again.")
    else:
        st.subheader("Cluster overview")
        st.caption("Open a cluster card to explore its details.")

        clusters = result.cluster_summary.reset_index(drop=True)
        for index, (_, cluster_row) in enumerate(clusters.iterrows()):
            if index % 3 == 0:
                cluster_columns = st.columns(3, gap="large")
            column = cluster_columns[index % 3]
            cluster_id = int(cluster_row["cluster"])
            if cluster_id < 0:
                cluster_heading = "Noise"
                cluster_label = None
            else:
                cluster_label = cluster_id + 1
                cluster_heading = f"Cluster {cluster_label}"

            with column:
                header_cols = st.columns([5, 1])
                with header_cols[0]:
                    st.markdown(f"**{cluster_heading}**")
                with header_cols[1]:
                    if st.button(
                        "View",
                        key=f"cluster_card_{cluster_id}",
                        type="secondary",
                        help="View cluster details",
                    ):
                        st.session_state.selected_cluster = cluster_id
                        st.session_state.pending_navigation_target = "Cluster Detail"
                        navigate_to("cluster_detail")
                        st.rerun()

                metric_cols = st.columns(2)
                with metric_cols[0]:
                    st.metric("Size", int(cluster_row["cluster_size"]))
                with metric_cols[1]:
                    st.metric(
                        "Mean likeness",
                        f"{cluster_row['mean_likeness']:.2f}",
                    )

                representatives = result.roster_parts.get(cluster_id, [])
                if representatives:
                    st.caption("Representative parts: " + ", ".join(representatives))

        st.subheader("Cluster metrics")
        cluster_metrics_display = result.cluster_summary.copy()
        cluster_metrics_display["Cluster"] = cluster_metrics_display["cluster"].apply(
            lambda value: "Noise" if int(value) < 0 else str(int(value) + 1)
        )
        cluster_metrics_display = cluster_metrics_display[
            ["Cluster", "cluster_size", "mean_likeness"]
        ]
        st.dataframe(
            cluster_metrics_display,
            use_container_width=True,
            column_config={
                "Cluster": st.column_config.Column("Cluster"),
                "cluster_size": st.column_config.NumberColumn("Size", format="%d"),
                "mean_likeness": st.column_config.NumberColumn(
                    "Mean likeness", format="%.3f"
                ),
            },
        )

    st.subheader("Candidate evaluation")
    st.dataframe(
        result.candidate_df,
        use_container_width=True,
        column_config={
            "eps": st.column_config.NumberColumn("Eps", format="%.3f"),
            "silhouette_score": st.column_config.NumberColumn(
                "Silhouette", format="%.3f"
            ),
            "num_clusters": st.column_config.NumberColumn("Clusters", format="%d"),
            "num_noise_points": st.column_config.NumberColumn(
                "Noise points", format="%d"
            ),
            "proportion_noise": st.column_config.NumberColumn(
                "Noise proportion", format="%.2f"
            ),
            "mean_cluster_likeness": st.column_config.NumberColumn(
                "Mean likeness", format="%.3f"
            ),
            "Score": st.column_config.NumberColumn("Score", format="%.3f"),
            "Recommended": st.column_config.CheckboxColumn(
                "Recommended", disabled=True
            ),
        },
    )

    excel_bytes, pdf_bytes = build_downloads(result)

    download_columns = st.columns(2)
    with download_columns[0]:
        st.download_button(
            "Download Excel output",
            data=excel_bytes,
            file_name="Clustered_Output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with download_columns[1]:
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name="Cluster_Report.pdf",
            mime="application/pdf",
        )
