"""Results screen showing clustering summaries."""

from __future__ import annotations

import streamlit as st

from ..models import RunResult
from ..state import navigate_to
from ..workflow import build_downloads


def render_results_screen() -> None:
    """Display the latest clustering results."""

    result: RunResult | None = st.session_state.get("last_result")
    if result is None:
        st.info("Run clustering from the setup screen to see results here.")
        return

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown("<div class='material-header'>Run summary</div>", unsafe_allow_html=True)
    st.caption(f"Dataset: {result.config.dataset_name}")
    summary_cols = st.columns(3)
    with summary_cols[0]:
        st.metric("Clusters", int(result.cluster_summary.shape[0]))
    with summary_cols[1]:
        st.metric("Metric", result.metric)
    with summary_cols[2]:
        st.metric("Eps selected", f"{result.eps_selected:.3f}")
    st.markdown("</div>", unsafe_allow_html=True)

    if result.cluster_summary.empty:
        st.warning("No clusters were identified. Adjust the configuration and try again.")
    else:
        st.markdown("<div class='material-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='material-header'>Cluster overview</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='material-subtitle'>Open a cluster card to explore its details.</div>",
            unsafe_allow_html=True,
        )

        clusters = result.cluster_summary.reset_index(drop=True)
        for index, (_, cluster_row) in enumerate(clusters.iterrows()):
            if index % 3 == 0:
                cluster_columns = st.columns(3, gap="large")
            column = cluster_columns[index % 3]
            cluster_id = int(cluster_row["cluster"])
            cluster_label = cluster_id + 1
            with column:
                st.markdown("<div class='cluster-card'>", unsafe_allow_html=True)
                header_cols = st.columns([3, 1])
                with header_cols[0]:
                    st.markdown(
                        f"<div class='material-header'>Cluster {cluster_label}</div>",
                        unsafe_allow_html=True,
                    )
                with header_cols[1]:
                    if st.button(
                        "Open cluster",
                        key=f"cluster_card_{cluster_id}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_cluster = cluster_id
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
                    st.caption(
                        "Representative parts: " + ", ".join(representatives)
                    )
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='material-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='material-header'>Cluster metrics</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            result.cluster_summary,
            use_container_width=True,
            column_config={
                "cluster": st.column_config.NumberColumn("Cluster"),
                "cluster_size": st.column_config.NumberColumn("Size", format="%d"),
                "mean_likeness": st.column_config.NumberColumn(
                    "Mean likeness", format="%.3f"
                ),
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>Candidate evaluation</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(result.candidate_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
