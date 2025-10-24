"""History screen showing previous clustering runs."""

from __future__ import annotations

import copy

import streamlit as st

from models import RunConfig
from state import trigger_clustering_run


def render_history_screen() -> None:
    """Display the recently executed clustering runs."""

    history = st.session_state.get("run_history", [])
    if not history:
        st.info("Run clustering to populate your history.")
        return

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>Recent runs</div>",
        unsafe_allow_html=True,
    )

    for entry in history:
        summary = entry["summary"]
        config: RunConfig = entry["config"]
        st.markdown("<div class='material-card'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='material-header'>Run from {entry['timestamp']}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Dataset: {summary['dataset']}")
        st.caption(
            "Filters: "
            + ", ".join(
                f"{label} = {value}" for label, value in summary["filters"].items()
            )
        )
        st.caption("Attributes: " + ", ".join(summary["attributes"]))
        history_cols = st.columns(3)
        with history_cols[0]:
            st.metric("Clusters", summary["clusters"])
        with history_cols[1]:
            st.metric("Metric", summary["metric"])
        with history_cols[2]:
            st.metric("Eps", f"{summary['eps']:.3f}")

        if st.button(
            "Run this configuration again",
            key=f"history_rerun_{entry['id']}",
            use_container_width=True,
        ):
            st.session_state.active_dataset = config.dataset_name
            trigger_clustering_run(copy.deepcopy(config))

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
