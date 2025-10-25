"""Streamlit entry point for the clustering workbench."""

from __future__ import annotations

import streamlit as st

from data_access import load_clustering_dataset
from screens import (
    render_cluster_detail_screen,
    render_history_screen,
    render_results_screen,
    render_setup_screen,
)
from state import (
    available_navigation_options,
    initialize_app_state,
    navigate_to,
)
from ui.styles import configure_page
from workflow import execute_pending_run


SCREEN_TITLES = {
    "setup": "Setup",
    "results": "Results",
    "cluster_detail": "Cluster Detail",
    "history": "History",
}


DATASET_OPTIONS = ["A", "B", "C"]


def ensure_attribute_state() -> None:
    if "attribute_config" not in st.session_state:
        st.session_state.attribute_config = None
        st.session_state.attribute_signature = None


def render_navigation() -> None:
    options = available_navigation_options()
    labels = [SCREEN_TITLES[option] for option in options]
    label_to_screen = dict(zip(labels, options))

    current_screen = st.session_state.screen
    if current_screen not in options:
        current_screen = st.session_state.last_screen
    if current_screen not in options:
        current_screen = options[0]

    default_label = SCREEN_TITLES.get(current_screen, labels[0])
    selected_label = st.sidebar.radio(
        "Navigate",
        options=labels,
        index=labels.index(st.session_state.get("navigation_menu", default_label)),
        key="navigation_menu",
    )
    selected_screen = label_to_screen[selected_label]

    if selected_screen != st.session_state.screen:
        navigate_to(selected_screen)
    else:
        st.session_state.last_screen = st.session_state.screen


def render_by_part(_: str, __: list[str], ___) -> None:
    st.info("The By Part configuration will be available soon.")


def render_by_commodity(dataset_name: str, dataset_options: list[str], consolidated_df) -> None:
    initialize_app_state()

    if st.session_state.screen == "loading":
        execute_pending_run(consolidated_df)
        return

    render_navigation()

    if st.session_state.screen == "setup":
        render_setup_screen(consolidated_df, dataset_name, dataset_options)
    elif st.session_state.screen == "results":
        render_results_screen()
    elif st.session_state.screen == "cluster_detail":
        render_cluster_detail_screen()
    elif st.session_state.screen == "history":
        render_history_screen()


def main() -> None:
    configure_page()
    st.title("Clustering Workbench")
    st.caption(
        "Interactive workflow for preparing clustering inputs using the pre-loaded Dataiku dataset."
    )

    ensure_attribute_state()

    dataset_options = DATASET_OPTIONS

    if not dataset_options:
        st.error("No datasets are configured for the Commodity selection.")
        st.stop()

    active_dataset = st.session_state.get("active_dataset")
    if active_dataset not in dataset_options:
        active_dataset = dataset_options[0]
        st.session_state.active_dataset = active_dataset

    try:
        consolidated_df = load_clustering_dataset(active_dataset)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    view_mode = st.radio(
        "Configuration mode",
        options=["By Commodity", "By Part"],
        horizontal=True,
    )

    if view_mode == "By Commodity":
        render_by_commodity(active_dataset, dataset_options, consolidated_df)
    else:
        render_by_part(active_dataset, dataset_options, consolidated_df)


if __name__ == "__main__":
    main()
