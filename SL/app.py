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


DATASET_ALIASES = {
    "Airflow": "Airflow",
    "Bearings": "Bearings",
    "Capacitors": "Capacitors",
    "Compressors": "Compressors",
    "Drives": "Drives",
    "Electric_Motors": "Electric Motors",
    "Fasteners_Hardware_Fittings": "Fasteners Hardware & Fittings",
    "Foams_Fiberglass": "Foams & Fiberglass",
    "Pulleys_Tensioners": "Pulleys Tensioners",
    "Pumps": "Pumps",
    "Valves": "Valves",
}


def ensure_attribute_state() -> None:
    if "attribute_config" not in st.session_state:
        st.session_state.attribute_config = None
        st.session_state.attribute_signature = None
    if "attribute_selection_confirmed" not in st.session_state:
        st.session_state.attribute_selection_confirmed = False
    if "attribute_selection_threshold" not in st.session_state:
        st.session_state.attribute_selection_threshold = 70


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

    pending_label = st.session_state.pop("pending_navigation_target", None)
    if pending_label in labels:
        st.session_state["navigation_menu"] = pending_label
    elif pending_label is not None:
        st.session_state["navigation_menu"] = default_label

    if "navigation_menu" not in st.session_state:
        st.session_state["navigation_menu"] = default_label

    current_label = st.session_state.get("navigation_menu", default_label)
    if current_label not in labels:
        current_label = default_label
        st.session_state["navigation_menu"] = current_label

    selected_label = st.sidebar.radio(
        "Navigate",
        options=labels,
        index=labels.index(current_label),
        key="navigation_menu",
    )
    selected_screen = label_to_screen[selected_label]

    if selected_screen != st.session_state.screen:
        navigate_to(selected_screen)
    else:
        st.session_state.last_screen = st.session_state.screen


def render_by_part(_: str, __: list[str], ___) -> None:
    st.info("The By Part configuration will be available soon.")


def render_by_commodity(
    dataset_name: str,
    dataset_options: list[str],
    dataset_aliases: dict[str, str],
    consolidated_df,
) -> None:
    render_navigation()

    if st.session_state.screen == "setup":
        render_setup_screen(
            consolidated_df, dataset_name, dataset_options, dataset_aliases
        )
    elif st.session_state.screen == "results":
        render_results_screen()
    elif st.session_state.screen == "cluster_detail":
        render_cluster_detail_screen()
    elif st.session_state.screen == "history":
        render_history_screen()


def main() -> None:
    configure_page()
    initialize_app_state()
    ensure_attribute_state()

    dataset_options = list(DATASET_ALIASES.keys())

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

    if st.session_state.screen == "loading":
        execute_pending_run(consolidated_df)
        return

    st.title("Clustering Workbench")
    st.caption(
        "Interactive workflow for preparing clustering inputs using the pre-loaded Dataiku dataset."
    )

    view_mode = st.radio(
        "Configuration mode",
        options=["By Commodity", "By Part"],
        horizontal=True,
    )

    if view_mode == "By Commodity":
        render_by_commodity(
            active_dataset,
            dataset_options,
            DATASET_ALIASES,
            consolidated_df,
        )
    else:
        render_by_part(active_dataset, dataset_options, consolidated_df)


if __name__ == "__main__":
    main()
