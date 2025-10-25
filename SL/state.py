"""Session state helpers for the clustering app."""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Dict, List
from uuid import uuid4

import streamlit as st

from .models import RunConfig, RunResult


def initialize_app_state() -> None:
    """Ensure expected keys exist in :mod:`streamlit` session state."""

    defaults: Dict[str, object] = {
        "screen": "setup",
        "pending_run": None,
        "last_result": None,
        "run_history": [],
        "selected_cluster": None,
        "last_screen": "setup",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_app_state() -> None:
    """Return the application to its default navigation and cache state."""

    st.session_state.screen = "setup"
    st.session_state.last_screen = "setup"
    for key in [
        "pending_run",
        "last_result",
        "selected_cluster",
        "attribute_config",
        "attribute_signature",
        "attribute_selection_confirmed",
        "attribute_selection_threshold",
    ]:
        st.session_state.pop(key, None)


def navigate_to(screen: str) -> None:
    """Switch to the provided screen and update navigation history."""

    st.session_state.screen = screen
    if screen != "loading":
        st.session_state.last_screen = screen


def available_navigation_options() -> List[str]:
    """Return the list of screens that should be visible in navigation."""

    return ["setup", "results", "cluster_detail", "history"]


def store_run_history(result: RunResult) -> None:
    """Persist a run summary to the Streamlit session history."""

    history_entry = {
        "id": str(uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "config": copy.deepcopy(result.config),
        "summary": {
            "clusters": int(result.cluster_summary.shape[0]),
            "eps": float(result.eps_selected),
            "metric": result.metric,
            "dataset": result.config.filters.get(
                "Commodity", result.config.dataset_name
            ),
            "filters": result.config.filters,
            "attributes": result.config.chosen_attributes,
        },
    }

    st.session_state.run_history.insert(0, history_entry)
    st.session_state.run_history = st.session_state.run_history[:10]


def trigger_clustering_run(config: RunConfig) -> None:
    """Queue a clustering run and jump to the loading overlay."""

    st.session_state.pending_run = config
    navigate_to("loading")
    st.rerun()


__all__ = [
    "initialize_app_state",
    "reset_app_state",
    "navigate_to",
    "available_navigation_options",
    "store_run_history",
    "trigger_clustering_run",
]
