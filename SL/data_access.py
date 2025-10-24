"""Data access helpers for the clustering app."""

from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_clustering_dataset(name: str = "Clustering") -> pd.DataFrame:
    """Fetch the consolidated Dataiku dataset."""

    try:
        import dataiku  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Dataiku Python package is required to load the 'Clustering' dataset."
        ) from exc

    try:
        dataset = dataiku.Dataset(name)
    except Exception as exc:  # pragma: no cover - defensive message for runtime
        raise RuntimeError(f"Unable to access Dataiku dataset '{name}': {exc}") from exc

    try:
        dataframe = dataset.get_dataframe()
    except Exception as exc:  # pragma: no cover - defensive message for runtime
        raise RuntimeError(f"Failed to read Dataiku dataset '{name}': {exc}") from exc

    return dataframe


@st.cache_data(show_spinner=False)
def list_streamlit_datasets(zone_name: str = "StreamLit") -> List[str]:
    """Return the datasets available in the specified Dataiku flow zone."""

    try:
        import dataiku  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Dataiku Python package is required to list datasets for the app."
        ) from exc

    client = dataiku.api_client()
    project = client.get_default_project()

    try:
        zones = project.list_flow_zones()
    except Exception as exc:  # pragma: no cover - defensive message for runtime
        raise RuntimeError(f"Unable to list flow zones: {exc}") from exc

    streamlit_zone_id = None
    for zone in zones:
        if zone.get("name") == zone_name:
            streamlit_zone_id = zone.get("id")
            break

    if streamlit_zone_id is None:
        raise RuntimeError(f"Flow zone '{zone_name}' was not found in the project.")

    try:
        dataset_summaries = project.list_datasets()
    except Exception as exc:  # pragma: no cover - defensive message for runtime
        raise RuntimeError(f"Unable to list datasets: {exc}") from exc

    dataset_names: List[str] = []
    for summary in dataset_summaries:
        zone_id = (
            summary.get("zone")
            or summary.get("zoneId")
            or summary.get("flowZone")
        )
        if zone_id is None:
            try:
                definition = project.get_dataset(summary["name"]).get_definition()
            except Exception:
                continue
            zone_id = (
                definition.get("zone")
                or definition.get("zoneId")
                or definition.get("flowZone")
            )
        if zone_id == streamlit_zone_id:
            dataset_names.append(summary["name"])

    unique_names = sorted(dict.fromkeys(dataset_names))
    if not unique_names:
        raise RuntimeError(
            f"No datasets were found in the '{zone_name}' flow zone."
        )

    return unique_names


def resolve_column(df: pd.DataFrame, label: str, candidates: Iterable[str]) -> str:
    """Return the first matching column from a list of candidate names."""

    normalized = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise KeyError(f"Column for '{label}' not found. Checked: {', '.join(candidates)}")


def apply_filters(
    df: pd.DataFrame, column_mapping: Dict[str, str], filters: Dict[str, str]
) -> pd.DataFrame:
    """Apply commodity filters to a dataframe and return the filtered copy."""

    filtered = df.copy()
    for label, value in filters.items():
        column_name = column_mapping.get(label)
        if column_name is None:
            continue
        if value != "All":
            filtered = filtered[filtered[column_name] == value]
    return filtered


__all__ = [
    "apply_filters",
    "list_streamlit_datasets",
    "load_clustering_dataset",
    "resolve_column",
]
