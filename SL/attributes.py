"""Attribute configuration helpers for the clustering app."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import pandas as pd
import streamlit as st

from feature_engineering import guess_attribute_type


IGNORED_COLUMNS = {"part name", "part_name", "partname"}


def _should_ignore(column: str) -> bool:
    return column.lower() in IGNORED_COLUMNS


def build_attribute_config(
    df: pd.DataFrame,
    excluded_columns: Iterable[str],
) -> pd.DataFrame:
    """Construct a dataframe describing available attributes for clustering."""

    excluded_lookup = {col.lower() for col in excluded_columns}
    attributes: List[str] = []
    for column in df.columns:
        column_key = column.lower()
        if column_key in excluded_lookup or _should_ignore(column):
            continue
        attributes.append(column)

    if not attributes:
        return pd.DataFrame(
            columns=["Attribute", "Type", "Unit Extraction", "Fill Ratio", "Include"]
        )

    fill_ratio = (df[attributes].notna().mean() * 100).round(1)

    config = pd.DataFrame(
        {
            "Attribute": attributes,
            "Type": [guess_attribute_type(df[attr]) for attr in attributes],
            "Unit Extraction": ["No"] * len(attributes),
            "Fill Ratio": fill_ratio.reindex(attributes).fillna(0.0).values,
            "Include": False,
        }
    )

    config = config[["Attribute", "Type", "Unit Extraction", "Fill Ratio", "Include"]]
    return config.sort_values("Fill Ratio", ascending=False).reset_index(drop=True)


def update_attribute_session_state(
    filtered_df: pd.DataFrame,
    excluded_columns: Iterable[str],
    signature: Tuple,
) -> None:
    """Keep the attribute configuration cached and refresh fill ratios on demand."""

    if (
        st.session_state.get("attribute_config") is None
        or st.session_state.get("attribute_signature") != signature
    ):
        st.session_state.attribute_config = build_attribute_config(
            filtered_df, excluded_columns
        )
        st.session_state.attribute_signature = signature
        return

    current_config = st.session_state.attribute_config
    refreshed = build_attribute_config(filtered_df, excluded_columns)
    merged = current_config.merge(
        refreshed[["Attribute", "Fill Ratio"]],
        on="Attribute",
        how="right",
        suffixes=("", "_new"),
    )
    merged["Fill Ratio"] = (
        merged["Fill Ratio_new"].fillna(merged["Fill Ratio"]).round(1)
    )
    merged = merged.drop(columns=["Fill Ratio_new"])
    merged = merged[["Attribute", "Type", "Unit Extraction", "Fill Ratio", "Include"]]
    st.session_state.attribute_config = (
        merged.sort_values("Fill Ratio", ascending=False).reset_index(drop=True)
    )
    st.session_state.attribute_signature = signature


__all__ = ["build_attribute_config", "update_attribute_session_state"]
