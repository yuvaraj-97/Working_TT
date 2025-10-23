"""Compatibility helpers for the original Excel-driven CLSING workflow."""
from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def process_attributes(excel_file: str) -> Tuple[List[str], List[str], List[str]]:
    """Extract attribute metadata from the legacy "Summary Tab" worksheet."""

    summary_df = pd.read_excel(
        excel_file,
        sheet_name="Summary Tab",
        usecols="P:R",
        skiprows=5,
        nrows=100,
    )
    summary_df.columns = [col.strip() for col in summary_df.columns]

    attributes = summary_df["Attribute"].dropna().tolist()
    types = summary_df["Type"].dropna().tolist()
    unit_flags = summary_df["Unit Extraction"].fillna("No").tolist()
    return attributes, types, unit_flags


def read_filtered_data(excel_file: str) -> pd.DataFrame:
    """Return the data prepared in the legacy "FilteredOutput" sheet."""

    return pd.read_excel(excel_file, sheet_name="FilteredOutput")
