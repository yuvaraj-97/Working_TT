"""Helper for creating downloadable Excel workbooks."""
from __future__ import annotations

from io import BytesIO
from typing import Sequence

import pandas as pd


def build_excel_workbook(
    detailed_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    sheet_names: Sequence[str] = ("All Data", "Grouped by Part Number"),
) -> BytesIO:
    """Return an in-memory Excel workbook with the clustering outputs."""

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        detailed_df.to_excel(writer, sheet_name=sheet_names[0], index=False)
        grouped_df.to_excel(writer, sheet_name=sheet_names[1], index=False)
    output.seek(0)
    return output
