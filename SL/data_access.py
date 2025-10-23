"""Data loading helpers for the Streamlit clustering app."""
from __future__ import annotations

from typing import IO, Union

import pandas as pd


def load_consolidated_sheet(
    excel_file: Union[str, bytes, IO[bytes]],
    sheet_name: str = "Consolidated",
) -> pd.DataFrame:
    """Return the consolidated worksheet as a DataFrame.

    Parameters
    ----------
    excel_file:
        Path, binary handle, or in-memory buffer pointing to an Excel workbook.
    sheet_name:
        Name of the worksheet containing the consolidated dataset.

    Raises
    ------
    ValueError
        If the requested sheet does not exist in the workbook.
    """

    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
    except ValueError as exc:  # Raised when the sheet cannot be found.
        raise ValueError(
            f"Sheet '{sheet_name}' not found in the uploaded workbook."
        ) from exc

    return df
