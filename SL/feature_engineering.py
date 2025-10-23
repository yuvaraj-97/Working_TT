"""Feature engineering utilities shared by the CLI and Streamlit app."""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


_NUMERIC_WEIGHT_DEFAULT = 10


def extract_numeric(val) -> float:
    """Extract the first numeric value embedded in ``val``.

    Non-numeric values yield ``NaN`` so downstream callers can decide how to
    handle them.
    """

    match = re.search(r"\d+(\.\d+)?", str(val))
    if match:
        try:
            return float(match.group())
        except ValueError:
            return np.nan
    return np.nan


def guess_attribute_type(series: pd.Series) -> str:
    """Infer a sensible default attribute type from a pandas series."""

    if pd.api.types.is_numeric_dtype(series):
        return "Numerical"

    series_as_str = series.dropna().astype(str)
    if series_as_str.empty:
        return "Categorical"

    if series_as_str.str.contains(r"\d").mean() > 0.3:
        return "Alpha Numeric"

    unique_ratio = series_as_str.nunique() / max(len(series_as_str), 1)
    return "Categorical" if unique_ratio < 0.4 else "Text"


def encode_features(
    df: pd.DataFrame,
    attributes: Sequence[str],
    types: Sequence[str],
    unit_flags: Sequence[str],
    numeric_weight: float = _NUMERIC_WEIGHT_DEFAULT,
) -> Tuple[np.ndarray, List[str], np.ndarray | None]:
    """Convert heterogeneous attributes into a numerical matrix for clustering."""

    encoded_features = []
    categorical_features = []
    all_feature_types = []

    for attr, typ, unit_flag in zip(attributes, types, unit_flags):
        vals = df[attr].fillna("Missing")

        if str(unit_flag).lower() == "yes":
            vals_num = vals.apply(extract_numeric)
            median_value = vals_num.median()
            if np.isnan(median_value):
                median_value = 0.0
            vals_num = vals_num.fillna(median_value)
            enc = (
                StandardScaler().fit_transform(vals_num.values.reshape(-1, 1))
                * numeric_weight
            )
            encoded_features.append(enc)
            all_feature_types.append("numerical")
        elif typ.lower() == "numerical":
            vals_num = pd.to_numeric(vals, errors="coerce")
            median_value = vals_num.median()
            if np.isnan(median_value):
                median_value = 0.0
            vals_num = vals_num.fillna(median_value)
            enc = (
                StandardScaler().fit_transform(vals_num.values.reshape(-1, 1))
                * numeric_weight
            )
            encoded_features.append(enc)
            all_feature_types.append("numerical")
        elif typ.lower() == "categorical":
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            vals_str = vals.astype(str)
            enc = encoder.fit_transform(vals_str.values.reshape(-1, 1))
            encoded_features.append(enc)
            categorical_features.append(enc)
            all_feature_types.append("categorical")
        elif typ.lower() == "alpha numeric":
            tfidf = TfidfVectorizer()
            enc = tfidf.fit_transform(vals.astype(str)).toarray()
            encoded_features.append(enc)
            all_feature_types.append("text")
        else:
            tfidf = TfidfVectorizer()
            enc = tfidf.fit_transform(vals.astype(str)).toarray()
            encoded_features.append(enc)
            all_feature_types.append("text")

    vectors = np.hstack(encoded_features) if encoded_features else np.empty((len(df), 0))
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)

    cat_vectors = None
    if categorical_features:
        cat_vectors = np.hstack(categorical_features)

    return vectors, all_feature_types, cat_vectors
