from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple
from uuid import uuid4

import copy

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from clustering import (
    add_likeness_score,
    cluster_and_score,
    enforce_min_cluster_size,
    recommend_dbscan_metric,
    recommend_eps,
)
from exporters import build_excel_workbook
from feature_engineering import encode_features, guess_attribute_type
from reporting import export_cluster_report_pdf

st.set_page_config(
    page_title="Clustering Workbench",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            color-scheme: dark;
        }
        .stApp {
            background: radial-gradient(circle at top, #1e293b, #0f172a 65%);
            color: #e2e8f0;
        }
        .material-card {
            background: rgba(15, 23, 42, 0.85);
            border-radius: 18px;
            padding: 1.5rem;
            box-shadow: 0 18px 48px rgba(15, 23, 42, 0.45);
            border: 1px solid rgba(148, 163, 184, 0.25);
            backdrop-filter: blur(6px);
        }
        .material-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #f8fafc;
            margin-bottom: 0.25rem;
        }
        .material-subtitle {
            color: #cbd5f5;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #f8fafc;
        }
        .stButton>button {
            border-radius: 999px;
            background: linear-gradient(135deg, #38bdf8, #6366f1);
            color: #0f172a;
            border: none;
            padding: 0.75rem 2.5rem;
            font-weight: 600;
            box-shadow: 0 18px 42px rgba(99, 102, 241, 0.45);
        }
        .stButton>button:hover {
            box-shadow: 0 22px 52px rgba(56, 189, 248, 0.55);
        }
        .stSelectbox label, .stSlider label, .stRadio label, .stToggle label {
            color: #e2e8f0 !important;
        }
        .stDataFrame, .stDataFrame [data-testid="stTable"] {
            color: #e2e8f0;
        }
        .stDataFrame thead tr th {
            background-color: rgba(15, 23, 42, 0.75);
        }
        .stDataFrame tbody tr td {
            background-color: rgba(30, 41, 59, 0.45);
        }
        .block-container {
            padding-top: 2rem;
        }
        .loading-overlay {
            position: fixed;
            inset: 0;
            background: rgba(15, 23, 42, 0.92);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            backdrop-filter: blur(12px);
        }
        .loading-card {
            width: min(420px, 90vw);
            padding: 2.5rem;
            border-radius: 24px;
            background: linear-gradient(
                145deg,
                rgba(30, 41, 59, 0.95),
                rgba(15, 23, 42, 0.95)
            );
            box-shadow: 0 24px 64px rgba(8, 47, 73, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.35);
            text-align: center;
        }
        .loading-card .loading-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: #f8fafc;
            margin-bottom: 0.5rem;
        }
        .loading-card .loading-subtitle {
            color: #cbd5f5;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }
        .loading-card .loading-status {
            color: #e2e8f0;
            font-size: 0.95rem;
            margin-bottom: 0.75rem;
        }
        .loading-overlay div[data-testid="stProgressBar"] > div {
            background-color: rgba(148, 163, 184, 0.2);
            border-radius: 999px;
        }
        .loading-overlay div[data-testid="stProgressBar"] div[role="progressbar"] {
            background: linear-gradient(135deg, #38bdf8, #6366f1);
            border-radius: 999px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Clustering Workbench")
st.caption(
    "Interactive workflow for preparing clustering inputs using the pre-loaded "
    "Dataiku dataset."
)

if "attribute_config" not in st.session_state:
    st.session_state.attribute_config = None
    st.session_state.attribute_signature = None


@dataclass
class RunConfig:
    dataset_name: str
    filters: Dict[str, str]
    column_mapping: Dict[str, str]
    part_number_col: str
    chosen_attributes: List[str]
    chosen_types: List[str]
    chosen_unit_flags: List[str]
    manual_configuration: bool
    eps_min: float | None
    eps_max: float | None
    eps_step: float | None
    min_samples: int | None
    numeric_weight: float
    metric: str
    manual_eps: float
    min_fill_ratio: int


@dataclass
class RunResult:
    config: RunConfig
    filtered_df: pd.DataFrame
    result_df: pd.DataFrame
    grouped_df: pd.DataFrame
    cluster_summary: pd.DataFrame
    candidate_df: pd.DataFrame
    eps_selected: float
    roster_parts: Dict[int, List[str]]
    metric: str
    part_number_col: str
    vectors: np.ndarray
    cat_vectors: np.ndarray | None


class LoadingScreen:
    def __init__(self, title: str, subtitle: str | None = None):
        self.container = st.empty()
        with self.container.container():
            st.markdown(
                "<div class='loading-overlay'><div class='loading-card'>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='loading-title'>{title}</div>",
                unsafe_allow_html=True,
            )
            if subtitle:
                st.markdown(
                    f"<div class='loading-subtitle'>{subtitle}</div>",
                    unsafe_allow_html=True,
                )
            self.status_placeholder = st.empty()
            self.progress_bar = st.progress(0)
            st.markdown("</div></div>", unsafe_allow_html=True)

    def update(self, message: str, percent_complete: int) -> None:
        percent_complete = max(0, min(100, percent_complete))
        self.status_placeholder.markdown(
            f"<div class='loading-status'>{message}</div>",
            unsafe_allow_html=True,
        )
        self.progress_bar.progress(percent_complete)

    def finalize(self, message: str | None = None) -> None:
        if message:
            self.status_placeholder.markdown(
                f"<div class='loading-status'>{message}</div>",
                unsafe_allow_html=True,
            )
        self.progress_bar.progress(100)

    def clear(self) -> None:
        self.container.empty()


def initialize_app_state() -> None:
    if "screen" not in st.session_state:
        st.session_state.screen = "setup"
    if "pending_run" not in st.session_state:
        st.session_state.pending_run = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "run_history" not in st.session_state:
        st.session_state.run_history = []
    if "selected_cluster" not in st.session_state:
        st.session_state.selected_cluster = None
    if "last_screen" not in st.session_state:
        st.session_state.last_screen = "setup"


def reset_app_state() -> None:
    st.session_state.screen = "setup"
    st.session_state.last_screen = "setup"
    for key in [
        "pending_run",
        "last_result",
        "selected_cluster",
        "attribute_config",
        "attribute_signature",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def navigate_to(screen: str) -> None:
    st.session_state.screen = screen
    if screen != "loading":
        st.session_state.last_screen = screen


def available_navigation_options() -> List[str]:
    options = ["setup"]
    if st.session_state.get("last_result") is not None:
        options.append("results")
        options.append("cluster_detail")
    if st.session_state.get("run_history"):
        options.append("history")
    return options


SCREEN_TITLES = {
    "setup": "Setup",
    "results": "Results",
    "cluster_detail": "Cluster Detail",
    "history": "History",
}


def apply_filters(
    df: pd.DataFrame, column_mapping: Dict[str, str], filters: Dict[str, str]
) -> pd.DataFrame:
    filtered = df.copy()
    for label, value in filters.items():
        column_name = column_mapping.get(label)
        if column_name is None:
            continue
        if value != "All":
            filtered = filtered[filtered[column_name] == value]
    return filtered


def store_run_history(result: RunResult) -> None:
    history_entry = {
        "id": str(uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "config": copy.deepcopy(result.config),
        "summary": {
            "clusters": int(result.cluster_summary.shape[0]),
            "eps": float(result.eps_selected),
            "metric": result.metric,
            "dataset": result.config.dataset_name,
            "filters": result.config.filters,
            "attributes": result.config.chosen_attributes,
        },
    }

    st.session_state.run_history.insert(0, history_entry)
    st.session_state.run_history = st.session_state.run_history[:10]


def trigger_clustering_run(config: RunConfig) -> None:
    st.session_state.pending_run = config
    navigate_to("loading")
    st.rerun()


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


def build_attribute_config(
    df: pd.DataFrame,
    excluded_columns: List[str],
) -> pd.DataFrame:
    attributes = [col for col in df.columns if col not in excluded_columns]
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
    excluded_columns: List[str],
    signature: Tuple,
) -> None:
    if (
        st.session_state.attribute_config is None
        or st.session_state.attribute_signature != signature
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
    merged = merged.sort_values("Fill Ratio", ascending=False).reset_index(drop=True)
    st.session_state.attribute_config = merged
    st.session_state.attribute_signature = signature


def perform_clustering(
    consolidated_df: pd.DataFrame, config: RunConfig, loader: LoadingScreen
) -> RunResult:
    filtered_df = apply_filters(
        consolidated_df, config.column_mapping, config.filters
    )

    if filtered_df.empty:
        raise ValueError("No rows remain after applying the selected filters.")

    loader.update("Encoding features for clustering…", 10)
    vectors, _, cat_vectors = encode_features(
        filtered_df,
        config.chosen_attributes,
        config.chosen_types,
        config.chosen_unit_flags,
        config.numeric_weight,
    )

    metric = config.metric
    loader.update("Validating metric selection…", 25)
    if metric == "jaccard" and (cat_vectors is None or cat_vectors.size == 0):
        st.warning(
            "Jaccard metric requires at least one categorical attribute. Falling back to Euclidean."
        )
        metric = "euclidean"

    loader.update("Preparing epsilon sweep…", 40)
    if config.manual_configuration:
        if None in (config.eps_min, config.eps_max, config.eps_step):
            raise ValueError("Manual configuration requires eps min, max, and step values.")
        if config.eps_min >= config.eps_max:
            raise ValueError("Eps maximum must be greater than eps minimum.")
        eps_values = np.arange(
            config.eps_min,
            config.eps_max + (config.eps_step or 0) / 2,
            config.eps_step,
        )
        min_samples = config.min_samples if config.min_samples is not None else 2
    else:
        feature_matrix = (
            cat_vectors if metric == "jaccard" and cat_vectors is not None else vectors
        )

        if feature_matrix.size == 0:
            eps_values = np.arange(0.1, 1.01, 0.05)
        else:
            sample_size = min(500, feature_matrix.shape[0])
            if sample_size >= 2:
                sample = feature_matrix[:sample_size]
                from sklearn.metrics import pairwise_distances

                distances = pairwise_distances(sample, sample, metric=metric)
                upper = distances[np.triu_indices_from(distances, k=1)]
                upper = upper[upper > 0]
                if upper.size == 0:
                    base_min, base_max = 0.05, 0.5
                else:
                    base_min = float(np.percentile(upper, 10))
                    base_max = float(np.percentile(upper, 90))
                    if base_min == base_max:
                        base_max = base_min + 0.05
                eps_step = max((base_max - base_min) / 20, 0.01)
                eps_values = np.arange(
                    max(0.01, base_min), base_max + eps_step / 2, eps_step
                )
            else:
                eps_values = np.arange(0.1, 1.01, 0.05)

        if eps_values.size == 0:
            eps_values = np.arange(0.1, 1.01, 0.05)

        if config.min_samples is not None:
            min_samples = config.min_samples
        else:
            min_samples = max(
                2,
                min(
                    10,
                    int(np.ceil(np.log10(max(len(filtered_df), 1))) + 1),
                ),
            )

    if len(eps_values) == 0:
        raise ValueError("Eps sweep produced no values. Adjust the range and try again.")

    loader.update("Evaluating clustering candidates…", 60)
    candidate_df = cluster_and_score(
        filtered_df,
        vectors,
        eps_values,
        metric,
        cat_vectors=cat_vectors,
        min_samples=min_samples,
    )

    if candidate_df.empty:
        raise ValueError("No valid clusters found with the current configuration.")

    loader.update("Selecting optimal parameters…", 80)
    best_eps = recommend_eps(candidate_df)
    eps_selected = config.manual_eps if config.manual_eps > 0 else best_eps

    loader.update("Building final clusters…", 90)
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=eps_selected, min_samples=min_samples, metric=metric)
    if metric == "jaccard" and cat_vectors is not None:
        labels = db.fit_predict(cat_vectors)
    else:
        labels = db.fit_predict(vectors)

    result_df = filtered_df.copy()
    result_df["cluster"] = labels
    result_df = add_likeness_score(result_df, vectors, labels, metric, cat_vectors)
    result_df = enforce_min_cluster_size(result_df, min_size=min_samples)

    grouped_df = (
        result_df.groupby(config.part_number_col)[
            config.chosen_attributes + ["cluster", "likeness_score"]
        ]
        .first()
        .reset_index()
    )

    cluster_summary = (
        result_df[result_df["cluster"] != -1]
        .groupby("cluster")
        .agg(
            cluster_size=("cluster", "count"),
            mean_likeness=("likeness_score", "mean"),
        )
        .reset_index()
    )
    cluster_summary["mean_likeness"] = cluster_summary["mean_likeness"].round(3)
    cluster_summary["cluster_size"] = cluster_summary["cluster_size"].astype(int)
    cluster_summary = cluster_summary.sort_values("cluster_size", ascending=False)

    roster_parts: Dict[int, List[str]] = {}
    for cluster_id in cluster_summary["cluster"].tolist():
        cluster_parts = (
            result_df[result_df["cluster"] == cluster_id]
            .nlargest(3, "likeness_score")
            .get(config.part_number_col)
            .astype(str)
            .tolist()
        )
        roster_parts[cluster_id] = cluster_parts

    loader.update("Finalizing outputs…", 95)

    return RunResult(
        config=config,
        filtered_df=filtered_df,
        result_df=result_df,
        grouped_df=grouped_df,
        cluster_summary=cluster_summary,
        candidate_df=candidate_df,
        eps_selected=float(eps_selected),
        roster_parts=roster_parts,
        metric=metric,
        part_number_col=config.part_number_col,
        vectors=vectors,
        cat_vectors=cat_vectors,
    )


def execute_pending_run(consolidated_df: pd.DataFrame) -> None:
    config: RunConfig | None = st.session_state.get("pending_run")
    if config is None:
        navigate_to("setup")
        st.rerun()

    loader = LoadingScreen(
        "Running clustering",
        "Please wait while we evaluate clustering candidates.",
    )

    try:
        result = perform_clustering(consolidated_df, config, loader)
    except ValueError as exc:
        loader.finalize("Unable to complete clustering.")
        loader.clear()
        st.error(str(exc))
        st.session_state.pending_run = None
        navigate_to("setup")
        st.rerun()

    loader.finalize("Clustering completed.")
    st.session_state.last_result = result
    st.session_state.pending_run = None
    st.session_state.selected_cluster = None
    store_run_history(result)
    navigate_to("results")
    st.rerun()


def render_setup_screen(
    consolidated_df: pd.DataFrame, dataset_name: str, dataset_options: List[str]
) -> None:
    try:
        commodity_col = resolve_column(consolidated_df, "Commodity", ["Commodity"])
        subcommodity_col = resolve_column(
            consolidated_df, "Sub-Commodity", ["Sub-Commodity", "Sub Commodity"]
        )
        detail_col = resolve_column(
            consolidated_df,
            "Detailed Commodity",
            ["Detailed Commodity", "Detailed commodity", "Detailed_Commodity"],
        )
        part_number_col = resolve_column(
            consolidated_df,
            "Part Number",
            [
                "Part Number",
                "PartNumber",
                "Part_Number",
                "Part No",
                "Part_No",
            ],
        )
    except KeyError as exc:
        st.error(str(exc))
        return

    column_mapping = {
        "Commodity": commodity_col,
        "Sub-Commodity": subcommodity_col,
        "Detailed Commodity": detail_col,
    }

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown("<div class='material-header'>1. Filter dataset</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-subtitle'>Refine the dataset using the commodity hierarchy before clustering.</div>",
        unsafe_allow_html=True,
    )

    filtered_df = consolidated_df.copy()
    selection_values: Dict[str, str] = {}

    top_filter_cols = st.columns(2)
    with top_filter_cols[0]:
        dataset_index = (
            dataset_options.index(dataset_name)
            if dataset_name in dataset_options
            else 0
        )
        chosen_dataset = st.selectbox(
            "Dataset",
            options=dataset_options,
            index=dataset_index,
            key="dataset_selector",
        )
    if chosen_dataset != dataset_name:
        st.session_state.active_dataset = chosen_dataset
        reset_app_state()
        st.rerun()

    st.caption(f"Using `{dataset_name}` from the StreamLit flow zone.")

    commodity_options = ["All"] + sorted(
        filtered_df[commodity_col].dropna().unique().tolist()
    )
    with top_filter_cols[1]:
        selection_values["Commodity"] = st.selectbox(
            "Commodity",
            options=commodity_options,
            index=0,
            key="filter_Commodity",
        )
    if selection_values["Commodity"] != "All":
        filtered_df = filtered_df[filtered_df[commodity_col] == selection_values["Commodity"]]

    sub_filter_cols = st.columns(2)
    subcommodity_options = ["All"] + sorted(
        filtered_df[subcommodity_col].dropna().unique().tolist()
    )
    with sub_filter_cols[0]:
        selection_values["Sub-Commodity"] = st.selectbox(
            "Sub-Commodity",
            options=subcommodity_options,
            index=0,
            key="filter_Sub-Commodity",
        )
    if selection_values["Sub-Commodity"] != "All":
        filtered_df = filtered_df[
            filtered_df[subcommodity_col] == selection_values["Sub-Commodity"]
        ]

    detail_options = ["All"] + sorted(
        filtered_df[detail_col].dropna().unique().tolist()
    )
    with sub_filter_cols[1]:
        selection_values["Detailed Commodity"] = st.selectbox(
            "Detailed Commodity",
            options=detail_options,
            index=0,
            key="filter_Detailed Commodity",
        )
    if selection_values["Detailed Commodity"] != "All":
        filtered_df = filtered_df[
            filtered_df[detail_col] == selection_values["Detailed Commodity"]
        ]

    st.metric("Rows after filtering", len(filtered_df))
    st.markdown("</div>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No rows remain after applying the selected filters.")
        return

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>2. Choose clustering attributes</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='material-subtitle'>Select the features to feed the clustering model. Attributes without data can be hidden from the table.</div>",
        unsafe_allow_html=True,
    )

    st.caption(f"Using `{part_number_col}` as the unique part identifier.")

    excluded_columns = [commodity_col, subcommodity_col, detail_col, part_number_col]

    hide_empty_attributes = st.toggle(
        "Hide attributes with no values",
        value=True,
        help="Only keep columns with data in the table.",
    )

    signature = (
        dataset_name,
        tuple(selection_values.items()),
        part_number_col,
        hide_empty_attributes,
        filtered_df.shape,
    )

    update_attribute_session_state(filtered_df, excluded_columns, signature)

    attribute_config = st.session_state.attribute_config
    if hide_empty_attributes and not attribute_config.empty:
        attribute_config = attribute_config[attribute_config["Fill Ratio"] > 0]

    min_fill_ratio = st.slider(
        "Minimum fill ratio required for clustering",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
    )

    attribute_editor = st.data_editor(
        attribute_config,
        key="attribute_editor",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Include": st.column_config.CheckboxColumn(
                "Include",
                help="Select to send attribute to the clustering model.",
            ),
            "Type": st.column_config.SelectboxColumn(
                "Type",
                options=["Numerical", "Categorical", "Alpha Numeric", "Text"],
            ),
            "Unit Extraction": st.column_config.SelectboxColumn(
                "Unit Extraction",
                options=["Yes", "No"],
                help="If Yes, the app will extract the first numeric value inside the field.",
            ),
            "Fill Ratio": st.column_config.ProgressColumn(
                "Fill Ratio",
                help="Share of non-null values in the filtered data",
                min_value=0.0,
                max_value=100.0,
                format="{:.0f}%",
            ),
        },
        disabled=["Attribute", "Fill Ratio"],
    )

    st.session_state.attribute_config = attribute_editor

    selected_attributes_df = attribute_editor[
        (attribute_editor["Include"]) & (attribute_editor["Fill Ratio"] >= min_fill_ratio)
    ]

    st.markdown("</div>", unsafe_allow_html=True)

    if selected_attributes_df.empty:
        st.info("Select at least one attribute with sufficient fill ratio to continue.")
        return

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown("<div class='material-header'>3. Configure clustering</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-subtitle'>Fine-tune DBSCAN parameters. The app will suggest a metric and epsilon value automatically.</div>",
        unsafe_allow_html=True,
    )

    chosen_attributes = selected_attributes_df["Attribute"].tolist()
    chosen_types = selected_attributes_df["Type"].tolist()
    chosen_unit_flags = selected_attributes_df["Unit Extraction"].tolist()

    recommended_metric = recommend_dbscan_metric(chosen_types)
    available_metrics = ["euclidean", "cosine", "jaccard"]

    manual_configuration = st.toggle(
        "Configure clustering parameters manually",
        value=False,
        help="Enable to fine-tune eps sweep, minimum samples, and weighting.",
    )

    eps_min: float | None
    eps_max: float | None
    eps_step: float | None
    min_samples_value: int | None

    if manual_configuration:
        parameter_cols = st.columns(4)
        eps_min = parameter_cols[0].number_input(
            "Eps minimum", min_value=0.01, value=0.10, step=0.01
        )
        eps_max = parameter_cols[1].number_input(
            "Eps maximum", min_value=eps_min + 0.01, value=1.00, step=0.01
        )
        eps_step = parameter_cols[2].number_input(
            "Eps step", min_value=0.01, value=0.05, step=0.01
        )
        min_samples_value = int(
            parameter_cols[3].number_input("Min samples", min_value=2, value=2, step=1)
        )
        numeric_weight = st.slider(
            "Weight applied to numeric attributes",
            min_value=1.0,
            max_value=25.0,
            value=10.0,
            step=1.0,
        )
        metric = st.selectbox(
            "Distance metric",
            options=available_metrics,
            index=available_metrics.index(recommended_metric)
            if recommended_metric in available_metrics
            else 0,
            help="Override the automatically suggested metric if desired.",
        )
        manual_eps = st.number_input(
            "Manual eps override (leave 0 for automatic recommendation)",
            min_value=0.0,
            value=0.0,
            step=0.01,
        )
    else:
        eps_min = eps_max = eps_step = None
        min_samples_value = None
        numeric_weight = 10.0
        metric = recommended_metric if recommended_metric in available_metrics else "euclidean"
        manual_eps = 0.0
        st.caption(
            "Automatic mode enabled: the app will infer clustering parameters from the data."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    run_button_columns = st.columns([3, 1, 3])
    with run_button_columns[1]:
        run_requested = st.button(
            "Run clustering",
            type="primary",
            use_container_width=False,
        )

    if not run_requested:
        return

    config = RunConfig(
        dataset_name=dataset_name,
        filters=selection_values.copy(),
        column_mapping=column_mapping,
        part_number_col=part_number_col,
        chosen_attributes=chosen_attributes,
        chosen_types=chosen_types,
        chosen_unit_flags=chosen_unit_flags,
        manual_configuration=manual_configuration,
        eps_min=eps_min,
        eps_max=eps_max,
        eps_step=eps_step,
        min_samples=min_samples_value,
        numeric_weight=numeric_weight,
        metric=metric,
        manual_eps=manual_eps,
        min_fill_ratio=min_fill_ratio,
    )

    trigger_clustering_run(config)


def render_results_screen() -> None:
    result: RunResult | None = st.session_state.get("last_result")
    if result is None:
        st.info("Run clustering from the setup screen to see results.")
        return

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>Clustering summary</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Dataset: {result.config.dataset_name}")

    metric_columns = st.columns(3)
    with metric_columns[0]:
        st.metric("Recommended eps", f"{result.eps_selected:.3f}")
    with metric_columns[1]:
        st.metric("Clusters identified", int(result.cluster_summary.shape[0]))
    with metric_columns[2]:
        st.metric("Rows analysed", len(result.filtered_df))

    st.markdown("</div>", unsafe_allow_html=True)

    if result.cluster_summary.empty:
        st.warning("No clusters were identified. Adjust the configuration and try again.")
    else:
        st.markdown("<div class='material-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='material-header'>Cluster overview</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='material-subtitle'>Select a cluster card to explore its details.</div>",
            unsafe_allow_html=True,
        )

        clusters = result.cluster_summary.reset_index(drop=True)
        cards_per_row = 3
        for start in range(0, len(clusters), cards_per_row):
            row = clusters.iloc[start : start + cards_per_row]
            card_columns = st.columns(len(row))
            for column, (_, cluster_row) in zip(card_columns, row.iterrows()):
                cluster_id = int(cluster_row["cluster"])
                cluster_label = cluster_id + 1
                with column:
                    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
                    st.markdown(
                        f"<div class='material-header'>Cluster {cluster_label}</div>",
                        unsafe_allow_html=True,
                    )
                    st.metric("Size", int(cluster_row["cluster_size"]))
                    st.metric("Mean likeness", f"{cluster_row['mean_likeness']:.2f}")
                    representatives = result.roster_parts.get(cluster_id, [])
                    if representatives:
                        st.caption(
                            "Representative parts: " + ", ".join(representatives)
                        )
                    if st.button(
                        "Open details",
                        key=f"cluster_card_{cluster_id}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_cluster = cluster_id
                        navigate_to("cluster_detail")
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='material-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='material-header'>Cluster metrics</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            result.cluster_summary,
            use_container_width=True,
            column_config={
                "cluster": st.column_config.NumberColumn("Cluster"),
                "cluster_size": st.column_config.NumberColumn("Size", format="%d"),
                "mean_likeness": st.column_config.NumberColumn(
                    "Mean likeness", format="%.3f"
                ),
            },
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>Candidate evaluation</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(result.candidate_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    excel_bytes = build_excel_workbook(result.result_df, result.grouped_df)
    pdf_bytes = export_cluster_report_pdf(
        result.result_df,
        candidate_eps_table=result.candidate_df,
        metric=result.metric,
        eps_selected=result.eps_selected,
        attributes=result.config.chosen_attributes,
        types=result.config.chosen_types,
        output=None,
        top_n=10,
    )

    download_columns = st.columns(2)
    with download_columns[0]:
        st.download_button(
            "Download Excel output",
            data=excel_bytes,
            file_name="Clustered_Output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with download_columns[1]:
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name="Cluster_Report.pdf",
            mime="application/pdf",
        )


def render_cluster_detail_screen() -> None:
    result: RunResult | None = st.session_state.get("last_result")
    if result is None:
        st.info("Results will appear after running clustering from the setup screen.")
        return

    cluster_id = st.session_state.get("selected_cluster")
    if cluster_id is None:
        st.info("Select a cluster from the results screen to view its details.")
        return

    cluster_summary = result.cluster_summary
    if cluster_summary.empty or cluster_id not in cluster_summary["cluster"].values:
        st.warning("The selected cluster is no longer available. Rerun clustering to refresh.")
        return

    st.button(
        "← Back to results",
        on_click=lambda: navigate_to("results"),
        type="secondary",
    )

    summary_row = cluster_summary[cluster_summary["cluster"] == cluster_id].iloc[0]
    cluster_label = int(cluster_id) + 1

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='material-header'>Cluster {cluster_label} overview</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Dataset: {result.config.dataset_name}")

    overview_columns = st.columns(3)
    with overview_columns[0]:
        st.metric("Parts", int(summary_row["cluster_size"]))
    with overview_columns[1]:
        st.metric("Mean likeness", f"{summary_row['mean_likeness']:.2f}")
    with overview_columns[2]:
        st.metric("Metric", result.metric)

    representatives = result.roster_parts.get(int(cluster_id), [])
    if representatives:
        st.caption("Representative parts: " + ", ".join(representatives))

    st.markdown("</div>", unsafe_allow_html=True)

    cluster_parts = result.grouped_df[result.grouped_df["cluster"] == cluster_id]
    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>Cluster members</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(cluster_parts, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    feature_matrix = (
        result.cat_vectors
        if result.metric == "jaccard" and result.cat_vectors is not None
        else result.vectors
    )
    cluster_mask = result.result_df["cluster"].to_numpy() == cluster_id
    cluster_vectors = feature_matrix[cluster_mask]
    cluster_parts_labels = (
        result.result_df.loc[cluster_mask, result.part_number_col].astype(str)
    )

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>Similarity diagnostics</div>",
        unsafe_allow_html=True,
    )

    if cluster_vectors.shape[0] > 1:
        from sklearn.metrics import pairwise_distances

        distances = pairwise_distances(
            cluster_vectors,
            metric=result.metric
            if result.metric != "jaccard" or result.cat_vectors is not None
            else "euclidean",
        )
        similarity = 1.0 / (1.0 + distances)
        heatmap_df = pd.DataFrame(
            similarity,
            index=cluster_parts_labels,
            columns=cluster_parts_labels,
        ).reset_index(names=result.part_number_col)
        heatmap_long = heatmap_df.melt(
            id_vars=result.part_number_col,
            var_name="Peer",
            value_name="Similarity",
        )
        heatmap_chart = (
            alt.Chart(heatmap_long)
            .mark_rect()
            .encode(
                x=alt.X("Peer:N", title="Peer part"),
                y=alt.Y(f"{result.part_number_col}:N", title="Part"),
                color=alt.Color(
                    "Similarity:Q",
                    scale=alt.Scale(scheme="blues"),
                    legend=alt.Legend(title="Similarity"),
                ),
                tooltip=[
                    alt.Tooltip(f"{result.part_number_col}:N", title="Part"),
                    alt.Tooltip("Peer:N", title="Peer"),
                    alt.Tooltip("Similarity:Q", format=".3f"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(heatmap_chart, use_container_width=True)
    else:
        st.info("Not enough parts in the selected cluster to compute a similarity heatmap.")

    likeness_distribution = result.result_df[result.result_df["cluster"] == cluster_id][
        "likeness_score"
    ]
    st.bar_chart(likeness_distribution.value_counts().sort_index())

    st.markdown("</div>", unsafe_allow_html=True)


def render_history_screen() -> None:
    history: List[Dict[str, object]] = st.session_state.get("run_history", [])
    if not history:
        st.info("Previous runs will appear here once clustering has been executed.")
        return

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>Previous clustering runs</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='material-subtitle'>Select a saved configuration to execute the clustering again.</div>",
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


def render_by_commodity(
    dataset_options: List[str], dataset_name: str, consolidated_df: pd.DataFrame
) -> None:
    initialize_app_state()

    if st.session_state.screen == "loading":
        execute_pending_run(consolidated_df)
        return

    options = available_navigation_options()
    labels = [SCREEN_TITLES[option] for option in options]
    label_to_screen = dict(zip(labels, options))

    default_screen = (
        st.session_state.screen
        if st.session_state.screen in options
        else st.session_state.last_screen
    )
    default_label = SCREEN_TITLES.get(default_screen, SCREEN_TITLES[options[0]])

    selected_label = st.sidebar.radio(
        "Navigate",
        options=labels,
        index=labels.index(default_label),
    )
    selected_screen = label_to_screen[selected_label]

    if selected_screen != st.session_state.screen:
        navigate_to(selected_screen)
    else:
        st.session_state.last_screen = st.session_state.screen

    if st.session_state.screen == "setup":
        render_setup_screen(consolidated_df, dataset_name, dataset_options)
    elif st.session_state.screen == "results":
        render_results_screen()
    elif st.session_state.screen == "cluster_detail":
        render_cluster_detail_screen()
    elif st.session_state.screen == "history":
        render_history_screen()


def render_by_part(_: pd.DataFrame) -> None:
    st.info("The By Part configuration will be available soon.")


def main() -> None:
    try:
        dataset_options = list_streamlit_datasets()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    if not dataset_options:
        st.error("No datasets are available in the StreamLit flow zone.")
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
        render_by_commodity(dataset_options, active_dataset, consolidated_df)
    else:
        render_by_part(consolidated_df)


if __name__ == "__main__":
    main()
