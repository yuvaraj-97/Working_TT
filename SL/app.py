from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .clustering import (
    add_likeness_score,
    cluster_and_score,
    enforce_min_cluster_size,
    recommend_dbscan_metric,
    recommend_eps,
)
from .exporters import build_excel_workbook
from .feature_engineering import encode_features, guess_attribute_type
from .reporting import export_cluster_report_pdf

st.set_page_config(
    page_title="CLSING Clustering Workbench",
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("CLSING Clustering Workbench")
st.caption(
    "Interactive workflow for preparing CLSING clustering inputs using the "
    "pre-loaded Dataiku dataset."
)

if "attribute_config" not in st.session_state:
    st.session_state.attribute_config = None
    st.session_state.attribute_signature = None


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

    fill_ratio = df[attributes].notna().mean().round(3)

    config = pd.DataFrame(
        {
            "Attribute": attributes,
            "Type": [guess_attribute_type(df[attr]) for attr in attributes],
            "Unit Extraction": ["No"] * len(attributes),
            "Fill Ratio": fill_ratio.reindex(attributes).fillna(0.0).values,
            "Include": False,
        }
    )

    return config[["Attribute", "Type", "Unit Extraction", "Fill Ratio", "Include"]]


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
    merged["Fill Ratio"] = merged["Fill Ratio_new"].fillna(merged["Fill Ratio"]).round(3)
    merged = merged.drop(columns=["Fill Ratio_new"])
    merged = merged[["Attribute", "Type", "Unit Extraction", "Fill Ratio", "Include"]]
    st.session_state.attribute_config = merged
    st.session_state.attribute_signature = signature


def render_by_commodity(consolidated_df: pd.DataFrame) -> None:
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
    except KeyError as exc:
        st.error(str(exc))
        st.stop()

    candidate_columns = consolidated_df.columns.tolist()

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown("<div class='material-header'>1. Filter dataset</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-subtitle'>Refine the dataset using the commodity "
        "hierarchy before clustering.</div>",
        unsafe_allow_html=True,
    )

    filtered_df = consolidated_df.copy()
    selection_values: Dict[str, str] = {}
    for column, label in [
        (commodity_col, "Commodity"),
        (subcommodity_col, "Sub-Commodity"),
        (detail_col, "Detailed Commodity"),
    ]:
        options = ["All"] + sorted(filtered_df[column].dropna().unique().tolist())
        selected_value = st.selectbox(
            f"{label}", options=options, index=0, key=f"filter_{label}"
        )
        selection_values[label] = selected_value
        if selected_value != "All":
            filtered_df = filtered_df[filtered_df[column] == selected_value]

    st.metric("Rows after filtering", len(filtered_df))
    st.markdown("</div>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No rows remain after applying the selected filters.")
        st.stop()

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>2. Choose clustering attributes</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='material-subtitle'>Select the features to feed the clustering "
        "model. Attributes without data can be hidden from the table.</div>",
        unsafe_allow_html=True,
    )

    part_number_col = st.selectbox(
        "Column representing the unique part identifier",
        options=candidate_columns,
        index=candidate_columns.index("Part Number")
        if "Part Number" in candidate_columns
        else 0,
    )

    excluded_columns = [commodity_col, subcommodity_col, detail_col, part_number_col]

    hide_empty_attributes = st.toggle(
        "Hide attributes with no values",
        value=True,
        help="Only keep columns with data in the table.",
    )

    signature = (
        "Clustering",
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
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
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
                max_value=1.0,
                format="{:.0%}",
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
        "<div class='material-subtitle'>Fine-tune DBSCAN parameters. The app will "
        "suggest a metric and epsilon value automatically.</div>",
        unsafe_allow_html=True,
    )

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
    min_samples = int(
        parameter_cols[3].number_input("Min samples", min_value=2, value=2, step=1)
    )

    numeric_weight = st.slider(
        "Weight applied to numeric attributes",
        min_value=1.0,
        max_value=25.0,
        value=10.0,
        step=1.0,
    )

    chosen_attributes = selected_attributes_df["Attribute"].tolist()
    chosen_types = selected_attributes_df["Type"].tolist()
    chosen_unit_flags = selected_attributes_df["Unit Extraction"].tolist()

    recommended_metric = recommend_dbscan_metric(chosen_types)
    metric = st.selectbox(
        "Distance metric",
        options=["euclidean", "cosine", "jaccard"],
        index=["euclidean", "cosine", "jaccard"].index(recommended_metric)
        if recommended_metric in ["euclidean", "cosine", "jaccard"]
        else 0,
        help="Override the automatically suggested metric if desired.",
    )

    manual_eps = st.number_input(
        "Manual eps override (leave 0 for automatic recommendation)",
        min_value=0.0,
        value=0.0,
        step=0.01,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    run_clustering = st.button("Run clustering", type="primary")

    if not run_clustering:
        return

    eps_values = np.arange(eps_min, eps_max + eps_step / 2, eps_step)
    if len(eps_values) == 0:
        st.error("Eps sweep produced no values. Adjust the range and try again.")
        return

    vectors, _, cat_vectors = encode_features(
        filtered_df, chosen_attributes, chosen_types, chosen_unit_flags, numeric_weight
    )

    if metric == "jaccard" and (cat_vectors is None or cat_vectors.size == 0):
        st.warning(
            "Jaccard metric requires at least one categorical attribute. Falling back to Euclidean."
        )
        metric = "euclidean"

    candidate_df = cluster_and_score(
        filtered_df,
        vectors,
        eps_values,
        metric,
        cat_vectors=cat_vectors,
        min_samples=min_samples,
    )

    if candidate_df.empty:
        st.warning("No valid clusters found with the current configuration.")
        return

    best_eps = recommend_eps(candidate_df)
    eps_selected = manual_eps if manual_eps > 0 else best_eps

    st.success("Clustering completed.")
    st.metric("Recommended eps", f"{best_eps:.3f}")
    st.dataframe(candidate_df, use_container_width=True)

    from sklearn.cluster import DBSCAN  # local import to avoid circular dependency

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
        result_df.groupby(part_number_col)[chosen_attributes + ["cluster", "likeness_score"]]
        .first()
        .reset_index()
    )

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>Results overview</div>", unsafe_allow_html=True
    )
    st.dataframe(result_df.head(100), use_container_width=True)

    cluster_counts = (
        result_df["cluster"].value_counts().rename_axis("cluster").reset_index(name="count")
    )
    st.bar_chart(cluster_counts.set_index("cluster"))
    st.markdown("</div>", unsafe_allow_html=True)

    excel_bytes = build_excel_workbook(result_df, grouped_df)
    st.download_button(
        "Download Excel output",
        data=excel_bytes,
        file_name="Clustered_Output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    pdf_bytes = export_cluster_report_pdf(
        result_df,
        candidate_eps_table=candidate_df,
        metric=metric,
        eps_selected=eps_selected,
        attributes=chosen_attributes,
        types=chosen_types,
        output=None,
        top_n=10,
    )
    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name="Cluster_Report.pdf",
        mime="application/pdf",
    )


def render_by_part(_: pd.DataFrame) -> None:
    st.info("The By Part configuration will be available soon.")


def main() -> None:
    try:
        consolidated_df = load_clustering_dataset()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    view_mode = st.radio(
        "Configuration mode",
        options=["By Commodity", "By Part"],
        horizontal=True,
    )

    if view_mode == "By Commodity":
        render_by_commodity(consolidated_df)
    else:
        render_by_part(consolidated_df)


if __name__ == "__main__":
    main()
