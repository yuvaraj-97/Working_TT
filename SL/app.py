from __future__ import annotations

from io import BytesIO
from typing import Dict, List

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
from .data_access import load_consolidated_sheet
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
        .stApp {
            background-color: #ffffff;
        }
        .material-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        .material-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }
        .material-subtitle {
            color: #475569;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
        }
        .stButton>button {
            border-radius: 999px;
            background: linear-gradient(135deg, #2563eb, #3b82f6);
            color: white;
            border: none;
            padding: 0.75rem 2.5rem;
            font-weight: 600;
            box-shadow: 0 12px 30px rgba(59, 130, 246, 0.35);
        }
        .stButton>button:hover {
            box-shadow: 0 16px 35px rgba(37, 99, 235, 0.45);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("CLSING Clustering Workbench")
st.caption(
    "Interactive workflow for preparing CLSING clustering inputs directly from the "
    "consolidated dataset."
)

if "attribute_config" not in st.session_state:
    st.session_state.attribute_config = None
    st.session_state.attribute_signature = None

uploaded_file = st.file_uploader(
    "Upload the consolidated Excel workbook", type=["xlsx", "xlsm", "xls"]
)

def build_attribute_config(df: pd.DataFrame, excluded_columns: List[str]) -> pd.DataFrame:
    attributes = [col for col in df.columns if col not in excluded_columns]
    fill_ratio = df[attributes].notna().mean().round(3) if attributes else pd.Series(dtype=float)

    config = pd.DataFrame(
        {
            "Attribute": attributes,
            "Include": False,
            "Type": [guess_attribute_type(df[attr]) for attr in attributes],
            "Unit Extraction": ["No"] * len(attributes),
            "Fill Ratio": fill_ratio.reindex(attributes).fillna(0.0).values,
        }
    )
    return config

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    workbook = BytesIO(file_bytes)

    try:
        consolidated_df = load_consolidated_sheet(workbook)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    signature = (uploaded_file.name, len(file_bytes), consolidated_df.shape)

    string_columns = (
        consolidated_df.select_dtypes(include=["object", "string", "category"]).columns
    )
    candidate_columns = consolidated_df.columns.tolist()

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown("<div class='material-header'>1. Filter dataset</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-subtitle'>Choose the hierarchy fields to replicate the "
        "Summary/Filtered tabs and narrow down the population you want to cluster."
        "</div>",
        unsafe_allow_html=True,
    )

    filter_col_container = st.columns(3)
    default_map: Dict[str, str | None] = {
        "Commodity": None,
        "Sub Commodity": None,
        "Detailed Commodity": None,
    }
    for label in default_map:
        if label in string_columns:
            default_map[label] = label

    commodity_col = filter_col_container[0].selectbox(
        "Commodity column",
        options=["(None)"] + list(string_columns),
        index=(
            (["(None)"] + list(string_columns)).index(default_map["Commodity"])
            if default_map["Commodity"]
            else 0
        ),
    )
    subcommodity_col = filter_col_container[1].selectbox(
        "Sub-commodity column",
        options=["(None)"] + list(string_columns),
        index=(
            (["(None)"] + list(string_columns)).index(default_map["Sub Commodity"])
            if default_map["Sub Commodity"]
            else 0
        ),
    )
    detail_col = filter_col_container[2].selectbox(
        "Detailed commodity column",
        options=["(None)"] + list(string_columns),
        index=(
            (["(None)"] + list(string_columns)).index(default_map["Detailed Commodity"])
            if default_map["Detailed Commodity"]
            else 0
        ),
    )

    filtered_df = consolidated_df.copy()
    def resolve_selection(column_name: str, label: str, df_in: pd.DataFrame) -> pd.DataFrame:
        if not column_name or column_name == "(None)":
            return df_in
        options = ["All"] + sorted(df_in[column_name].dropna().unique().tolist())
        selected_value = st.selectbox(
            f"{label} value", options=options, index=0, key=f"filter_{label}"
        )
        if selected_value != "All":
            return df_in[df_in[column_name] == selected_value]
        return df_in

    filtered_df = resolve_selection(commodity_col, "Commodity", filtered_df)
    filtered_df = resolve_selection(subcommodity_col, "Sub-commodity", filtered_df)
    filtered_df = resolve_selection(detail_col, "Detailed commodity", filtered_df)

    st.metric("Rows after filtering", len(filtered_df))
    st.markdown("</div>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No rows remain after applying the selected filters.")
        st.stop()

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown("<div class='material-header'>2. Choose clustering attributes</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-subtitle'>Toggle the features that should feed the "
        "clustering model and adjust the detected metadata.</div>",
        unsafe_allow_html=True,
    )

    part_number_col = st.selectbox(
        "Column representing the unique part identifier",
        options=candidate_columns,
        index=candidate_columns.index("Part Number") if "Part Number" in candidate_columns else 0,
    )

    excluded_columns = [col for col in [commodity_col, subcommodity_col, detail_col] if col and col != "(None)"]
    excluded_columns.append(part_number_col)

    if (
        st.session_state.attribute_config is None
        or st.session_state.attribute_signature != signature
    ):
        st.session_state.attribute_config = build_attribute_config(filtered_df, excluded_columns)
        st.session_state.attribute_signature = signature
    else:
        current_config = st.session_state.attribute_config
        valid_attributes = [col for col in filtered_df.columns if col in current_config["Attribute"].tolist()]
        refreshed = build_attribute_config(filtered_df, excluded_columns)
        merged = current_config.merge(
            refreshed[["Attribute", "Fill Ratio"]], on="Attribute", how="left", suffixes=("", "_new")
        )
        merged["Fill Ratio"] = merged["Fill Ratio_new"].fillna(merged["Fill Ratio"]).round(3)
        merged = merged.drop(columns=["Fill Ratio_new"])
        merged = merged[merged["Attribute"].isin(valid_attributes)]
        st.session_state.attribute_config = merged

    min_fill_ratio = st.slider(
        "Minimum fill ratio", min_value=0.0, max_value=1.0, value=0.7, step=0.05
    )

    attribute_editor = st.data_editor(
        st.session_state.attribute_config,
        key="attribute_editor",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Include": st.column_config.CheckboxColumn(
                "Include", help="Select to send attribute to the clustering model."
            ),
            "Type": st.column_config.SelectboxColumn(
                "Type",
                options=["Numerical", "Categorical", "Alpha Numeric", "Text"],
            ),
            "Unit Extraction": st.column_config.SelectboxColumn(
                "Unit Extraction", options=["Yes", "No"],
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

    if selected_attributes_df.empty:
        st.info("Select at least one attribute with sufficient fill ratio to continue.")
        st.stop()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown("<div class='material-header'>3. Configure clustering</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-subtitle'>Fine-tune DBSCAN parameters. The app will "
        "suggest a metric and epsilon value automatically.</div>",
        unsafe_allow_html=True,
    )

    parameter_cols = st.columns(4)
    eps_min = parameter_cols[0].number_input("Eps minimum", min_value=0.01, value=0.10, step=0.01)
    eps_max = parameter_cols[1].number_input("Eps maximum", min_value=eps_min + 0.01, value=1.00, step=0.01)
    eps_step = parameter_cols[2].number_input("Eps step", min_value=0.01, value=0.05, step=0.01)
    min_samples = int(parameter_cols[3].number_input("Min samples", min_value=2, value=2, step=1))

    numeric_weight = st.slider(
        "Weight applied to numeric attributes", min_value=1.0, max_value=25.0, value=10.0, step=1.0
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

    if run_clustering:
        eps_values = np.arange(eps_min, eps_max + eps_step / 2, eps_step)
        if len(eps_values) == 0:
            st.error("Eps sweep produced no values. Adjust the range and try again.")
            st.stop()

        vectors, feature_types, cat_vectors = encode_features(
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
            st.stop()

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

        cluster_counts = result_df["cluster"].value_counts().rename_axis("cluster").reset_index(name="count")
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

else:
    st.info("Upload the consolidated workbook to begin.")
