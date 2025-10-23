from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

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

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown("<div class='material-header'>1. Filter dataset</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-subtitle'>Refine the dataset using the commodity "
        "hierarchy before clustering.</div>",
        unsafe_allow_html=True,
    )

    filtered_df = consolidated_df.copy()
    selection_values: Dict[str, str] = {}

    top_filter_cols = st.columns(2)
    commodity_options = ["All"] + sorted(
        filtered_df[commodity_col].dropna().unique().tolist()
    )
    with top_filter_cols[0]:
        commodity_value = st.selectbox(
            "Commodity",
            options=commodity_options,
            index=0,
            key="filter_Commodity",
        )
    selection_values["Commodity"] = commodity_value
    if commodity_value != "All":
        filtered_df = filtered_df[filtered_df[commodity_col] == commodity_value]

    subcommodity_options = ["All"] + sorted(
        filtered_df[subcommodity_col].dropna().unique().tolist()
    )
    with top_filter_cols[1]:
        subcommodity_value = st.selectbox(
            "Sub-Commodity",
            options=subcommodity_options,
            index=0,
            key="filter_Sub-Commodity",
        )
    selection_values["Sub-Commodity"] = subcommodity_value
    if subcommodity_value != "All":
        filtered_df = filtered_df[filtered_df[subcommodity_col] == subcommodity_value]

    bottom_filter_cols = st.columns(2)
    detail_options = ["All"] + sorted(
        filtered_df[detail_col].dropna().unique().tolist()
    )
    with bottom_filter_cols[0]:
        detail_value = st.selectbox(
            "Detailed Commodity",
            options=detail_options,
            index=0,
            key="filter_Detailed Commodity",
        )
    selection_values["Detailed Commodity"] = detail_value
    if detail_value != "All":
        filtered_df = filtered_df[filtered_df[detail_col] == detail_value]


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

    try:
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
        st.stop()

    st.caption(f"Using `{part_number_col}` as the unique part identifier.")

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
        "<div class='material-subtitle'>Fine-tune DBSCAN parameters. The app will "
        "suggest a metric and epsilon value automatically.</div>",
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
        eps_min = eps_max = eps_step = 0.0  # placeholders for automatic mode
        min_samples = None
        numeric_weight = 10.0
        metric = recommended_metric if recommended_metric in available_metrics else "euclidean"
        manual_eps = 0.0
        st.caption(
            "Automatic mode enabled: the app will infer clustering parameters from the data."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    run_clustering = st.button("Run clustering", type="primary")

    if not run_clustering:
        return

    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.write("Encoding features for clustering…")
    progress_bar.progress(10)

    vectors, _, cat_vectors = encode_features(
        filtered_df, chosen_attributes, chosen_types, chosen_unit_flags, numeric_weight
    )

    progress_text.write("Validating metric selection…")
    progress_bar.progress(25)
    if metric == "jaccard" and (cat_vectors is None or cat_vectors.size == 0):
        st.warning(
            "Jaccard metric requires at least one categorical attribute. Falling back to Euclidean."
        )
        metric = "euclidean"

    if manual_configuration:
        eps_values = np.arange(eps_min, eps_max + eps_step / 2, eps_step)
        min_samples = min_samples if min_samples is not None else 2
    else:
        progress_text.write("Estimating eps sweep automatically…")
        progress_bar.progress(40)
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

        if min_samples is None:
            min_samples = max(
                2,
                min(
                    10,
                    int(np.ceil(np.log10(max(len(filtered_df), 1))) + 1),
                ),
            )

    if len(eps_values) == 0:
        st.error("Eps sweep produced no values. Adjust the range and try again.")
        return

    progress_text.write("Evaluating clustering candidates…")
    progress_bar.progress(60)

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

    progress_text.write("Selecting optimal parameters…")
    progress_bar.progress(80)

    best_eps = recommend_eps(candidate_df)
    eps_selected = manual_eps if manual_eps > 0 else best_eps

    progress_text.write("Building final clusters…")
    progress_bar.progress(90)

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

    progress_bar.progress(100)
    progress_text.write("Clustering completed.")
    st.success("Clustering completed.")
    st.metric("Recommended eps", f"{best_eps:.3f}")
    st.dataframe(candidate_df, use_container_width=True)

    grouped_df = (
        result_df.groupby(part_number_col)[chosen_attributes + ["cluster", "likeness_score"]]
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
            .get(part_number_col)
            .astype(str)
            .tolist()
        )
        roster_parts[cluster_id] = cluster_parts

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>Results overview</div>", unsafe_allow_html=True
    )
    summary_tab, explorer_tab, diagnostics_tab = st.tabs(
        ["Summary", "Cluster Explorer", "Diagnostics"]
    )

    with summary_tab:
        if not cluster_summary.empty:
            top_summary = cluster_summary.head(3)
            metric_columns = st.columns(len(top_summary))
            for idx, (_, row) in enumerate(top_summary.iterrows()):
                with metric_columns[idx]:
                    st.metric(
                        f"Cluster {int(row['cluster'])}",
                        f"{int(row['cluster_size'])} parts",
                        help="Top clusters ranked by size",
                        delta=f"Avg likeness {row['mean_likeness']:.2f}",
                    )

        scatter_source = cluster_summary.rename(
            columns={"cluster_size": "Cluster Size", "mean_likeness": "Mean Likeness"}
        )
        if not scatter_source.empty:
            scatter_source["Cluster"] = scatter_source["cluster"].astype(str)
            scatter_chart = (
                alt.Chart(scatter_source)
                .mark_circle(size=220)
                .encode(
                    x=alt.X("Cluster Size:Q", title="Cluster size"),
                    y=alt.Y("Mean Likeness:Q", title="Mean likeness"),
                    color=alt.Color(
                        "Mean Likeness:Q",
                        scale=alt.Scale(scheme="turbo"),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("Cluster", title="Cluster"),
                        alt.Tooltip("Cluster Size:Q", title="Size"),
                        alt.Tooltip("Mean Likeness:Q", title="Mean likeness", format=".2f"),
                    ],
                )
                .properties(height=360)
                .interactive()
            )
            st.altair_chart(scatter_chart, use_container_width=True)

        cluster_counts = (
            result_df["cluster"].value_counts().rename_axis("cluster").reset_index(name="count")
        )
        st.bar_chart(cluster_counts.set_index("cluster"))
        st.dataframe(
            cluster_summary,
            use_container_width=True,
            column_config={
                "cluster": st.column_config.NumberColumn(
                    "Cluster", help="Cluster label assigned by DBSCAN"
                ),
                "cluster_size": st.column_config.NumberColumn(
                    "Size", format="%d", help="Number of rows in the cluster"
                ),
                "mean_likeness": st.column_config.NumberColumn(
                    "Mean likeness", format="%.3f"
                ),
            },
        )

    with explorer_tab:
        if cluster_summary.empty:
            st.info("No clusters were identified for exploration.")
        else:
            cards_per_row = 3
            rows = [
                cluster_summary.iloc[i : i + cards_per_row]
                for i in range(0, len(cluster_summary), cards_per_row)
            ]
            for row in rows:
                card_columns = st.columns(len(row))
                for col, (_, cluster_row) in zip(card_columns, row.iterrows()):
                    cluster_id = int(cluster_row["cluster"])
                    with col:
                        st.markdown("<div class='material-card'>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='material-header'>Cluster {cluster_id}</div>",
                            unsafe_allow_html=True,
                        )
                        st.metric("Size", int(cluster_row["cluster_size"]))
                        st.metric("Mean likeness", f"{cluster_row['mean_likeness']:.2f}")
                        representatives = roster_parts.get(cluster_id, [])
                        if representatives:
                            st.caption(
                                "Representative parts: "
                                + ", ".join(representatives)
                            )
                        st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            selected_cluster = st.selectbox(
                "Inspect cluster", ["All"]
                + [str(int(c)) for c in cluster_summary["cluster"].tolist()],
            )
            if selected_cluster == "All":
                st.dataframe(grouped_df, use_container_width=True)
            else:
                cluster_id = int(selected_cluster)
                filtered_group = grouped_df[grouped_df["cluster"] == cluster_id]
                st.dataframe(filtered_group, use_container_width=True)

    with diagnostics_tab:
        cluster_options = [
            c for c in sorted(result_df["cluster"].unique()) if int(c) != -1
        ]
        if cluster_options:
            selected_heatmap_cluster = st.selectbox(
                "Cluster for similarity heatmap", cluster_options
            )
            heatmap_matrix = (
                cat_vectors
                if metric == "jaccard" and cat_vectors is not None
                else vectors
            )
            cluster_mask = result_df["cluster"].to_numpy() == selected_heatmap_cluster
            cluster_vectors = heatmap_matrix[cluster_mask]
            cluster_parts = result_df.loc[cluster_mask, part_number_col].astype(str)
            if cluster_vectors.shape[0] > 1:
                from sklearn.metrics import pairwise_distances

                distances = pairwise_distances(
                    cluster_vectors,
                    metric=metric if metric != "jaccard" or cat_vectors is not None else "euclidean",
                )
                similarity = 1.0 / (1.0 + distances)
                heatmap_df = pd.DataFrame(
                    similarity,
                    index=cluster_parts,
                    columns=cluster_parts,
                ).reset_index(names=part_number_col)
                heatmap_long = heatmap_df.melt(
                    id_vars=part_number_col,
                    var_name="Peer",
                    value_name="Similarity",
                )
                heatmap_chart = (
                    alt.Chart(heatmap_long)
                    .mark_rect()
                    .encode(
                        x=alt.X("Peer:N", title="Peer part"),
                        y=alt.Y(f"{part_number_col}:N", title="Part"),
                        color=alt.Color(
                            "Similarity:Q",
                            scale=alt.Scale(scheme="blues"),
                            legend=alt.Legend(title="Similarity"),
                        ),
                        tooltip=[
                            alt.Tooltip(f"{part_number_col}:N", title="Part"),
                            alt.Tooltip("Peer:N", title="Peer"),
                            alt.Tooltip("Similarity:Q", format=".3f"),
                        ],
                    )
                    .properties(height=360)
                )
                st.altair_chart(heatmap_chart, use_container_width=True)
            else:
                st.info("Not enough parts in the selected cluster to compute a heatmap.")
        else:
            st.info("No clusters available for heatmap diagnostics.")

        from sklearn.metrics import silhouette_samples

        feature_matrix = (
            cat_vectors if metric == "jaccard" and cat_vectors is not None else vectors
        )
        valid_mask = result_df["cluster"].to_numpy() != -1
        unique_clusters = np.unique(result_df.loc[valid_mask, "cluster"])
        if len(unique_clusters) >= 2 and valid_mask.sum() >= len(unique_clusters):
            sil_values = silhouette_samples(
                feature_matrix[valid_mask],
                result_df.loc[valid_mask, "cluster"].to_numpy(),
                metric=metric,
            )
            silhouette_df = pd.DataFrame(
                {
                    "Silhouette": sil_values,
                    "Cluster": result_df.loc[valid_mask, "cluster"].astype(str).to_numpy(),
                }
            )
            silhouette_chart = (
                alt.Chart(silhouette_df)
                .mark_bar()
                .encode(
                    x=alt.X("Silhouette:Q", bin=alt.Bin(maxbins=25), title="Silhouette score"),
                    y=alt.Y("count()", title="Count"),
                    color=alt.Color("Cluster:N", legend=alt.Legend(title="Cluster")),
                    tooltip=[
                        alt.Tooltip("Cluster:N", title="Cluster"),
                        alt.Tooltip("count()", title="Parts"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(silhouette_chart, use_container_width=True)
        else:
            st.info(
                "Silhouette scores require at least two populated clusters without noise labels."
            )

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
