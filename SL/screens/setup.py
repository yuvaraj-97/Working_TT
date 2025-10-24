"""Setup screen for configuring clustering runs."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import streamlit as st

from clustering import recommend_dbscan_metric

from data_access import resolve_column
from models import RunConfig
from state import reset_app_state, trigger_clustering_run
from workflow import prepare_attribute_config


def render_setup_screen(
    consolidated_df: pd.DataFrame, dataset_name: str, dataset_options: List[str]
) -> None:
    """Render the form used to configure a clustering run."""

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
    st.markdown("<div class='material-header'>2. Choose clustering attributes</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-subtitle'>Select the features to feed the clustering model. Attributes without data can be hidden from the table.</div>",
        unsafe_allow_html=True,
    )

    st.caption(f"Using `{part_number_col}` as the unique part identifier.")

    excluded_columns = (commodity_col, subcommodity_col, detail_col, part_number_col)

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

    attribute_config = prepare_attribute_config(filtered_df, excluded_columns, signature)

    if hide_empty_attributes and attribute_config is not None and not attribute_config.empty:
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

    st.session_state.attribute_config = attribute_editor.copy()

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
        )
    else:
        eps_min = eps_max = eps_step = None
        min_samples_value = None
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
            help="The app will automatically refine eps based on the chosen metric.",
        )

    manual_eps = st.slider(
        "Manual epsilon adjustment",
        min_value=0.01,
        max_value=5.0,
        value=0.5,
        step=0.01,
        help="Provides a hint when the app automatically recommends eps values.",
    )

    min_fill_ratio_value = min_fill_ratio

    run_button_cols = st.columns([1, 1])
    with run_button_cols[0]:
        trigger_run = st.button("Run clustering", use_container_width=True)
    with run_button_cols[1]:
        reset_clicked = st.button(
            "Reset configuration", use_container_width=True, type="secondary"
        )

    if reset_clicked:
        reset_app_state()
        st.rerun()

    if trigger_run:
        run_config = RunConfig(
            dataset_name=dataset_name,
            filters=selection_values,
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
            min_fill_ratio=min_fill_ratio_value,
        )
        trigger_clustering_run(run_config)

    st.markdown("</div>", unsafe_allow_html=True)
