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
    consolidated_df: pd.DataFrame,
    dataset_name: str,
    dataset_options: List[str],
    dataset_aliases: Dict[str, str],
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

    st.session_state.setdefault("attribute_selection_confirmed", False)
    st.session_state.setdefault("attribute_selection_threshold", 70)

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown("<div class='material-header'>Filter dataset</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-subtitle'>Refine the dataset using the commodity hierarchy before clustering.</div>",
        unsafe_allow_html=True,
    )

    filtered_df = consolidated_df.copy()
    selection_values: Dict[str, str] = {}

    alias_options = [dataset_aliases.get(name, name) for name in dataset_options]
    alias_to_dataset = {dataset_aliases.get(name, name): name for name in dataset_options}
    current_alias = dataset_aliases.get(dataset_name, dataset_name)
    if current_alias not in alias_options:
        alias_options = [current_alias] + alias_options

    filter_cols = st.columns(3)
    with filter_cols[0]:
        alias_index = (
            alias_options.index(current_alias) if current_alias in alias_options else 0
        )
        chosen_alias = st.selectbox(
            "Commodity",
            options=alias_options,
            index=alias_index,
            key="dataset_selector",
        )

    chosen_dataset = alias_to_dataset.get(chosen_alias, dataset_name)
    if chosen_dataset != dataset_name:
        st.session_state.active_dataset = chosen_dataset
        reset_app_state()
        st.rerun()

    selection_values["Commodity"] = chosen_alias

    if commodity_col in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df[commodity_col].astype(str) == chosen_alias
        ]

    sub_options = ["All"] + sorted(
        filtered_df[subcommodity_col].dropna().astype(str).unique().tolist()
    )
    with filter_cols[1]:
        selection_values["Sub-Commodity"] = st.selectbox(
            "Sub-Commodity",
            options=sub_options,
            index=0,
            key="filter_Sub-Commodity",
        )
    if selection_values["Sub-Commodity"] != "All":
        filtered_df = filtered_df[
            filtered_df[subcommodity_col].astype(str)
            == selection_values["Sub-Commodity"]
        ]

    detail_options = ["All"] + sorted(
        filtered_df[detail_col].dropna().astype(str).unique().tolist()
    )
    with filter_cols[2]:
        selection_values["Detailed Commodity"] = st.selectbox(
            "Detailed Commodity",
            options=detail_options,
            index=0,
            key="filter_Detailed Commodity",
        )
    if selection_values["Detailed Commodity"] != "All":
        filtered_df = filtered_df[
            filtered_df[detail_col].astype(str)
            == selection_values["Detailed Commodity"]
        ]

    st.metric("Parts after filtering", len(filtered_df))
    st.markdown("</div>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No rows remain after applying the selected filters.")
        return

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>Select clustering attributes</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='material-subtitle'>Select the features to feed the clustering model. Attributes without data can be hidden from the table.</div>",
        unsafe_allow_html=True,
    )

    excluded_columns = (commodity_col, subcommodity_col, detail_col, part_number_col)

    hide_empty_attributes = st.toggle(
        "Hide attributes with no values",
        value=True,
        help="Only keep columns with data in the table.",
        key="hide_empty_attributes_toggle",
    )

    signature = (
        dataset_name,
        tuple(selection_values.items()),
        part_number_col,
        hide_empty_attributes,
        filtered_df.shape,
    )

    previous_signature = st.session_state.get("attribute_signature")
    attribute_config = prepare_attribute_config(
        filtered_df, excluded_columns, signature
    )

    if previous_signature != signature:
        st.session_state.attribute_selection_confirmed = False

    display_config = attribute_config
    if (
        hide_empty_attributes
        and display_config is not None
        and not display_config.empty
    ):
        display_config = display_config[display_config["Fill Ratio"] > 0].copy()

    if display_config is None or display_config.empty:
        st.info("No eligible attributes were found. Adjust your filters and try again.")
        st.session_state.attribute_selection_confirmed = False
        st.markdown("</div>", unsafe_allow_html=True)
        return

    previous_threshold = st.session_state.get("attribute_selection_threshold", 70)
    min_fill_ratio = st.slider(
        "Minimum fill ratio required for clustering",
        min_value=0,
        max_value=100,
        value=int(previous_threshold),
        step=5,
        key="min_fill_ratio_slider",
    )
    if min_fill_ratio != previous_threshold:
        st.session_state.attribute_selection_threshold = min_fill_ratio
        st.session_state.attribute_selection_confirmed = False
    else:
        st.session_state.attribute_selection_threshold = min_fill_ratio

    configured_attributes = st.session_state.get("attribute_config")
    if configured_attributes is None and attribute_config is not None:
        configured_attributes = attribute_config.copy()

    if not st.session_state.attribute_selection_confirmed:
        visible_attributes = display_config[
            display_config["Fill Ratio"] >= min_fill_ratio
        ].copy()

        if visible_attributes.empty:
            st.info(
                "No attributes meet the minimum fill ratio. Lower the threshold or adjust filters."
            )
            st.session_state.attribute_selection_confirmed = False
            st.markdown("</div>", unsafe_allow_html=True)
            return

        visible_attributes = visible_attributes.reset_index(drop=True)

        attribute_editor = st.data_editor(
            visible_attributes,
            key="attribute_editor",
            use_container_width=True,
            hide_index=True,
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

        attribute_editor = attribute_editor.reset_index(drop=True)
        base_config = (
            configured_attributes.copy()
            if configured_attributes is not None
            else attribute_config.copy()
        )
        updated_config = attribute_config.set_index("Attribute")
        editor_updates = attribute_editor.set_index("Attribute")
        updated_config.loc[
            editor_updates.index, ["Include", "Type", "Unit Extraction"]
        ] = editor_updates[["Include", "Type", "Unit Extraction"]]
        updated_config = updated_config.reset_index()
        st.session_state.attribute_config = updated_config

        if base_config is None:
            previous_subset = updated_config.set_index("Attribute")[[
                "Include",
                "Type",
                "Unit Extraction",
            ]].sort_index()
        else:
            previous_subset = base_config.set_index("Attribute")[[
                "Include",
                "Type",
                "Unit Extraction",
            ]].sort_index()
        new_subset = updated_config.set_index("Attribute")[[
            "Include",
            "Type",
            "Unit Extraction",
        ]].sort_index()
        if not new_subset.equals(previous_subset):
            st.session_state.attribute_selection_confirmed = False

        selected_attributes_df = updated_config[
            (updated_config["Include"])
            & (updated_config["Fill Ratio"] >= min_fill_ratio)
        ]

        finalize_clicked = st.button(
            "Finalize attribute selection",
            key="finalize_attribute_selection",
            type="secondary",
        )
        if finalize_clicked:
            if selected_attributes_df.empty:
                st.warning(
                    "Select at least one attribute above the minimum fill ratio before finalizing."
                )
            else:
                st.session_state.attribute_selection_confirmed = True

        if (
            st.session_state.attribute_selection_confirmed
            and selected_attributes_df.empty
        ):
            st.session_state.attribute_selection_confirmed = False

        if st.session_state.attribute_selection_confirmed:
            st.success(
                "Attribute selection confirmed. You can configure clustering below."
            )
        else:
            st.markdown("</div>", unsafe_allow_html=True)
            st.info(
                "Finalize the attribute selection to configure clustering parameters."
            )
            return
    else:
        configured_attributes = st.session_state.get("attribute_config")
        if configured_attributes is None:
            configured_attributes = attribute_config.copy()

        selected_attributes_df = configured_attributes[
            (configured_attributes["Include"])
            & (configured_attributes["Fill Ratio"] >= min_fill_ratio)
        ]

        if selected_attributes_df.empty:
            st.warning(
                "No attributes remain selected. Click Edit attribute selection to make changes."
            )
            if st.button(
                "Edit attribute selection", key="edit_attribute_selection_empty"
            ):
                st.session_state.attribute_selection_confirmed = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            return

        st.success("Attribute selection confirmed. You can configure clustering below.")
        st.dataframe(
            selected_attributes_df[
                ["Attribute", "Type", "Unit Extraction", "Fill Ratio"]
            ],
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Edit attribute selection", key="edit_attribute_selection"):
            st.session_state.attribute_selection_confirmed = False
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    configured_attributes = st.session_state.get("attribute_config")
    if configured_attributes is None:
        st.info("Finalize the attribute selection to configure clustering parameters.")
        return

    selected_attributes_df = configured_attributes[
        (configured_attributes["Include"])
        & (configured_attributes["Fill Ratio"] >= min_fill_ratio)
    ]

    if selected_attributes_df.empty:
        st.info("Finalize the attribute selection to configure clustering parameters.")
        return

    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='material-header'>Configure clustering</div>",
        unsafe_allow_html=True,
    )
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
        manual_eps = st.slider(
            "Manual epsilon adjustment",
            min_value=0.01,
            max_value=5.0,
            value=0.5,
            step=0.01,
            help="Provides a hint when overriding the automatically recommended eps value.",
        )
    else:
        eps_min = eps_max = eps_step = None
        min_samples_value = None
        numeric_weight = 10.0
        metric = recommended_metric
        manual_eps = None
        st.caption(
            f"Automatic configuration will use the recommended `{recommended_metric}` distance metric and a numeric weight of 10."
        )

    run_button = st.button("Run clustering", key="run_clustering", type="primary")

    if run_button:
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
            min_fill_ratio=min_fill_ratio,
        )
        trigger_clustering_run(run_config)

    st.markdown("</div>", unsafe_allow_html=True)
