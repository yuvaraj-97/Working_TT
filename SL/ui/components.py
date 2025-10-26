"""Reusable UI components for Streamlit layouts."""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st


@contextmanager
def material_card(
    title: str | None = None,
    subtitle: str | None = None,
    card_class: str = "material-card",  # <-- ADD THIS ARGUMENT
):
    """Render content inside a styled material card container.

    Parameters
    ----------
    title:
        Optional title rendered using the ``material-header`` class.
    subtitle:
        Optional subtitle rendered below the title.
    card_class:  # <-- ADD THIS DOCSTRING
        The CSS class to apply (e.g., 'material-card' or 'cluster-card').
    """

    card = st.container()
    with card:
        # Use the new 'card_class' variable here
        card.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
        if title:
            card.markdown(
                f"<div class='material-header'>{title}</div>",
                unsafe_allow_html=True,
            )
        if subtitle:
            card.markdown(
                f"<div class='material-subtitle'>{subtitle}</div>",
                unsafe_allow_html=True,
            )
        yield card
        card.markdown("</div>", unsafe_allow_html=True)


__all__ = ["material_card"]
