"""Reusable UI components for Streamlit layouts."""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st


@contextmanager
def material_card(title: str | None = None, subtitle: str | None = None):
    """Render content inside a styled material card container.

    Parameters
    ----------
    title:
        Optional title rendered using the ``material-header`` class.
    subtitle:
        Optional subtitle rendered below the title.
    """

    card = st.container()
    with card:
        card.markdown("<div class='material-card'>", unsafe_allow_html=True)
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
