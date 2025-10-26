"""Loading overlay used while clustering is running."""

from __future__ import annotations

import streamlit as st


class LoadingScreen:
    """Render a branded loading overlay and expose update hooks."""

    def __init__(self, title: str, subtitle: str | None = None):
        self.container = st.empty()
        with self.container.container():
            st.markdown("<div class='loading-screen'>", unsafe_allow_html=True)
            if title:
                st.markdown(
                    f"<div class='loading-status-title'>{title}</div>",
                    unsafe_allow_html=True,
                )
            if subtitle:
                st.markdown(
                    f"<div class='loading-status-subtitle'>{subtitle}</div>",
                    unsafe_allow_html=True,
                )
            self.status_placeholder = st.empty()
            self.progress_bar = st.progress(0)
            st.markdown("</div>", unsafe_allow_html=True)

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


__all__ = ["LoadingScreen"]
