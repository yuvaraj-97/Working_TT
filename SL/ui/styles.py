"""Styling helpers for the clustering Streamlit app."""

from __future__ import annotations

import streamlit as st

PRIMARY_COLOR = "#6400FF"
SECONDARY_COLOR = "#bfbfbf"
TEXT_COLOR = "#FFFFFF"
CARD_BORDER_COLOR = "rgba(255, 255, 255, 0.18)"


def configure_page() -> None:
    """Configure the Streamlit page and apply shared CSS."""

    st.set_page_config(
        page_title="Clustering Workbench",
        page_icon="📊",
        layout="wide",
    )

    st.markdown(
        f"""
        <style>
            :root {{
                color-scheme: dark;
            }}
            .stApp {{
                background: radial-gradient(circle at top, #1a1b24, #050507 70%);
                color: {TEXT_COLOR};
                font-family: 'Inter', sans-serif;
            }}
            .block-container {{
                padding-top: 1.5rem;
                padding-bottom: 4rem;
            }}
            .material-card {{
                position: relative;
                background: rgba(10, 11, 18, 0.88);
                border-radius: 18px;
                border: none;
                padding: 1.5rem;
                box-shadow: 0 18px 38px rgba(0, 0, 0, 0.45);
            }}
            .material-header {{
                font-size: 1.05rem;
                font-weight: 600;
                color: {TEXT_COLOR};
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 0.35rem;
            }}
            .material-subtitle {{
                color: {SECONDARY_COLOR};
                font-size: 0.9rem;
                margin-bottom: 1.2rem;
            }}
            .cluster-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                gap: 1rem;
            }}
            .cluster-card {{
                border-radius: 18px;
                border: 1px solid rgba(140, 140, 152, 0.55);
                padding: 1.35rem;
                background: rgba(19, 20, 30, 0.82);
                box-shadow: 0 18px 36px rgba(0, 0, 0, 0.35);
                height: 100%;
                transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
            }}
            .cluster-card:hover {{
                transform: translateY(-2px);
                border-color: rgba(192, 192, 204, 0.85);
                box-shadow: 0 22px 44px rgba(0, 0, 0, 0.45);
            }}
            .cluster-card h4 {{
                margin: 0;
                font-size: 1rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                color: {TEXT_COLOR};
            }}
            .cluster-card .cluster-button {{
                border: 1px solid {CARD_BORDER_COLOR};
                background: transparent;
                color: {TEXT_COLOR};
                border-radius: 999px;
                padding: 0.35rem 0.9rem;
                font-size: 0.85rem;
                cursor: pointer;
            }}
            .cluster-card .cluster-button:hover {{
                border-color: {PRIMARY_COLOR};
                color: {PRIMARY_COLOR};
            }}
            .cluster-card .cluster-metric {{
                margin-top: 1rem;
                display: grid;
                gap: 0.5rem;
            }}
            .stButton > button,
            .stDownloadButton button {{
                border-radius: 999px;
                border: 1px solid {PRIMARY_COLOR};
                background: linear-gradient(135deg, {PRIMARY_COLOR}, #3b0bff);
                color: {TEXT_COLOR};
                font-weight: 600;
                padding: 0.6rem 1.6rem;
                box-shadow: 0 14px 32px rgba(100, 0, 255, 0.35);
                width: fit-content;
                min-width: 0;
            }}
            .stButton > button:hover,
            .stDownloadButton button:hover {{
                box-shadow: 0 18px 36px rgba(100, 0, 255, 0.55);
            }}
            .stButton > button[data-testid="baseButton-secondary"],
            .stButton > button[kind="secondary"] {{
                border: 1px solid {SECONDARY_COLOR};
                background: transparent;
                color: {SECONDARY_COLOR};
                box-shadow: none;
            }}
            .stButton > button[data-testid="baseButton-secondary"]:hover,
            .stButton > button[kind="secondary"]:hover {{
                border-color: {TEXT_COLOR};
                color: {TEXT_COLOR};
            }}
            div[data-testid="stMetricValue"] {{
                color: {TEXT_COLOR};
            }}
            .stDataFrame, .stDataFrame [data-testid="stTable"] {{
                color: {TEXT_COLOR};
            }}
            .stDataFrame thead tr th {{
                background-color: rgba(16, 18, 28, 0.8);
            }}
            .stDataFrame tbody tr td {{
                background-color: rgba(8, 10, 18, 0.6);
            }}
            .loading-screen {{
                min-height: 65vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                gap: 1.5rem;
                text-align: center;
                padding: 2rem 1.5rem;
            }}
            .loading-status-title {{
                font-size: 1.35rem;
                font-weight: 600;
                color: {TEXT_COLOR};
            }}
            .loading-status-subtitle {{
                color: {SECONDARY_COLOR};
                font-size: 0.95rem;
            }}
            .loading-status {{
                color: {TEXT_COLOR};
                font-size: 1rem;
            }}
            .loading-screen div[data-testid="stProgressBar"] > div {{
                background-color: rgba(255, 255, 255, 0.12);
                border-radius: 999px;
            }}
            .loading-screen div[data-testid="stProgressBar"] div[role="progressbar"] {{
                background: linear-gradient(135deg, {PRIMARY_COLOR}, #2f00b5);
                border-radius: 999px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
