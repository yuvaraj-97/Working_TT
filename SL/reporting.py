"""PDF export helpers for sharing clustering results."""
from __future__ import annotations

from io import BytesIO
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def export_cluster_report_pdf(
    df: pd.DataFrame,
    candidate_eps_table: pd.DataFrame,
    metric: str,
    eps_selected: float,
    attributes: Sequence[str],
    types: Sequence[str],
    output: BytesIO | None = None,
    top_n: int = 5,
) -> BytesIO:
    """Generate a PDF summarising the clustering run and return it as bytes."""

    if output is None:
        output = BytesIO()

    cluster_summary = (
        df.groupby("cluster").agg(
            cluster_size=("cluster", "count"),
            mean_likeness=("likeness_score", "mean"),
        )
    ).reset_index()
    cluster_summary = cluster_summary[cluster_summary["cluster"] != -1]
    cluster_summary["score"] = (
        cluster_summary["mean_likeness"] * cluster_summary["cluster_size"]
    )
    best_clusters = cluster_summary.sort_values("score", ascending=False).head(top_n)

    metrics_table = pd.DataFrame(
        {
            "Metric": [
                "Records analysed",
                "Attributes used",
                "Distance metric",
                "Selected eps",
            ],
            "Value": [
                f"{len(df):,}",
                f"{len(attributes)}",
                metric,
                f"{eps_selected:.3f}",
            ],
        }
    )

    attribute_rows = [[attribute, attr_type] for attribute, attr_type in zip(attributes, types)]

    page_size = (11.69, 8.27)

    with PdfPages(output) as pdf:
        with plt.style.context("seaborn-v0_8"):
            fig, ax = plt.subplots(figsize=page_size)
            fig.patch.set_facecolor("#f8fafc")
            ax.axis("off")
            ax.set_title(
                "Clustering Run Summary",
                loc="left",
                fontsize=20,
                fontweight="bold",
                color="#0f172a",
                pad=20,
            )

            summary_table = ax.table(
                cellText=metrics_table.values,
                colLabels=metrics_table.columns,
                loc="upper left",
                colWidths=[0.35, 0.35],
            )
            summary_table.auto_set_font_size(False)
            summary_table.set_fontsize(12)
            summary_table.scale(1, 1.3)
            for (row, col), cell in summary_table.get_celld().items():
                cell.set_edgecolor("#cbd5f5")
                if row == 0:
                    cell.set_facecolor("#0f172a")
                    cell.get_text().set_color("#f8fafc")
                    cell.get_text().set_fontweight("bold")
                else:
                    cell.set_facecolor("#ffffff")
                    cell.get_text().set_color("#1f2937")

            ax.text(
                0.02,
                0.55,
                "Clustering attributes",
                ha="left",
                va="top",
                fontsize=16,
                fontweight="bold",
                color="#0f172a",
                transform=ax.transAxes,
            )
            if attribute_rows:
                attribute_table = ax.table(
                    cellText=attribute_rows,
                    colLabels=["Attribute", "Type"],
                    loc="upper left",
                    colWidths=[0.35, 0.35],
                    bbox=[0.02, 0.1, 0.66, 0.42],
                )
                attribute_table.auto_set_font_size(False)
                attribute_table.set_fontsize(11)
                for (row, col), cell in attribute_table.get_celld().items():
                    cell.set_edgecolor("#e2e8f0")
                    if row == 0:
                        cell.set_facecolor("#1e293b")
                        cell.get_text().set_color("#f8fafc")
                        cell.get_text().set_fontweight("bold")
                    else:
                        cell.set_facecolor("#ffffff")
                        cell.get_text().set_color("#1f2937")
            else:
                ax.text(
                    0.02,
                    0.5,
                    "No attributes were selected for clustering.",
                    ha="left",
                    va="top",
                    fontsize=12,
                    color="#475569",
                    transform=ax.transAxes,
                )

            if not best_clusters.empty:
                ax.text(
                    0.7,
                    0.55,
                    f"Top {min(top_n, len(best_clusters))} clusters",
                    ha="left",
                    va="top",
                    fontsize=16,
                    fontweight="bold",
                    color="#0f172a",
                    transform=ax.transAxes,
                )
                highlights = best_clusters.round(2)[
                    ["cluster", "cluster_size", "mean_likeness"]
                ]
                highlight_table = ax.table(
                    cellText=highlights.values,
                    colLabels=["Cluster", "Size", "Mean likeness"],
                    loc="upper left",
                    colWidths=[0.18, 0.18, 0.22],
                    bbox=[0.7, 0.1, 0.28, 0.42],
                )
                highlight_table.auto_set_font_size(False)
                highlight_table.set_fontsize(11)
                for (row, col), cell in highlight_table.get_celld().items():
                    cell.set_edgecolor("#cbd5f5")
                    if row == 0:
                        cell.set_facecolor("#1e293b")
                        cell.get_text().set_color("#f8fafc")
                        cell.get_text().set_fontweight("bold")
                    else:
                        cell.set_facecolor("#ffffff")
                        cell.get_text().set_color("#1f2937")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            if not candidate_eps_table.empty:
                fig, ax = plt.subplots(figsize=page_size)
                fig.patch.set_facecolor("#f8fafc")
                ax.axis("off")
                ax.set_title(
                    "Candidate epsilon sweep",
                    loc="left",
                    fontsize=18,
                    fontweight="bold",
                    color="#0f172a",
                    pad=20,
                )
                candidate_table = ax.table(
                    cellText=candidate_eps_table.round(3).values,
                    colLabels=candidate_eps_table.columns.tolist(),
                    loc="upper left",
                    colWidths=[1.0 / max(len(candidate_eps_table.columns), 1)]
                    * len(candidate_eps_table.columns),
                )
                candidate_table.auto_set_font_size(False)
                candidate_table.set_fontsize(11)
                candidate_table.scale(1, 1.2)
                for (row, col), cell in candidate_table.get_celld().items():
                    cell.set_edgecolor("#e2e8f0")
                    if row == 0:
                        cell.set_facecolor("#1e293b")
                        cell.get_text().set_color("#f8fafc")
                        cell.get_text().set_fontweight("bold")
                    else:
                        cell.set_facecolor("#ffffff")
                        cell.get_text().set_color("#1f2937")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            if not best_clusters.empty:
                fig, ax = plt.subplots(figsize=page_size)
                fig.patch.set_facecolor("#f8fafc")
                ax.axis("off")
                ax.set_title(
                    f"Top {min(top_n, len(best_clusters))} potential clusters",
                    loc="left",
                    fontsize=18,
                    fontweight="bold",
                    color="#0f172a",
                    pad=20,
                )
                best_table = ax.table(
                    cellText=best_clusters.round(3).values,
                    colLabels=best_clusters.columns.tolist(),
                    loc="upper left",
                    colWidths=[
                        0.15,
                        0.2,
                        0.2,
                        0.25,
                    ][: len(best_clusters.columns)],
                )
                best_table.auto_set_font_size(False)
                best_table.set_fontsize(11)
                best_table.scale(1, 1.2)
                for (row, col), cell in best_table.get_celld().items():
                    cell.set_edgecolor("#cbd5f5")
                    if row == 0:
                        cell.set_facecolor("#1e293b")
                        cell.get_text().set_color("#f8fafc")
                        cell.get_text().set_fontweight("bold")
                    else:
                        cell.set_facecolor("#ffffff")
                        cell.get_text().set_color("#1f2937")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            if not cluster_summary.empty:
                fig, axes = plt.subplots(1, 2, figsize=page_size)
                fig.patch.set_facecolor("#f8fafc")
                scatter_ax, hist_ax = axes
                scatter = scatter_ax.scatter(
                    cluster_summary["cluster_size"],
                    cluster_summary["mean_likeness"],
                    c=cluster_summary["mean_likeness"],
                    cmap="viridis",
                    s=120,
                    edgecolor="#1f2937",
                    linewidths=0.5,
                )
                for _, row in cluster_summary.iterrows():
                    scatter_ax.text(
                        row["cluster_size"],
                        row["mean_likeness"] + 0.01,
                        str(int(row["cluster"])),
                        fontsize=9,
                        ha="center",
                    )
                scatter_ax.set_title("Cluster size vs. likeness", fontsize=14)
                scatter_ax.set_xlabel("Cluster size")
                scatter_ax.set_ylabel("Mean likeness score")
                scatter_ax.grid(True, alpha=0.3)
                cbar = fig.colorbar(scatter, ax=scatter_ax)
                cbar.set_label("Mean likeness")

                hist_ax.hist(
                    cluster_summary["cluster_size"],
                    bins=min(12, len(cluster_summary)),
                    color="#38bdf8",
                    edgecolor="#0f172a",
                    alpha=0.9,
                )
                hist_ax.set_title("Cluster size distribution", fontsize=14)
                hist_ax.set_xlabel("Cluster size")
                hist_ax.set_ylabel("Clusters")
                hist_ax.grid(True, alpha=0.3)

                fig.suptitle("Cluster diagnostics", fontsize=18, fontweight="bold", color="#0f172a")
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    output.seek(0)
    return output
