"""PDF export helpers for sharing clustering results."""
from __future__ import annotations

from io import BytesIO
from typing import Iterable, Sequence

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

    summary_text = (
        "\n".join(
            [
                "Data set summary:",
                f"- Number of records (rows): {len(df)}",
                f"- Number of attributes used: {len(attributes)}",
                f"- Distance metric for DBSCAN: {metric}",
                f"- Cluster eps value used: {eps_selected}",
            ]
        )
        + "\n"
    )

    page_size = (10, 6)

    with PdfPages(output) as pdf:
        fig, ax = plt.subplots(figsize=page_size)
        ax.axis("off")
        ax.text(
            0.01,
            1,
            summary_text,
            ha="left",
            va="top",
            fontsize=12,
            wrap=True,
            transform=ax.transAxes,
        )

        col_labels = ["Attribute", "Type"]
        table_data = [[a, t] for a, t in zip(attributes, types)]
        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="left",
            colWidths=[0.4, 0.3],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        ax.set_title("Input Data Summary & Clustering Attributes", fontsize=15, pad=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if not candidate_eps_table.empty:
            fig, ax = plt.subplots(figsize=page_size)
            ax.axis("tight")
            ax.axis("off")
            table_data = candidate_eps_table.round(3).values.tolist()
            col_labels = candidate_eps_table.columns.tolist()
            table = ax.table(
                cellText=table_data,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            ax.set_title("Clustering Candidate eps Table", fontsize=15, pad=12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if not best_clusters.empty:
            fig, ax = plt.subplots(figsize=page_size)
            ax.axis("tight")
            ax.axis("off")
            table_data = best_clusters.round(3).values.tolist()
            col_labels = best_clusters.columns.tolist()
            table = ax.table(
                cellText=table_data,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            ax.set_title(f"Top {top_n} Potential Clusters", fontsize=15, pad=12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        if not cluster_summary.empty:
            fig, ax = plt.subplots(figsize=page_size)
            scatter = ax.scatter(
                cluster_summary["cluster_size"],
                cluster_summary["mean_likeness"],
                c=cluster_summary["mean_likeness"],
                cmap="viridis",
                s=100,
                edgecolor="k",
            )
            for _, row in cluster_summary.iterrows():
                ax.text(
                    row["cluster_size"] + 0.5,
                    row["mean_likeness"],
                    str(int(row["cluster"])),
                    fontsize=10,
                )
            ax.set_xlabel("Cluster Size")
            ax.set_ylabel("Mean Likeness Score")
            ax.set_title("Cluster Size vs. Mean Likeness Score")
            plt.colorbar(scatter, label="Mean Likeness Score")
            ax.grid(True)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=page_size)
            ax.hist(
                cluster_summary["cluster_size"],
                bins=min(15, len(cluster_summary)),
                color="steelblue",
                edgecolor="black",
            )
            ax.set_xlabel("Cluster Size")
            ax.set_ylabel("Number of Clusters")
            ax.set_title("Histogram of Cluster Sizes")
            ax.grid(True)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    output.seek(0)
    return output
