from __future__ import annotations

import os
import numpy as np
from sklearn.cluster import DBSCAN

from SL.clustering import (
    add_likeness_score,
    cluster_and_score,
    enforce_min_cluster_size,
    recommend_dbscan_metric,
    recommend_eps,
)
from SL.exporters import build_excel_workbook
from SL.feature_engineering import encode_features
from SL.legacy_io import process_attributes, read_filtered_data
from SL.reporting import export_cluster_report_pdf


def main() -> None:
    print("Enter full path to your Excel file (including .xlsm):")
    file_path = input().strip()
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    print(
        "Enter path for saving results (e.g., Clustered_Output.xlsx) or press Enter for default:"
    )
    out_path = input().strip() or "Clustered_Output.xlsx"

    part_number_column = "Part Number"
    eps_values = np.arange(0.1, 1.05, 0.05)
    numeric_weight = 10

    attributes, types, unit_flags = process_attributes(file_path)
    df_data = read_filtered_data(file_path)
    vectors, _, cat_vectors = encode_features(
        df_data, attributes, types, unit_flags, numeric_weight=numeric_weight
    )

    metric_rec = recommend_dbscan_metric(types)
    print(f"\nRecommended DBSCAN metric based on feature types: {metric_rec}")
    print(
        "If you want to override, input metric ('euclidean', 'cosine', 'jaccard'), else press Enter:"
    )
    metric_input = input().strip().lower()
    metric = metric_input if metric_input else metric_rec

    result_df = cluster_and_score(
        df_data,
        vectors,
        eps_values,
        metric,
        cat_vectors,
        min_samples=2,
    )

    print("\n==== Candidate eps Table (with similarity) ====")
    print(result_df)
    try:
        from IPython.display import display

        display(result_df)
    except Exception:
        pass

    if result_df.empty:
        print("No valid clusters found with given parameter sweep.")
        return

    best_eps = recommend_eps(result_df)
    print(f"\nRecommended best eps: {best_eps:.3f}")
    print("You may manually override this if desired.")
    print("Enter eps value for clustering (or press Enter to accept recommended):")
    eps_input = input().strip()
    eps_selected = float(eps_input) if eps_input else float(best_eps)

    db = DBSCAN(eps=eps_selected, min_samples=2, metric=metric)
    if metric == "jaccard" and cat_vectors is not None:
        labels = db.fit_predict(cat_vectors)
    else:
        labels = db.fit_predict(vectors)

    df_data = df_data.copy()
    df_data["cluster"] = labels
    df_data = add_likeness_score(df_data, vectors, labels, metric, cat_vectors)
    df_data = enforce_min_cluster_size(df_data, min_size=2)

    grouped_df = (
        df_data.groupby(part_number_column)[attributes + ["cluster", "likeness_score"]]
        .first()
        .reset_index()
    )

    workbook_bytes = build_excel_workbook(df_data, grouped_df)
    with open(out_path, "wb") as fh:
        fh.write(workbook_bytes.getvalue())

    pdf_buffer = export_cluster_report_pdf(
        df_data,
        candidate_eps_table=result_df,
        metric=metric,
        eps_selected=eps_selected,
        attributes=attributes,
        types=types,
        output=None,
        top_n=10,
    )
    with open("Cluster_Report.pdf", "wb") as fh:
        fh.write(pdf_buffer.getvalue())

    print(f"Results saved to {out_path}")
    print("Cluster summary report saved to Cluster_Report.pdf")


if __name__ == "__main__":
    main()
