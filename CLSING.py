import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def extract_numeric(val):
    match = re.search(r"\d+(\.\d+)?", str(val))
    if match:
        try:
            return float(match.group())
        except ValueError:
            return np.nan
    return np.nan


def process_attributes(excel_file):
    summary_df = pd.read_excel(
        excel_file,
        sheet_name="Summary Tab",
        usecols="P:R",
        skiprows=5,
        nrows=100,
    )
    summary_df.columns = [col.strip() for col in summary_df.columns]

    attributes = summary_df["Attribute"].dropna().tolist()
    types = summary_df["Type"].dropna().tolist()
    unit_flags = summary_df["Unit Extraction"].fillna("No").tolist()
    return attributes, types, unit_flags


def read_filtered_data(excel_file):
    return pd.read_excel(excel_file, sheet_name="FilteredOutput")


def encode_features(df, attributes, types, unit_flags, numeric_weight=10):
    encoded_features = []
    categorical_features = []
    all_feature_types = []

    for attr, typ, unit_flag in zip(attributes, types, unit_flags):
        vals = df[attr].fillna("Missing")

        if str(unit_flag).lower() == "yes":
            vals_num = vals.apply(extract_numeric)
            median_value = vals_num.median()
            if np.isnan(median_value):
                median_value = 0.0
            vals_num = vals_num.fillna(median_value)
            enc = (
                StandardScaler().fit_transform(vals_num.values.reshape(-1, 1))
                * numeric_weight
            )
            encoded_features.append(enc)
            all_feature_types.append("numerical")
        elif typ.lower() == "numerical":
            vals_num = pd.to_numeric(vals, errors="coerce")
            median_value = vals_num.median()
            if np.isnan(median_value):
                median_value = 0.0
            vals_num = vals_num.fillna(median_value)
            enc = (
                StandardScaler().fit_transform(vals_num.values.reshape(-1, 1))
                * numeric_weight
            )
            encoded_features.append(enc)
            all_feature_types.append("numerical")
        elif typ.lower() == "categorical":
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            vals_str = vals.astype(str)
            enc = encoder.fit_transform(vals_str.values.reshape(-1, 1))
            encoded_features.append(enc)
            categorical_features.append(enc)
            all_feature_types.append("categorical")
        elif typ.lower() == "alpha numeric":
            tfidf = TfidfVectorizer()
            enc = tfidf.fit_transform(vals.astype(str)).toarray()
            encoded_features.append(enc)
            all_feature_types.append("text")
        else:
            tfidf = TfidfVectorizer()
            enc = tfidf.fit_transform(vals.astype(str)).toarray()
            encoded_features.append(enc)
            all_feature_types.append("text")

    vectors = np.hstack(encoded_features)
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)

    cat_vectors = None
    if categorical_features:
        cat_vectors = np.hstack(categorical_features)

    return vectors, all_feature_types, cat_vectors


def recommend_dbscan_metric(types):
    n_numeric = sum(t.lower() == "numerical" for t in types)
    n_cat = sum(t.lower() == "categorical" for t in types)
    n_text = sum(t.lower() in ["alpha numeric", "text"] for t in types)
    n_total = len(types)

    ratio_numeric = n_numeric / n_total
    ratio_text = n_text / n_total
    ratio_cat = n_cat / n_total

    if ratio_numeric > 0.5 and ratio_text == 0 and ratio_cat == 0:
        return "euclidean"
    if ratio_text > 0.4 and ratio_text > ratio_numeric:
        return "cosine"
    if ratio_cat > 0.5:
        return "jaccard"
    return "euclidean"


def similarity_within_cluster(cluster_vecs, metric, cat_vectors_subset=None):
    if cluster_vecs.shape[0] < 2:
        return 1.0

    if metric == "euclidean":
        dists = euclidean_distances(cluster_vecs)
        upper = dists[np.triu_indices_from(dists, k=1)]
        max_dist = upper.max() if upper.max() > 0 else 1
        likeness = 1 - (upper.mean() / max_dist)
        return max(likeness, 0)

    if metric == "cosine":
        sims = cosine_similarity(cluster_vecs)
        upper = sims[np.triu_indices_from(sims, k=1)]
        return np.mean(upper)

    if metric == "jaccard":
        if cat_vectors_subset is None:
            return 0

        sims = []
        n = cat_vectors_subset.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                a = cat_vectors_subset[i]
                b = cat_vectors_subset[j]
                intersect = np.logical_and(a, b).sum()
                union = np.logical_or(a, b).sum()
                sim = intersect / union if union else 1.0
                sims.append(sim)
        return np.mean(sims) if sims else 1.0

    return 1.0


def cluster_and_score(df, vectors, eps_values, metric, cat_vectors=None, min_samples=2):
    records = []

    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

        if metric == "jaccard":
            labels = db.fit_predict(cat_vectors)
        else:
            labels = db.fit_predict(vectors)

        num_noise_points = (labels == -1).sum()
        proportion_noise = num_noise_points / len(labels)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        try:
            if num_clusters > 1 and proportion_noise < 0.5:
                if metric == "jaccard":
                    sil = silhouette_score(cat_vectors, labels, metric=metric)
                else:
                    sil = silhouette_score(vectors, labels, metric=metric)

                cluster_likeness = []
                for label in set(labels):
                    if label == -1:
                        continue

                    indices = np.where(labels == label)[0]
                    if metric == "jaccard":
                        cluster_vecs = vectors[indices]
                        cat_vectors_subset = (
                            cat_vectors[indices] if cat_vectors is not None else None
                        )
                        likeness = similarity_within_cluster(
                            cluster_vecs, metric, cat_vectors_subset
                        )
                    else:
                        cluster_vecs = vectors[indices]
                        likeness = similarity_within_cluster(cluster_vecs, metric)

                    cluster_likeness.append(likeness)

                avg_likeness = np.mean(cluster_likeness) if cluster_likeness else 0.0

                records.append(
                    {
                        "eps": eps,
                        "silhouette_score": sil,
                        "num_clusters": num_clusters,
                        "num_noise_points": num_noise_points,
                        "proportion_noise": proportion_noise,
                        "mean_cluster_likeness": avg_likeness,
                    }
                )
        except Exception:
            continue

    return pd.DataFrame(records)


def recommend_eps(result_df):
    result_df["score"] = (
        result_df["silhouette_score"]
        * (1 - result_df["proportion_noise"])
        * np.log(result_df["num_clusters"] + 1)
    )
    best_row = result_df.loc[result_df["score"].idxmax()]
    return best_row["eps"]


def add_likeness_score(df, vectors, labels, metric, cat_vectors=None):
    likeness_scores = []

    for idx, _ in df.iterrows():
        cluster_label = labels[idx]
        if cluster_label == -1:
            likeness_scores.append(0)
            continue

        indices = df.index[df["cluster"] == cluster_label].tolist()

        if metric == "jaccard":
            cluster_vecs = vectors[indices]
            cat_vectors_subset = (
                cat_vectors[indices] if cat_vectors is not None else None
            )
            likeness = similarity_within_cluster(
                cluster_vecs, metric, cat_vectors_subset
            )
        else:
            cluster_vecs = vectors[indices]
            likeness = similarity_within_cluster(cluster_vecs, metric)

        likeness_scores.append(likeness)

    df["likeness_score"] = likeness_scores
    return df


def enforce_min_cluster_size(df, min_size=2):
    cluster_counts = df["cluster"].value_counts()
    small_clusters = cluster_counts[cluster_counts < min_size].index
    df.loc[df["cluster"].isin(small_clusters), "cluster"] = -1
    return df


def export_cluster_report_pdf(
    df,
    candidate_eps_table,
    metric,
    eps_selected,
    attributes,
    types,
    filename="Cluster_Report.pdf",
    top_n=5,
):
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

    with PdfPages(filename) as pdf:
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

        fig, ax = plt.subplots(figsize=page_size)
        ax.scatter(
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
        plt.colorbar(ax.collections[0], label="Mean Likeness Score")
        ax.grid(True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=page_size)
        ax.hist(
            cluster_summary["cluster_size"],
            bins=15,
            color="steelblue",
            edgecolor="black",
        )
        ax.set_xlabel("Cluster Size")
        ax.set_ylabel("Number of Clusters")
        ax.set_title("Histogram of Cluster Sizes")
        ax.grid(True)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Cluster report PDF saved to {filename}")


def main():
    print("Enter full path to your Excel file (including .xlsm):")
    file_path = input().strip()
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    print(
        "Enter path for saving results (e.g., Clustered_Output.xlsx) or press Enter for default:"
    )
    out_path = input().strip()
    if not out_path:
        out_path = "Clustered_Output.xlsx"

    part_number_column = "Part Number"
    eps_values = np.arange(0.1, 1.05, 0.05)
    numeric_weight = 10

    attributes, types, unit_flags = process_attributes(file_path)
    df_data = read_filtered_data(file_path)
    vectors, feature_types, cat_vectors = encode_features(
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
    if metric == "jaccard":
        labels = db.fit_predict(cat_vectors)
    else:
        labels = db.fit_predict(vectors)

    df_data["cluster"] = labels
    df_data = add_likeness_score(df_data, vectors, labels, metric, cat_vectors)
    df_data = enforce_min_cluster_size(df_data, min_size=2)

    grouped_df = (
        df_data.groupby(part_number_column)[attributes + ["cluster", "likeness_score"]]
        .first()
        .reset_index()
    )

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_data.to_excel(writer, sheet_name="All Data", index=False)
        grouped_df.to_excel(writer, sheet_name="Grouped by Part Number", index=False)

    export_cluster_report_pdf(
        df_data,
        candidate_eps_table=result_df,
        metric=metric,
        eps_selected=eps_selected,
        attributes=attributes,
        types=types,
        filename="Cluster_Report.pdf",
        top_n=10,
    )

    print(f"Results saved to {out_path}")
    print("Cluster summary report saved to Cluster_Report.pdf")


if __name__ == "__main__":
    main()
