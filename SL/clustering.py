"""Clustering utilities extracted from the legacy CLI workflow."""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def recommend_dbscan_metric(types: Sequence[str]) -> str:
    n_numeric = sum(t.lower() == "numerical" for t in types)
    n_cat = sum(t.lower() == "categorical" for t in types)
    n_text = sum(t.lower() in ["alpha numeric", "text"] for t in types)
    n_total = max(len(types), 1)

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


def similarity_within_cluster(
    cluster_vecs: np.ndarray,
    metric: str,
    cat_vectors_subset: np.ndarray | None = None,
) -> float:
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


def cluster_and_score(
    df: pd.DataFrame,
    vectors: np.ndarray,
    eps_values: Iterable[float],
    metric: str,
    cat_vectors: np.ndarray | None = None,
    min_samples: int = 2,
) -> pd.DataFrame:
    records = []

    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

        if metric == "jaccard":
            if cat_vectors is None:
                continue
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


def recommend_eps(result_df: pd.DataFrame) -> float:
    result_df = result_df.copy()
    result_df["score"] = (
        result_df["silhouette_score"]
        * (1 - result_df["proportion_noise"])
        * np.log(result_df["num_clusters"] + 1)
    )
    best_row = result_df.loc[result_df["score"].idxmax()]
    return float(best_row["eps"])


def add_likeness_score(
    df: pd.DataFrame,
    vectors: np.ndarray,
    labels: np.ndarray,
    metric: str,
    cat_vectors: np.ndarray | None = None,
) -> pd.DataFrame:
    likeness_scores = []

    for idx in range(len(df)):
        cluster_label = labels[idx]
        if cluster_label == -1:
            likeness_scores.append(0)
            continue

        indices = np.where(labels == cluster_label)[0]

        if metric == "jaccard":
            cluster_vecs = vectors[indices]
            cat_vectors_subset = cat_vectors[indices] if cat_vectors is not None else None
            likeness = similarity_within_cluster(cluster_vecs, metric, cat_vectors_subset)
        else:
            cluster_vecs = vectors[indices]
            likeness = similarity_within_cluster(cluster_vecs, metric)

        likeness_scores.append(likeness)

    df = df.copy()
    df["likeness_score"] = likeness_scores
    return df


def enforce_min_cluster_size(df: pd.DataFrame, min_size: int = 2) -> pd.DataFrame:
    df = df.copy()
    cluster_counts = df["cluster"].value_counts()
    small_clusters = cluster_counts[cluster_counts < min_size].index
    df.loc[df["cluster"].isin(small_clusters), "cluster"] = -1
    return df
