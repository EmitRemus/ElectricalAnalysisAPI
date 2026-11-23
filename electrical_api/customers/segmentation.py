"""Segment customers using PCA diagnostics, k-means, and Ward linkage."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from ..datasets.storage import DatasetError
from .profiles import customer_frame


FEATURES = ["avg_daily_kwh", "night_share", "seasonality", "weekend_ratio", "stability"]
PCA_FEATURES = FEATURES + ["annual_charge_uah", "avg_price_uah_kwh", "voltage_dev_mean", "loss_index_mean"]
MAX_CLUSTERS = 8


def prepare_customer_features(table):
    customers = customer_frame(table)
    if (customers["avg_daily_kwh"] <= 0).any():
        raise DatasetError("Segmentation requires positive avg_daily_kwh", "nonpositive_avg_daily_kwh", 422)

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            standard_data = customers[FEATURES].copy()
            standard_data["avg_daily_kwh"] = np.log(standard_data["avg_daily_kwh"])
            features = StandardScaler().fit_transform(standard_data)
    except FloatingPointError as error:
        raise DatasetError("Segmentation is outside the numeric range", "segmentation_numeric_range", 422) from error
    return customers, features


def principal_components(customers):
    standardized = StandardScaler().fit_transform(customers[PCA_FEATURES])
    pca = PCA().fit(standardized)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    kaiser = pca.explained_variance_ > 1
    loadings = pd.DataFrame(pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2]),
                            index=PCA_FEATURES, columns=["PC1", "PC2"])
    return {
        "eigenvalues": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance_ratio": cumulative.tolist(),
        "kaiser_components": int(kaiser.sum()),
        "kaiser_explained_variance_ratio": float(pca.explained_variance_ratio_[kaiser].sum()),
        "loadings": loadings.to_dict(orient="index"),
    }


def validate_cluster_count(features, k):
    max_k = min(MAX_CLUSTERS, len(features) - 1, len(np.unique(features, axis=0)))
    if k > max_k:
        raise DatasetError("Segmentation requires at least k + 1 profiles and k distinct feature vectors",
                           "insufficient_segmentation_data", 422)
    return max_k


def train_kmeans(features, k):
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit(features)


def fit_kmeans(features, k):
    max_k = validate_cluster_count(features, k)
    diagnoses = []
    labels = None
    for candidate in range(2, max_k + 1):
        model = train_kmeans(features, candidate)
        if len(np.unique(model.labels_)) < candidate:
            continue
        diagnoses.append({"k": candidate, "inertia": float(model.inertia_),
                          "silhouette": float(silhouette_score(features, model.labels_))})
        if candidate == k:
            labels = model.labels_
    if labels is None:
        raise DatasetError("K-means could not form k distinct segments",
                           "insufficient_segmentation_data", 422)
    return labels, diagnoses


def segment_profiles(customers, labels):
    segments = customers.assign(segment_id=labels, ind_1985=customers["meter_type"] == "IND_1985")
    segments["churned"] = segments["churned"].astype("Float64")
    columns = FEATURES + ["churned", "ind_1985"]
    grouped = segments.groupby("segment_id")
    profiles = grouped[columns].mean()
    profiles.insert(0, "count", grouped.size())
    profiles = profiles.astype(object).where(profiles.notna(), None).reset_index()
    base = segments[columns].mean()
    baseline = {"count": len(segments), **base.astype(object).where(base.notna(), None).to_dict()}
    return profiles.to_dict(orient="records"), baseline


def segment_customers(table, k):
    customers, features = prepare_customer_features(table)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            labels, diagnostics = fit_kmeans(features, k)
            pca = principal_components(customers)
            ward_tree = linkage(features, "ward")
            ward = fcluster(ward_tree, k, "maxclust")
            profiles, baseline = segment_profiles(customers, labels)
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise DatasetError("Segmentation is outside the numeric range", "segmentation_numeric_range", 422) from error

    contingency = pd.crosstab(labels, ward)
    ward_cut_gaps = [
        {"k": candidate, "height_gap": float(ward_tree[-(candidate - 1), 2] - ward_tree[-candidate, 2])}
        for candidate in range(3, min(6, len(ward_tree)) + 1)
    ]
    assignments = [
        {"customer_id": customer_id, "segment_id": segment_id, "ward_id": ward_id}
        for customer_id, segment_id, ward_id in zip(customers["customer_id"], labels.tolist(), ward.tolist())
    ]
    return {
        "k": k,
        "customer_count": len(customers),
        "pca": pca,
        "kmeans_candidates": diagnostics,
        "ward_cut_gaps": ward_cut_gaps,
        "adjusted_rand_index": float(adjusted_rand_score(labels, ward)),
        "contingency": {"segment_ids": contingency.index.tolist(), "ward_ids": contingency.columns.tolist(),
                        "counts": contingency.to_numpy().tolist()},
        "profiles": profiles,
        "baseline": baseline,
        "assignments": assignments,
    }
