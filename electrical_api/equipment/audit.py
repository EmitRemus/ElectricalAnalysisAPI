"""Audit equipment failure data for leakage, coverage, and drift."""
from math import isfinite

import numpy as np
from sklearn.metrics import roc_auc_score

from ..datasets.storage import DatasetError


def equipment_frames(registry_table, observations_table):
    if registry_table is None or not registry_table.num_rows:
        raise DatasetError("Dataset has no equipment table", "equipment_missing", 422)
    if observations_table is None or not observations_table.num_rows:
        raise DatasetError("Dataset has no equipment observations table", "equipment_observations_missing", 422)

    registry = registry_table.to_pandas()
    observations = observations_table.to_pandas()
    if registry["equipment_id"].duplicated().any():
        raise DatasetError("Expected one registry record per equipment_id", "duplicate_equipment_id", 422)
    if not observations["equipment_id"].isin(registry["equipment_id"]).all():
        raise DatasetError("An observation references equipment outside the registry", "unknown_equipment_id", 422)

    joined = observations.merge(registry, on="equipment_id", how="left", validate="m:1")
    train = joined.loc[joined["role"] == "training"]
    prediction = joined.loc[joined["role"] == "prediction"]
    return registry, train, prediction


def finite_statistic(value):
    return float(value) if isfinite(value) else None


def check_leakage(train):
    threshold = 0.9
    y = train["failed"].astype(bool)
    can_score = y.nunique() == 2
    columns = train.drop(columns="failed").select_dtypes(include=["number", "bool"]).columns

    scores = []
    for column in sorted(columns):
        auc = float(roc_auc_score(y, train[column])) if can_score else None
        adjusted = max(auc, 1 - auc) if can_score else None
        scores.append({"column": column, "roc_auc": auc, "roc_auc_adjusted": adjusted})
    if can_score:
        scores.sort(key=lambda entry: entry["roc_auc_adjusted"], reverse=True)
    flagged = [entry["column"] for entry in scores
               if entry["roc_auc_adjusted"] is not None and entry["roc_auc_adjusted"] > threshold]
    return {"detected": bool(flagged) if can_score else None, "threshold": threshold,
            "scores": scores, "flagged_columns": flagged}


def check_history_coverage(registry, train):
    threshold = 0.9
    history = registry.loc[registry["cohort"] == "history"]
    in_training = history["equipment_id"].isin(train["equipment_id"])
    covered = int(in_training.sum())
    coverage = covered / len(history) if len(history) else None
    return {
        "detected": None if coverage is None else coverage < threshold,
        "threshold": threshold,
        "coverage": coverage,
        "history_count": len(history),
        "covered_history_count": covered,
        "training_outside_history_count": int(train["equipment_id"].nunique()) - covered,
        "mean_install_year_covered": finite_statistic(history.loc[in_training, "install_year"].mean()),
        "mean_install_year_uncovered": finite_statistic(history.loc[~in_training, "install_year"].mean()),
    }


def check_load_drift(train, prediction):
    threshold = 0.5
    feature = "load_mean"
    with np.errstate(over="ignore", invalid="ignore"):
        train_mean = finite_statistic(train[feature].mean())
        prediction_mean = finite_statistic(prediction[feature].mean())
        train_std = finite_statistic(train[feature].std())
    drift = None
    if train_mean is not None and prediction_mean is not None and train_std is not None and train_std > 0:
        drift = finite_statistic((prediction_mean - train_mean) / train_std)
    return {
        "feature": feature,
        "detected": None if drift is None else abs(drift) > threshold,
        "threshold": threshold,
        "train_mean": train_mean,
        "prediction_mean": prediction_mean,
        "train_std": train_std,
        "drift": drift,
    }


def report_equipment_audit(registry_table, observations_table):
    registry, train, prediction = equipment_frames(registry_table, observations_table)
    failed_count = int(train["failed"].sum())
    return {
        "equipment_count": len(registry),
        "training_count": len(train),
        "prediction_count": len(prediction),
        "failed_count": failed_count,
        "failure_rate": failed_count / len(train) if len(train) else None,
        "m1": check_leakage(train),
        "m2": check_history_coverage(registry, train),
        "m3": check_load_drift(train, prediction),
    }
