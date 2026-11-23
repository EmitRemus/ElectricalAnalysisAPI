"""Prepare equipment features and validate one failure model."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ..datasets.storage import DatasetError
from .audit import check_leakage, equipment_frames


DEFAULT_MODEL = "perceptron"
DEFAULT_PERCEPTRON_ALPHA = 1.0
CATEGORICAL = ["equipment_type", "voltage_class", "region"]
VALIDATION_FRACTION = 0.25


@dataclass
class PreparedFailureSplit:
    train_features: np.ndarray
    validation_features: np.ndarray
    train_labels: np.ndarray
    validation_labels: np.ndarray
    feature_columns: list[str]
    excluded_features: list[str]


def prepare_failure_split(registry_table, observations_table):
    _, training, _ = equipment_frames(registry_table, observations_table)
    if training["equipment_id"].duplicated().any():
        raise DatasetError("Expected one training observation per equipment_id", "duplicate_training_equipment", 422)
    training = training.sort_values("equipment_id")
    labels = training["failed"].astype(int)
    try:
        train, validation, y_train, y_validation = train_test_split(
            training, labels, test_size=VALIDATION_FRACTION, stratify=labels, random_state=0,
        )
    except ValueError as error:
        raise DatasetError("Need enough observations of both classes for stratified validation",
                           "insufficient_failure_data", 422) from error
    if y_train.nunique() != 2 or y_validation.nunique() != 2:
        raise DatasetError("Both classes must be present in the training and validation folds",
                           "insufficient_failure_data", 422)

    metadata_columns = ["equipment_id", "cohort", "role", "failed"]
    X_train = pd.get_dummies(train.drop(columns=metadata_columns), columns=CATEGORICAL, dtype=float)
    X_validation = pd.get_dummies(validation.drop(columns=metadata_columns), columns=CATEGORICAL, dtype=float)

    # Screen numeric and one-hot features for leakage using the fitting fold.
    leakage = check_leakage(X_train.assign(failed=y_train))["flagged_columns"]
    excluded = sorted({"work_order_opened", *leakage})
    X_train = X_train.drop(columns=excluded)
    if not len(X_train.columns):
        raise DatasetError("No model features remain after leakage screening", "no_model_features", 422)
    X_validation = X_validation.reindex(columns=X_train.columns, fill_value=0.0)

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scaler = StandardScaler().fit(X_train)
            train_features = scaler.transform(X_train)
            validation_features = scaler.transform(X_validation)
    except (ValueError, FloatingPointError) as error:
        raise DatasetError("Model features are outside the numeric range", "model_numeric_range", 422) from error

    return PreparedFailureSplit(train_features, validation_features, y_train.to_numpy(), y_validation.to_numpy(),
                                X_train.columns.tolist(), excluded)


def calibrated_svm(labels):
    folds = min(5, int(np.bincount(labels).min()))
    if folds < 2:
        raise DatasetError("SVM calibration requires two training observations per class",
                           "insufficient_failure_data", 422)
    return CalibratedClassifierCV(SVC(), cv=folds, ensemble=False)


def perceptron(_labels, *, alpha=DEFAULT_PERCEPTRON_ALPHA):
    return MLPClassifier(hidden_layer_sizes=(32,), alpha=alpha, max_iter=1000, random_state=0)


MODEL_FACTORIES = {
    "decision_tree": lambda labels: DecisionTreeClassifier(max_depth=5, random_state=0),
    "svm": calibrated_svm,
    "knn": lambda labels: KNeighborsClassifier(n_neighbors=min(25, len(labels))),
    "naive_bayes": lambda labels: GaussianNB(),
    "perceptron": perceptron,
}


def failure_metrics(model, features, labels):
    probability = model.predict_proba(features)[:, 1]
    return {"count": len(labels), "failure_rate": float(labels.mean()),
            "pr_auc": float(average_precision_score(labels, probability)),
            "roc_auc": float(roc_auc_score(labels, probability))}


def evaluate_failure_model(registry_table, observations_table, model_name, *, alpha=None):
    split = prepare_failure_split(registry_table, observations_table)
    options = {"alpha": alpha} if alpha is not None else {}
    model = MODEL_FACTORIES[model_name](split.train_labels, **options)
    if model_name == "naive_bayes" and not np.any(split.train_features):
        raise DatasetError("Gaussian naive Bayes requires variation in the training features", "no_model_features", 422)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            model.fit(split.train_features, split.train_labels)
            training = failure_metrics(model, split.train_features, split.train_labels)
            validation = failure_metrics(model, split.validation_features, split.validation_labels)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        raise DatasetError("The model could not evaluate the feature values", "model_numeric_range", 422) from error

    evaluation = {
        "model": model_name,
        "validation_fraction": VALIDATION_FRACTION,
        "feature_columns": split.feature_columns,
        "excluded_features": split.excluded_features,
        "training": training,
        "validation": validation,
        "pr_auc_gap": training["pr_auc"] - validation["pr_auc"],
    }
    if model_name == "perceptron":
        evaluation["alpha"] = model.alpha
    return evaluation
