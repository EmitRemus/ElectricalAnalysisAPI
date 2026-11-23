"""Subtractive clustering and the Sugeno cost model from coursework steps 5–8."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from skfuzzy.cluster import cmeans

from ..datasets.storage import DatasetError


FEATURES = ["reactive_energy_mvarh", "heat_energy_gcal", "contracted_power_kw", "outage_hours"]
TARGET = "service_cost_kuah"
SERVICE_COST_COLUMNS = ["case_id", *FEATURES, TARGET]
DEFAULT_RADIUS = 0.5
DEFAULT_RULES = 3
TEST_FRACTION = 0.25


@dataclass
class CostSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    t_train: np.ndarray
    t_test: np.ndarray
    y_min: float
    y_span: float
    prediction_ids: list[str]
    x_prediction: np.ndarray


def case_order(case_id):
    source, separator, suffix = case_id.rpartition(":")
    if separator and suffix.isascii() and suffix.isdecimal():
        digits = suffix.lstrip("0")
        return source, len(digits), digits, case_id
    return case_id, -1, "", case_id


def prepare_cost_split(table):
    if table is None or not table.num_rows:
        raise DatasetError("Dataset has no service cases table", "service_cases_missing", 422)
    cases = table.to_pandas().set_index("case_id")
    if not cases.index.is_unique:
        raise DatasetError("Expected one record per case_id", "duplicate_case_id", 422)
    cases = cases.loc[sorted(cases.index, key=case_order)]
    labeled = cases.loc[cases[TARGET].notna()]
    prediction = cases.loc[cases[TARGET].isna()]
    if len(labeled) < 3:
        raise DatasetError("At least three labeled cases are needed for a train/test split",
                           "insufficient_cost_data", 422)

    X_train, X_test, y_train, y_test = train_test_split(
        labeled[FEATURES].to_numpy(), labeled[TARGET].to_numpy(), test_size=TEST_FRACTION, random_state=0,
    )
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            x_min = X_train.min(axis=0)
            x_span = X_train.max(axis=0) - x_min
            x_span[x_span == 0] = 1
            y_min = float(y_train.min())
            y_span = float(y_train.max() - y_min)
            if y_span == 0:
                raise DatasetError("Training costs must vary to normalize the target", "constant_service_cost", 422)
            return CostSplit(
                (X_train - x_min) / x_span, (X_test - x_min) / x_span,
                (y_train - y_min) / y_span, (y_test - y_min) / y_span, y_min, y_span,
                prediction.index.tolist(), (prediction[FEATURES].to_numpy() - x_min) / x_span,
            )
    except FloatingPointError as error:
        raise DatasetError("Service case values are outside the numeric range", "cost_model_numeric_range", 422) from error


def subclust(data, radius, accept=0.5, reject=0.15):
    a, b = 4 / radius ** 2, 4 / (1.5 * radius) ** 2
    potential = np.exp(-a * ((data[:, None, :] - data[None, :, :]) ** 2).sum(axis=2)).sum(axis=1)
    first, centers = potential.max(), []
    while True:
        k = int(potential.argmax())
        top = potential[k]
        if top < reject * first:
            break
        if top < accept * first:
            nearest = min(np.sqrt(((data[k] - data[c]) ** 2).sum()) for c in centers)
            if nearest / radius + top / first < 1:
                potential[k] = 0
                continue
        centers.append(k)
        potential = np.maximum(potential - top * np.exp(-b * ((data - data[k]) ** 2).sum(axis=1)), 0)
    return data[centers]


def report_subtractive_clustering(table, radius):
    split = prepare_cost_split(table)
    data = np.column_stack([split.x_train, split.t_train])
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            centers = subclust(data, radius)
    except (OverflowError, ZeroDivisionError, FloatingPointError) as error:
        raise DatasetError("Radius is outside the numeric range", "cost_model_numeric_range", 422) from error
    return {"radius": radius, "rule_count": len(centers), "training_count": len(split.x_train),
            "test_count": len(split.x_test)}


def unpack(theta, count, feature_count):
    k = count * feature_count
    return theta[:k].reshape(count, -1), np.exp(theta[k:2 * k]).reshape(count, -1), theta[2 * k:]


def activation(theta, x, count):
    centers, widths, _ = unpack(theta, count, x.shape[1])
    z = -0.5 * (((x[:, None, :] - centers[None]) / widths[None]) ** 2).sum(axis=2)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def predict_normalized(theta, x, count):
    return activation(theta, x, count) @ unpack(theta, count, x.shape[1])[2]


def rmse(theta, x, target, count):
    return float(np.sqrt(np.mean((predict_normalized(theta, x, count) - target) ** 2)))


def pack(centers, widths, split):
    count, features = centers.shape
    theta = np.concatenate([centers.ravel(), np.log(widths.ravel()), np.zeros(count)])
    weights = activation(theta, split.x_train, count)
    theta[2 * count * features:] = np.linalg.solve(
        weights.T @ weights + 1e-8 * np.eye(count), weights.T @ split.t_train,
    )
    return theta


def cmeans_rules(split, count):
    data = np.column_stack([split.x_train, split.t_train])
    # Match the notebook's seed without changing NumPy's global random state.
    init = np.random.RandomState(0).rand(count, len(data))
    init /= init.sum(axis=0, keepdims=True)
    centers, memberships, *_ = cmeans(data.T, c=count, m=2.0, error=1e-6, maxiter=1000, init=init)
    centers = centers[:, :split.x_train.shape[1]]
    widths = np.zeros_like(centers)
    for k in range(count):
        weights = memberships[k] ** 2
        widths[k] = np.sqrt((weights @ (split.x_train - centers[k]) ** 2) / weights.sum())
    return pack(centers, np.clip(widths, 0.08, 0.8), split)


def tune(theta, split, count):
    n = count * split.x_train.shape[1]
    bounds = [(-0.05, 1.05)] * n + [(np.log(0.08), np.log(0.8))] * n + [(-0.05, 1.05)] * count
    return minimize(rmse, np.clip(theta, *np.array(bounds).T), args=(split.x_train, split.t_train, count),
                    method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})


def rmse_summary(train_rmse, test_rmse):
    return {"train_rmse_kuah": train_rmse, "test_rmse_kuah": test_rmse,
            "rmse_gap_kuah": test_rmse - train_rmse}


def model_errors(theta, split, count):
    train = rmse(theta, split.x_train, split.t_train, count) * split.y_span
    test = rmse(theta, split.x_test, split.t_test, count) * split.y_span
    return rmse_summary(train, test)


def report_cost_model(table, count):
    split = prepare_cost_split(table)
    if count > len(split.x_train):
        raise DatasetError("The training split must contain at least as many cases as rules",
                           "insufficient_cost_data", 422)

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean = float(split.t_train.mean())
            train = np.sqrt(np.mean((split.t_train - mean) ** 2)) * split.y_span
            test = np.sqrt(np.mean((split.t_test - mean) ** 2)) * split.y_span
            baseline = {"mean_prediction_kuah": split.y_min + mean * split.y_span,
                        **rmse_summary(float(train), float(test))}
            theta = cmeans_rules(split, count)
            initial = model_errors(theta, split, count)
            optimized = tune(theta, split, count)
            tuned = model_errors(optimized.x, split, count)
            predictions = []
            if split.prediction_ids:
                normalized = predict_normalized(optimized.x, split.x_prediction, count)
                predictions = (split.y_min + normalized * split.y_span).tolist()
            values = [*baseline.values(), *initial.values(), *tuned.values(), *predictions]
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite cost model output")
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise DatasetError("The cost model is outside the numeric range", "cost_model_numeric_range", 422) from error

    return {"rule_count": count, "training_count": len(split.x_train), "test_count": len(split.x_test),
            "baseline": baseline, "initial": initial, "tuned": tuned,
            "optimization": {"converged": bool(optimized.success), "iterations": int(optimized.nit)},
            "predictions": [{"case_id": case_id, "predicted_service_cost_kuah": value}
                            for case_id, value in zip(split.prediction_ids, predictions)]}
