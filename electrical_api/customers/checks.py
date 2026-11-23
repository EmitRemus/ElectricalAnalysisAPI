"""Check three statistical pitfalls in customer profiles."""
from math import isfinite

import numpy as np

from .profiles import customer_frame


PITFALL_COLUMNS = [
    "customer_id", "district", "voltage_dev_mean", "loss_index_mean", "meter_type",
    "seasonality", "annual_charge_uah", "avg_price_uah_kwh", "churned",
]


def finite_number(value):
    value = float(value)
    return value if isfinite(value) else None


def pearson(left, right):
    if left.nunique() < 2 or right.nunique() < 2:
        return None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return finite_number(left.corr(right))
    except FloatingPointError:
        return None


def seasonality_mean(series):
    try:
        with np.errstate(over="raise", invalid="raise"):
            return finite_number(series.mean())
    except FloatingPointError:
        return None


def check_district_correlations(customers):
    total_threshold = 0.2
    district_threshold = 0.1
    r_total = pearson(customers["voltage_dev_mean"], customers["loss_index_mean"])
    r_district = {
        district: pearson(group["voltage_dev_mean"], group["loss_index_mean"])
        for district, group in customers.groupby("district")
    }
    detected = None
    if r_total is not None and all(value is not None for value in r_district.values()):
        detected = abs(r_total) > total_threshold and all(
            abs(value) < district_threshold for value in r_district.values()
        )
    return {
        "detected": detected,
        "total_correlation": r_total,
        "district_correlations": r_district,
        "total_threshold": total_threshold,
        "district_threshold": district_threshold,
    }


def check_meter_seasonality(customers):
    lower_threshold, upper_threshold = 0.8, 1.2
    meter_old = customers["meter_type"] == "IND_1985"
    mean_old = seasonality_mean(customers.loc[meter_old, "seasonality"])
    mean_other = seasonality_mean(customers.loc[~meter_old, "seasonality"])
    ratio = None
    if mean_old is not None and mean_other is not None and mean_other != 0:
        ratio = finite_number(mean_old / mean_other)
    return {
        "detected": None if ratio is None else ratio < lower_threshold or ratio > upper_threshold,
        "mean_ind_1985": mean_old,
        "mean_other": mean_other,
        "ratio": ratio,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
    }


def price_churn_group(customers):
    labeled = customers.loc[customers["churned"].notna()]
    return {
        "customer_count": len(customers),
        "labeled_count": len(labeled),
        "correlation": pearson(labeled["avg_price_uah_kwh"], labeled["churned"].astype(float)),
    }


def check_price_churn(customers):
    sorted_price = customers.sort_values("annual_charge_uah", ascending=False, kind="stable")
    top_count = len(customers) * 3 // 10
    top = price_churn_group(sorted_price.iloc[:top_count])
    other = price_churn_group(sorted_price.iloc[top_count:])
    r_top, r_other = top["correlation"], other["correlation"]
    detected = None
    if r_top is not None and r_other is not None:
        detected = r_top * r_other < 0
    return {"detected": detected, "top_fraction": 0.30, "top": top, "other": other}


def report_customer_pitfalls(table):
    customers = customer_frame(table)
    return {
        "customer_count": len(customers),
        "t1": check_district_correlations(customers),
        "t2": check_meter_seasonality(customers),
        "t3": check_price_churn(customers),
    }
