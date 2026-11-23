"""Train a linear regression for daily household consumption."""
from math import isfinite

import pyarrow.compute as pc
from sklearn.linear_model import LinearRegression

from ..datasets.storage import DatasetError
from .deduplication import deduplicate_readings
from .quality import resolve_window_end, summarize_reading_quality


REGISTRY_COLUMNS = ["meter_id", "customer_type", "electric_heating", "tariff"]


def household_regression(raw, meters, thresholds, window_end):
    if meters is None or meters.num_rows == 0:
        raise DatasetError("Dataset has no meters table", "meters_missing", 422)

    readings = deduplicate_readings(raw)
    window_end = resolve_window_end(readings, window_end)
    hourly = readings.filter(pc.equal(readings["interval_minutes"], 60))
    if hourly.num_rows == 0:
        raise DatasetError("Dataset has no hourly readings", "hourly_readings_missing", 422)

    registry = meters.select(REGISTRY_COLUMNS).to_pandas().drop_duplicates()
    if registry["meter_id"].duplicated().any():
        raise DatasetError("Meter registry has conflicting attributes", "meter_registry_conflict", 422)

    quality = summarize_reading_quality(hourly, meters, thresholds, window_end, raw_count=hourly.num_rows)
    silent = {meter["meter_id"] for meter in quality["silent_meters"]}
    suspected = {meter["meter_id"] for meter in quality["unit_check"]["suspected_meters"]}

    result = hourly.select(["meter_id", "energy", "energy_unit"]).to_pandas()
    result = result.merge(registry, on="meter_id", how="inner", validate="m:1")
    in_wh = result["energy_unit"] == "Wh"
    rescaled = result["meter_id"].isin(suspected) & ~in_wh
    rescaled_ids = set(result.loc[rescaled, "meter_id"])
    result["consumption_kwh"] = result["energy"]
    result.loc[in_wh | rescaled, "consumption_kwh"] /= 1000

    # Check corrected values before fitting one row per household.
    consumption = result.groupby("meter_id")["consumption_kwh"]
    unique_ratio = consumption.nunique() / consumption.size()
    frozen = unique_ratio[unique_ratio < thresholds["frozen_ratio"]].index
    homes = result[(result["customer_type"] == "household")
                   & (~result["meter_id"].isin(silent))
                   & (~result["meter_id"].isin(frozen))
                   & result["tariff"].isin(["single", "dual_zone"])]

    per_meter = homes.groupby("meter_id").agg(
        kwh_per_day=("consumption_kwh", "mean"),
        electric_heating=("electric_heating", "first"),
        tariff=("tariff", "first"),
    )
    if len(per_meter) < 3:
        raise DatasetError("Regression requires at least three eligible household meters",
                           "insufficient_regression_data", 422)

    per_meter["kwh_per_day"] *= 24
    per_meter["dual_zone"] = (per_meter["tariff"] == "dual_zone").astype(int)
    x, y = per_meter[["electric_heating", "dual_zone"]], per_meter["kwh_per_day"]
    if not y.map(isfinite).all():
        raise DatasetError("Daily energy is outside the numeric range", "regression_numeric_range", 422)

    try:
        model = LinearRegression().fit(x, y)
        r_squared = float(model.score(x, y))
    except (ValueError, ArithmeticError) as error:
        raise DatasetError("Regression results are outside the numeric range",
                           "regression_numeric_range", 422) from error
    if model.rank_ != x.shape[1]:
        raise DatasetError("Electric heating and tariff must vary independently", "insufficient_regression_data", 422)

    intercept = float(model.intercept_)
    heating, dual_zone = map(float, model.coef_)
    if not all(map(isfinite, (intercept, heating, dual_zone, r_squared))):
        raise DatasetError("Regression results are outside the numeric range", "regression_numeric_range", 422)

    return {
        "window_end": window_end.isoformat(),
        "thresholds": thresholds,
        "meter_count": len(per_meter),
        "rescaled_meter_ids": sorted(rescaled_ids.intersection(per_meter.index)),
        "intercept": intercept,
        "coefficients": {"electric_heating": heating, "dual_zone": dual_zone},
        "r_squared": r_squared,
    }
