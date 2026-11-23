"""Report quality checks for archived electricity readings."""
from collections import defaultdict
from itertools import groupby
from math import isfinite
from operator import itemgetter
from statistics import mean

import pyarrow.compute as pc

from ..datasets.contract import InputError
from .deduplication import deduplicate_readings


QUALITY_THRESHOLDS = {"silence_hours": 48.0, "frozen_ratio": 0.65, "unit_ratio": 100.0}


def customer_types(meters):
    types = {}
    if meters is not None:
        for row in meters.to_pylist():
            meter_id = row["meter_id"]
            customer_type = row["customer_type"]
            if types.setdefault(meter_id, customer_type) != customer_type:
                types[meter_id] = None
    return types


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return mean(ordered[middle - 1:middle + 1])


def report_unit_check(meter_medians, cohort_values, types, threshold):
    cohort_medians = {kind: median(values) for kind, values in cohort_values.items()}
    evaluated = 0
    suspected = []
    for meter_id, meter_median in meter_medians.items():
        customer_type = types[meter_id]
        cohort_median = cohort_medians[customer_type]
        if cohort_median <= 0:
            continue

        evaluated += 1
        ratio = meter_median / cohort_median
        if ratio > threshold:
            suspected.append({
                "meter_id": meter_id,
                "customer_type": customer_type,
                "meter_median_kwh": meter_median,
                "customer_type_median_kwh": cohort_median,
                "ratio": ratio if isfinite(ratio) else None,
            })
    return {"evaluated_meters": evaluated, "suspected_meters": suspected}


def resolve_window_end(readings, window_end):
    latest_interval = pc.max(readings["interval_start"]).as_py()
    if window_end is None:
        return latest_interval
    if window_end < latest_interval:
        raise InputError("Expected a timestamp at or after the latest interval start", path="/window_end")
    return window_end


def reading_energy_kwh(row):
    return row["energy"] / 1000 if row["energy_unit"] == "Wh" else row["energy"]


def report_reading_quality(raw, meters, thresholds, window_end):
    readings = deduplicate_readings(raw)
    return summarize_reading_quality(readings, meters, thresholds, window_end, raw_count=raw.num_rows)


def summarize_reading_quality(readings, meters, thresholds, window_end, *, raw_count):
    window_end = resolve_window_end(readings, window_end)

    types = customer_types(meters)
    meter_medians = {}
    cohort_values = defaultdict(list)
    silent = []
    frozen = []
    meter_count = 0
    analysis = readings.select(["meter_id", "interval_start", "interval_minutes", "energy", "energy_unit"])
    rows = (row for batch in analysis.to_batches(max_chunksize=8192) for row in batch.to_pylist())
    for meter_id, group in groupby(rows, key=itemgetter("meter_id")):
        meter_count += 1
        customer_type = types.get(meter_id)
        distinct_values = set()
        hourly_values = []
        for reading_count, row in enumerate(group, start=1):
            energy = reading_energy_kwh(row)
            distinct_values.add(energy)
            if customer_type is not None and row["interval_minutes"] == 60:
                hourly_values.append(energy)

        if hourly_values:
            meter_medians[meter_id] = median(hourly_values)
            cohort_values[customer_type].extend(hourly_values)

        last_interval = row["interval_start"]
        hours = (window_end - last_interval).total_seconds() / 3600
        if hours > thresholds["silence_hours"]:
            silent.append({"meter_id": meter_id, "last_interval_start": last_interval.isoformat(),
                           "hours_since_last_reading": hours})

        unique_count = len(distinct_values)
        unique_ratio = unique_count / reading_count
        if unique_ratio < thresholds["frozen_ratio"]:
            frozen.append({"meter_id": meter_id, "unique_values": unique_count,
                           "reading_count": reading_count, "unique_ratio": unique_ratio})

    unit_check = None
    if meters is not None:
        unit_check = report_unit_check(meter_medians, cohort_values, types, thresholds["unit_ratio"])
    return {
        "window_end": window_end.isoformat(),
        "thresholds": thresholds,
        "counts": {
            "readings": raw_count,
            "deduplicated_readings": readings.num_rows,
            "duplicate_readings": raw_count - readings.num_rows,
            "meters_with_readings": meter_count,
        },
        "silent_meters": silent,
        "frozen_meters": frozen,
        "unit_check": unit_check,
    }
