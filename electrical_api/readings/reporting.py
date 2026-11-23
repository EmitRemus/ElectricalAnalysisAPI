"""Browse deduplicated meter readings and summarize observed energy."""
from collections import defaultdict
from fractions import Fraction
from math import fsum
from statistics import mean

import pyarrow.compute as pc

from ..datasets.storage import DatasetError
from .deduplication import deduplicate_readings
from .quality import median, reading_energy_kwh


PAGE_SIZE = 100
MAX_PAGE_SIZE = 1000
MAX_OFFSET = 2**63 - 1
DEFAULT_VALUE_FIELD = "energy_kwh"
VALUE_FIELDS = (DEFAULT_VALUE_FIELD, "interval_minutes")
VALUE_COMPARISONS = {
    "gt": pc.greater,
    "gte": pc.greater_equal,
    "lt": pc.less,
    "lte": pc.less_equal,
    "eq": pc.equal,
}


def select_readings(raw, *, meter_id=None, start=None, end=None):
    if meter_id is not None:
        raw = raw.filter(pc.equal(raw["meter_id"], meter_id))
    if start is not None:
        raw = raw.filter(pc.greater_equal(raw["interval_start"], start))
    if end is not None:
        raw = raw.filter(pc.less(raw["interval_start"], end))
    return deduplicate_readings(raw)


def filter_reading_values(readings, field, comparisons):
    if not comparisons:
        return readings

    if field == "energy_kwh":
        energy = readings["energy"]
        values = pc.if_else(pc.equal(readings["energy_unit"], "Wh"), pc.divide(energy, 1000.0), energy)
    else:
        values = readings[field]

    mask = None
    for operator, threshold in comparisons.items():
        matches = VALUE_COMPARISONS[operator](values, threshold)
        mask = matches if mask is None else pc.and_(mask, matches)
    return readings.filter(mask)


def readings_page(readings, limit, offset):
    ordered = readings.sort_by([("interval_start", "ascending"), ("meter_id", "ascending")])
    page = ordered.slice(min(offset, ordered.num_rows), limit)
    rows = [
        {"meter_id": row["meter_id"], "interval_start": row["interval_start"].isoformat(),
         "interval_minutes": row["interval_minutes"], "received_at": row["received_at"].isoformat(),
         "energy_kwh": reading_energy_kwh(row)}
        for row in page.to_pylist()
    ]
    next_offset = offset + len(rows)
    return {"total": ordered.num_rows, "limit": limit, "offset": offset,
            "next_offset": next_offset if next_offset < ordered.num_rows else None, "readings": rows}


def energy_total(values):
    try:
        return fsum(values)
    except OverflowError:
        # Opposite-sign readings can cancel after an intermediate sum overflows.
        try:
            return float(sum(map(Fraction, values)))
        except OverflowError as error:
            raise DatasetError("Total energy is outside the numeric range", "readings_numeric_range", 422) from error


def energy_statistics(values):
    if not values:
        return {"reading_count": 0, "total_kwh": 0.0, "average_kwh": None, "median_kwh": None,
                "min_kwh": None, "max_kwh": None}
    return {"reading_count": len(values), "total_kwh": energy_total(values),
            "average_kwh": mean(values), "median_kwh": median(values),
            "min_kwh": min(values), "max_kwh": max(values)}


def summarize_readings(readings):
    energy = readings.select(["energy", "energy_unit"])
    values = [
        reading_energy_kwh(row)
        for batch in energy.to_batches(max_chunksize=8192)
        for row in batch.to_pylist()
    ]
    return {"meter_count": pc.count_distinct(readings["meter_id"]).as_py(), **energy_statistics(values)}


def daily_readings(readings):
    values_by_day = defaultdict(list)
    daily_columns = readings.select(["interval_start", "energy", "energy_unit"])
    for batch in daily_columns.to_batches(max_chunksize=8192):
        for row in batch.to_pylist():
            values_by_day[row["interval_start"].date()].append(reading_energy_kwh(row))
    return {"days": [
        {"date": day.isoformat(), **energy_statistics(values)}
        for day, values in sorted(values_by_day.items())
    ]}
