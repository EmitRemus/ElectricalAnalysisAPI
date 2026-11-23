"""Mamdani inspection priority based on the quality of meter readings."""
import numpy as np
import pyarrow.compute as pc
from skfuzzy import defuzz, trimf

from ..datasets.storage import DatasetError
from .deduplication import deduplicate_readings
from .quality import QUALITY_THRESHOLDS, reading_energy_kwh, resolve_window_end


SILENCE_REFERENCE_HOURS = QUALITY_THRESHOLDS["silence_hours"]
REPETITION_REFERENCE = 1 - QUALITY_THRESHOLDS["frozen_ratio"]
MAX_SILENCE_HOURS = 2 * SILENCE_REFERENCE_HOURS
PRIORITY_SCALE = np.linspace(0, 100, 201)
LOW_PRIORITY = trimf(PRIORITY_SCALE, [0, 0, 50])
HIGH_PRIORITY = trimf(PRIORITY_SCALE, [50, 100, 100])


def calculate_priority(hours, repetition):
    delay = float(np.interp(hours, [0, SILENCE_REFERENCE_HOURS, MAX_SILENCE_HOURS], [0, 0.5, 1]))
    repeats = float(np.interp(repetition, [0, REPETITION_REFERENCE, 1], [0, 0.5, 1]))
    routine = min(1 - delay, 1 - repeats)
    escalate = max(delay, repeats)
    # Mamdani clips each consequent by its rule strength and aggregates them by maximum.
    membership = np.maximum(np.minimum(routine, LOW_PRIORITY), np.minimum(escalate, HIGH_PRIORITY))
    return float(defuzz(PRIORITY_SCALE, membership, "centroid")), {"routine": routine, "escalate": escalate}


def report_meter_priority(raw, meter_id, window_end):
    selected = raw.filter(pc.equal(raw["meter_id"], meter_id))
    if not selected.num_rows:
        raise DatasetError("No readings for this meter", "meter_readings_missing", 404)

    readings = deduplicate_readings(selected)
    window_end = resolve_window_end(raw, window_end)
    last_interval = readings["interval_start"][-1].as_py()
    energy = readings.select(["energy", "energy_unit"])
    unique_values = len({
        reading_energy_kwh(row)
        for batch in energy.to_batches(max_chunksize=8192)
        for row in batch.to_pylist()
    })
    unique_ratio = unique_values / readings.num_rows
    hours = (window_end - last_interval).total_seconds() / 3600
    repetition = 1 - unique_ratio
    score, strengths = calculate_priority(hours, repetition)
    return {
        "meter_id": meter_id,
        "window_end": window_end.isoformat(),
        "last_interval_start": last_interval.isoformat(),
        "hours_since_last_reading": hours,
        "reading_count": readings.num_rows,
        "unique_values": unique_values,
        "unique_ratio": unique_ratio,
        "repetition_ratio": repetition,
        "priority_score": score,
        "rule_strengths": strengths,
    }


def report_priority_surface():
    hours = [MAX_SILENCE_HOURS * step / 20 for step in range(21)]
    repetition = [step / 20 for step in range(21)]
    scores = [[calculate_priority(delay, ratio)[0] for delay in hours] for ratio in repetition]
    return {"hours_since_last_reading": hours, "repetition_ratio": repetition, "priority_scores": scores}
