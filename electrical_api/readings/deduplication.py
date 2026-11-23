"""Derive one reading per meter and interval start from archived transmissions."""
import pyarrow as pa


class ReadingConflict(ValueError):
    def __init__(self, row):
        super().__init__("Latest transmissions disagree for this meter and interval")
        self.details = {
            "meter_id": row["meter_id"],
            "interval_start": row["interval_start"].isoformat(),
            "received_at": row["received_at"].isoformat(),
        }


def deduplicate_readings(readings: pa.Table) -> pa.Table:
    ordered = readings.sort_by([
        ("meter_id", "ascending"),
        ("interval_start", "ascending"),
        ("received_at", "descending"),
    ])
    selected = []
    current_key = None
    latest = None
    offset = 0

    for batch in ordered.to_batches(max_chunksize=8192):
        for index, row in enumerate(batch.to_pylist(), start=offset):
            key = (row["meter_id"], row["interval_start"])
            if key != current_key:
                selected.append(index)
                current_key = key
                latest = row
            elif row["received_at"] == latest["received_at"] and row != latest:
                raise ReadingConflict(row)
        offset += batch.num_rows

    return ordered.take(pa.array(selected, type=pa.int64()))
