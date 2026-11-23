"""Stage a complete dataset privately before publishing it in one rename."""
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from .contract import SCHEMAS, TABLES


class DatasetError(ValueError):
    def __init__(self, message, code, status):
        super().__init__(message)
        self.code = code
        self.status = status


def read_dataset_table(dataset: Path, kind, *, columns=None):
    source = dataset / f"{TABLES[kind]}.parquet"
    return pq.read_table(source, columns=columns) if source.exists() else None


def load_dataset(root: Path, dataset_id):
    try:
        dataset = root / str(UUID(dataset_id))
    except ValueError as error:
        raise DatasetError("Dataset not found", "dataset_not_found", 404) from error
    if not (dataset / "metadata.json").is_file():
        raise DatasetError("Dataset not found", "dataset_not_found", 404)
    return dataset


def load_readings(root: Path, dataset_id):
    dataset = load_dataset(root, dataset_id)
    readings = read_dataset_table(dataset, "reading")
    if readings is None or readings.num_rows == 0:
        raise DatasetError("Dataset has no readings table", "readings_missing", 422)
    return dataset, readings


def save_dataset(root: Path, metadata, rows, batch_size=8192):
    root.mkdir(parents=True, exist_ok=True)
    dataset_id = str(uuid4())
    counts = Counter()
    buffers = defaultdict(list)
    writers = {}

    with TemporaryDirectory(dir=root, prefix=".import-") as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        with ExitStack() as stack:
            def flush(kind):
                if kind not in writers:
                    filename = staging / (TABLES[kind] + ".parquet")
                    writers[kind] = stack.enter_context(pq.ParquetWriter(filename, SCHEMAS[kind]))
                writers[kind].write_table(pa.Table.from_pylist(buffers[kind], schema=SCHEMAS[kind]))
                buffers[kind].clear()

            for kind, row in rows:
                buffers[kind].append(row)
                counts[kind] += 1
                if len(buffers[kind]) >= batch_size:
                    flush(kind)
            for kind, buffer in buffers.items():
                if buffer:
                    flush(kind)

        receipt = {"dataset_id": dataset_id, "counts": dict(counts)}
        (staging / "metadata.json").write_text(json.dumps({**metadata, **receipt}), encoding="utf-8")
        staging.rename(root / dataset_id)
    return receipt
