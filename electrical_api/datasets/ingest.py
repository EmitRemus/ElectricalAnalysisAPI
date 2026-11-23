"""Decode the two HTTP representations into the same validated row stream."""
import json
from itertools import chain

from .contract import InputError, normalize, validate_header


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Repeated JSON key: {key}")
        obj[key] = value
    return obj


def invalid_constant(value):
    raise ValueError(f"Invalid JSON number: {value}")


def decode(raw, line=None):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=invalid_constant)
    except (ValueError, RecursionError) as exc:
        raise InputError(str(exc), status=400, line=line) from exc


def read_dataset(stream, media_type):
    if media_type == "application/json":
        obj = decode(stream.read())
        if not isinstance(obj, dict) or "records" not in obj:
            raise InputError("Expected a dataset with records", path="/records")
        records = obj.pop("records")
        if not isinstance(records, list) or not records:
            raise InputError("Expected a nonempty array", path="/records")
        if "record_type" in obj:
            raise InputError("Unexpected field", path="/record_type")
        validate_header({"record_type": "dataset", **obj})
        return obj, (normalize(r, f"/records/{i}", None) for i, r in enumerate(records))

    if media_type != "application/x-ndjson":
        raise InputError("Use application/json or application/x-ndjson", status=415)
    lines = iter(stream.readline, b"")
    header = decode(next(lines, b""), 1)
    if not isinstance(header, dict) or header.get("record_type") != "dataset":
        raise InputError("Expected a dataset header", status=400, line=1)
    validate_header(header, 1)
    header.pop("record_type")
    first = next(lines, None)
    if first is None:
        raise InputError("Expected at least one record", status=400, line=2)

    def rows():
        for index, raw in enumerate(chain((first,), lines), start=2):
            record = decode(raw, index)
            if isinstance(record, dict) and record.get("record_type") == "dataset":
                raise InputError("Unexpected second header", status=400, line=index)
            yield normalize(record, "", index)

    return header, rows()
