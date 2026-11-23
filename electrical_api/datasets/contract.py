"""Validation and Arrow types derived from the versioned input contract."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import fastjsonschema
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
DEFINITIONS = json.loads((ROOT / "schemas/electrical-dataset-v1.schema.json").read_text("utf-8"))["$defs"]
TABLES = {
    "meter": "meters", "reading": "readings", "customer_profile": "customer_profiles",
    "customer_event": "customer_events", "equipment": "equipment",
    "equipment_observation": "equipment_observations", "service_case": "service_cases",
}
# The individual record definitions use only keywords supported by this validator.
VALIDATORS = {name: fastjsonschema.compile(DEFINITIONS[name]) for name in (*TABLES, "header")}


class InputError(ValueError):
    def __init__(self, message, *, status=422, path="", line=None):
        super().__init__(message)
        self.status = status
        self.details = {"code": {400: "invalid_input", 415: "unsupported_media_type",
                                 422: "validation_error"}[status], "message": message, "path": path}
        if line is not None:
            self.details["line"] = line


def validate(name, obj, path="", line=None):
    try:
        VALIDATORS[name](obj)
    except fastjsonschema.JsonSchemaValueException as exc:
        suffix = "".join("/" + str(key).replace("~", "~0").replace("/", "~1") for key in exc.path[1:])
        raise InputError(exc.message, path=path + suffix, line=line) from exc


def validate_header(obj, line=None):
    validate("header", obj, line=line)
    if "source_timezone" in obj:
        try:
            ZoneInfo(obj["source_timezone"])
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise InputError("Unknown IANA time zone", path="/source_timezone", line=line) from exc


def arrow_type(spec):
    if spec.get("format") == "date-time":
        return pa.timestamp("us", tz="UTC")
    return {"number": pa.float64(), "integer": pa.int64(), "boolean": pa.bool_(),
            "string": pa.string()}[spec.get("type", "string")]


def table_schema(name):
    definition = DEFINITIONS[name]
    return pa.schema([pa.field(key, arrow_type(spec), nullable=key not in definition["required"])
                      for key, spec in definition["properties"].items() if key != "record_type"])


SCHEMAS = {name: table_schema(name) for name in TABLES}


def normalize(obj, path, line):
    kind = obj.get("record_type") if isinstance(obj, dict) else None
    if not isinstance(kind, str) or kind not in TABLES:
        raise InputError("Expected a known record_type", path=path + "/record_type", line=line)
    validate(kind, obj, path, line)
    row = {}
    for field in SCHEMAS[kind]:
        value = obj.get(field.name)
        try:
            if value is not None:
                if pa.types.is_timestamp(field.type):
                    value = datetime.fromisoformat(value).astimezone(timezone.utc)
                elif pa.types.is_int64(field.type):
                    if not -(2**63) <= value < 2**63:
                        raise ValueError("Integer outside int64 range")
                    value = int(value)
                elif pa.types.is_float64(field.type):
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError("Expected a finite float64")
                elif pa.types.is_string(field.type):
                    value.encode("utf-8")
        except (ValueError, OverflowError) as exc:
            raise InputError(str(exc), path=f"{path}/{field.name}", line=line) from exc
        row[field.name] = value
    return kind, row
