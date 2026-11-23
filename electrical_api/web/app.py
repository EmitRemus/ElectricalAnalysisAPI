"""HTTP entry points for electrical datasets and analysis."""
import os
import re
from datetime import datetime, timezone
from io import BytesIO
from math import isfinite
from pathlib import Path

import pyarrow.parquet as pq
from flask import Flask, jsonify, request, send_file, send_from_directory

from ..customers.associations import DEFAULT_CONFIDENCE, DEFAULT_SUPPORTS, report_churn_rules
from ..customers.checks import PITFALL_COLUMNS, report_customer_pitfalls
from ..customers.segmentation import FEATURES, MAX_CLUSTERS, segment_customers
from ..customers.som import customer_maps
from ..datasets.contract import DEFINITIONS, ROOT, InputError
from ..datasets.ingest import read_dataset
from ..datasets.storage import DatasetError, load_dataset, load_readings, read_dataset_table, save_dataset
from ..equipment.audit import report_equipment_audit
from ..equipment.models import DEFAULT_MODEL, MODEL_FACTORIES, evaluate_failure_model
from ..readings.deduplication import ReadingConflict, deduplicate_readings
from ..readings.priority import report_meter_priority, report_priority_surface
from ..readings.quality import QUALITY_THRESHOLDS, report_reading_quality
from ..readings.regression import REGISTRY_COLUMNS, household_regression
from ..readings.reporting import (
    DEFAULT_VALUE_FIELD, MAX_OFFSET, MAX_PAGE_SIZE, PAGE_SIZE, VALUE_COMPARISONS, VALUE_FIELDS,
    daily_readings, filter_reading_values, readings_page, select_readings, summarize_readings,
)
from ..service_cases.cost import (
    DEFAULT_RADIUS, DEFAULT_RULES, SERVICE_COST_COLUMNS, report_cost_model, report_subtractive_clustering,
)
from .documentation import documentation


def quality_options(args):
    options = QUALITY_THRESHOLDS.copy()
    for name in options:
        if name not in args:
            continue
        try:
            value = float(args[name])
        except ValueError as error:
            raise InputError("Expected a finite number", path=f"/{name}") from error
        if not isfinite(value) or value < 0:
            raise InputError("Expected a finite nonnegative number", path=f"/{name}")
        if name == "frozen_ratio" and value > 1:
            raise InputError("Expected a value from 0 to 1", path=f"/{name}")
        options[name] = value

    return options, timestamp_arg(args.get("window_end"), "window_end")


def timestamp_arg(value, name, *, allow_date=False):
    if value is None:
        return None
    message = "Expected a valid timestamp with an offset or Z"
    if allow_date:
        message += ", or a date YYYY-MM-DD in UTC"
    pattern = DEFINITIONS["reading"]["properties"]["interval_start"]["pattern"]
    try:
        if allow_date and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        if not re.fullmatch(pattern, value):
            raise ValueError("Expected an offset or Z")
        if not value.endswith("Z") and int(value[-2:]) >= 60:
            raise ValueError("Invalid offset minutes")
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except (ValueError, OverflowError) as error:
        raise InputError(message, path=f"/{name}") from error


def single_query_arg(args, name, default=None):
    values = args.getlist(name)
    if len(values) > 1 or (values and not values[0]):
        raise InputError("Expected one nonempty value", path=f"/{name}")
    return values[0] if values else default


def page_query_arg(args, name, default, minimum, maximum):
    raw = single_query_arg(args, name, str(default))
    message = f"Expected an integer from {minimum} to {maximum}"
    try:
        value = int(raw)
    except ValueError as error:
        raise InputError(message, path=f"/{name}") from error
    if not minimum <= value <= maximum:
        raise InputError(message, path=f"/{name}")
    return value


def reading_value_options(args):
    field = single_query_arg(args, "filter_field", DEFAULT_VALUE_FIELD)
    if field not in VALUE_FIELDS:
        raise InputError(f"Expected one of: {', '.join(VALUE_FIELDS)}", path="/filter_field")

    whole_minutes = field == "interval_minutes"
    parse = int if whole_minutes else float
    message = "Expected a signed 64-bit integer" if whole_minutes else "Expected a finite number"
    comparisons = {}
    for name in VALUE_COMPARISONS:
        raw = single_query_arg(args, name)
        if raw is None:
            continue
        try:
            value = parse(raw)
        except ValueError as error:
            raise InputError(message, path=f"/{name}") from error
        valid = -(2**63) <= value < 2**63 if whole_minutes else isfinite(value)
        if not valid:
            raise InputError(message, path=f"/{name}")
        comparisons[name] = value
    return field, comparisons


def cluster_count(args):
    message = f"Expected an integer from 2 to {MAX_CLUSTERS}"
    try:
        k = int(args.get("k", "4"))
    except ValueError as error:
        raise InputError(message, path="/k") from error
    if not 2 <= k <= MAX_CLUSTERS:
        raise InputError(message, path="/k")
    return k


def probability_arg(args, name, default=None, *, positive=False):
    raw = args.get(name)
    if raw is None:
        return default
    interval = "(0, 1]" if positive else "[0, 1]"
    message = f"Expected a number in {interval}"
    try:
        value = float(raw)
    except ValueError as error:
        raise InputError(message, path=f"/{name}") from error
    if not 0 <= value <= 1 or (positive and value == 0):
        raise InputError(message, path=f"/{name}")
    return value


def create_app(data_dir=None):
    app = Flask(__name__, root_path=str(ROOT / "electrical_api"))
    app.register_blueprint(documentation)
    root = Path(data_dir or os.environ.get("ELECTRICAL_DATA_DIR", ROOT / "data"))

    @app.errorhandler(InputError)
    def input_error(error):
        return jsonify(error.details), error.status

    @app.errorhandler(DatasetError)
    def dataset_error(error):
        return jsonify(code=error.code, message=str(error)), error.status

    @app.errorhandler(ReadingConflict)
    def reading_conflict(error):
        return jsonify(code="reading_conflict", message=str(error), **error.details), 409

    @app.post("/v1/datasets")
    def import_dataset():
        header, rows = read_dataset(request.stream, request.mimetype)
        return jsonify(save_dataset(root, header, rows)), 201

    def load_reading_selection(dataset_id):
        meter_id = single_query_arg(request.args, "meter_id")
        start = timestamp_arg(single_query_arg(request.args, "start"), "start", allow_date=True)
        end = timestamp_arg(single_query_arg(request.args, "end"), "end", allow_date=True)
        if start is not None and end is not None and end <= start:
            raise InputError("Expected end after start", path="/end")
        field, comparisons = reading_value_options(request.args)
        dataset, raw = load_readings(root, dataset_id)
        readings = select_readings(raw, meter_id=meter_id, start=start, end=end)
        return dataset.name, filter_reading_values(readings, field, comparisons)

    @app.get("/v1/datasets/<dataset_id>/readings")
    def list_readings(dataset_id):
        limit = page_query_arg(request.args, "limit", PAGE_SIZE, 1, MAX_PAGE_SIZE)
        offset = page_query_arg(request.args, "offset", 0, 0, MAX_OFFSET)
        dataset, readings = load_reading_selection(dataset_id)
        return jsonify(dataset_id=dataset, **readings_page(readings, limit, offset))

    @app.get("/v1/datasets/<dataset_id>/readings/summary")
    def reading_summary(dataset_id):
        dataset, readings = load_reading_selection(dataset_id)
        return jsonify(dataset_id=dataset, **summarize_readings(readings))

    @app.get("/v1/datasets/<dataset_id>/readings/daily")
    def reading_daily(dataset_id):
        dataset, readings = load_reading_selection(dataset_id)
        return jsonify(dataset_id=dataset, **daily_readings(readings))

    @app.get("/v1/datasets/<dataset_id>/readings/deduplicated")
    def download_deduplicated_readings(dataset_id):
        _, raw = load_readings(root, dataset_id)
        readings = deduplicate_readings(raw)
        output = BytesIO()
        pq.write_table(readings, output)
        output.seek(0)
        return send_file(output, mimetype="application/vnd.apache.parquet",
                         as_attachment=True, download_name="readings-deduplicated.parquet")

    @app.get("/v1/datasets/<dataset_id>/readings/quality")
    def reading_quality(dataset_id):
        thresholds, window_end = quality_options(request.args)
        dataset, readings = load_readings(root, dataset_id)
        meters = read_dataset_table(dataset, "meter", columns=["meter_id", "customer_type"])
        report = report_reading_quality(readings, meters, thresholds, window_end)
        return jsonify(dataset_id=dataset.name, **report)

    @app.get("/v1/datasets/<dataset_id>/readings/household-regression")
    def reading_household_regression(dataset_id):
        thresholds, window_end = quality_options(request.args)
        dataset, readings = load_readings(root, dataset_id)
        meters = read_dataset_table(dataset, "meter", columns=REGISTRY_COLUMNS)
        model = household_regression(readings, meters, thresholds, window_end)
        return jsonify(dataset_id=dataset.name, **model)

    @app.get("/v1/datasets/<dataset_id>/customers/segmentation")
    def customer_segmentation(dataset_id):
        k = cluster_count(request.args)
        dataset = load_dataset(root, dataset_id)
        profiles = read_dataset_table(dataset, "customer_profile")
        return jsonify(dataset_id=dataset.name, **segment_customers(profiles, k))

    @app.get("/v1/datasets/<dataset_id>/customers/som")
    def customer_som(dataset_id):
        k = cluster_count(request.args)
        dataset = load_dataset(root, dataset_id)
        profiles = read_dataset_table(dataset, "customer_profile", columns=["customer_id", *FEATURES])
        return jsonify(dataset_id=dataset.name, **customer_maps(profiles, k))

    @app.get("/v1/datasets/<dataset_id>/customers/churn-rules")
    def customer_churn_rules(dataset_id):
        support = probability_arg(request.args, "min_support", positive=True)
        confidence = probability_arg(request.args, "min_confidence", DEFAULT_CONFIDENCE)
        supports = DEFAULT_SUPPORTS if support is None else (support,)
        dataset = load_dataset(root, dataset_id)
        profiles = read_dataset_table(dataset, "customer_profile", columns=["customer_id", "churned"])
        events = read_dataset_table(dataset, "customer_event", columns=["customer_id", "event_type"])
        report = report_churn_rules(profiles, events, supports, confidence)
        return jsonify(dataset_id=dataset.name, **report)

    @app.get("/v1/datasets/<dataset_id>/customers/pitfalls")
    def customer_pitfalls(dataset_id):
        dataset = load_dataset(root, dataset_id)
        profiles = read_dataset_table(dataset, "customer_profile", columns=PITFALL_COLUMNS)
        return jsonify(dataset_id=dataset.name, **report_customer_pitfalls(profiles))

    @app.get("/v1/datasets/<dataset_id>/equipment/audit")
    def equipment_audit(dataset_id):
        dataset = load_dataset(root, dataset_id)
        registry = read_dataset_table(dataset, "equipment")
        observations = read_dataset_table(dataset, "equipment_observation")
        return jsonify(dataset_id=dataset.name, **report_equipment_audit(registry, observations))

    @app.get("/v1/datasets/<dataset_id>/equipment/model-evaluation")
    def equipment_model_evaluation(dataset_id):
        model = request.args.get("model", DEFAULT_MODEL)
        if model not in MODEL_FACTORIES:
            raise InputError(f"Expected one of: {', '.join(MODEL_FACTORIES)}", path="/model")
        alpha_values = request.args.getlist("alpha")
        alpha = None
        if alpha_values:
            if model != "perceptron":
                raise InputError("alpha is supported only for the perceptron", path="/alpha")
            message = "Expected one finite nonnegative alpha"
            if len(alpha_values) != 1:
                raise InputError(message, path="/alpha")
            try:
                alpha = float(alpha_values[0])
            except ValueError as error:
                raise InputError(message, path="/alpha") from error
            if not isfinite(alpha) or alpha < 0:
                raise InputError(message, path="/alpha")
        dataset = load_dataset(root, dataset_id)
        registry = read_dataset_table(dataset, "equipment")
        observations = read_dataset_table(dataset, "equipment_observation")
        evaluation = evaluate_failure_model(registry, observations, model, alpha=alpha)
        return jsonify(dataset_id=dataset.name, **evaluation)

    @app.get("/v1/datasets/<dataset_id>/readings/inspection-priority")
    def meter_inspection_priority(dataset_id):
        meter_ids = request.args.getlist("meter_id")
        if len(meter_ids) != 1 or not meter_ids[0]:
            raise InputError("Expected one nonempty meter identifier", path="/meter_id")
        window_end = timestamp_arg(request.args.get("window_end"), "window_end")
        dataset, readings = load_readings(root, dataset_id)
        return jsonify(dataset_id=dataset.name, **report_meter_priority(readings, meter_ids[0], window_end))

    @app.get("/v1/readings/inspection-priority/surface")
    def meter_priority_surface():
        return jsonify(report_priority_surface())

    @app.get("/v1/datasets/<dataset_id>/service-cases/subtractive-clustering")
    def service_subtractive_clustering(dataset_id):
        radii = request.args.getlist("radius") or [str(DEFAULT_RADIUS)]
        message = "Expected one finite positive radius"
        if len(radii) != 1:
            raise InputError(message, path="/radius")
        try:
            radius = float(radii[0])
        except ValueError as error:
            raise InputError(message, path="/radius") from error
        if not isfinite(radius) or radius <= 0:
            raise InputError(message, path="/radius")
        dataset = load_dataset(root, dataset_id)
        cases = read_dataset_table(dataset, "service_case", columns=SERVICE_COST_COLUMNS)
        return jsonify(dataset_id=dataset.name, **report_subtractive_clustering(cases, radius))

    @app.get("/v1/datasets/<dataset_id>/service-cases/cost-model")
    def service_cost_model(dataset_id):
        settings = request.args.getlist("rules") or [str(DEFAULT_RULES)]
        message = "Expected one integer rule count from 2 to 6"
        if len(settings) != 1:
            raise InputError(message, path="/rules")
        try:
            count = int(settings[0])
        except ValueError as error:
            raise InputError(message, path="/rules") from error
        if not 2 <= count <= 6:
            raise InputError(message, path="/rules")
        dataset = load_dataset(root, dataset_id)
        cases = read_dataset_table(dataset, "service_case", columns=SERVICE_COST_COLUMNS)
        return jsonify(dataset_id=dataset.name, **report_cost_model(cases, count))

    @app.get("/docs/openapi.json")
    def specification():
        return send_from_directory(ROOT / "docs", "openapi.json")

    @app.get("/schemas/electrical-dataset-v1.schema.json")
    def schema():
        return send_from_directory(ROOT / "schemas", "electrical-dataset-v1.schema.json")

    return app
