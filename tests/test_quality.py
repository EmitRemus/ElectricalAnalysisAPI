import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from electrical_api.app import create_app
from electrical_api.datasets.contract import ROOT


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
SAMPLE = json.loads((ROOT / "examples/dataset.json").read_bytes())
METER = next(row for row in SAMPLE["records"] if row["record_type"] == "meter")


def reading(meter_id, hour, energy, *, unit="kWh", minutes=60, delay=60):
    start = START + timedelta(hours=hour)
    return {
        "record_type": "reading",
        "meter_id": meter_id,
        "interval_start": start.isoformat(),
        "interval_minutes": minutes,
        "received_at": (start + timedelta(minutes=delay)).isoformat(),
        "energy": energy,
        "energy_unit": unit,
    }


class ReadingQualityTests(unittest.TestCase):
    def setUp(self):
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        self.client = create_app(storage.name).test_client()

    def import_records(self, records, media="application/json"):
        if media == "application/x-ndjson":
            header = {"record_type": "dataset", "schema_version": "1.0"}
            body = "\n".join(json.dumps(row) for row in [header, *records])
        else:
            body = json.dumps({"schema_version": "1.0", "records": records})
        with self.client.post("/v1/datasets", data=body, content_type=media) as response:
            result = response.get_json()
            self.assertEqual(response.status_code, 201, result)
        return f"/v1/datasets/{result['dataset_id']}/readings/quality"

    def read_quality(self, path, **params):
        with self.client.get(path, query_string=params) as response:
            return response.status_code, response.get_json()

    def test_default_quality_report_for_both_formats(self):
        records = [
            reading("A", 0, 1.2), reading("D", 50, 1000, minutes=30),
            reading("A", 0, 0.5, delay=10), reading("A", 1, 1200, unit="Wh"),
            reading("B", 50, 160), reading("C", 2, 1200, unit="Wh"),
            reading("E", 50, 1000),
            *(dict(METER, meter_id=meter) for meter in "ABCD"),
        ]
        for media in ["application/json", "application/x-ndjson"]:
            with self.subTest(media=media):
                path = self.import_records(records, media)
                status, result = self.read_quality(path)
                self.assertEqual(status, 200, result)
                self.assertEqual(result["dataset_id"], path.split("/")[3])
                self.assertEqual(result["window_end"], "2026-01-03T02:00:00+00:00")
                self.assertEqual(result["counts"], {"readings": 7, "deduplicated_readings": 6,
                                                    "duplicate_readings": 1, "meters_with_readings": 5})
                self.assertEqual(result["silent_meters"], [{"meter_id": "A", "hours_since_last_reading": 49,
                                                            "last_interval_start": "2026-01-01T01:00:00+00:00"}])
                self.assertEqual(result["frozen_meters"], [{"meter_id": "A", "unique_values": 1,
                                                            "reading_count": 2, "unique_ratio": 0.5}])
                unit_check = result["unit_check"]
                self.assertEqual(unit_check["evaluated_meters"], 3)
                suspect, = unit_check["suspected_meters"]
                self.assertEqual(suspect["meter_id"], "B")
                self.assertEqual(suspect["customer_type"], METER["customer_type"])
                self.assertEqual(suspect["meter_median_kwh"], 160)
                self.assertEqual(suspect["customer_type_median_kwh"], 1.2)
                self.assertAlmostEqual(suspect["ratio"], 160 / 1.2)

    def test_reports_without_registry_and_with_custom_parameters(self):
        path = self.import_records([reading("A", 0, 1), reading("A", 1, 1), reading("B", 50, 2)])
        status, result = self.read_quality(path, window_end="2026-01-03T05:00:00+02:00",
                                           silence_hours=50, frozen_ratio=0.5, unit_ratio=200)
        self.assertEqual(status, 200, result)
        self.assertEqual(result["window_end"], "2026-01-03T03:00:00+00:00")
        self.assertEqual(result["thresholds"], {"silence_hours": 50, "frozen_ratio": 0.5, "unit_ratio": 200})
        self.assertEqual(result["silent_meters"], [])
        self.assertEqual(result["frozen_meters"], [])
        self.assertIsNone(result["unit_check"])

    def test_unit_check_excludes_ambiguous_types_and_nonpositive_cohorts(self):
        records = [reading("A", 0, 10), reading("B", 0, 0), reading("C", 0, 1),
                   dict(METER, meter_id="A"), dict(METER, meter_id="A", customer_type="household"),
                   dict(METER, meter_id="B", customer_type="industrial"), dict(METER, meter_id="C")]
        path = self.import_records(records)
        status, result = self.read_quality(path, unit_ratio=0.5)
        self.assertEqual(status, 200, result)
        self.assertEqual(result["unit_check"]["evaluated_meters"], 1)
        self.assertEqual([entry["meter_id"] for entry in result["unit_check"]["suspected_meters"]], ["C"])

    def test_query_validation(self):
        path = self.import_records([reading("A", 0, 1)])
        for parameter, value in [("window_end", "2025-12-31T23:59:59Z"),
                                 ("window_end", "2026-01-01T00:00:00"),
                                 ("window_end", "2026-01-01T00:00:00-00:60"),
                                 ("silence_hours", "nan"), ("frozen_ratio", "1.1"), ("unit_ratio", "-1")]:
            with self.subTest(parameter=parameter, value=value):
                status, result = self.read_quality(path, **{parameter: value})
                self.assertEqual(status, 422, result)
                self.assertEqual(result["code"], "validation_error")
                self.assertEqual(result["path"], f"/{parameter}")

    def test_reports_missing_data_and_reading_conflicts(self):
        status, result = self.read_quality("/v1/datasets/invalid/readings/quality")
        self.assertEqual((status, result["code"]), (404, "dataset_not_found"))
        status, result = self.read_quality(self.import_records([METER]))
        self.assertEqual((status, result["code"]), (422, "readings_missing"))
        conflict_path = self.import_records([reading("A", 0, 1), reading("A", 0, 2)])
        status, result = self.read_quality(conflict_path)
        self.assertEqual((status, result["code"], result["meter_id"]), (409, "reading_conflict", "A"))
