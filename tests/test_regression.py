import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from electrical_api.app import create_app


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def meter(meter_id, heating=False, tariff="single", customer_type="household"):
    return {"record_type": "meter", "meter_id": meter_id, "customer_type": customer_type,
            "electric_heating": heating, "tariff": tariff, "connection_year": 2000, "district": "Obolon"}


def reading(meter_id, hour, energy, *, unit="kWh", minutes=60, delay=60):
    interval_start = START + timedelta(hours=hour)
    received_at = interval_start + timedelta(minutes=delay)
    return {"record_type": "reading", "meter_id": meter_id, "interval_start": interval_start.isoformat(),
            "interval_minutes": minutes, "received_at": received_at.isoformat(),
            "energy": energy, "energy_unit": unit}


class HouseholdRegressionTests(unittest.TestCase):
    def setUp(self):
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        self.data = Path(storage.name)
        self.client = create_app(self.data).test_client()

    def import_records(self, records):
        body = json.dumps({"schema_version": "1.0", "records": records})
        response = self.client.post("/v1/datasets", data=body, content_type="application/json")
        self.assertEqual(response.status_code, 201, response.get_json())
        return response.get_json()["dataset_id"]

    def regression(self, dataset_id, **params):
        path = f"/v1/datasets/{dataset_id}/readings/household-regression"
        response = self.client.get(path, query_string=params)
        return response.status_code, response.get_json()

    def test_fits_model_after_filtering_and_rescaling(self):
        records = [
            meter("a"), meter("b", True), meter("c", False, "dual_zone"),
            meter("d", True, "dual_zone"), meter("d", True, "dual_zone"),
            meter("silent"), meter("frozen"), meter("unknown", tariff="other"),
            meter("business", customer_type="small_business"),
            reading("a", 100, 90, delay=15), reading("a", 100, 0.4375), reading("a", 101, 0.5625),
            reading("b", 100, 0.6875), reading("b", 101, 0.8125),
            reading("c", 100, 0.5625), reading("c", 101, 0.6875),
            reading("d", 100, 812.5), reading("d", 101, 937.5, unit="Wh"),
            reading("silent", 0, 2), reading("silent", 1, 3),
            reading("frozen", 100, 2), reading("frozen", 101, 2),
            reading("unknown", 100, 4), reading("unknown", 101, 5),
            reading("business", 101, 10), reading("a", 102, 33, minutes=30),
        ]
        dataset_id = self.import_records(records)
        raw_file = self.data / dataset_id / "readings.parquet"
        original = raw_file.read_bytes()

        status, model = self.regression(dataset_id, window_end="2026-01-05T07:00:00Z")
        self.assertEqual(status, 200, model)
        self.assertEqual(model["dataset_id"], dataset_id)
        self.assertEqual(model["window_end"], "2026-01-05T07:00:00+00:00")
        self.assertEqual(model["meter_count"], 4)
        self.assertEqual(model["rescaled_meter_ids"], ["d"])
        self.assertAlmostEqual(model["intercept"], 12)
        self.assertAlmostEqual(model["coefficients"]["electric_heating"], 6)
        self.assertAlmostEqual(model["coefficients"]["dual_zone"], 3)
        self.assertAlmostEqual(model["r_squared"], 1)
        self.assertEqual(raw_file.read_bytes(), original)

    def test_returns_diagnostics_for_unusable_model_inputs(self):
        cases = [
            ([reading("a", 1, 1)], 422, "meters_missing"),
            ([meter("a"), reading("a", 1, 1, minutes=30)], 422, "hourly_readings_missing"),
            ([meter("a"), reading("a", 1, 1)], 422, "insufficient_regression_data"),
            ([meter("a"), meter("a", True), reading("a", 1, 1)], 422, "meter_registry_conflict"),
            ([meter("a"), reading("a", 1, 1), reading("a", 1, 2)], 409, "reading_conflict"),
        ]
        collinear = [meter("a"), meter("b", True, "dual_zone"), meter("c", True, "dual_zone"),
                     reading("a", 1, 1), reading("b", 1, 2), reading("c", 1, 3)]
        cases.append((collinear, 422, "insufficient_regression_data"))
        too_large = [meter("a"), meter("b", True), meter("c", tariff="dual_zone"),
                     reading("a", 1, 1e308), reading("b", 1, 1e308), reading("c", 1, 1e308)]
        cases.append((too_large, 422, "regression_numeric_range"))

        for records, expected_status, code in cases:
            with self.subTest(code=code, records=len(records)):
                dataset_id = self.import_records(records)
                status, result = self.regression(dataset_id)
                self.assertEqual((status, result["code"]), (expected_status, code))
