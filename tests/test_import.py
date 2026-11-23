import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pyarrow.parquet as pq

from electrical_api.app import create_app
from electrical_api.datasets.contract import ROOT, TABLES


class ImportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.data = Path(temp.name)
        self.client = create_app(self.data).test_client()

    def post(self, body, media):
        response = self.client.post("/v1/datasets", data=body, content_type=media)
        return response.status_code, response.get_json()

    def test_formats_match_on_disk(self):
        directories = []
        for ext, media in [("json", "application/json"), ("ndjson", "application/x-ndjson")]:
            status, result = self.post((ROOT / f"examples/dataset.{ext}").read_bytes(), media)
            self.assertEqual(status, 201, result)
            self.assertEqual(sum(result["counts"].values()), 8)
            directories.append(self.data / result["dataset_id"])
        self.assertNotEqual(*directories)
        for name in TABLES.values():
            tables = [pq.read_table(d / f"{name}.parquet") for d in directories]
            self.assertTrue(tables[0].equals(tables[1]))
        readings = pq.read_table(directories[0] / "readings.parquet")
        self.assertEqual(readings["energy"].to_pylist(), [1.42, 1.45])
        self.assertEqual(readings.schema.field("interval_start").type.tz, "UTC")

    def test_failure_after_written_batch(self):
        lines = (ROOT / "examples/dataset.ndjson").read_bytes().splitlines()
        raw = b"\n".join([lines[0]] + [lines[2]] * 8193 + [b'{"record_type":"reading"}'])
        status, result = self.post(raw, "application/x-ndjson")
        self.assertEqual(status, 422, result)
        self.assertEqual(result["line"], 8195)
        self.assertEqual(list(self.data.iterdir()), [])

    def test_bad_input(self):
        for media, body, status in [
            ("text/plain", b"x", 415),
            ("application/json", b'{"a":0,"a":1}', 400),
            ("application/json", b'{"schema_version":"1.0","records":[]}', 422),
            ("application/x-ndjson", b'{"record_type":"dataset","schema_version":"1.0"}\n\n', 400),
        ]:
            with self.subTest(media=media, body=body):
                self.assertEqual(self.post(body, media)[0], status)
        self.assertEqual(list(self.data.iterdir()), [])

    def test_download_deduplicated_sample(self):
        for extension, media_type in [("json", "application/json"),
                                      ("ndjson", "application/x-ndjson")]:
            with self.subTest(media_type=media_type):
                body = (ROOT / f"examples/dataset.{extension}").read_bytes()
                status, imported = self.post(body, media_type)
                self.assertEqual(status, 201, imported)
                dataset_id = imported["dataset_id"]
                path = f"/v1/datasets/{dataset_id}/readings/deduplicated"
                raw_path = self.data / dataset_id / "readings.parquet"
                original = raw_path.read_bytes()

                with self.client.get(path) as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.mimetype, "application/vnd.apache.parquet")
                    self.assertIn("readings-deduplicated.parquet", response.headers["Content-Disposition"])
                    cleaned = pq.read_table(pa.BufferReader(response.data))

                raw = pq.read_table(pa.BufferReader(original))
                self.assertTrue(cleaned.equals(raw.slice(1, 1)))
                self.assertEqual(cleaned["energy"].to_pylist(), [1.45])
                self.assertEqual(raw_path.read_bytes(), original)

    def test_deduplication_uses_latest_instant_per_meter_and_interval(self):
        sample = json.loads((ROOT / "examples/dataset.json").read_bytes())
        earlier, later = [row for row in sample["records"] if row["record_type"] == "reading"]
        sample["records"] = [
            later,
            earlier,
            dict(earlier, energy=9),
            dict(later, received_at="2026-01-15T07:07:00Z"),
            dict(later, meter_id="other-meter", energy=3),
            dict(later, interval_start="2026-01-15T09:00:00+02:00", energy=4),
        ]
        status, imported = self.post(json.dumps(sample), "application/json")
        self.assertEqual(status, 201, imported)
        path = f"/v1/datasets/{imported['dataset_id']}/readings/deduplicated"
        with self.client.get(path) as response:
            self.assertEqual(response.status_code, 200)
            cleaned = pq.read_table(pa.BufferReader(response.data))
        self.assertEqual(cleaned["energy"].to_pylist(), [1.45, 4.0, 3.0])

    def test_deduplication_reports_conflicting_latest_transmissions(self):
        sample = json.loads((ROOT / "examples/dataset.json").read_bytes())
        later = [row for row in sample["records"] if row["record_type"] == "reading"][-1]
        for differences in [{"energy": 2}, {"energy_unit": "Wh"}, {"interval_minutes": 30}]:
            with self.subTest(differences=differences):
                sample["records"] = [later, later, dict(later, **differences)]
                status, imported = self.post(json.dumps(sample), "application/json")
                self.assertEqual(status, 201, imported)
                path = f"/v1/datasets/{imported['dataset_id']}/readings/deduplicated"
                with self.client.get(path) as response:
                    self.assertEqual(response.status_code, 409)
                    error = response.get_json()
                self.assertEqual(error["code"], "reading_conflict")
                self.assertEqual(error["meter_id"], later["meter_id"])
                self.assertEqual(error["interval_start"], "2026-01-15T06:00:00+00:00")
                self.assertEqual(error["received_at"], "2026-01-15T07:07:00+00:00")

    def test_deduplication_requires_existing_dataset_with_readings(self):
        for dataset_id in [str(uuid4()), "invalid-id"]:
            with self.subTest(dataset_id=dataset_id):
                path = f"/v1/datasets/{dataset_id}/readings/deduplicated"
                with self.client.get(path) as response:
                    self.assertEqual(response.status_code, 404)
                    self.assertEqual(response.get_json()["code"], "dataset_not_found")

        sample = json.loads((ROOT / "examples/dataset.json").read_bytes())
        sample["records"] = [row for row in sample["records"] if row["record_type"] == "meter"]
        status, imported = self.post(json.dumps(sample), "application/json")
        self.assertEqual(status, 201, imported)
        path = f"/v1/datasets/{imported['dataset_id']}/readings/deduplicated"
        with self.client.get(path) as response:
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.get_json()["code"], "readings_missing")

    def test_openapi_and_schema(self):
        for path, field in [("/docs/openapi.json", "openapi"),
                            ("/schemas/electrical-dataset-v1.schema.json", "$defs")]:
            with self.client.get(path) as response:
                self.assertIn(field, response.get_json())

    def test_html_documentation_covers_the_api_contract(self):
        with self.client.get("/docs/openapi.json") as response:
            specification = response.get_json()
        reference = self.client.get("/docs")
        self.assertEqual((reference.status_code, reference.mimetype), (200, "text/html"))
        content = reference.get_data(as_text=True)
        for methods in specification["paths"].values():
            for operation in methods.values():
                self.assertIn(f'id="{operation["operationId"]}"', content)

        data_guide = self.client.get("/docs/data")
        self.assertEqual((data_guide.status_code, data_guide.mimetype), (200, "text/html"))
        content = data_guide.get_data(as_text=True)
        with self.client.get("/schemas/electrical-dataset-v1.schema.json") as response:
            schema = response.get_json()
        for record_type in TABLES:
            self.assertIn(f'id="{record_type}"', content)
            section = content.split(f'id="{record_type}"', 1)[1].partition("</section>")[0]
            for field in schema["$defs"][record_type]["properties"]:
                self.assertIn(f"<code>{field}</code>", section)

    def test_swagger_serves_the_existing_spec_and_local_assets(self):
        with self.client.get("/docs/swagger") as response:
            self.assertEqual((response.status_code, response.mimetype), (200, "text/html"))
            page = response.get_data(as_text=True)
        self.assertIn('data-spec-url="/docs/openapi.json"', page)
        for asset in ("vendor/swagger-ui/swagger-ui.css", "vendor/swagger-ui/swagger-ui-bundle.js", "swagger.js"):
            url = f"/static/{asset}"
            self.assertIn(url, page)
            with self.client.get(url) as response:
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.data)
