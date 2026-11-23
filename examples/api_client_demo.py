"""Run one reading-quality report and print its results."""
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main(api_url="http://127.0.0.1:5000"):
    data = Path(__file__).with_name("meter_readings.ndjson")
    upload = Request(
        f"{api_url}/v1/datasets",
        data=data.read_bytes(),
        headers={"Content-Type": "application/x-ndjson"},
        method="POST",
    )
    with urlopen(upload, timeout=30) as response:
        dataset_id = json.load(response)["dataset_id"]

    with urlopen(f"{api_url}/v1/datasets/{dataset_id}/readings/quality", timeout=30) as response:
        report = json.load(response)

    counts = report["counts"]
    rows = [
        ("Покази у файлі", counts["readings"]),
        ("Покази після дедуплікації", counts["deduplicated_readings"]),
        ("Повторні передачі", counts["duplicate_readings"]),
        ("Лічильники з показами", counts["meters_with_readings"]),
        ("Замовклі лічильники", ", ".join(row["meter_id"] for row in report["silent_meters"]) or "—"),
        ("Завмерлі лічильники", ", ".join(row["meter_id"] for row in report["frozen_meters"]) or "—"),
    ]
    print(f"{'Показник':<30} | Результат")
    print("-" * 43)
    for label, value in rows:
        print(f"{label:<30} | {value}")


if __name__ == "__main__":
    main()
