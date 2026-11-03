"""Synthetic CPU-only checks for lossless Bank device stream preparation."""

import csv
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

from edge_rca.data import prepare_bank


SCHEMAS = {
    "metric/metric_container.csv": ["timestamp", "cmdb_id", "kpi_name", "value"],
    "metric/metric_app.csv": ["timestamp", "rr", "sr", "cnt", "mrt", "tc"],
    "log/log_service.csv": ["timestamp", "cmdb_id", "log_id", "log_name", "value"],
    "trace/trace_span.csv": [
        "timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"
    ],
}


def milliseconds(local_time):
    parsed = datetime.fromisoformat(local_time).replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    return int(parsed.timestamp() * 1000)


def seconds(local_time):
    return str(milliseconds(local_time) // 1000)


class BankStreamPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.telemetry = self.root / "Bank" / "telemetry"
        self.output = self.root / "prepared"
        self.start = "2021-03-04 18:00:00"
        self.end = "2021-03-04 18:01:00"

    def write_csv(self, name, rows, day="2021_03_04"):
        path = self.telemetry / day / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SCHEMAS[name])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def metric(self, when, device="Tomcat01", value="12.50"):
        return {"timestamp": seconds(when), "cmdb_id": device,
                "kpi_name": "container_cpu_used", "value": value}

    def trace(self, when, duration, device="MG01", span="s1"):
        return {"timestamp": str(milliseconds(when)), "cmdb_id": device,
                "parent_id": "parent0", "span_id": span,
                "trace_id": "trace0", "duration": str(duration)}

    def prepare(self, **kwargs):
        return prepare_bank(self.telemetry, self.start, self.end, self.output, **kwargs)

    def events(self, manifest):
        records = []
        for node in manifest["nodes"]:
            path = self.output / node["path"]
            self.assertTrue(path.is_file(), node)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            self.assertEqual(sorted(rows, key=lambda row: (
                row["available_at_ms"], row["event_id"])), rows)
            self.assertTrue(all(row["node_id"] == node["node_id"] for row in rows))
            records.extend(rows)
        self.assertEqual(len(records), len({row["event_id"] for row in records}))
        return records

    def test_seconds_timezone_unsorted_rows_and_duplicates_are_preserved(self):
        duplicate = self.metric("2021-03-04 18:00:10")
        self.write_csv("metric/metric_container.csv", [
            self.metric("2021-03-04 18:00:50"), duplicate, duplicate,
            self.metric(self.start), self.metric(self.end),
            self.metric("2021-03-04 17:59:59"),
        ])
        manifest = self.prepare()
        self.assertEqual(1, manifest["schema_version"])
        self.assertEqual(milliseconds(self.start), manifest["window"]["start_ms"])
        self.assertEqual(milliseconds(self.end), manifest["window"]["end_ms"])
        self.assertEqual("Asia/Shanghai", manifest["window"]["timezone"])
        events = self.events(manifest)
        self.assertEqual(4, len(events))
        self.assertEqual([milliseconds(self.start), milliseconds("2021-03-04 18:00:10"),
                          milliseconds("2021-03-04 18:00:10"),
                          milliseconds("2021-03-04 18:00:50")],
                         [event["available_at_ms"] for event in events])
        self.assertEqual(2, sum(event["raw"] == duplicate for event in events))
        self.assertTrue(all(event["modality"] == "metric" for event in events))
        self.assertEqual(4, manifest["totals"]["prepared"])
        self.assertEqual(4, manifest["totals"]["deliverable"])
        self.assertEqual(0, manifest["totals"]["pending"])
        source = manifest["source_files"][0]
        self.assertEqual(6, source["rows_scanned"])
        self.assertEqual(4, source["rows_prepared"])
        self.assertEqual(2, source["rows_excluded"])
        self.assertTrue(all(isinstance(row["source_row"], int) for row in events))

    def test_multiline_log_text_and_business_identity_are_lossless(self):
        message = 'A long diagnostic record: "quoted", commas\n' + "detail " * 1000
        log = {"timestamp": seconds("2021-03-04 18:00:03"), "cmdb_id": "Tomcat01",
               "log_id": "00023", "log_name": "gc.log", "value": message}
        business = {"timestamp": seconds("2021-03-04 18:00:04"), "rr": "0.0",
                    "sr": "99.50", "cnt": "0012", "mrt": "12.00", "tc": "ServiceTest1"}
        self.write_csv("log/log_service.csv", [log])
        self.write_csv("metric/metric_app.csv", [business])
        manifest = self.prepare()
        events = self.events(manifest)
        by_modality = {event["modality"]: event for event in events}
        self.assertEqual(log, by_modality["log"]["raw"])
        self.assertEqual(business, by_modality["business_metric"]["raw"])
        self.assertEqual("device:Tomcat01", by_modality["log"]["node_id"])
        self.assertEqual("business:ServiceTest1", by_modality["business_metric"]["node_id"])
        self.assertEqual({"device", "business"}, {node["kind"] for node in manifest["nodes"]})
        self.assertEqual({"log": 1, "business_metric": 1},
                         {key: value for key, value in manifest["totals"]["by_modality"].items()
                          if value})

    def test_trace_is_only_available_when_finished_and_end_boundary_is_pending(self):
        rows = [
            self.trace("2021-03-04 17:59:59", 1000, span="finishes_at_start"),
            self.trace("2021-03-04 18:00:01", 23, span="short"),
            self.trace("2021-03-04 18:00:59", 1000, span="finishes_at_end"),
            self.trace("2021-03-04 18:00:58", 3000, span="finishes_after_end"),
            self.trace("2021-03-04 17:59:58", 1, span="finished_before_start"),
            self.trace(self.end, 0, span="starts_at_end"),
        ]
        self.write_csv("trace/trace_span.csv", rows)
        manifest = self.prepare()
        events = self.events(manifest)
        self.assertEqual(4, len(events))
        self.assertEqual(2, manifest["totals"]["deliverable"])
        self.assertEqual(2, manifest["totals"]["pending"])
        self.assertEqual({"finishes_at_start", "short", "finishes_at_end", "finishes_after_end"},
                         {event["raw"]["span_id"] for event in events})
        for event in events:
            self.assertEqual(int(event["raw"]["timestamp"]), event["observed_at_ms"])
            self.assertEqual(event["observed_at_ms"] + int(event["raw"]["duration"]),
                             event["available_at_ms"])

    def test_cross_midnight_reads_existing_neighbor_directories(self):
        self.start, self.end = "2021-03-04 23:59:59", "2021-03-05 00:00:02"
        self.write_csv("metric/metric_container.csv", [self.metric(self.start)])
        self.write_csv("metric/metric_container.csv", [
            self.metric("2021-03-05 00:00:01", device="Mysql01")], day="2021_03_05")
        # Export directory and observation date need not be the same. Neighbors are scanned too.
        self.write_csv("trace/trace_span.csv", [
            self.trace("2021-03-04 23:59:58", 1500)], day="2021_03_03")
        self.write_csv("metric/metric_container.csv", [
            self.metric("2021-03-05 00:00:00", device="Redis01")], day="2021_03_06")
        manifest = self.prepare()
        events = self.events(manifest)
        self.assertEqual(4, len(events))
        self.assertEqual(4, len(manifest["source_files"]))
        self.assertEqual({"device:Tomcat01", "device:Mysql01", "device:MG01", "device:Redis01"},
                         {event["node_id"] for event in events})

    def test_node_inventory_can_include_window_external_rows_without_delivering_them(self):
        self.write_csv("metric/metric_container.csv", [
            self.metric(self.start), self.metric("2021-03-04 19:00:00", device="Mysql02")])
        manifest = self.prepare()
        events = self.events(manifest)
        self.assertEqual(1, len(events))
        self.assertEqual("device:Tomcat01", events[0]["node_id"])
        self.assertEqual(1, manifest["totals"]["prepared"])

    def test_invalid_identity_timestamp_and_duration_fail_explicitly(self):
        invalid = [
            ("metric/metric_container.csv", {**self.metric(self.start), "cmdb_id": ""}),
            ("metric/metric_container.csv", {**self.metric(self.start), "timestamp": "not-a-time"}),
            ("trace/trace_span.csv", self.trace(self.start, -1)),
            ("metric/metric_app.csv", {"timestamp": seconds(self.start), "rr": "0", "sr": "1",
                                       "cnt": "1", "mrt": "1", "tc": ""}),
        ]
        for index, (name, row) in enumerate(invalid):
            with self.subTest(name=name, index=index):
                case_root = self.root / str(index)
                self.telemetry = case_root / "telemetry"
                self.output = case_root / "prepared"
                self.write_csv(name, [row])
                with self.assertRaises((ValueError, RuntimeError)):
                    self.prepare()

    def test_empty_source_and_invalid_window_are_rejected(self):
        self.telemetry.mkdir(parents=True)
        with self.assertRaises((ValueError, FileNotFoundError, RuntimeError)):
            self.prepare()
        self.write_csv("metric/metric_container.csv", [self.metric(self.start)])
        self.end = self.start
        with self.assertRaises((ValueError, RuntimeError)):
            self.prepare()

    def test_existing_output_is_never_overwritten(self):
        self.write_csv("metric/metric_container.csv", [self.metric(self.start)])
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("existing results", encoding="utf-8")
        with self.assertRaises((FileExistsError, ValueError, RuntimeError)):
            self.prepare()
        self.assertEqual("existing results", sentinel.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
