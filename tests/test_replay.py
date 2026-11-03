"""Synthetic tests for causal edge-device replay; no dataset/model required."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from edge_rca.replay import ReplaySession


def observation(event_id, node_id="device:Tomcat01", modality="metric", observed=1000, available=None):
    return {
        "event_id": event_id,
        "modality": modality,
        "node_id": node_id,
        "source_id": node_id.split(":", 1)[1],
        "observed_at_ms": observed,
        "available_at_ms": observed if available is None else available,
        "source_file": f"{modality}.csv",
        "source_row": 2,
        "raw": {"cmdb_id": node_id.split(":", 1)[1], "value": "001.500"},
    }


class ReplaySessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def prepare(self, node_events, start=1000, end=2000):
        nodes = []
        (self.directory / "nodes").mkdir(exist_ok=True)
        for index, (node_id, events) in enumerate(node_events.items()):
            path = f"nodes/{index}.jsonl"
            (self.directory / path).write_text(
                "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
            )
            nodes.append({
                "node_id": node_id,
                "kind": node_id.split(":", 1)[0],
                "source_id": node_id.split(":", 1)[1],
                "path": path,
                "counts": {modality: sum(event["modality"] == modality for event in events)
                           for modality in ("metric", "log", "trace", "business_metric")},
            })
        (self.directory / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "window": {"start_ms": start, "end_ms": end, "timezone": "Asia/Shanghai"},
            "nodes": nodes,
        }), encoding="utf-8")
        return self.directory

    def session(self):
        session = ReplaySession(self.directory)
        self.addCleanup(session.close)
        return session

    def test_boundary_and_future_records_are_never_exposed(self):
        self.prepare({"device:Tomcat01": [
            observation("at-start"), observation("at-1100", observed=1100),
            observation("at-end", modality="trace", observed=1900, available=2000),
            observation("after-end", modality="trace", observed=1950, available=2200),
        ]})
        replay = self.session()
        self.assertEqual(replay.query("device:Tomcat01"), [])
        self.assertEqual(replay.advance_to(1000)["events"], [])
        self.assertEqual([event["event_id"] for event in replay.advance_to(1100)["events"]], ["at-start"])
        self.assertEqual([event["event_id"] for event in replay.advance_to(2000)["events"]], ["at-1100"])
        self.assertTrue(replay.paused)
        self.assertEqual(replay.total_delivered, 2)
        self.assertEqual(len(replay.query("device:Tomcat01", limit=None)), 2)
        self.assertEqual(replay.tick(100)["events"], [])

    def test_trace_becomes_visible_at_completion_not_start(self):
        self.prepare({"device:Tomcat01": [
            observation("already-started", modality="trace", observed=900, available=1200),
            observation("later-call", modality="trace", observed=1100, available=1500),
        ]})
        replay = self.session()
        replay.advance_to(1200)
        self.assertEqual(replay.query("device:Tomcat01", "trace"), [])
        replay.advance_to(1201)
        self.assertEqual(replay.query("device:Tomcat01", "trace")[0]["event_id"], "already-started")
        self.assertEqual(replay.query("device:Tomcat01", start_ms=1000), [])
        replay.advance_to(1501)
        self.assertEqual(len(replay.query("device:Tomcat01", start_ms=1000, end_ms=1200)), 1)

    def test_shared_ticks_include_empty_slices_and_stop_at_end(self):
        self.prepare({"device:Tomcat01": [observation("late", observed=1700)]})
        ticks = list(self.session().iter_ticks(300))
        self.assertEqual([tick["clock_ms"] for tick in ticks], [1300, 1600, 1900, 2000])
        self.assertEqual([tick["delivered"] for tick in ticks], [0, 0, 1, 0])
        self.assertEqual([tick["paused"] for tick in ticks], [False, False, False, True])

    def test_historical_alert_query_hides_later_records_in_the_same_batch(self):
        self.prepare({"device:Tomcat01": [
            observation("early", observed=1100), observation("later", observed=1900),
        ]})
        replay = self.session()
        replay.advance_to(2000)
        self.assertEqual(len(replay.query("device:Tomcat01")), 2)
        self.assertEqual([item["event_id"] for item in replay.query(
            "device:Tomcat01", available_before_ms=1500)], ["early"])
        with self.assertRaises(ValueError):
            replay.query("device:Tomcat01", available_before_ms=2001)

    def test_merging_preserves_identity_raw_fields_and_all_modalities(self):
        events = {
            "device:Tomcat01": [observation("b"), observation("d", modality="log", observed=1100)],
            "device:Mysql01": [observation("a", "device:Mysql01"),
                               observation("c", "device:Mysql01", "trace", observed=1050, available=1100)],
            "business:ServiceTest1": [observation("e", "business:ServiceTest1", "business_metric", observed=1200)],
        }
        self.prepare(events)
        replay = self.session()
        batch = replay.advance_to(2000)
        self.assertEqual([event["event_id"] for event in batch["events"]], ["a", "b", "c", "d", "e"])
        self.assertEqual(batch["counts"], {"metric": 2, "log": 1, "trace": 1, "business_metric": 1})
        self.assertEqual(sum(len(replay.query(node_id, limit=None)) for node_id in events), 5)
        self.assertEqual(replay.query("device:Tomcat01", "log")[0]["raw"]["value"], "001.500")
        self.assertEqual(replay.query("business:ServiceTest1")[0]["node_id"], "business:ServiceTest1")

    def test_unsorted_stream_is_rejected_without_committing_tick(self):
        self.prepare({"device:Tomcat01": [observation("b", observed=1100), observation("a")]})
        replay = self.session()
        with self.assertRaisesRegex(ValueError, "strictly sorted"):
            replay.advance_to(2000)
        self.assertEqual(replay.total_delivered, 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            replay.query("device:Tomcat01")

    def test_wrong_node_and_source_identity_are_rejected(self):
        event = observation("a", "device:Mysql01")
        self.prepare({"device:Tomcat01": [event]})
        with self.assertRaisesRegex(ValueError, "different node"):
            ReplaySession(self.directory)
        event = observation("a")
        event["source_id"] = "Mysql01"
        self.prepare({"device:Tomcat01": [event]})
        with self.assertRaisesRegex(ValueError, "source_id"):
            ReplaySession(self.directory)

    def test_invalid_time_and_query_arguments_fail_explicitly(self):
        self.prepare({"device:Tomcat01": []})
        replay = self.session()
        replay.advance_to(1200)
        for target in (1199, 2001, 1300.0, True):
            with self.assertRaises(ValueError):
                replay.advance_to(target)
        for step in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                replay.tick(step)
        with self.assertRaises(KeyError):
            replay.query("device:missing")
        with self.assertRaises(ValueError):
            replay.query("device:Tomcat01", modality="unknown")
        with self.assertRaises(ValueError):
            replay.query("device:Tomcat01", start_ms=1500, end_ms=1400)
        self.assertEqual(replay.query("device:Tomcat01", limit=0), [])

    def test_persistent_inbox_contains_only_delivered_records(self):
        self.prepare({"device:Tomcat01": [observation("a"), observation("b", observed=1500)]})
        path = self.directory / "received.sqlite3"
        with ReplaySession(self.directory, inbox_path=path) as replay:
            replay.advance_to(1200)
        with sqlite3.connect(path) as connection:
            self.assertEqual(connection.execute("SELECT event_id FROM events").fetchall(), [("a",)])
        with self.assertRaisesRegex(ValueError, "already contains"):
            ReplaySession(self.directory, inbox_path=path)

    def test_empty_node_and_window_work(self):
        self.prepare({"device:Tomcat01": []})
        replay = self.session()
        self.assertEqual(len(list(replay.iter_ticks(400))), 3)
        self.assertTrue(replay.paused)
        self.assertEqual(replay.total_delivered, 0)


if __name__ == "__main__":
    unittest.main()
