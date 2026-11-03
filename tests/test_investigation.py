"""Checks for exact CPU evidence, causal scopes and model-page preservation."""

from collections import defaultdict
import json
from pathlib import Path
import tempfile
import unittest

from edge_rca.investigation import FixedSOP, log_summaries, metric_summary, model_pages, trace_summary
from edge_rca.preprocessing import build_preprocessor_prompt, validate_preprocessor_response
from edge_rca.replay import ReplaySession
from edge_rca.trace_index import TraceIndex


def event(eid, kind="metric", at=0, node="device:Tomcat01", available=None, **raw):
    return {"event_id": eid, "node_id": node, "source_id": node.split(":", 1)[1],
            "modality": kind, "observed_at_ms": at,
            "available_at_ms": at if available is None else available, "raw": raw}


class ToolStatisticsTests(unittest.TestCase):
    def test_latencies_compare_actual_windows_and_keep_slow_time(self):
        rows = [event("fast1", "trace", at=0, available=10, duration="10"),
                event("fast2", "trace", at=100, available=120, duration="20"),
                event("slow1", "trace", at=900, available=1300, duration="400"),
                event("slow2", "trace", at=1000, available=1600, duration="600"),
                event("invalid", "trace", at=1001, available=1700, duration="nan")]
        result = trace_summary(rows, 0, 1000, 2000)
        self.assertEqual(result["reference"]["p95"], 20)
        self.assertEqual(result["current"]["p50"], 400)
        self.assertEqual(result["current"]["p95"], 600)
        self.assertEqual(result["p95_change"]["ratio"], 30)
        self.assertEqual(result["current"]["completed_calls_per_second"], 2)
        self.assertTrue(result["current"]["p95_low_sample"])
        self.assertEqual(result["slow_examples"][-1]["start_ms"], 1000)
        self.assertEqual(result["invalid_count"], 1)
        self.assertEqual(sum(b["count"] for b in result["minute_buckets"]), 4)

    def test_metrics_do_not_flip_free_memory_direction(self):
        rows = [event(str(i), at=i * 60_000, value=v) for i, v in enumerate([100, 100, 100, 20, 10])]
        result = metric_summary("JVMFreeMemory", rows, 0, 180_000, 300_000)
        self.assertEqual(result["change"]["direction"], "down")
        self.assertIn("剩余", result["semantics"]["meaning"])
        self.assertEqual(result["reference"]["median"], 100)
        self.assertEqual(result["current"]["median"], 15)
        self.assertEqual(result["sample_interval_median_ms"], 60_000)
        self.assertEqual(result["comparison_status"], "limited_history")

    def test_within_window_early_late_comparison_is_explicit(self):
        rows = [event(str(i), at=i * 60_000, value=v)
                for i, v in enumerate([10, 10, 10, 10, 50, 50, 50, 50])]
        result = metric_summary("unknown", rows, 0, 0, 480_000)
        self.assertEqual(result["early"]["median"], 10)
        self.assertEqual(result["late"]["median"], 50)
        self.assertEqual(result["early_late_change"]["direction"], "up")
        self.assertEqual(result["early_late_window_ms"], 300_000)
        text = __import__("edge_rca.report_cards", fromlist=["card"]).card(
            {"id": "f1", "kind": "metric", "metric": "unknown", "meaning": "测试",
             "unit": "source_unit_unverified", "reference": result["reference"],
             "current": result["current"], "change": result["change"],
             "early": result["early"], "late": result["late"],
             "early_late_change": result["early_late_change"], "extremes": result["extremes"],
             "quality": {"invalid": 0, "conflicts": 0, "counter_resets": 0}},
            {"scope": {"start": "18:00:00", "end": "18:08:00"}})["tool_fact"]
        self.assertIn("窗口前后5分钟", text)
        self.assertIn("极值观测", text)

    def test_counters_convert_rates_and_skip_resets(self):
        rows = [event(str(i), at=i * 1000, value=v) for i, v in enumerate([10, 30, 5, 15])]
        result = metric_summary("Redis(total_commands_processed)", rows, 0, 2000, 4000)
        self.assertEqual(result["reference"]["median"], 20)
        self.assertEqual(result["current"]["median"], 10)
        self.assertEqual(result["counter_reset_count"], 1)
        self.assertEqual(result["semantics"]["analysis_unit"], "raw_units/second")

    def test_conflicting_duplicate_samples_are_not_arbitrarily_selected(self):
        rows = [event("a", at=0, value=10), event("b", at=0, value=99),
                event("c", at=1000, value=5), event("d", at=1000, value=5),
                event("e", at=2000, value="NaN")]
        result = metric_summary("unknown", rows, 0, 500, 3000)
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(result["conflicting_record_count"], 2)
        self.assertEqual(result["invalid_count"], 1)
        self.assertIsNone(result["reference"]["median"])
        self.assertEqual(result["change"]["direction"], "unknown")

    def test_http_grouping_retains_status_path_requester_but_merges_sizes(self):
        values = ['IG01 POST /ServiceTest1.json HTTP/1.1 200 12 - client 38',
                  'IG01 POST /ServiceTest1.json HTTP/1.1 200 99 - client 42',
                  'IG01 POST /ServiceTest1.json HTTP/1.1 500 99 - client 42',
                  'IG02 POST /ServiceTest1.json HTTP/1.1 200 99 - client 42',
                  'IG01 POST /ServiceTest2.json HTTP/1.1 200 99 - client 42']
        rows = [event(str(i), "log", at=i * 100, log_name="access", value=value) for i, value in enumerate(values)]
        result = log_summaries(rows, 0, 200, 1000)
        self.assertEqual(len(result), 4)
        self.assertEqual(sum(f["count"] for f, _ in result), 5)
        self.assertEqual(sum(f["has_explicit_error"] for f, _ in result), 1)
        merged = next(f for f, _ in result if f["count"] == 2)
        self.assertEqual(merged["reference_count"], 2)
        self.assertEqual(merged["current_count"], 0)

    def test_gc_uses_outer_pause_and_does_not_call_allocation_failure_oom(self):
        text = '[GC (Allocation Failure) [ParNew: 10K->2K(12K), 0.02 secs] 20K->5K(25K), 0.03 secs] [Times: user=0.0 sys=0.0, real=0.03 secs]'
        rows = [event("gc", "log", value=text, log_name="gc"),
                event("oom", "log", at=1, value="java.lang.OutOfMemoryError: Java heap space", log_name="gc")]
        result = {f["category"]: f for f, _ in log_summaries(rows, 0, 1, 2)}
        self.assertEqual(result["gc"]["gc_pause_ms"]["max"], 30)
        self.assertFalse(result["gc"]["has_explicit_error"])
        self.assertTrue(result["oom_text"]["has_explicit_error"])


class TraceIndexTests(unittest.TestCase):
    def setUp(self):
        self.index = TraceIndex()
        self.addCleanup(self.index.close)
        self.child = event("child", "trace", at=100, available=200, node="device:Tomcat01",
                           trace_id="t", span_id="c", parent_id="p", duration="100")
        self.parent = event("parent", "trace", at=50, available=500, node="device:IG01",
                            trace_id="t", span_id="p", parent_id="p", duration="450")

    def test_late_parent_is_linked_only_after_arrival_with_both_latencies(self):
        self.index.ingest([self.child], 300)
        early = self.index.snapshot(300, 0, 100)
        self.assertEqual(early["links"], [])
        self.assertEqual(early["counts"]["unmatched_parent"], 1)
        self.index.ingest([self.parent], 600)
        self.assertEqual(self.index.snapshot(300, 0, 100), early)
        result = self.index.snapshot(600, 0, 100)
        self.assertEqual(len(result["links"]), 1)
        example = result["links"][0]["examples"][0]
        self.assertEqual((example["parent_duration_ms"], example["child_duration_ms"]), (450, 100))
        self.assertTrue(example["child_within_parent_interval"])
        self.assertEqual(example["known_at_ms"], 500)

    def test_trace_identity_prevents_cross_request_join(self):
        self.parent["raw"]["trace_id"] = "other-trace"
        self.index.ingest([self.child, self.parent], 600)
        result = self.index.snapshot(600, 0, 100)
        self.assertEqual(result["links"], [])

    def test_new_ambiguous_parent_retracts_link_but_does_not_rewrite_past(self):
        self.index.ingest([self.child, self.parent], 600)
        original = self.index.snapshot(600, 0, 100)
        duplicate = dict(self.parent, event_id="duplicate", available_at_ms=650)
        self.index.ingest([duplicate], 700)
        self.assertEqual(self.index.snapshot(600, 0, 100), original)
        self.assertEqual(self.index.snapshot(700, 0, 100)["links"], [])

    def test_self_reference_same_node_and_duplicates_do_not_inflate_edges(self):
        self.child["node_id"] = self.parent["node_id"]
        self.index.ingest([self.child, self.parent], 600)
        result = self.index.snapshot(600, 0, 100)
        self.assertEqual(result["links"], [])
        self.assertEqual(result["counts"]["self_reference"], 1)
        self.assertEqual(result["counts"]["same_node_parent"], 1)
        with self.assertRaises(Exception):
            self.index.ingest([self.child], 700)
        self.assertEqual(self.index.snapshot(600, 0, 100), result)

    def test_future_batch_rolls_back_atomically(self):
        with self.assertRaisesRegex(ValueError, "future"):
            self.index.ingest([self.child, self.parent], 400)
        self.index.ingest([self.child], 300)
        with self.assertRaisesRegex(ValueError, "future"):
            self.index.snapshot(600, 0, 100)


class FixedSOPTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name)

    def replay(self, events, end=3000):
        grouped = defaultdict(list)
        for e in events:
            grouped[e["node_id"]].append(e)
        nodes = []
        for i, (node, rows) in enumerate(grouped.items()):
            rows.sort(key=lambda e: (e["available_at_ms"], e["event_id"]))
            path = f"{i}.jsonl"
            (self.path / path).write_text("".join(json.dumps(e) + "\n" for e in rows))
            nodes.append({"node_id": node, "source_id": node.split(":")[1], "path": path})
        (self.path / "manifest.json").write_text(json.dumps({"schema_version": 1,
            "window": {"start_ms": 0, "end_ms": end}, "nodes": nodes}))
        session = ReplaySession(self.path)
        self.addCleanup(session.close)
        session.advance_to(end)
        return session

    def test_historical_cutoff_and_unlimited_query_preserve_all_123_records(self):
        rows = [event(str(i), at=i, value=i, kpi_name="cpu") for i in range(123)]
        rows += [event("future", at=2000, value=999999, kpi_name="cpu")]
        session = self.replay(rows)
        result = FixedSOP(session).build("device:Tomcat01", 1000, 100)
        metric = next(f for f in result["observations"] if f["modality"] == "metric")
        self.assertEqual(metric["record_count"], 123)
        self.assertEqual(metric["current"]["max"], 122)
        self.assertNotIn("future", json.dumps(result))
        self.assertEqual(set(result["evidence_index"][metric["observation_id"]]), {str(i) for i in range(123)})
        self.assertEqual(len(session.query("device:Tomcat01", limit=None)), 124)
        absence = [f for f in result["observations"] if f["status"] == "no_arrived_records_in_scope"]
        self.assertEqual(len(absence), 2)

    def test_pages_cover_every_fact_and_are_valid_for_v2_prompt(self):
        rows = [event(str(i), at=i, value=i, kpi_name=f"series-{i}") for i in range(20)]
        result = FixedSOP(self.replay(rows)).build("device:Tomcat01", 1000, 500)
        pages = model_pages(result, max_observations=3, max_characters=7000)
        ids = [f["observation_id"] for p in pages for f in p["observations"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {f["observation_id"] for f in result["observations"]})
        for page in pages:
            self.assertFalse(page["scope"]["oversized_single_page"])
            self.assertLessEqual(len(json.dumps(page, ensure_ascii=False, separators=(",", ":"))), 7000)
            build_preprocessor_prompt(page)
            validate_preprocessor_response({"groups": [], "retain_local": [f["observation_id"] for f in page["observations"]],
                                            "defer": [], "questions": []}, page)

    def test_alerts_are_mandatory_and_cross_node_or_future_evidence_rejected(self):
        session = self.replay([event("a", value=1, kpi_name="x")])
        transition = {"node_id": "device:Tomcat01", "at_ms": 1, "action": "open", "rule": "metric.high",
                      "signal": "x", "details": {"value": 1}, "evidence_ids": ["a"]}
        sop = FixedSOP(session)
        result = sop.build("device:Tomcat01", 1000, 1, [transition])
        self.assertEqual(len(result["scope"]["must_report_ids"]), 1)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            sop.build("device:Tomcat01", 1000, 1, [dict(transition, evidence_ids=["unseen"])])
        with self.assertRaisesRegex(ValueError, "causal"):
            sop.build("device:Tomcat01", 1000, 1, [dict(transition, node_id="device:Other")])

    def test_business_remains_separate_and_future_requested_time_fails(self):
        node = "business:ServiceTest1"
        session = self.replay([event("b", "business_metric", node=node, rr=100, sr=98, cnt=40, mrt=25)])
        sop = FixedSOP(session)
        result = sop.build(node, 1000, 500)
        self.assertEqual({f["name"] for f in result["observations"]}, {"rr", "sr", "cnt", "mrt"})
        self.assertNotIn("cmdb_id", json.dumps(result))
        with self.assertRaises(ValueError):
            sop.build(node, 4000, 500)

    def test_alert_task_uses_original_episode_not_evaluation_or_current_clock(self):
        session = self.replay([event("a", value=1, kpi_name="x"),
                               event("later", at=2000, value=100, kpi_name="x")])
        task = {"task_id": "task", "node_id": "device:Tomcat01",
                "data_access": {"node_id": "device:Tomcat01", "available_before_ms": 1000},
                "transitions": [{"node_id": "device:Tomcat01", "at_ms": 900, "alert_id": "ep",
                    "action": "escalated", "rule": "metric.high", "signal": "x",
                    "details": {}, "evidence_ids": ["a"]}]}
        sop = FixedSOP(session)
        result = sop.build_task(task, {"ep": 0})
        self.assertEqual(result["cutoff_ms"], 1000)
        self.assertEqual(result["scope"]["current"]["start_ms"], 0)
        self.assertNotIn("later", json.dumps(result))
        with self.assertRaisesRegex(ValueError, "missing original"):
            sop.build_task(task, {})


if __name__ == "__main__":
    unittest.main()
