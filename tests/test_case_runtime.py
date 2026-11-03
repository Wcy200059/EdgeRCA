from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from edge_rca import case_runtime as runtime
from edge_rca.case_eval import evaluate_case
from edge_rca.coordination import make_view
from edge_rca.node_reports import ReportInbox
from edge_rca.node_probe import decode_object


START = 1614852000000
CPU = "OSLinux-CPU_CPU_CPUCpuUtil"


class FixtureModel:
    """A scripted contract fixture; no claim about real RCA model accuracy."""
    class CUDA:
        @staticmethod
        def max_memory_allocated():
            return 0

        @staticmethod
        def empty_cache():
            pass

        @staticmethod
        def memory_allocated():
            return 0

    class Torch:
        pass

    def __init__(self, path, journal, **kwargs):
        self.path, self.journal, self.calls, self.model = str(path), journal, [], object()
        self.torch = self.Torch()
        self.torch.cuda = self.CUDA()

    def invoke(self, messages, tools, maximum, phase):
        if phase.startswith("final_commit"):
            value = {"faults": [{"component": "MG01", "reason": "high CPU usage", "time": None,
                                "evidence_ids": ["e1"], "explanation": "合成流程检查。"}]}
        elif "/overview" in phase:
            value = {"summary": "本地观测已整理，原因仍需结合其他节点核实。"}
        elif phase.startswith("case/"):
            value = {"hypotheses": [], "requests": []}
        else:
            data = json.JSONDecoder().raw_decode(messages[-1]["content"].split("\n", 1)[1])[0]
            value = {"groups": [{"observation_ids": data["allowed_ids"], "summary": "已有本地观测，需结合对照。",
                                  "priority": "routine"}], "retain_local": [], "defer": [], "questions": []}
        item = {"phase": phase, "seconds": 0, "hit_output_limit": False, "messages": deepcopy(messages),
                "raw_reply": json.dumps(value, ensure_ascii=False)}
        self.calls.append(item)
        with self.journal.open("a") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return item["raw_reply"]


class RequestingCoordinator(FixtureModel):
    """Fixture: one checkpoint asks two bounded queries; only one is executed."""
    def invoke(self, messages, tools, maximum, phase):
        if phase == "case/600/round0/0":
            return json.dumps({"hypotheses": [], "requests": [
                {"node_id": "device:MG01", "start": "18:00:00", "end": "18:05:00",
                 "tools": ["list_metrics"], "metrics": [], "focus": "核实 MG01 同期指标。", "basis_ids": []},
                {"node_id": "device:Tomcat04", "start": "18:00:00", "end": "18:05:00",
                 "tools": ["list_metrics"], "metrics": [], "focus": "核实 Tomcat04 同期指标。", "basis_ids": []}]})
        return super().invoke(messages, tools, maximum, phase)


class TemplateFixture(FixtureModel):
    def invoke(self, messages, tools, maximum, phase):
        if not phase.startswith("node_template/"):
            return super().invoke(messages, tools, maximum, phase)
        data = json.loads(messages[-1]["content"])
        value = {"items": [{"id": r["id"], "reading": "uncertain", "use": "background", "related_ids": []} for r in data["forms"]],
                 "request_id": None}
        item = {"phase": phase, "seconds": 0, "hit_output_limit": False, "messages": deepcopy(messages),
                "raw_reply": json.dumps(value, ensure_ascii=False)}
        self.calls.append(item)
        with self.journal.open("a") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        return item["raw_reply"]


class DoubleFaultFixture(TemplateFixture):
    def invoke(self, messages, tools, maximum, phase):
        if phase.startswith("final_commit"):
            return json.dumps({"faults": [
                {"component": node, "reason": None, "time": None,
                 "evidence_ids": ["e1"], "explanation": "合成双故障接口检查，不代表真实模型能力。"}
                for node in ("MG01", "Tomcat04")]})
        return super().invoke(messages, tools, maximum, phase)


class CaseRuntimeTests(unittest.TestCase):
    def test_double_fault_task_reaches_investigation_and_final_prediction(self):
        instruction = ("On March 4, 2021, between 18:00 and 18:30, two system failures occurred. "
                       "Identify the root cause components.")
        with self.query.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=["instruction", "scoring_points", "task_index"])
            writer.writeheader()
            writer.writerow({"instruction": instruction, "scoring_points": "SECRET_LABEL_NOT_FOR_MODELS",
                             "task_index": "task_3"})
        self.frames()
        with patch.object(runtime, "LocalModel", DoubleFaultFixture):
            runtime.nodes(self.root, self.prepared, 1800, Path("fixture4b"), max_self_requests=0, node_mode="template")
            runtime.coordinate(self.root, self.prepared, 1800, Path("fixture9b"), Path("fixture4b"),
                               max_queries=0, final=True, node_mode="template")
        central = self.root / "coordinator_1800"
        view = json.loads((central / "input_0.json").read_text())["view"]
        self.assertEqual(view["public_task"], {"instruction": instruction, "expected_failures": 2})
        prediction = json.loads((central / "prediction.json").read_text())
        self.assertEqual([p["root cause component"] for p in prediction], ["MG01", "Tomcat04"])
        scorer = self.root / "scorer.py"
        scorer.write_text("def evaluate(prediction, scoring_points):\n    return [], [], 0.0\n")
        evaluate_case(self.root, scorer)
        report = (self.root / "evaluation/结果.md").read_text()
        self.assertIn("故障1：MG01", report)
        self.assertIn("故障2：Tomcat04", report)
        for path in self.root.glob("**/*calls*.jsonl"):
            self.assertNotIn("SECRET_LABEL", path.read_text())

    def test_new_investigation_priority_can_replace_stale_fifo_order(self):
        from edge_rca.followup import request_key
        view = {"case_id": "fixture", "window_start_ms": START, "cutoff_ms": START + 1200000}
        old = {"node_id": "device:MG01", "start": "18:00:00", "end": "18:05:00",
               "tools": ["get_metrics"], "metrics": [CPU], "focus": "旧疑点", "basis_ids": []}
        other = {**old, "node_id": "device:Tomcat04"}
        fresh = {**other, "focus": "依据新观测重新确认", "basis_ids": ["e3"]}
        result = runtime.prioritize_requests([old, other], [fresh], view)
        self.assertEqual(result, [fresh, old])
        self.assertEqual(other["basis_ids"], [])
        self.assertEqual(runtime.prioritize_requests(result, [fresh], view,
                         {request_key(fresh, view)}), [old])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        nodes = []
        for node in ("device:MG01", "device:Tomcat04", "business:ServiceTest1"):
            events = []
            for i in range(30):
                raw = {"kpi_name": CPU, "value": 99999 if i == 25 else 30}
                modality = "metric"
                if node.startswith("business:"):
                    modality, raw = "business_metric", {"cnt": 10, "mrt": 20, "rr": 100, "sr": 100}
                events.append({"event_id": node + f"-{i}", "node_id": node, "source_id": node.split(":")[1],
                               "modality": modality, "observed_at_ms": START + i * 60000,
                               "available_at_ms": START + i * 60000, "raw": raw})
            path = node.replace(":", "_") + ".jsonl"
            (self.prepared / path).write_text("".join(json.dumps(e) + "\n" for e in events))
            nodes.append({"node_id": node, "source_id": node.split(":")[1], "path": path})
        (self.prepared / "manifest.json").write_text(json.dumps({"schema_version": 1,
            "window": {"start_ms": START, "end_ms": START + 1800000}, "nodes": nodes}))
        self.query = self.root / "queries.csv"
        self.instruction = "On March 4, 2021, between 18:00 and 18:30, identify the root cause component and reason."
        with self.query.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=["instruction", "scoring_points", "task_index"])
            writer.writeheader()
            writer.writerow({"instruction": self.instruction, "scoring_points": "SECRET_LABEL_NOT_FOR_MODELS", "task_index": "task_6"})

    def frames(self):
        runtime.prepare(self.prepared, self.root / "frames", self.query, expected_nodes=2)

    def test_all_causal_frames_and_public_input_exclude_labels(self):
        self.frames()
        meta = json.loads((self.root / "frames" / "complete.json").read_text())
        self.assertEqual(meta["device_nodes"], 2)
        self.assertEqual(meta["delivered_rows"], 90)
        for seconds, count in ((600,30),(1200,60),(1800,90)):
            info = json.loads((self.root / "frames" / f"c{seconds}" / "ready.json").read_text())
            self.assertEqual(info["delivered_rows"], count)
            self.assertEqual(len(info["nodes"]), 2)
        early = (self.root / "frames" / "c600" / "device_MG01.json").read_text()
        self.assertNotIn("99999", early)
        for path in (self.root / "frames").rglob("*.json"):
            self.assertNotIn("SECRET_LABEL", path.read_text())

    def test_incomplete_inventory_or_wrong_window_fails_before_models(self):
        with self.assertRaisesRegex(ValueError, "expected 18"):
            runtime.prepare(self.prepared, self.root / "bad", self.query)

    def test_template_node_to_coordinator_without_prose_resummarization(self):
        self.frames()
        with patch.object(runtime, "LocalModel", TemplateFixture):
            runtime.nodes(self.root, self.prepared, 1800, Path("fixture4b"), max_self_requests=0, node_mode="template")
            runtime.coordinate(self.root, self.prepared, 1800, Path("fixture9b"), Path("fixture4b"),
                               max_queries=0, final=True, node_mode="template")
        report = json.loads((self.root / "nodes_1800/reports.json").read_text())
        self.assertEqual(report["node_mode"], "template")
        self.assertTrue(all(r["template_rows"] for r in report["reports"]))
        view = json.loads((self.root / "coordinator_1800/input_0.json").read_text())["view"]
        self.assertEqual(view["report_detail_mode"], "template_facts")
        self.assertTrue(any(e["node_id"] == "device:MG01" for e in view["evidence"]))
        journal = (self.root / "nodes_1800/calls.jsonl").read_text()
        self.assertNotIn("/overview", journal)
        self.assertNotIn("SECRET_LABEL", journal)

    def test_all_nodes_coordinator_and_final_commit_contract(self):
        self.frames()
        with patch.object(runtime, "LocalModel", FixtureModel):
            for seconds in (600,1200,1800):
                runtime.nodes(self.root, self.prepared, seconds, Path("fixture4b"), max_self_requests=0)
                runtime.coordinate(self.root, self.prepared, seconds, Path("fixture9b"), Path("fixture4b"),
                                   max_queries=0, final=seconds==1800)
        last = self.root / "coordinator_1800"
        final = json.loads((last / "final_answer.json").read_text())
        self.assertIsNotNone(final["reply"])
        self.assertFalse(final["labels_used"])
        record = json.loads((last / "complete.json").read_text())
        self.assertEqual(record["device_nodes_received"], 2)
        report = json.loads((self.root / "nodes_1800" / "reports.json").read_text())["reports"][0]
        inbox = ReportInbox()
        inbox.receive(report)
        view, _ = make_view(inbox, report["case_id"], report["cutoff_ms"], None, include_report_details=False)
        self.assertEqual(len(view["node_summaries"]), 1)
        self.assertEqual(view["node_summaries"][0]["origin"], "node_overview")
        for path in self.root.glob("**/*calls*.jsonl"):
            self.assertNotIn("SECRET_LABEL", path.read_text())

    def test_previous_checkpoint_supplements_and_pending_are_carried(self):
        self.frames()
        with patch.object(runtime, "LocalModel", RequestingCoordinator):
            runtime.nodes(self.root, self.prepared, 600, Path("fixture4b"), max_self_requests=0)
            runtime.coordinate(self.root, self.prepared, 600, Path("fixture9b"), Path("fixture4b"),
                               max_queries=1, final=False)
            runtime.nodes(self.root, self.prepared, 1200, Path("fixture4b"), max_self_requests=0)
            runtime.coordinate(self.root, self.prepared, 1200, Path("fixture9b"), Path("fixture4b"),
                               max_queries=1, final=False)
        first = json.loads((self.root / "coordinator_600" / "state.json").read_text())
        self.assertTrue(first["executed"])
        self.assertEqual(len(first["pending"]), 1)
        self.assertTrue(any(answer.get("node_id") == "device:MG01" for answer in first["supplements"]))
        second = json.loads((self.root / "coordinator_1200" / "state.json").read_text())
        # The carried pending request was executed at 1200, not forgotten.
        self.assertEqual(len(second["pending"]), 0)
        self.assertEqual(len(second["executed"]), 2)
        view = json.loads((self.root / "coordinator_1200" / "input_0.json").read_text())["view"]
        self.assertEqual(len(view["pending_requests"]), 1)
        self.assertEqual(view["pending_requests"][0]["node_id"], "device:Tomcat04")
        self.assertTrue(view["evidence"])

    def test_1800_only_outputs_are_evaluable(self):
        self.frames()
        with patch.object(runtime, "LocalModel", FixtureModel):
            runtime.nodes(self.root, self.prepared, 1800, Path("fixture4b"), max_self_requests=0)
            runtime.coordinate(self.root, self.prepared, 1800, Path("fixture9b"), Path("fixture4b"),
                               max_queries=0, final=True)
        scorer = self.root / "scorer.py"
        scorer.write_text("def evaluate(prediction, scoring_points):\n    return [], [], 0.0\n")
        evaluate_case(self.root, scorer)
        result = json.loads((self.root / "evaluation" / "result.json").read_text())
        self.assertEqual([item["seconds"] for item in result["checkpoints"]], [1800])
        self.assertIn("第30分钟", (self.root / "evaluation" / "结果.md").read_text())

    def test_evaluator_rejects_unfinished_cluster(self):
        self.frames()
        with self.assertRaises(FileNotFoundError):
            evaluate_case(self.root, self.root / "unused_scorer.py")

    def test_cosmetic_json_fences_do_not_force_a_new_model_call(self):
        self.assertEqual(decode_object('{"x":1}`<|im_end|>'), {"x": 1})
        self.assertEqual(decode_object('```json\n{"x":1}\n```'), {"x": 1})
        for raw in ('{"x":1}{"x":2}', '{"x":1} arbitrary text', 'prefix {"x":1}'):
            with self.assertRaises(ValueError):
                decode_object(raw)

    def test_requested_pages_are_accumulated_across_rounds(self):
        self.assertEqual(runtime.merge_requested_pages(["e1:0"], ["e2:0"]),
                         ["e1:0", "e2:0"])
        self.assertEqual(runtime.merge_requested_pages(["e1:0", "e2:0"], ["e2:0", "directory:1"]),
                         ["e1:0", "e2:0", "directory:1"])

    def test_hypothesis_ledger_keeps_a_temporarily_dropped_candidate(self):
        redis = {"component": "Redis02", "reason": "high memory usage", "support_ids": ["e1"]}
        tomcat = {"component": "Tomcat02", "reason": "JVM Out of Memory (OOM) Heap", "support_ids": []}
        merged = runtime.merge_hypotheses([redis], [tomcat])
        keys = {(item["component"], item["reason"]) for item in merged}
        self.assertIn(("Redis02", "high memory usage"), keys)
        self.assertIn(("Tomcat02", "JVM Out of Memory (OOM) Heap"), keys)
        updated = {"component": "Redis02", "reason": "high memory usage", "support_ids": ["e2"]}
        merged = runtime.merge_hypotheses(merged, [updated])
        kept = next(item for item in merged if item["component"] == "Redis02")
        self.assertEqual(kept["support_ids"], ["e2"])

    def test_coordinator_state_prefers_latest_prior_checkpoint(self):
        for seconds, cutoff in ((600, START + 600000), (1200, START + 1200000)):
            directory = self.root / f"coordinator_{seconds}"
            directory.mkdir()
            (directory / "state.json").write_text(json.dumps({
                "case_id": "case", "cutoff_ms": cutoff, "supplements": [],
                "previous": [{"id": f"h{seconds}"}], "pending": [], "executed": {}}))
        self.assertEqual(runtime.coordinator_state(self.root, 1800)["cutoff_ms"], START + 1200000)
        self.assertEqual(runtime.coordinator_state(self.root, 1200)["cutoff_ms"], START + 600000)

    def test_assembled_pending_query_is_rebound_to_report_fact_ids(self):
        directory = self.root / "node_pages"
        page = directory / "p0"
        page.mkdir(parents=True)
        fact = {"id": "f1", "kind": "log", "status": "available", "category": "other",
                "count": 1, "first": "18:00:00", "last": "18:01:00", "details": {},
                "explicit_error": False, "gc_pause_max_ms": 0}
        (page / "input.json").write_text(json.dumps({"request_id": "r1", "facts": [fact]}))
        (page / "node_answer.json").write_text(json.dumps({"answer": {
            "groups": [], "pending_fact_ids": [], "questions": []}}))
        request = {"node_id": "device:MG01", "start": "18:00:00", "end": "18:05:00",
                   "tools": ["list_metrics"], "metrics": [], "focus": "核实内存。",
                   "basis_ids": ["f1"]}
        seed = {"case_id": "case", "node_id": "device:MG01", "window_start_ms": START,
                "cutoff_ms": START + 600000, "scope": {"start": "18:00:00", "end": "18:10:00"},
                "cpu_only_metric_ids": []}
        local = {"files": ["p0/node_answer.json"],
                 "pending_requests": [{"node_id": "device:MG01", "parent_request_id": "r1",
                                       "request": request}]}
        report = runtime.assemble_report(seed, local, directory, "总览")
        pending = report["pending_self_queries"][0]
        self.assertEqual(pending["parent_request_id"], f"device:MG01@{START + 600000}")
        self.assertEqual(pending["request"]["basis_ids"], ["s0_f1"])

    def test_shared_limits_are_not_repeated_as_data_loss(self):
        original = {"facts": [{"tool_fact": "实际观测=123。业务级观测，不能强行归属设备；中位数不是总计，极值时刻不是已证实故障时间。"}]}
        compact = runtime.compact_globals(original)
        self.assertEqual(compact["facts"][0]["tool_fact"], "实际观测=123")
        self.assertIn("业务级观测", original["facts"][0]["tool_fact"])
        self.assertIn("业务级观测", runtime.GLOBAL_LIMITS)


if __name__ == "__main__":
    unittest.main()
