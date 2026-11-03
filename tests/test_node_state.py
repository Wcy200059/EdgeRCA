"""Check lossless incremental state without a model, API or background worker."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from edge_rca.node_state import NodeState
from edge_rca.prompts import build_node_prompt


def observation(identifier, at, description="CPU使用率95%，超过90%阈值。", kind="metric.cpu_high"):
    return {"observation_id": identifier, "kind": kind, "observed_at_ms": at,
            "available_at_ms": at, "description": description, "evidence_ids": ["e-"+identifier]}


def evidence(row, node="device:Tomcat01"):
    return {row["evidence_ids"][0]: {"node_id": node, "modality": "metric",
            "observed_at_ms": row["observed_at_ms"], "available_at_ms": row["available_at_ms"]}}


class NodeStateTests(unittest.TestCase):
    def setUp(self):
        self.state = NodeState("case1", "device:Tomcat01", 0, 1_800_000)

    def receive(self, identifier, at, *, cutoff=None, description=None, kind=None):
        kwargs = {}
        if description is not None:
            kwargs["description"] = description
        if kind is not None:
            kwargs["kind"] = kind
        row = observation(identifier, at, **kwargs)
        self.state.receive(row, evidence(row), at+1 if cutoff is None else cutoff)
        return row

    def response(self, *identifiers, summary="CPU偏高，根因仍未确定。"):
        return {"notes": [{"observation_id": key, "summary": summary} for key in identifiers],
                "questions": []}

    def test_cpu_observation_report_does_not_wait_for_or_require_a_model(self):
        self.receive("o1", 1000)
        reports = self.state.reports()
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["type"], "cpu_observation")
        self.assertEqual(reports[0]["observation"]["evidence_ids"], ["e-o1"])
        self.assertNotIn("model_note", reports[0])
        self.assertTrue(self.state.needs_update)

    def test_busy_node_keeps_receiving_and_old_response_cannot_ack_new_observation(self):
        self.receive("o1", 1000)
        request = self.state.begin_update(2000)
        old = deepcopy(request)
        self.receive("o2", 3000)
        self.assertTrue(self.state.busy)
        with self.assertRaisesRegex(RuntimeError, "in flight"):
            self.state.begin_update(4000)
        result = self.state.complete_update(request["request_id"], self.response("o1"))
        self.assertEqual(result["remaining"], 1)
        self.assertEqual(self.state.snapshot()["pending_ids"], ["o2"])
        self.assertEqual(request, old)
        self.assertEqual(len(self.state.reports()), 3)
        next_request = self.state.begin_update(4000)
        self.assertEqual([r["observation_id"] for r in next_request["packet"]["observations"]], ["o2"])
        self.assertEqual(next_request["packet"]["previous_findings"][0]["observation_id"], "o1")

    def test_partial_summary_keeps_omitted_observations_pending(self):
        for index in range(3):
            self.receive(f"o{index}", index*1000)
        request = self.state.begin_update(4000)
        result = self.state.complete_update(request["request_id"], self.response("o1"))
        self.assertEqual(result["remaining"], 2)
        self.assertEqual(self.state.snapshot()["pending_ids"], ["o0", "o2"])
        self.assertEqual(sum(row["type"] == "cpu_observation" for row in self.state.reports()), 3)

    def test_pages_are_fifo_and_never_top_k_drop_the_remaining_history(self):
        for index in range(15):
            self.receive(f"o{index}", index*1000)
        pages = []
        while self.state.needs_update:
            request = self.state.begin_update(20_000)
            ids = [row["observation_id"] for row in request["packet"]["observations"]]
            pages.append(ids)
            self.state.complete_update(request["request_id"], self.response(*ids))
        self.assertEqual([len(ids) for ids in pages], [6, 6, 3])
        self.assertEqual([key for ids in pages for key in ids], [f"o{i}" for i in range(15)])
        self.assertEqual(len(self.state.snapshot()["observations"]), 15)

    def test_problem_then_recovery_survive_while_model_is_busy(self):
        self.receive("started", 1000, kind="metric.cpu_high.open")
        request = self.state.begin_update(2000)
        self.receive("cleared", 3000, kind="metric.cpu_high.cleared", description="CPU降至40%，规则不再命中。")
        self.state.complete_update(request["request_id"], self.response("started"))
        reports = [row for row in self.state.reports() if row["type"] == "cpu_observation"]
        self.assertEqual([row["observation"]["kind"] for row in reports],
                         ["metric.cpu_high.open", "metric.cpu_high.cleared"])
        self.assertEqual(self.state.snapshot()["pending_ids"], ["cleared"])

    def test_empty_notes_do_not_starve_later_pages(self):
        for index in range(8):
            self.receive(f"o{index}", index*1000)
        first = self.state.begin_update(10_000)
        self.state.complete_update(first["request_id"], self.response())
        second = self.state.begin_update(10_000)
        ids = [row["observation_id"] for row in second["packet"]["observations"]]
        self.assertEqual(ids[:2], ["o6", "o7"])
        self.assertEqual(len(self.state.snapshot()["pending_ids"]), 8)

    def test_questions_are_reported_and_not_cleared_by_later_empty_output(self):
        self.receive("o1", 1000)
        first = self.state.begin_update(2000)
        response = self.response("o1")
        response["questions"] = ["同期请求延迟是否变化？"]
        self.state.complete_update(first["request_id"], response)
        report = self.state.reports()[-1]
        self.assertEqual(report["type"], "node_questions")
        self.assertIn("not confirmed facts", report["note"])
        self.receive("o2", 3000)
        second = self.state.begin_update(4000)
        self.state.complete_update(second["request_id"], self.response("o2"))
        self.assertEqual(self.state.snapshot()["questions"], ["同期请求延迟是否变化？"])

    def test_invalid_response_leaves_request_and_facts_intact_for_retry(self):
        self.receive("o1", 1000)
        request = self.state.begin_update(2000)
        before = self.state.snapshot()
        for response in (self.response("invented"), self.response("o1", "o1"),
                         {"answer": "CPU fault"}, self.response("o1", summary="x"*161)):
            with self.subTest(response=response), self.assertRaises(ValueError):
                self.state.complete_update(request["request_id"], response)
            self.assertEqual(self.state.snapshot(), before)
        self.state.fail_update(request["request_id"], "no model has been connected")
        self.assertFalse(self.state.busy)
        self.assertEqual(self.state.snapshot()["pending_ids"], ["o1"])

    def test_foreign_future_or_invented_evidence_is_rejected(self):
        row = observation("o1", 1000)
        candidates = [evidence(row, node="device:Mysql01"), {}, None]
        future = evidence(row)
        future["e-o1"]["available_at_ms"] = 3000
        candidates.append(future)
        for catalog in candidates:
            with self.subTest(catalog=catalog), self.assertRaises(ValueError):
                self.state.receive(row, catalog, 2000)
        with self.assertRaises(ValueError):
            self.state.receive(row, evidence(row), 1000)
        self.assertEqual(self.state.snapshot()["observations"], {})

    def test_duplicate_ids_cannot_replace_old_measurements(self):
        row = self.receive("o1", 1000)
        self.assertFalse(self.state.receive(row, evidence(row), 2000))
        changed = {**row, "description": "changed old data"}
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.state.receive(changed, evidence(changed), 2000)
        self.assertEqual(len(self.state.reports()), 1)

    def test_wrong_model_wording_cannot_overwrite_cpu_values(self):
        self.receive("o1", 1000)
        request = self.state.begin_update(2000)
        self.state.complete_update(request["request_id"], self.response("o1", summary="CPU下降。"))
        snapshot = self.state.snapshot()
        self.assertIn("95%", snapshot["observations"]["o1"]["observation"]["description"])
        self.assertEqual(snapshot["model_notes"]["o1"], "CPU下降。")
        self.assertIn("semantics not verified", self.state.reports()[-1]["validation"])

    def test_half_hour_history_stays_in_state_without_being_repeated_in_every_prompt(self):
        self.receive("early", 1000)
        request = self.state.begin_update(2000)
        self.state.complete_update(request["request_id"], self.response("early"))
        self.receive("late", 1_750_000, description="25分钟后出现新的超时日志。", kind="log.error")
        request = self.state.begin_update(1_800_000, previous_limit=0)
        self.assertEqual(len(self.state.snapshot()["observations"]), 2)
        self.assertEqual(len(request["packet"]["observations"]), 1)
        self.assertEqual(request["packet"]["previous_findings"], [])
        self.assertIn("user_prompt", build_node_prompt(request["packet"]))

    def test_journal_preserves_received_facts_and_model_notes_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"node.jsonl"
            self.state = NodeState("case1", "device:Tomcat01", 0, 1_800_000, path)
            self.receive("o1", 1000)
            request = self.state.begin_update(2000)
            self.state.complete_update(request["request_id"], self.response("o1"))
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["type"] for row in records],
                             ["observation", "request_started", "model_annotations"])
            with self.assertRaises(FileExistsError):
                NodeState("case1", "device:Tomcat01", 0, 1_800_000, path)

    def test_finding_identity_is_isolated_between_cases(self):
        self.receive("o1", 1000)
        another = NodeState("case2", "device:Tomcat01", 0, 1_800_000)
        row = observation("o1", 1000)
        another.receive(row, evidence(row), 2000)
        self.assertNotEqual(self.state.reports()[0]["finding_id"], another.reports()[0]["finding_id"])


if __name__ == "__main__":
    unittest.main()
