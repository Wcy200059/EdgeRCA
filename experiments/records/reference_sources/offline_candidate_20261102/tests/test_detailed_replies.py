"""Check partial answers and their evidence binding, without an actual LLM."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from edge_rca import case_runtime as runtime
from edge_rca import node_templates
from edge_rca.case_eval import evaluate_case
from edge_rca.coordination import make_view
from edge_rca.node_explanations import validate as validate_interpretation
from edge_rca.node_reports import ReportInbox
from test_case_runtime import FixtureModel
import test_case_runtime as runtime_fixtures
from test_node_templates import metric, packet


def explanation(meaning="本范围存在指标变化"):
    return {"meaning": meaning, "can_answer": "能确认已提供采样的变化，不能直接确定根因",
            "cannot_answer": "没有逐点曲线，尚不能确认高值在采样间隙是否持续",
            "next_check": "需要相关时段的完整采样及同期调用对照"}


class DetailedFixture(FixtureModel):
    def invoke(self, messages, tools, maximum, phase):
        data = json.loads(messages[-1]["content"])
        if phase.startswith("node_catalogue/"):
            value = {"expand": [r["id"] for r in data["catalogue"]], "background": [], "defer": []}
        elif phase.startswith("node_template/"):
            value = {"items": [], "request_id": None}
            for row in data["forms"]:
                item = {"id": row["id"], "reading": "uncertain", "use": "report", "related_ids": [],
                        "interpretation": explanation()}
                if data.get("investigation_request") is not None:
                    item["relation_to_request"] = "insufficient"
                value["items"].append(item)
        elif phase.startswith("final_commit"):
            identifier = data["evidence"][0][0]
            def fault(component):
                return {"component": component, "reason": None, "time": None,
                        "evidence_ids": [identifier], "explanation": "合成合同检查，不是模型能力证明。"}
            value = {"faults": [fault("MG01")], "alternatives": [{"faults": [fault("Tomcat04")]}]}
        elif phase == "case/1800/round0/0":
            value = {"hypotheses": [], "requests": [{
                "node_id": "device:MG01", "start": "18:20:00", "end": "18:30:00",
                "tools": ["get_metrics", "get_logs"], "metrics": [runtime_fixtures.CPU],
                "focus": "核对CPU采样能否支持高占用，日志是否提供原因，说明不能回答的部分。", "basis_ids": []}]}
        else:
            return super().invoke(messages, tools, maximum, phase)
        raw = json.dumps(value, ensure_ascii=False)
        self.calls.append({"phase": phase, "messages": deepcopy(messages), "seconds": 0,
                           "hit_output_limit": False, "raw_reply": raw})
        with self.journal.open("a") as handle:
            handle.write(json.dumps(self.calls[-1], ensure_ascii=False) + "\n")
        return raw


class DetailedReplyTests(unittest.TestCase):
    def test_interpretation_requires_known_and_unknown_parts(self):
        value = explanation()
        self.assertEqual(validate_interpretation(value), value)
        for field in ("meaning", "can_answer", "cannot_answer", "next_check"):
            bad = dict(value)
            del bad[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_interpretation(bad)
        with self.assertRaises(ValueError):
            validate_interpretation({**value, "cannot_answer": " "})
        with self.assertRaises(ValueError):
            validate_interpretation({**value, "meaning": "长" * 41})

    def test_directed_query_explains_all_returned_facts_and_keeps_original_question(self):
        class Model:
            def __init__(self):
                self.calls = []

            def invoke(self, messages, tools, maximum, phase):
                self.calls.append((phase, deepcopy(messages)))
                self_data = json.loads(messages[-1]["content"])
                self.assertion = self_data
                return json.dumps({"items": [
                    {"id": row["id"], "reading": "uncertain", "use": "uncertain", "related_ids": [],
                     "relation_to_request": "outside_scope", "interpretation": explanation()}
                    for row in self_data["forms"]], "request_id": None}, ensure_ascii=False)

        p = packet([metric(f"f{i}") for i in range(1, 15)])
        p["request"] = {"focus": "请同时核对变化持续情况与日志能否解释调用变慢。"}
        original = deepcopy(p)
        model = Model()
        with tempfile.TemporaryDirectory() as directory:
            result = node_templates.answer(model, p, Path(directory), catalogue=True, detailed=True)
        self.assertEqual(p, original)
        self.assertEqual(result["screening"]["model_candidates"], 14)
        self.assertEqual(result["catalogue_audit"]["omitted_count"], 0)
        self.assertTrue(all(phase.startswith("node_template/") for phase, _ in model.calls))
        self.assertTrue(all(json.loads(prompt[-1]["content"])["investigation_request"] == p["request"]
                            for _, prompt in model.calls))
        response = result["question_response"]
        self.assertEqual(response["question"], p["request"]["focus"])
        self.assertEqual(len(response["interpretations_unverified"]), 14)
        self.assertTrue(all(i["interpretation"]["cannot_answer"] for i in response["interpretations_unverified"]))
        self.assertEqual(result["facts"][0]["detail_fields"]["current"]["max"], 98)

    def test_new_reply_passes_to_coordinator_and_ranked_eval_without_labels(self):
        case = runtime_fixtures.CaseRuntimeTests("test_all_causal_frames_and_public_input_exclude_labels")
        case.setUp()
        self.addCleanup(case.doCleanups)
        # Only the evaluation process may see this label text and marker.
        label = "The only predicted root cause component is Tomcat04\nPRIVATE_SCORING_MARKER"
        # Use a CSV writer because the scoring field now contains a newline.
        import csv
        with case.query.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=["instruction", "scoring_points", "task_index"])
            writer.writeheader()
            writer.writerow({"instruction": case.instruction, "scoring_points": label, "task_index": "task_3"})
        case.frames()
        with patch.object(runtime, "LocalModel", DetailedFixture):
            runtime.nodes(case.root, case.prepared, 1800, Path("fixture4b"),
                          max_self_requests=0, node_mode="catalogue", detailed=True)
            runtime.coordinate(case.root, case.prepared, 1800, Path("fixture9b"), Path("fixture4b"),
                               max_queries=1, final=True, node_mode="catalogue", detailed=True, max_candidates=3)
        reports = json.loads((case.root / "nodes_1800/reports.json").read_text())["reports"]
        inbox = ReportInbox()
        for report in reports:
            inbox.receive(report)
        view, _ = make_view(inbox, reports[0]["case_id"], reports[0]["cutoff_ms"], None,
                            include_report_details=False)
        self.assertEqual(view["node_reply_schema"], "node_slots_v2")
        central = case.root / "coordinator_1800"
        journal = (central / "calls.jsonl").read_text()
        self.assertIn("can_answer", journal)
        self.assertIn("cannot_answer", journal)
        followup = json.loads((central / "input_1.json").read_text())["view"]["investigation_updates"][0]
        self.assertIn("说明不能回答的部分", followup["question"])
        self.assertTrue(followup["interpretations_unverified"])
        self.assertTrue(all(i["id"].startswith("e") and i["interpretation"]["cannot_answer"]
                            for i in followup["interpretations_unverified"]))
        self.assertIn("can_answer", (case.root / "nodes_1800/calls.jsonl").read_text())
        scorer = case.root / "scorer.py"
        scorer.write_text("# Existing scorer path required by the evaluation entry\n")
        evaluate_case(case.root, scorer)
        result = json.loads((case.root / "evaluation/result.json").read_text())
        self.assertFalse(result["fully_correct"])
        self.assertTrue(result["top3_fully_correct"])
        self.assertEqual(result["ranked_evaluation"]["first_correct_rank"], 2)
        self.assertEqual(len(result["prediction"]), 1)
        for path in case.root.glob("**/*calls*.jsonl"):
            self.assertNotIn("PRIVATE_SCORING_MARKER", path.read_text())


if __name__ == "__main__":
    unittest.main()
