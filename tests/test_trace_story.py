"""The walkthrough must distinguish saved evidence, visibility and executed work."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tools.trace_story import enrich_trace, metric_series, visible_evidence


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def call(phase, payload):
    return {"phase": phase, "messages": [{"role": "user", "content": json.dumps(payload)}],
            "raw_reply": "{}", "call": 1}


class TraceStoryTests(unittest.TestCase):
    def test_only_actual_payload_and_system_excerpts_are_visible(self):
        call_record = call("final_commit/0", {"evidence": [["e1", "device:A", "displayed", ["10:00"]]]})
        call_record["messages"].insert(0, {"role": "system", "content":
            '再输出：[{"evidence_brief":[{"id":"e1","fact":"another visible excerpt","times":["10:02"]},'
            '{"id":"e3","fact":"literal excerpt","times":["10:01"]}]}]'})
        binding = {"e1": {"source_id": "device:A@1/s0_f1", "origin": "node_template"},
                   "e2": {"source_id": "hidden", "origin": "node_template"},
                   "e3": {"source_id": "q1/f1", "node_id": "device:A", "origin": "followup"}}
        facts = visible_evidence(call_record, binding)
        self.assertEqual([item["id"] for item in facts], ["e1", "e3"])
        self.assertEqual(facts[0]["system_briefs"], [{"fact": "another visible excerpt", "time_marks": ["10:02"]}])
        self.assertEqual(facts[1]["fact"], "literal excerpt")
        self.assertEqual(facts[1]["shown_in"], "system_brief")

    def test_round_order_and_node_visibility_follow_real_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            node = case / "nodes_1800/device_A"
            save(node / "manifest.json", {"files": ["step000/node_answer.json", "step001/node_answer.json"]})
            save(node / "step000/node_answer.json", {"answer": {"request_id": "request-0"}})
            save(node / "step001/node_answer.json", {"answer": {"request_id": "request-1"}})
            save(node / "report.json", {"template_rows": [
                {"id": "s0_f1", "fields": {}, "model_selection": {"use": "report"}},
                {"id": "s1_f1", "fields": {}}]})
            journal = [call("node_catalogue/request-0/0/0", {"node": "device:A", "catalogue": [{"id": "f1"}]}),
                       call("node_template/request-0/0/0", {"node": "device:A", "forms": [{"id": "f1"}]}),
                       call("node_catalogue/request-1/0/0", {"node": "device:A", "catalogue": [{"id": "f2"}]})]
            (case / "nodes_1800/calls.jsonl").write_text("\n".join(json.dumps(c) for c in journal))
            central = case / "coordinator_1800"
            save(central / "input_0.json", {"view": {"evidence": [{"id": "e1"}, {"id": "e2"}]},
                 "binding": {"e1": {"source_id": "device:A@1/s0_f1", "origin": "node_template"}}})
            save(central / "input_1.json", {"view": {"evidence": [{"id": "e3"}]},
                 "binding": {"e3": {"source_id": "q1/f1", "origin": "followup"}}})
            save(central / "node_followup_0/manifest.json", {"files": ["step000/node_answer.json"]})
            save(central / "node_followup_0/step000/node_answer.json", {"answer": {
                "node_id": "device:A", "request_id": "q1", "facts": [{"id": "f1"}]}})
            save(central / "queries/q1/request.json", {"request": {"node_id": "device:A", "focus": "核查"}})
            saved_calls = [call("case/1800/round0/0", {"evidence": [["e1", "device:A", "visible", []]]}),
                           call("case/1800/round1/0", {"evidence": [["e3", "device:A", "returned", []]]})]
            (central / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in saved_calls))
            data = {"nodes": [{"node": "A", "template_rows": [
                        {"id": "s0_f1", "subject": "x", "use": "report", "protected": False},
                        {"id": "s1_f1", "subject": "x", "use": None, "protected": False}]}],
                    "central": {"decisions": [{"round": "0", "reply": {"requests": [{}, {}]}, "attempts": []},
                                               {"round": "1", "reply": {}, "attempts": []}],
                                "complete": {}}, "case": {}, "flow": {}}
            story = enrich_trace(case, data)
            first, second = story["nodes"][0]["template_rows"]
            self.assertTrue(first["catalogue_seen"])
            self.assertTrue(first["expanded"])
            self.assertFalse(second["catalogue_seen"])
            self.assertFalse(second["expanded"])
            self.assertEqual(story["flow"]["initial_visible"], 1)
            self.assertEqual(story["story"]["rounds"][0]["snapshot"]["archive_count"], 2)
            self.assertEqual(len(story["story"]["executed"]), 1)
            self.assertEqual(story["story"]["executed"][0]["loaded_in_next_round"], ["e3"])
            self.assertEqual([event["kind"] for event in story["story"]["events"]],
                             ["raw", "facts", "nodes", "decision", "request", "return", "decision", "final"])

    def test_raw_metric_samples_are_matched_by_node_and_metric_and_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case / "frames").mkdir()
            connection = sqlite3.connect(case / "frames/inbox.sqlite")
            connection.execute("CREATE TABLE events (node_id TEXT, modality TEXT, observed_at_ms INTEGER, payload TEXT)")
            for node, metric, value in (("device:A", "metric1", "98"), ("device:A", "metric1", "NaN"),
                                        ("device:A", "metric2", "7"), ("device:B", "metric1", "3")):
                event = {"raw": {"kpi_name": metric, "value": value}, "source_file": "original.csv", "source_row": 8}
                connection.execute("INSERT INTO events VALUES (?, 'metric', 1, ?)", (node, json.dumps(event)))
            connection.commit()
            connection.close()
            nodes = [{"node": "A", "template_rows": [{"subject": "metric1", "template": "metric_change",
                                                         "use": "report", "protected": True, "expanded": True}]}]
            metric_series(case, nodes)
            samples = nodes[0]["template_rows"][0]["series"]
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["value"], 98)
            self.assertEqual(samples[0]["source_file"], "original.csv")


if __name__ == "__main__":
    unittest.main()
