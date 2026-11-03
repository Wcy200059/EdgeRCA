"""The validator must agree with the exact evidence shown in both messages."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from edge_rca.bank_context import public_bank_context
from edge_rca.case_runtime import final_answer, register_displayed_briefs, validate_final_answers


START = 1614852000000


class FinalContextTests(unittest.TestCase):
    def setUp(self):
        self.view = {"window_start_ms": START, "cutoff_ms": START + 1800000,
                     "system_context": public_bank_context(), "evidence": []}
        self.refs = {"e1": {"id": "e1", "node_id": "device:MG01", "tool_fact": "visible prefix; hidden tail",
                            "time_marks": ["18:05:00", "18:06:00"]},
                     "e2": {"id": "e2", "node_id": "device:MG01", "tool_fact": "not shown",
                            "time_marks": ["18:07:00"]}}
        self.candidates = [{"evidence_brief": [{"id": "e1", "fact": "visible prefix", "times": ["18:05:00"]}]}]
        self.fault = {"component": "MG01", "reason": "high CPU usage", "time": "18:05:00.000",
                      "evidence_ids": ["e1"], "explanation": "基于当前可见证据。"}

    def test_only_literal_brief_and_displayed_times_are_registered(self):
        shown = register_displayed_briefs(self.view, self.candidates, self.refs)
        self.assertEqual(shown["evidence"][0]["tool_fact"], "visible prefix")
        self.assertEqual(shown["evidence"][0]["time_marks"], ["18:05:00"])
        self.assertEqual(len(shown["evidence"]), 1)
        validate_final_answers({"faults": [self.fault]}, shown, 1)
        for time, ids in (("18:06:00", ["e1"]), ("18:07:00", ["e2"]), ("18:05:00.001", ["e1"])):
            with self.subTest(time=time, ids=ids), self.assertRaises(ValueError):
                validate_final_answers({"faults": [{**self.fault, "time": time, "evidence_ids": ids}]}, shown, 1)
        self.assertEqual(self.view["evidence"], [])
        self.assertEqual(self.refs["e1"]["time_marks"], ["18:05:00", "18:06:00"])

    def test_final_validator_accepts_a_brief_still_present_after_user_payload_eviction(self):
        base = deepcopy(self.view)
        base["evidence"] = list(self.refs.values())
        base["previous_hypotheses"] = [{"component": "MG01", "reason": "high CPU usage",
            "support_ids": ["e1"], "assessment": "unresolved", "time": None, "explanation": "待复核。"}]

        class Model:
            def invoke(model, messages, *args):
                self.assertIn('"id": "e1"', messages[0]["content"])
                self.assertNotIn("e2", messages[0]["content"])
                return json.dumps({"faults": [self.fault]})

        def pack(model, full, system, budget, question):
            return [{"role": "system", "content": system}, {"role": "user", "content": "empty working set"}], deepcopy(self.view)

        with tempfile.TemporaryDirectory() as directory, patch("edge_rca.case_runtime.model_input", pack):
            result = final_answer(Model(), base, "A single failure occurred.", Path(directory))
            self.assertIsNotNone(result)
            visible = json.loads((Path(directory) / "final_input.json").read_text())["view"]
            self.assertEqual([e["id"] for e in visible["evidence"]], ["e1"])

    def test_long_explanation_does_not_discard_otherwise_valid_fields(self):
        shown = register_displayed_briefs(self.view, self.candidates, self.refs)
        validate_final_answers({"faults": [{**self.fault, "explanation": "完整说明。" * 60}]}, shown, 1)
        with self.assertRaises(ValueError):
            validate_final_answers({"faults": [{**self.fault, "explanation": "  "}]}, shown, 1)
