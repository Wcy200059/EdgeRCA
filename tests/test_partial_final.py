"""Unsupported time must remain unknown while independent valid fields survive."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from edge_rca.bank_context import public_bank_context
from edge_rca.case_runtime import final_answer, retain_valid_fields_with_unknown_time


class PartialFinalTests(unittest.TestCase):
    def setUp(self):
        self.view = {"window_start_ms": 1614852000000, "cutoff_ms": 1614853800000,
                     "system_context": public_bank_context(),
                     "evidence": [{"id": "e1", "time_marks": ["18:05:00"]}]}
        self.fault = {"component": "MG01", "reason": "high CPU usage", "time": "18:06:00",
                      "evidence_ids": ["e1"], "explanation": "模型原始解释。"}

    def test_only_unsupported_time_changes_to_null(self):
        raw = {"faults": [deepcopy(self.fault)]}
        result, changes = retain_valid_fields_with_unknown_time(raw, self.view, 1)
        self.assertEqual(result["faults"][0], {**self.fault, "time": None})
        self.assertEqual(changes[0]["original_value"], "18:06:00")
        self.assertEqual(raw["faults"][0]["time"], "18:06:00")

    def test_foreign_evidence_invalid_component_and_wrong_fault_count_still_fail(self):
        for changes in ({"evidence_ids": ["e999"]}, {"evidence_ids": []},
                        {"component": "invented"}, {"reason": "invented"}, {"explanation": ""},
                        {"time": 180500}, {"time": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                retain_valid_fields_with_unknown_time({"faults": [{**self.fault, **changes}]}, self.view, 1)
        with self.assertRaises(ValueError):
            retain_valid_fields_with_unknown_time({"faults": [self.fault]}, self.view, 2)

    def test_valid_time_on_other_fault_is_preserved(self):
        second = {**self.fault, "component": "MG02", "time": "18:05:00.000"}
        result, changes = retain_valid_fields_with_unknown_time({"faults": [self.fault, second]}, self.view, 2)
        self.assertEqual(result["faults"][1], second)
        self.assertEqual([change["fault_index"] for change in changes], [0])

    def test_retries_happen_before_partial_recovery_and_audit_is_saved(self):
        fault = self.fault

        class Model:
            calls = 0

            def invoke(model, *args):
                model.calls += 1
                return json.dumps({"faults": [fault]})

        model = Model()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = final_answer(model, self.view, "A single failure occurred.", output)
            saved = json.loads((output / "final_answer.json").read_text())
            self.assertIsNone(result["faults"][0]["time"])
            self.assertEqual(model.calls, 2)
            self.assertEqual(saved["attempts"][-1]["source"], "retained_fields_with_unknown_time")
            self.assertEqual(saved["field_repairs"][0]["field"], "time")
