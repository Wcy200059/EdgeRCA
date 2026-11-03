"""A repair may reload a known archived fact, never invent missing evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from edge_rca.bank_context import public_bank_context
from edge_rca.case_runtime import final_answer


class FinalReloadTests(unittest.TestCase):
    def check_reload(self, identifier):
        archive = {"window_start_ms": 1614852000000, "cutoff_ms": 1614853800000,
                   "system_context": public_bank_context(), "previous_hypotheses": [],
                   "evidence": [{"id": "e1", "node_id": "device:MG01", "tool_fact": "本题已保存的观测",
                                 "time_marks": ["18:05:00"]}]}
        packing = []

        def pack(model, full, system, budget, question):
            packing.append(deepcopy(full.get("requested_pages", [])))
            visible = deepcopy(full)
            visible["evidence"] = full["evidence"] if "e1:0" in full.get("requested_pages", []) else []
            return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(visible)}], visible

        class Model:
            calls = 0

            def invoke(model, messages, *args):
                model.calls += 1
                if model.calls == 2 and identifier == "e1":
                    self.assertIn("本题已保存的观测", json.loads(messages[1]["content"].split("\n请修正：")[0])["evidence"][0]["tool_fact"])
                return json.dumps({"faults": [{"component": "MG01", "reason": "high CPU usage",
                    "time": "18:05:00", "evidence_ids": [identifier], "explanation": "依据所引用的观测。"}]})

        model = Model()
        with tempfile.TemporaryDirectory() as directory, patch("edge_rca.case_runtime.model_input", pack):
            result = final_answer(model, archive, "A single failure occurred.", Path(directory))
            audit = json.loads((Path(directory) / "final_answer.json").read_text())
        self.assertEqual(model.calls, 2)
        self.assertNotIn("requested_pages", archive)
        return result, audit, packing

    def test_known_evicted_evidence_is_loaded_for_the_existing_retry(self):
        result, audit, packing = self.check_reload("e1")
        self.assertIsNotNone(result)
        self.assertEqual(packing, [[], ["e1:0"]])
        self.assertEqual(audit["attempts"][0]["requested_existing_evidence"], ["e1"])

    def test_unknown_id_is_never_fabricated_or_requested(self):
        result, audit, packing = self.check_reload("e999")
        self.assertIsNone(result)
        self.assertEqual(packing, [[]])
        self.assertNotIn("requested_existing_evidence", audit["attempts"][0])
