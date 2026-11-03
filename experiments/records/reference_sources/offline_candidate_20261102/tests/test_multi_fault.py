"""Regression checks for question-driven multi-fault output, without a GPU."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from edge_rca.bank_context import public_bank_context
from edge_rca.case_runtime import final_answer, validate_final_answers
from edge_rca.query import parse_failure_count


START = 1614852000000


class ReplyModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def invoke(self, messages, tools, maximum, phase):
        self.calls.append({"messages": deepcopy(messages), "maximum": maximum, "phase": phase})
        return json.dumps(next(self.replies))


class MultiFaultTests(unittest.TestCase):
    def setUp(self):
        self.view = {"window_start_ms": START, "cutoff_ms": START + 1800000,
                     "system_context": public_bank_context(),
                     "evidence": [{"id": "e1", "time_marks": ["18:05:00"]},
                                  {"id": "e2", "time_marks": ["18:15:00"]}]}
        self.faults = [
            {"component": "MG01", "reason": "high CPU usage", "time": "18:05:00",
             "evidence_ids": ["e1"], "explanation": "合成证据一。"},
            {"component": "Tomcat04", "reason": "high memory usage", "time": "18:15:00",
             "evidence_ids": ["e2"], "explanation": "合成证据二。"}]

    def test_public_count_words_and_adjectives(self):
        for text, expected in (("a single failure", 1), ("one documented failure", 1),
                               ("a recorded failure", 1), ("one identified failure", 1),
                               ("Two system failures", 2), ("two recorded failures", 2),
                               ("2 failures", 2), ("monitor this system", None),
                               ("failures may occur", None), ("18 nodes and 30 minutes", None)):
            with self.subTest(text=text):
                self.assertEqual(parse_failure_count(text), expected)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            parse_failure_count("one failure, but two failures were reported")

    def test_count_pairing_retries_and_official_object_serialization(self):
        model = ReplyModel([{"faults": self.faults[:1]}, {"faults": self.faults[::-1]}])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            final_answer(model, self.view, "Two system failures occurred; identify time, component and reason.", output)
            record = json.loads((output / "final_answer.json").read_text())
            self.assertEqual([a["valid"] for a in record["attempts"]], [False, True])
            self.assertEqual(record["expected_failures"], 2)
            prediction = json.loads((output / "prediction.json").read_text())
            self.assertEqual(len(prediction), 2)
            self.assertEqual(prediction[0], {"root cause occurrence datetime": "2021-03-04 18:15:00",
                                            "root cause component": "Tomcat04", "root cause reason": "high memory usage"})
            self.assertEqual(prediction[1]["root cause component"], "MG01")
        self.assertEqual(model.calls[0]["maximum"], 1400)
        self.assertIn("2个故障", model.calls[0]["messages"][0]["content"])
        self.assertNotIn("此题仅涉及单故障", model.calls[0]["messages"][0]["content"])

    def test_single_fault_still_writes_one_object(self):
        model = ReplyModel([{"faults": self.faults[:1]}])
        with tempfile.TemporaryDirectory() as directory:
            final_answer(model, self.view, "A single failure occurred.", Path(directory))
            self.assertEqual(len(json.loads((Path(directory) / "prediction.json").read_text())), 1)
        self.assertEqual(model.calls[0]["maximum"], 700)

    def test_cannot_fill_two_faults_with_one_duplicate_or_foreign_evidence(self):
        bad = [deepcopy(self.faults), [deepcopy(self.faults[0])] * 2,
               [deepcopy(self.faults[0])]]
        bad[0][1]["evidence_ids"] = ["e1"]  # e1 does not support the second time.
        for faults in bad:
            with self.subTest(faults=faults), self.assertRaises(ValueError):
                validate_final_answers({"faults": faults}, self.view, 2)
        bad_fields = deepcopy(self.faults)
        bad_fields[0]["component"] = "MG01, Tomcat04"
        with self.assertRaises(ValueError):
            validate_final_answers({"faults": bad_fields}, self.view, 2)

    def test_unknown_second_fault_is_explicit_without_fabricating_answer(self):
        unknown = {"component": None, "reason": None, "time": None,
                   "evidence_ids": [], "explanation": "第二件故障缺少证据，暂不能判断。"}
        value = {"faults": [self.faults[0], unknown]}
        self.assertEqual(validate_final_answers(value, self.view, 2), value)

    def test_supported_hypotheses_fill_all_null_two_fault_slots(self):
        view = deepcopy(self.view)
        view["previous_hypotheses"] = [
            {"component": "MG01", "reason": "high CPU usage", "assessment": "supported",
             "time": "18:05:00", "support_ids": ["e1"], "explanation": "协调者支持一。"},
            {"component": "Tomcat04", "reason": "high memory usage", "assessment": "supported",
             "time": "18:15:00", "support_ids": ["e2"], "explanation": "协调者支持二。"},
        ]
        blank = {"component": None, "reason": None, "time": None,
                 "evidence_ids": [], "explanation": "暂未填写。"}
        model = ReplyModel([{"faults": [deepcopy(blank), deepcopy(blank)]}])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            final_answer(model, view, "Two system failures occurred.", output)
            prediction = json.loads((output / "prediction.json").read_text())
            self.assertEqual([item["root cause component"] for item in prediction], ["MG01", "Tomcat04"])
            record = json.loads((output / "final_answer.json").read_text())
            self.assertTrue(any(attempt.get("source") == "merged_supported_seed" for attempt in record["attempts"]))

    def test_invalid_response_is_not_silently_scored_as_a_valid_single_fault(self):
        model = ReplyModel([{"faults": self.faults[:1]}] * 2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            final_answer(model, self.view, "Two failures occurred.", output)
            record = json.loads((output / "final_answer.json").read_text())
            self.assertIsNone(record["reply"])
            predictions = json.loads((output / "prediction.json").read_text())
            self.assertEqual(len(predictions), 2)
            self.assertTrue(all(not any(p.values()) for p in predictions))


if __name__ == "__main__":
    unittest.main()
