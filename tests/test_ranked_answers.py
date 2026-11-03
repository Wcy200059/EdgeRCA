"""Whole-fault-set ranking and independent Top-3 scoring."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from edge_rca.case_runtime import final_answer, validate_final_answers
from edge_rca.ranked_answers import build, score
from test_multi_fault import ReplyModel
import test_multi_fault as fault_fixtures


class RankedAnswerTests(unittest.TestCase):
    def setUp(self):
        fixture = fault_fixtures.MultiFaultTests()
        fixture.setUp()
        self.view, self.faults = fixture.view, fixture.faults

    def test_single_fault_alternatives_are_separate_from_the_first_prediction(self):
        value = {"faults": self.faults[:1], "alternatives": [{"faults": self.faults[1:]}]}
        model = ReplyModel([value])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            final_answer(model, self.view, "A single failure occurred.", output, max_candidates=3)
            first = json.loads((output / "prediction.json").read_text())
            record = json.loads((output / "ranked_predictions.json").read_text())
            self.assertEqual(len(first), 1)
            self.assertEqual(len(record["candidates"]), 2)
            points = ("The only predicted root cause component is Tomcat04\n"
                      "The only predicted root cause reason is high memory usage\n")
            result = score(record, first, points, time_tolerance_seconds=60)
            self.assertFalse(result["candidates"][0]["fully_correct"])
            self.assertTrue(result["fully_correct_within_top3"])
            self.assertEqual(result["first_correct_rank"], 2)
            changed = deepcopy(first)
            changed[0]["root cause component"] = "Tomcat04"
            with self.assertRaises(ValueError):
                score(record, changed, points, time_tolerance_seconds=60)

    def test_double_fault_permutations_are_duplicates_not_new_guesses(self):
        primary = {"faults": self.faults}
        alternate = deepcopy(primary)
        alternate["faults"][0]["component"] = "MG02"
        record = build(primary, [{"faults": self.faults[::-1]}, alternate], self.view, 2, 3,
                       validate_final_answers)
        self.assertEqual(len(record["candidates"]), 2)
        self.assertEqual(len(record["candidates"][1]["prediction"]), 2)
        self.assertEqual(record["candidates"][1]["prediction"][1]["root cause component"], "Tomcat04")
        self.assertIn("duplicate", record["rejected"][0]["error"])

    def test_invalid_alternative_does_not_destroy_valid_first_choice(self):
        bad = deepcopy(self.faults[1:])
        bad[0]["evidence_ids"] = ["foreign_id"]
        record = build({"faults": self.faults[:1]}, [{"faults": bad}, {"faults": self.faults[1:]}],
                       self.view, 1, 3, validate_final_answers)
        self.assertEqual(record["candidates"][0]["faults"], self.faults[:1])
        self.assertEqual(len(record["candidates"]), 2)
        self.assertEqual(len(record["rejected"]), 1)
        self.assertEqual(record["candidates"][1]["rank"], 2)


if __name__ == "__main__":
    unittest.main()
