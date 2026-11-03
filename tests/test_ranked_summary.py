"""Old runs without alternative answers must not inflate the Top-3 denominator."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import bank_summary


class RankedSummaryTests(unittest.TestCase):
    def test_candidate_hit_rate_uses_only_runs_with_candidate_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row, extra in ((0, {}), (1, {"top3_fully_correct": True,
                                          "ranked_evaluation": {"candidate_count": 2}})):
                target = root / f"bank_case_fixture_{row}" / "evaluation"
                target.mkdir(parents=True)
                (target / "result.json").write_text(json.dumps({
                    "score": 0, "fully_correct": False, "any_field_correct": False,
                    "model_calls": 1, "failed": [], **extra}))
            with patch.object(sys, "argv", ["bank_summary", "--root", str(root), "--job", "fixture"]):
                bank_summary.main()
            report = json.loads((root / "bank_accuracy_fixture/summary.json").read_text())
            self.assertEqual(report["questions_with_evaluation"], 2)
            self.assertEqual(report["questions_with_ranked_candidates"], 1)
            self.assertEqual(report["top3_exact_match_rate"], 1)
            self.assertEqual(report["exact_match_rate"], 0)


if __name__ == "__main__":
    unittest.main()
