"""Reports must preserve frozen answers, missing rows and scoring tolerances."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.bank_combine import combine
from tools.bank_summary import summarize, write_summary


def create_case(root: Path, job: str, row: int, tolerance: int = 300) -> Path:
    case = root / f"bank_case_{job}_{row}"
    (case / "evaluation").mkdir(parents=True)
    (case / "coordinator_1800").mkdir()
    prediction = [{"root cause occurrence datetime": "2021-03-24 19:31:01",
                   "root cause component": "MG01", "root cause reason": "network latency"}]
    raw = json.dumps(prediction).encode()
    (case / "coordinator_1800/prediction.json").write_bytes(raw)
    result = {"row": row, "score": 1, "fully_correct": True, "any_field_correct": True,
              "final_reply_valid": True, "prediction": prediction,
              "prediction_sha256": hashlib.sha256(raw).hexdigest(),
              "time_tolerance_seconds": tolerance,
              "scoring_points": "The only root cause occurrence time is within 1 minutes (i.e., <=1min) of 2021-03-24 19:30:00"}
    (case / "evaluation/result.json").write_text(json.dumps(result))
    return case


class BankSummaryTests(unittest.TestCase):
    def test_rescore_keeps_both_tolerances_and_frozen_bytes_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = create_case(root, "a", 0)
            before = (case / "evaluation/result.json").read_bytes()
            official = summarize(root, "a", time_tolerance_seconds=60)
            relaxed = summarize(root, "a", time_tolerance_seconds=300)
            self.assertEqual(official["mean_official_field_score"], 0)
            self.assertEqual(relaxed["mean_field_score"], 1)
            self.assertIsNone(relaxed["mean_official_field_score"])
            self.assertEqual((case / "evaluation/result.json").read_bytes(), before)

    def test_missing_directories_are_counted_when_expected_rows_supplied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_case(root, "a", 1)
            report = summarize(root, "a", expected_rows=3)
            self.assertEqual(report["missing_or_failed"], [0, 2])
            self.assertEqual(report["questions_with_evaluation"], 1)
            self.assertNotIn("unseen_questions", report)

    def test_mixed_saved_tolerances_require_explicit_rescore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_case(root, "a", 0, 60)
            create_case(root, "a", 1, 300)
            with self.assertRaisesRegex(ValueError, "mixed"):
                summarize(root, "a")
            self.assertEqual(summarize(root, "a", time_tolerance_seconds=60)["fully_correct_count"], 0)

    def test_ranked_hits_are_not_reused_with_different_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = create_case(root, "a", 0)
            path = case / "evaluation/result.json"
            data = json.loads(path.read_text())
            data.update(top3_fully_correct=True, ranked_evaluation={"candidate_count": 2})
            path.write_text(json.dumps(data))
            report = summarize(root, "a", time_tolerance_seconds=60)
            self.assertNotIn("top3_correct_count", report)

    def test_existing_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_case(root, "a", 0)
            report = summarize(root, "a")
            output = root / "report"
            write_summary(report, output)
            with self.assertRaises(FileExistsError):
                write_summary(report, output)


class BankCombineTests(unittest.TestCase):
    def test_mixed_source_versions_are_rejected_before_creating_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for job, row, source in (("a", 0, "old = True\n"), ("b", 1, "old = False\n")):
                case = create_case(root, job, row)
                package = case / "source/edge_rca"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text(source)
            with self.assertRaisesRegex(ValueError, "mixed inference"):
                combine(root, root / "new", "a", "b", split_row=1, expected_rows=2)
            self.assertFalse((root / "new").exists())

    def test_combines_complete_arrays_without_touching_old_broken_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_case(root, "a", 0)
            create_case(root, "b", 1)
            old = root / "bank_case_old_0"
            old.mkdir()
            marker = old / "keep.txt"
            marker.write_text("old evidence")
            output = root / "new"
            report = combine(root, output, "a", "b", split_row=1, expected_rows=2)
            self.assertEqual(report["60"]["questions_with_evaluation"], 2)
            self.assertEqual(report["300"]["fully_correct_count"], 2)
            self.assertTrue((output / "bank_case_combined_0").is_symlink())
            self.assertEqual(marker.read_text(), "old evidence")
            with self.assertRaises(FileExistsError):
                combine(root, output, "a", "b", split_row=1, expected_rows=2)

    def test_missing_source_fails_before_creating_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_case(root, "a", 0)
            with self.assertRaises(FileNotFoundError):
                combine(root, root / "new", "a", "b", split_row=1, expected_rows=2)
            self.assertFalse((root / "new").exists())

    def test_changed_prediction_is_rejected_before_creating_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = create_case(root, "a", 0)
            (case / "coordinator_1800/prediction.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "frozen prediction"):
                combine(root, root / "new", "a", "b", split_row=1, expected_rows=1)
            self.assertFalse((root / "new").exists())


if __name__ == "__main__":
    unittest.main()
