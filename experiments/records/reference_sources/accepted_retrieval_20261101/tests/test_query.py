"""The independent CLI needs only public times, never diagnostic labels."""

import csv
from pathlib import Path
import tempfile
import unittest

from edge_rca.query import parse_question_window, read_query_window


class QueryWindowTests(unittest.TestCase):
    def test_window_does_not_need_task_or_failure_count(self):
        instruction = "On March 4, 2021, between 18:00 and 18:30, monitor this system."
        self.assertEqual(parse_question_window(instruction),
                         ("2021-03-04 18:00:00", "2021-03-04 18:30:00"))

    def test_single_date_midnight_crossing(self):
        self.assertEqual(parse_question_window("On March 4, 2021, from 23:50 to 00:20."),
                         ("2021-03-04 23:50:00", "2021-03-05 00:20:00"))

    def test_explicit_end_date_is_respected(self):
        self.assertEqual(parse_question_window(
            "From March 4, 2021 at 23:50 to March 5, 2021 at 00:20."),
            ("2021-03-04 23:50:00", "2021-03-05 00:20:00"))
        with self.assertRaises(ValueError):
            parse_question_window("From March 5, 2021 at 18:00 to March 4, 2021 at 18:30.")

    def test_missing_and_empty_windows_fail(self):
        for instruction in ("between 18:00 and 18:30", "On March 4, 2021 at 18:00",
                            "On March 4, 2021 from 18:00 to 18:00"):
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                parse_question_window(instruction)

    def test_only_selected_instruction_affects_exported_window(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["instruction", "scoring_points"])
                writer.writeheader()
                writer.writerow({"instruction": "bad question in another row", "scoring_points": "unused"})
                writer.writerow({"instruction": "On March 4, 2021 from 18:00 to 18:30.",
                                 "scoring_points": "DO_NOT_EXPORT: January 1, 2000 01:00 02:00"})
            start, end, source = read_query_window(path, 1)
            self.assertEqual((start, end), ("2021-03-04 18:00:00", "2021-03-04 18:30:00"))
            self.assertEqual(set(source), {"query_file", "row_index"})
            self.assertNotIn("DO_NOT_EXPORT", str((start, end, source)))
            for index in (-1, True, 3):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    read_query_window(path, index)

    def test_missing_instruction_column_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "query.csv"
            path.write_text("answer\nnot a question\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "instruction"):
                read_query_window(path, 0)


if __name__ == "__main__":
    unittest.main()
