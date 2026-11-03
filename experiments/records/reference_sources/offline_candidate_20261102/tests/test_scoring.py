"""Scorer window checks: same object matching, configurable time tolerance."""

import unittest

from edge_rca.scoring import evaluate


POINTS = ("The only predicted root cause component is MG01\n"
          "The only predicted root cause reason is network latency\n"
          "The only root cause occurrence time is within 1 minutes (i.e., <=1min) of 2021-03-24 19:30:00")


def prediction(when):
    return ('[{"root cause occurrence datetime": "%s", '
            '"root cause component": "MG01", "root cause reason": "network latency"}]' % when)


class ScoringTests(unittest.TestCase):
    def test_five_minute_window_accepts_a_sixty_one_second_gap(self):
        passed, failed, score = evaluate(prediction("2021-03-24 19:31:01"), POINTS)
        self.assertEqual(score, 1.0)
        self.assertEqual(failed, [])

    def test_original_one_minute_window_still_rejects_it(self):
        passed, failed, score = evaluate(prediction("2021-03-24 19:31:01"), POINTS,
                                         time_tolerance_seconds=60)
        self.assertEqual(score, 0.67)
        self.assertIn("2021-03-24 19:30:00", failed)

    def test_five_minutes_does_not_accept_fourteen_minutes(self):
        passed, failed, score = evaluate(prediction("2021-03-24 19:44:00"), POINTS)
        self.assertLess(score, 1.0)

    def test_two_fault_permutations_still_match_objects(self):
        points = ("The 1-th predicted root cause component is MG01\n"
                  "The 1-th predicted root cause reason is network latency\n"
                  "The 1-th root cause occurrence time is within 1 minutes (i.e., <=1min) of 2021-03-24 19:30:00\n"
                  "The 2-th predicted root cause component is Redis02\n"
                  "The 2-th predicted root cause reason is high memory usage\n"
                  "The 2-th root cause occurrence time is within 1 minutes (i.e., <=1min) of 2021-03-24 19:35:00")
        value = ('[{"root cause occurrence datetime": "2021-03-24 19:35:30", "root cause component": "Redis02", "root cause reason": "high memory usage"},'
                 '{"root cause occurrence datetime": "2021-03-24 19:30:00", "root cause component": "MG01", "root cause reason": "network latency"}]')
        self.assertEqual(evaluate(value, points)[2], 1.0)


if __name__ == "__main__":
    unittest.main()
