from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from edge_rca.fact_cards import basic_fact_checks, fact_card, revise_inputs


def alert(details, rule="metric.memory_high", action="open"):
    return {"id": "a1", "kind": "alert_evaluation", "rule": rule,
            "action": action, "signal": "sample_signal",
            "time": "2021-03-04T18:10:00.000+08:00", "details": details}


def reply(text):
    return {"findings": [{"ids": ["a1"], "text": text}]}


class FactCardTests(unittest.TestCase):
    def test_absolute_threshold_does_not_supply_a_fake_mean(self):
        fact = alert({"value": 98, "threshold": 90, "unit": "%"})
        card = fact_card(fact, {}, "get_metrics")
        self.assertIn("采样=98%", card)
        self.assertIn("阈值=90%", card)
        self.assertIn("没有历史基线", card)
        self.assertTrue(basic_fact_checks(reply("均值98%，上升"), [fact], "get_metrics"))
        self.assertFalse(basic_fact_checks(reply("当前98%，阈值90%，均值未计算"), [fact], "get_metrics"))

    def test_equal_baseline_and_clear_not_described_as_new_drop(self):
        fact = alert({"value": 1, "baseline_median": 1, "direction": "unchanged"},
                     "metric.relative_down", "cleared")
        card = fact_card(fact, {}, "get_metrics")
        self.assertIn("相等未变化", card)
        self.assertIn("条件已解除", card)
        self.assertTrue(basic_fact_checks(reply("值1，中位数1"), [fact], "get_metrics"))
        self.assertFalse(basic_fact_checks(reply("条件已解除，当前值1，与基线1相同"), [fact], "get_metrics"))

    def test_numeric_mean_must_match_its_source(self):
        fact = alert({"value": 7054, "baseline_mean": 1667.6666667,
                      "direction": "up"}, "metric.std_up")
        self.assertFalse(basic_fact_checks(reply("此前均值1667.67，当前7054"), [fact], "get_metrics"))
        self.assertTrue(basic_fact_checks(reply("平均7054"), [fact], "get_metrics"))

    def test_rate_and_escalation_cannot_be_silently_dropped(self):
        fact = alert({"value": .03, "kind": "rate_per_second",
                      "baseline_median": .01, "direction": "up"},
                     "metric.relative_up", "escalated")
        self.assertEqual(len(basic_fact_checks(reply("当前.03"), [fact], "get_metrics")), 2)
        self.assertFalse(basic_fact_checks(reply("告警加重，每秒速率.03，基线.01"), [fact], "get_metrics"))

    def test_missing_logs_does_not_mean_missing_traces(self):
        fact = {"id": "a1", "kind": "missing"}
        self.assertIn("没有日志", fact_card(fact, {}, "get_logs"))
        self.assertTrue(basic_fact_checks(reply("没有已到达记录"), [fact], "get_logs"))
        self.assertFalse(basic_fact_checks(reply("本范围没有日志，不代表没有其他数据"), [fact], "get_logs"))

    def test_trace_alert_keeps_baseline_and_escalation(self):
        fact = alert({"calls": 1424, "p95_ms": 503, "max_ms": 3191,
                      "baseline_p95_median_ms": 67, "baseline_windows": 18},
                     "trace.latency", "escalated")
        card = fact_card(fact, {}, "get_calls")
        self.assertIn("加重", card)
        self.assertIn("P95中位数=67", card)

    def test_input_revision_keeps_original_and_updates_bound_copy(self):
        fact = alert({"value": 1, "baseline_median": 1, "direction": "down"},
                     "metric.relative_down", "cleared")
        request = {"request_id": "r1", "bundle": {
            "tools": {"get_metrics": [deepcopy(fact)]},
            "catalog": {"a1": {"cpu_fact": deepcopy(fact)}}},
            "selected_pages": [{"facts": [deepcopy(fact)]}]}
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "old", Path(tmp) / "new"
            src.mkdir()
            original = json.dumps(request)
            (src / "r1.json").write_text(original)
            (src / "manifest.json").write_text(json.dumps({"requests": [{"path": "r1.json"}]}))
            revise_inputs(src, dst)
            updated = json.loads((dst / "r1.json").read_text())
            self.assertEqual((src / "r1.json").read_text(), original)
            self.assertEqual(updated["selected_pages"][0]["facts"][0]["details"]["direction"], "unchanged")
            self.assertEqual(updated["bundle"]["catalog"]["a1"]["cpu_fact"]["details"]["direction"], "unchanged")
            self.assertEqual(len(json.loads((dst / "manifest.json").read_text())["corrections"]), 1)


if __name__ == "__main__":
    unittest.main()
