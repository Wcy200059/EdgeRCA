from copy import deepcopy
import json
import unittest

from edge_rca.coordinator import make_packet, occurrence_ms, validate
from edge_rca.node_reports import messages, ReportInbox
from edge_rca.report_cards import card
from edge_rca.report_probe import report_pages


START = 1614852000000


def node_report(node="device:Tomcat04", cutoff=START + 600000, case="case"):
    return {"case_id": case, "node_id": node, "cutoff_ms": cutoff, "window_start_ms": START,
            "groups": [{"source_ids": ["f1"], "summary": "待核实的模型解释"}],
            "tool_card_catalog": {"f1": {"id": "f1", "tool_fact": "P95=98毫秒；最大值10020毫秒。"}},
            "retain_local": [], "defer": [], "cpu_only_metric_ids": ["other"]}


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.packet = {"node_id": "device:Tomcat04", "scope": {"start": "18:00:00", "end": "18:10:00"},
                       "limitations": [], "catalog": {}}
        self.fact = {"id": "f1", "kind": "latency_over_analysis_window",
                     "current": {"count": 12047, "p50": 0, "p95": 98, "max": 10020},
                     "reference": {"count": 0, "p50": None, "p95": None, "max": None},
                     "slow_calls": [{"start": "18:00:20.598", "end": "18:00:30.618", "duration_ms": 10020}],
                     "invalid_count": 0}

    def test_exact_statistic_names_and_event_time_survive_merge(self):
        candidate = {"id": "g1", "source_ids": ["f1"], "summary": "错把P95写成10020", "cpu_facts": [self.fact]}
        prompt = messages(self.packet, [candidate], merge=True, readable=True)
        payload = json.loads(prompt[-1]["content"].split("\n", 1)[1])
        text = payload["facts"][0]["tool_facts"][0]
        for required in ("全部调用12047次", "P95=98毫秒", "最大单次耗时=10020毫秒", "18:00:20.598"):
            self.assertIn(required, text)
        self.assertNotIn('"f1"', json.dumps(payload))
        self.assertEqual(payload["facts"][0]["id"], "g1")
        self.assertEqual(candidate["cpu_facts"], [self.fact])

    def test_connection_stats_are_not_lost_at_merge(self):
        fact = {"id": "f3", "kind": "observed_connection", "from": "device:MG01", "to": "device:dockerA2",
                "child_latency_ms": {"calls": 1339, "p95": 628, "max": 2036}, "interval_mismatch_count": 1339}
        prompt = messages(self.packet, [{"id": "g1", "summary": "连接", "cpu_facts": [fact]}], merge=True, readable=True)
        text = prompt[-1]["content"]
        for item in ("1339", "628", "2036", "时间区间", "device:MG01", "device:dockerA2"):
            self.assertIn(item, text)

    def test_missing_modality_is_explicit_and_not_guessed(self):
        fact = {"id": "f1", "kind": "missing"}
        with self.assertRaisesRegex(ValueError, "modality"):
            card(fact, self.packet)
        self.packet["catalog"] = {"f1": {"cpu_fact": {"modality": "trace"}}}
        self.assertIn("没有已到达的已完成调用", card(fact, self.packet)["tool_fact"])

    def test_unknown_units_do_not_become_gigabytes(self):
        fact = {"id": "f1", "metric": "JVMFreeMemory", "meaning": "剩余量", "unit": "source_unit_unverified",
                "reference": {"count": 0, "median": None},
                "current": {"count": 2, "median": 2.8e9, "min": 2.44735e9, "max": 3.1796e9},
                "change": {"direction": "unknown"}, "quality": {"invalid": 0, "conflicts": 0, "counter_resets": 0}}
        text = card(fact, self.packet)["tool_fact"]
        self.assertIn("2.44735e+09", text)
        self.assertIn("不能自行换算", text)
        self.assertIn("无法判断变化", text)

    def test_readable_paging_preserves_all_facts_and_never_truncates(self):
        facts = [dict(self.fact, id=f"f{i}") for i in range(19)]
        self.assertEqual([f for chunk in report_pages(self.packet, facts, 4, 2000) for f in chunk], facts)
        with self.assertRaisesRegex(ValueError, "never truncate"):
            report_pages(self.packet, facts, max_chars=5)


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.inbox = ReportInbox()
        self.inbox.receive(node_report())
        self.packet = make_packet(self.inbox, "case", START + 600000)
        self.ref = self.packet["evidence"][0]["id"]
        self.reply = {"hypotheses": [{"id": "h1", "component": "Tomcat04", "time": "18:00:20.598",
                       "reason": None, "support_ids": [self.ref], "counter_ids": [],
                       "uncertainties": ["缺少下游报告"], "explanation": "见较长调用，原因未知。"}], "questions": []}

    def test_prefix_uses_historical_report_not_future_latest_or_other_case(self):
        self.inbox.receive(node_report(cutoff=START + 1800000))
        self.inbox.receive(node_report(case="other"))
        self.inbox.receive(node_report("device:dockerA2", START + 1800000))
        packet = make_packet(self.inbox, "case", START + 600000)
        self.assertEqual(len(packet["evidence"]), 1)
        self.assertEqual(packet["evidence"][0]["id"], self.ref)
        self.assertEqual(next(n["status"] for n in packet["node_status"] if n["node_id"] == "device:dockerA2"), "not_received")

    def test_new_cumulative_report_replaces_old_instead_of_adding_counts(self):
        new = node_report(cutoff=START + 1800000)
        new["tool_card_catalog"]["f1"]["tool_fact"] = "累计32940次"
        self.inbox.receive(new)
        packet = make_packet(self.inbox, "case", START + 1800000)
        self.assertEqual(len(packet["evidence"]), 1)
        self.assertEqual(packet["evidence"][0]["tool_fact"], "累计32940次")

    def test_note_is_not_the_evidence_and_missing_coverage_is_visible(self):
        self.assertNotIn("待核实", self.packet["evidence"][0]["tool_fact"])
        self.assertIn("待核实", self.packet["node_groups"][0]["node_note_unverified"])
        statuses = self.packet["node_status"]
        self.assertEqual(len(statuses), 18)
        self.assertEqual(sum(n["status"] == "not_received" for n in statuses), 17)

    def test_known_wrong_node_note_is_withheld_but_tool_facts_remain(self):
        report = node_report()
        report["groups"][0]["summary"] = "共有25366次慢调用"
        report["groups"][0]["cpu_facts"] = [{"current": {"count": 25366, "p95": 22, "max": 2036}}]
        inbox = ReportInbox()
        inbox.receive(report)
        packet = make_packet(inbox, "case", START + 600000)
        self.assertIsNone(packet["node_groups"][0]["node_note_unverified"])
        self.assertTrue(packet["node_groups"][0]["note_warnings"])
        self.assertEqual(len(packet["evidence"]), 1)
        self.assertEqual(inbox.history[0]["groups"][0]["summary"], "共有25366次慢调用")

    def test_known_candidate_without_report_remains_a_candidate(self):
        self.assertIn("Mysql01", self.packet["system_context"]["root_cause_components"])
        self.assertNotIn("dockerA2", self.packet["system_context"]["root_cause_components"])

    def test_valid_incomplete_hypothesis_and_empty_hypotheses_allowed(self):
        self.assertEqual(validate(self.reply, self.packet), self.reply)
        self.assertEqual(validate({"hypotheses": [], "questions": []}, self.packet)["hypotheses"], [])

    def test_nonexistent_future_or_other_node_citations_rejected(self):
        for ref in ("f1", "device:Mysql01@0/f1", f"device:Tomcat04@{START + 1800000}/f1"):
            reply = deepcopy(self.reply)
            reply["hypotheses"][0]["support_ids"] = [ref]
            with self.assertRaisesRegex(ValueError, "citations"):
                validate(reply, self.packet)

    def test_future_time_wrong_candidate_and_unsupported_hypothesis_rejected(self):
        for field, value in (("time", "18:11:00"), ("component", "dockerA2"), ("reason", "made_up"), ("support_ids", [])):
            reply = deepcopy(self.reply)
            reply["hypotheses"][0][field] = value
            with self.assertRaises(ValueError):
                validate(reply, self.packet)

    def test_same_fact_cannot_be_both_support_and_counter(self):
        self.reply["hypotheses"][0]["counter_ids"] = [self.ref]
        with self.assertRaisesRegex(ValueError, "separate"):
            validate(self.reply, self.packet)

    def test_questions_only_target_known_nodes(self):
        self.reply["questions"] = [{"node_id": "made_up", "focus": "查下游", "basis_ids": []}]
        with self.assertRaisesRegex(ValueError, "known node"):
            validate(self.reply, self.packet)

    def test_cross_midnight_time_keeps_correct_day_and_cutoff(self):
        start = 1614873000000  # 2021-03-04 23:50 Shanghai
        self.assertEqual(occurrence_ms("00:05:00", start, start + 1800000), start + 900000)
        with self.assertRaises(ValueError):
            occurrence_ms("00:20:00", start, start + 1800000)


if __name__ == "__main__":
    unittest.main()
