from copy import deepcopy
import unittest

from edge_rca.alarm_pages import choose_representatives, enrich_alert_facts, plan_pages
from edge_rca.alert_coverage import inspect_task
from edge_rca.staged_probe import brief_facts


class AlarmPagesTests(unittest.TestCase):
    def bundle(self, count=5):
        facts = [{"id": f"a{i}", "kind": "alert_evaluation", "rule": "metric.std_up", "action": "open",
                  "time": "18:08", "signal": f"m{i}", "details": {"value": i}} for i in range(count)]
        return {"context": {"node_id": "device:x", "request_scope": {"cutoff": "18:08"}, "alerts": []},
            "tools": {"get_metrics": facts, "get_logs": [], "get_calls": []}, "links": [],
            "catalog": {f["id"]: {"observation_id": f["id"], "evidence_ids": [f"row{i}"]} for i,f in enumerate(facts)},
            "cpu_only_observation_ids": ["retained_metric"]}

    def test_dense_update_splits_all_34_targets_without_truncation(self):
        b = self.bundle(34)
        p = plan_pages(b, "r1")
        self.assertEqual(len(p["pages"]), 17)
        ids = [i for page in p["pages"] for i in page["mandatory_ids"]]
        self.assertEqual(len(ids), 34)
        self.assertEqual(set(ids), set(b["catalog"]))
        self.assertEqual(p["cpu_only_ids"], ["retained_metric"])

    def test_character_limit_does_not_cut_one_oversized_fact(self):
        b = self.bundle(1)
        b["tools"]["get_metrics"][0]["details"]["note"] = "原始事实" * 500
        p = plan_pages(b, "r1", max_characters=500)
        self.assertTrue(p["pages"][0]["oversized_single_fact"])
        self.assertEqual(p["pages"][0]["facts"], b["tools"]["get_metrics"])

    def test_unplanned_control_is_explicitly_deferred(self):
        b = self.bundle(1)
        b["tools"]["get_logs"] = [{"id": "l1", "kind": "missing", "note": "没有已到达日志"}]
        b["catalog"]["l1"] = {"observation_id": "no_logs"}
        p = plan_pages(b, "r1")
        self.assertEqual(p["deferred_ids"], ["l1"])
        self.assertEqual(plan_pages(b, "r1", extra_required=["l1"])["deferred_ids"], [])
        with self.assertRaises(ValueError):
            plan_pages(b, "r1", extra_required=["not_a_real_fact"])

    def test_trigger_spike_is_retained_beside_later_median(self):
        b = self.bundle(0)
        f = {"observation_id": "alarm", "modality": "alert", "rule": "metric.std_up", "action": "open",
             "at_ms": 0, "signal": "cpu", "details": {"value": 98, "baseline_mean": 20}, "evidence_ids": ["spike"]}
        new = enrich_alert_facts({"observations": [f]}, b)
        alert = new["tools"]["get_metrics"][0]
        self.assertEqual(alert["details"]["value"], 98)
        self.assertEqual(new["catalog"]["a1"]["evidence_ids"], ["spike"])
        text = brief_facts({"scope": {}, "observed_links": [], "facts": [alert]})["facts"][0]["observation"]
        self.assertIn("98", text)
        self.assertEqual(b["tools"]["get_metrics"], [])

    def test_case_selection_covers_observed_nodes_rules_and_actions(self):
        candidates = [{"page_id": "a", "characters": 20, "features": ["node:x", "rule:up"]},
                      {"page_id": "b", "characters": 10, "features": ["node:y", "rule:down", "action:clear"]},
                      {"page_id": "c", "characters": 30, "features": ["node:x", "rule:up"]}]
        chosen = choose_representatives(candidates)
        self.assertEqual({c["page_id"] for c in chosen}, {"a", "b"})

    def test_coverage_check_catches_missing_raw_or_cpu_fact_or_alarm(self):
        b = self.bundle(1)
        result = {"observations": [{"observation_id": "a0"}, {"observation_id": "retained_metric"}],
                  "evidence_index": {"a0": ["row0"], "retained_metric": ["row1"]}}
        plan = plan_pages(b, "r1")
        self.assertEqual(inspect_task(result, b, plan, [{}], ["row0", "row1"])["errors"], [])
        self.assertIn("raw_to_cpu_source_membership", inspect_task(result,b,plan,[{}],["row0","row1","lost"])["errors"])
        bad = deepcopy(plan); bad["pages"] = []
        self.assertIn("alert_snapshot_not_planned", inspect_task(result,b,bad,[{}],["row0","row1"])["errors"])


if __name__ == "__main__":
    unittest.main()
