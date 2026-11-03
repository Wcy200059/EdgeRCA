from copy import deepcopy
import unittest

from edge_rca.node_handoff import build_handoff


class NodeHandoffTests(unittest.TestCase):
    def setUp(self):
        self.request = {"request_id": "r1", "cutoff_ms": 100,
            "bundle": {"context": {"node_id": "device:x", "request_scope": {"cutoff": "18:01"},
                "alerts": [{"rule": "trace.latency", "action": "open"}]},
                "tools": {"get_metrics": [], "get_logs": [], "get_calls": [
                    {"id": "t1", "kind": "latency_over_analysis_window", "current": {"max": 5001}}]},
                "links": [{"from": "device:x", "to": "device:y"}],
                "catalog": {"t1": {"observation_id": "original:trace", "evidence_ids": ["row:18"]}},
                "cpu_only_observation_ids": ["original:metric"]}}
        self.report = {"request_id": "r1", "cutoff_ms": 100, "steps": []}

    def test_failed_or_unrun_steps_remain_outstanding_and_keep_connections(self):
        result = build_handoff(self.request, self.report)
        self.assertEqual(result["alert_evidence_not_summarized"], ["t1"])
        self.assertEqual(result["not_summarized_ids"]["get_calls"], ["t1"])
        self.assertEqual(result["observed_links"][0]["to"], "device:y")
        self.assertEqual(result["cpu_only_model_unread_ids"], ["original:metric"])

    def test_model_wording_does_not_overwrite_source_facts(self):
        self.report["steps"] = [{"tool": "get_calls", "tool_attempts": [{"accepted": True}],
            "reply": {"findings": [{"ids": ["t1"], "text": "最大延迟999毫秒。"}], "questions": []}}]
        result = build_handoff(self.request, self.report)
        self.assertFalse(result["semantics_verified"])
        self.assertEqual(result["findings"][0]["tool_facts"][0]["current"]["max"], 5001)
        self.assertEqual(result["findings"][0]["source_references"][0]["evidence_ids"], ["row:18"])
        self.assertIn("999", result["findings"][0]["model_text_unverified"])

    def test_wrong_request_or_cutoff_cannot_reuse_short_ids(self):
        for key, value in (("request_id", "r2"), ("cutoff_ms", 101)):
            report = deepcopy(self.report)
            report[key] = value
            with self.assertRaises(ValueError):
                build_handoff(self.request, report)

    def test_summary_without_successful_tool_execution_is_rejected(self):
        self.report["steps"] = [{"tool": "get_calls", "tool_attempts": [{"accepted": False}],
            "reply": {"findings": [{"ids": ["t1"], "text": "有延迟"}], "questions": []}}]
        with self.assertRaises(ValueError):
            build_handoff(self.request, self.report)


if __name__ == "__main__":
    unittest.main()
