"""CPU checks for the model-only global-evidence relevance filter."""
import json
import unittest

from edge_rca.coordination import apply_global_filter, global_filter


class Model:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def count_input_tokens(self, messages):
        return len(json.dumps(messages)) // 4

    def invoke(self, messages, tools, maximum, phase):
        self.calls.append(messages)
        return self.reply


def view():
    return {"window_start_ms": 0, "cutoff_ms": 1,
            "public_task": {"instruction": "find the failure", "expected_failures": 1},
            "evidence": [
                {"id": "e1", "node_id": "device:MG02", "tool_fact": "local"},
                {"id": "e2", "node_id": "cluster:observations", "tool_fact": "global call a"},
                {"id": "e3", "node_id": "cluster:observations", "tool_fact": "global call b"}],
            "node_summaries": [{"node_id": "device:MG02", "available_fact_count": 2,
                                "loaded_fact_count": 1, "text_unverified": "x"}],
            "system_context": {"root_cause_components": ["MG02"]},
            "node_status": [{"node_id": "device:MG02", "status": "received"}]}


class GlobalFilterTests(unittest.TestCase):
    def test_keeps_only_selected_global_evidence_and_all_node_evidence(self):
        reply = json.dumps({"ids": ["e2"], "notes": "call a relevant"})
        triage = global_filter(Model(reply), view(), "case/1/global")
        self.assertEqual(triage["ids"], ["e2"])
        self.assertFalse(triage["fallback"])
        filtered = apply_global_filter(view(), triage)
        self.assertEqual([e["id"] for e in filtered["evidence"]], ["e1", "e2"])

    def test_invalid_filter_retains_all_global_evidence_instead_of_silent_drop(self):
        triage = global_filter(Model("{}"), view(), "case/1/global")
        self.assertTrue(triage["fallback"])
        self.assertEqual(set(triage["ids"]), {"e2", "e3"})
        self.assertEqual(len(apply_global_filter(view(), triage)["evidence"]), 3)

    def test_filter_output_cannot_change_root_cause_or_final_answer(self):
        triage = global_filter(Model(json.dumps({"ids": ["e2"], "notes": "x", "component": "MG02"})),
                               view(), "case/1/global")
        self.assertTrue(triage["fallback"])  # extra keys are rejected


if __name__ == "__main__":
    unittest.main()
