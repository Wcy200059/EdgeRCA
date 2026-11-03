"""CPU checks for paged evidence and the node summary restoration."""

import json
import unittest

from edge_rca.evidence_digest import detail_pages
from edge_rca.evidence_memory import fit, validate_page_requests


START = 1614852000000


class CountModel:
    max_context = 4000

    def count_input_tokens(self, prompt):
        return sum(len(message["content"]) for message in prompt) // 2


def fact(identifier, node="device:Redis02", high=98.0, median=23.0, unit="%"):
    return {"id": identifier, "node_id": node, "tool_fact": "指标观测", "time_marks": ["18:10:00.000"],
            "template_payload": {"subject": "MEM", "pattern": "variation_observed",
                                 "fields": {"unit": unit,
                                            "whole_prefix": {"min": 11.0, "max": high, "median": median}}},
            "initial_display": True}


def view(evidence, summaries=None, requested=None):
    return {"case_id": "c", "window_start_ms": START, "cutoff_ms": START + 1800000,
            "system_context": {}, "public_task": {"instruction": "single failure", "expected_failures": 1},
            "node_status": [], "evidence": evidence,
            "node_summaries": summaries or [{"node_id": "device:Redis02", "available_fact_count": 1,
                                             "loaded_fact_count": 1, "unloaded_fact_count": 0,
                                             "text_unverified": "节点总览"}],
            "node_questions": ["是否还有日志"], "followup_status": [{"node_id": "device:Redis02"}],
            "previous_hypotheses": [], "pending_requests": [],
            "requested_pages": requested or []}


class EvidenceMemoryTests(unittest.TestCase):
    def test_accumulated_pages_preserve_earlier_details(self):
        expanded = fact("e1")
        expanded["detail_fields"] = {"samples": [{"time": f"18:{i:02d}:00.000", "text": "x" * 250}
                                                 for i in range(20)]}
        parts = detail_pages(expanded["detail_fields"])
        self.assertGreater(len(parts), 1)
        base = view([expanded, fact("e2", high=99), fact("e3", high=100)],
                    requested=["e1:0", "e1:1", "e2:0", "e3:0"])
        model = CountModel()
        model.max_context = 20000
        prompt, _ = fit(model, base, "system", 100)
        payload = json.loads(prompt[1]["content"])
        shown = json.loads(next(row[2] for row in payload["evidence"] if row[0] == "e1"))
        self.assertEqual(shown["loaded_pages"], [0, 1])
        self.assertEqual(shown["leaves"], parts[0] + parts[1])
        with self.assertRaises(ValueError):
            validate_page_requests(base["requested_pages"],
                                   {"evidence_page_index": {"e1": len(parts), "e2": 1, "e3": 1}})

    def test_packing_keeps_a_second_node_before_duplicate_node_coverage(self):
        class TwoFactModel(CountModel):
            max_context = 2000

            def count_input_tokens(self, prompt):
                return 2000 if len(json.loads(prompt[1]["content"])["evidence"]) > 2 else 10

        first = fact("e1", node="device:MG01")
        second = fact("e2", node="device:Mysql01", high=90)
        third = fact("e3", node="device:Mysql01", high=91)
        prompt, visible = fit(TwoFactModel(), view([first, second, third]), "system", 200)
        self.assertEqual({item["node_id"] for item in visible["evidence"]},
                         {"device:MG01", "device:Mysql01"})
        self.assertEqual(len(visible["evidence"]), 2)
        self.assertEqual(len(json.loads(prompt[1]["content"])["evidence"]), 2)

    def test_v2_loaded_counts_follow_actual_display_not_the_full_archive(self):
        hidden = fact("e2", high=99)
        hidden["initial_display"] = False
        base = view([fact("e1"), hidden])
        base["node_reply_schema"] = "node_slots_v2"
        base["node_summaries"][0]["loaded_fact_count"] = 999
        prompt, _ = fit(CountModel(), base, "system", 200)
        item = json.loads(prompt[1]["content"])["node_summaries"][0]
        self.assertEqual(item["available_fact_count"], 2)
        self.assertEqual(item["loaded_fact_count"], 1)
        self.assertEqual(item["unloaded_fact_count"], 1)

    def test_summaries_and_numeric_leads_survive_paging(self):
        prompt, visible = fit(CountModel(), view([fact("e1")]), "system", 200)
        payload = json.loads(prompt[1]["content"])
        self.assertEqual(payload["node_summaries"][0]["node_id"], "device:Redis02")
        self.assertNotIn("key_numbers", payload["node_summaries"][0])
        self.assertIn("text_unverified", payload["node_summaries"][0])
        self.assertEqual(payload["node_questions"], ["是否还有日志"])
        self.assertEqual(payload["followup_status"][0]["node_id"], "device:Redis02")
        self.assertIn("evidence_directory", payload)

    def test_over_budget_evicts_visible_rows_but_keeps_directory(self):
        facts = [fact(f"e{i}", high=20.0 + i) for i in range(1, 31)]
        prompt, visible = fit(CountModel(), view(facts), "system", 300)
        payload = json.loads(prompt[1]["content"])
        self.assertIn("context_omissions", payload)
        self.assertGreater(payload["context_omissions"]["unloaded_count"], 0)
        self.assertEqual(payload["directory_pages"], 1)
        self.assertTrue(payload["evidence_directory"])
        self.assertIn("关键ID", payload["directory_columns"])
        self.assertLess(len(visible["evidence"]), len(facts))

    def test_requested_detail_page_is_loaded_and_page_requests_validate(self):
        base = view([fact("e1")], requested=["e1:0"])
        prompt, visible = fit(CountModel(), base, "system", 100)
        payload = json.loads(prompt[1]["content"])
        self.assertIn("detail_page", payload["evidence"][0][2])
        validate_page_requests(["e1:0", "directory:0"],
                               {"evidence_page_index": {"e1": 1}, "directory_pages": 1})
        with self.assertRaises(ValueError):
            validate_page_requests(["e1:1"], {"evidence_page_index": {"e1": 1}, "directory_pages": 1})
        with self.assertRaises(ValueError):
            validate_page_requests(["e1:0", "e1:0"],
                                   {"evidence_page_index": {"e1": 1}, "directory_pages": 1})


if __name__ == "__main__":
    unittest.main()
