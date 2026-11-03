"""CPU checks for the catalogue mode before it is sent to a GPU."""

import json
import unittest

from edge_rca.node_catalogue import choose


class CatalogueModel:
    def __init__(self):
        self.pages = 0

    def invoke(self, prompt, tools, maximum, phase):
        self.pages += 1
        payload = json.loads(prompt[-1]["content"])
        ids = [item["id"] for item in payload["catalogue"]]
        return json.dumps({"expand": ids[:1], "defer": []})


def row(index, subject="MEM"):
    return {"id": f"f{index}", "template": "metric", "subject": f"{subject}{index}",
            "pattern_description": "观测到波动", "fields": {"unit": "%",
                                                             "whole_prefix": {"count": 10, "min": 1, "max": 90, "median": 20},
                                                             "extremes": []}}


class NodeCatalogueTests(unittest.TestCase):
    def test_invalid_reply_is_unprocessed_not_fabricated_background(self):
        class InvalidModel:
            def invoke(self, prompt, tools, maximum, phase):
                return "not-json"

        rows = [row(i) for i in range(3)]
        decisions, audit = choose(InvalidModel(), rows,
                                  {"node_id": "device:Redis02", "scope": {}, "request_id": "r1"})
        self.assertEqual(set(decisions.values()), {"unprocessed"})
        self.assertTrue(all(not attempt["valid"] for attempt in audit["pages"][0]["attempts"]))

    def test_pages_and_all_ids_get_one_decision(self):
        rows = [row(i) for i in range(14)]
        model = CatalogueModel()
        decisions, audit = choose(model, rows, {"node_id": "device:Redis02", "scope": {}, "request_id": "r1"})
        self.assertEqual(len(decisions), 14)
        self.assertEqual(audit["total_observations"], 14)
        self.assertEqual(len(audit["pages"]), 2)
        self.assertEqual(audit["omitted_count"], 0)
        self.assertEqual(model.pages, 2)

    def test_protected_change_enters_the_one_page_even_when_balanced_slots_are_full(self):
        rows = [row(i) for i in range(14)]
        rows[-1]["protected"] = True
        rows[-1]["pattern_key"] = "variation_observed"
        decisions, audit = choose(CatalogueModel(), rows,
                                  {"node_id": "device:Redis02", "scope": {}, "request_id": "r1"})
        shown = [identifier for page in audit["pages"] for identifier in page["ids"]]
        self.assertIn(rows[-1]["id"], shown)
        self.assertEqual(len(shown), 14)

    def test_exact_duplicate_is_an_alias_not_a_silent_drop(self):
        original = row(0)
        duplicate = dict(original, id="f1")
        decisions, audit = choose(CatalogueModel(), [original, duplicate],
                                  {"node_id": "device:Redis02", "scope": {}, "request_id": "r1"})
        self.assertIn("f1", audit["aliases"].get("f0", []))


if __name__ == "__main__":
    unittest.main()
