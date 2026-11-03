from copy import deepcopy
import json
import unittest

from edge_rca.paged_probe import execute_page_call, page_prompt


class PageBindingTests(unittest.TestCase):
    def setUp(self):
        fact = {"id": "t1", "kind": "missing", "note": "没有已到达记录，不能判正常"}
        self.page = {"page_id": "r1-p1", "request_id": "r1", "tool": "get_calls",
                     "facts": [fact], "mandatory_ids": ["t1"]}
        self.request = {"request_id": "r1", "node_id": "device:x", "bundle": {
            "context": {"node_id": "device:x", "request_scope": {"cutoff": "18:05"}},
            "tools": {"get_calls": [deepcopy(fact)]}}}
        self.call = {"name": "get_calls", "arguments": {"request_id": "r1-p1"}}

    def test_wrong_page_arbitrary_time_and_stale_node_are_rejected(self):
        for call in ({"name": "get_metrics", "arguments": {"request_id": "r1-p1"}},
                     {"name": "get_calls", "arguments": {"request_id": "r2-p1"}},
                     {"name": "get_calls", "arguments": {"request_id": "r1-p1", "cutoff": "18:30"}}):
            with self.assertRaises(ValueError):
                execute_page_call(call, self.request, self.page)
        self.request["node_id"] = "device:other"
        with self.assertRaises(ValueError):
            execute_page_call(self.call, self.request, self.page)

    def test_tampered_or_omitted_fact_cannot_pass_as_frozen_input(self):
        bad = deepcopy(self.page)
        bad["facts"][0]["note"] = "伪造事实"
        with self.assertRaises(ValueError):
            execute_page_call(self.call, self.request, bad)
        bad = deepcopy(self.page); bad["mandatory_ids"] = []
        with self.assertRaises(ValueError):
            execute_page_call(self.call, self.request, bad)

    def test_page_prompt_does_not_demand_unprovided_modalities_or_future(self):
        value = execute_page_call(self.call, self.request, self.page)
        body = json.loads(page_prompt(value, self.page)[-1]["content"])
        self.assertNotIn("最近完整一分钟", body["task"])
        self.assertEqual(body["scope"]["cutoff"], "18:05")
        self.assertEqual(body["must_mention_ids"], ["t1"])
        value["facts"].clear()
        self.assertEqual(len(self.page["facts"]), 1)


if __name__ == "__main__":
    unittest.main()
