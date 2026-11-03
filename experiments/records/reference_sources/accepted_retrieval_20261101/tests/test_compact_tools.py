"""Bound tool permissions, explicit retained evidence and busy-node updates."""

from copy import deepcopy
import json
import unittest

from edge_rca.compact_tools import BoundToolSession, PendingNodeUpdates, TOOL_ORDER, build_compact_tools, response_packet
from edge_rca.node_probe import decode_object


def task(eid, action="open", at=100, episode="ep"):
    return {"task_id": eid, "node_id": "device:Tomcat04", "ready_at_ms": at + 10,
            "data_access": {"node_id": "device:Tomcat04", "available_before_ms": at + 10},
            "transitions": [{"alert_id": episode, "node_id": "device:Tomcat04", "action": action,
                             "at_ms": at, "details": {"window_start_ms": at - 60},
                             "evidence_ids": [eid]}]}


class ToolBindingTests(unittest.TestCase):
    def setUp(self):
        self.bundle = {"context": {"node_id": "device:Tomcat04", "request_scope": {"cutoff": "18:01:10"}},
                       "tools": {name: [{"id": name}] for name in TOOL_ORDER}, "links": []}

    def test_arbitrary_tools_time_ranges_and_other_requests_are_rejected(self):
        bound = BoundToolSession("r1", self.bundle)
        for name, args in (("execute_python", {"request_id": "r1"}),
                           ("get_calls", {"request_id": "r1"}),
                           ("get_metrics", {"request_id": "r2"}),
                           ("get_metrics", {"request_id": "r1", "end_ms": 999999})):
            with self.assertRaises(ValueError):
                bound.execute(name, args)
        self.assertEqual(bound.next_tool, "get_metrics")
        for name in TOOL_ORDER:
            result = bound.execute(name, {"request_id": "r1"})
            self.assertEqual(result["facts"][0]["id"], name)
        self.assertIsNone(bound.next_tool)
        with self.assertRaises(ValueError):
            bound.execute("get_metrics", {"request_id": "r1"})

    def test_frozen_tool_result_cannot_be_mutated_by_caller(self):
        bound = BoundToolSession("r1", self.bundle)
        self.bundle["tools"]["get_metrics"].clear()
        self.assertEqual(len(bound.execute("get_metrics", {"request_id": "r1"})["facts"]), 1)

    def test_native_and_json_calls_decode_but_extra_objects_do_not(self):
        obj = {"name": "get_metrics", "arguments": {"request_id": "r1"}}
        self.assertEqual(decode_object("<tool_call>" + json.dumps(obj) + "</tool_call><|im_end|>", True), obj)
        with self.assertRaises(ValueError):
            decode_object(json.dumps(obj) + json.dumps(obj), True)


class PendingUpdatesTests(unittest.TestCase):
    def test_busy_worker_does_not_drop_later_clear_or_reopen(self):
        pending = PendingNodeUpdates("device:Tomcat04")
        pending.offer(task("first"))
        in_flight = pending.begin()
        pending.offer(task("cleared", "cleared", 200))
        pending.offer(task("reopened", "open", 300, "ep2"))
        pending.finish(in_flight["task_id"])
        update = pending.begin()
        self.assertEqual(update["coalesced_task_ids"], ["cleared", "reopened"])
        self.assertEqual([t["action"] for t in update["transitions"]], ["cleared", "open"])
        self.assertEqual(update["data_access"]["available_before_ms"], 310)
        self.assertEqual(pending.episode_anchors["ep"], 40)
        self.assertEqual(pending.episode_anchors["ep2"], 240)

    def test_identical_task_is_idempotent_and_unknown_episode_rejected(self):
        pending = PendingNodeUpdates("device:Tomcat04")
        self.assertTrue(pending.offer(task("first")))
        self.assertFalse(pending.offer(task("first")))
        with self.assertRaises(ValueError):
            pending.offer(task("bad", "cleared", 200, "unknown"))
        with self.assertRaises(ValueError):
            pending.finish("unstarted")


if __name__ == "__main__":
    unittest.main()
