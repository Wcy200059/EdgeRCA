from copy import deepcopy
import unittest

from edge_rca.rolling_memory import RollingNodeMemory, memory_messages, validate_memory_reply


def transition(aid, action, at, rule="metric.relative_up", signal="cpu"):
    return {"alert_id": aid, "node_id": "device:x", "action": action,
            "at_ms": at, "rule": rule, "signal": signal,
            "details": {"value": 20, "baseline_median": 10, "direction": "up"},
            "evidence_ids": [f"e-{aid}-{at}"]}


def snapshot(aid, start, last=None, closed=None, rule="metric.relative_up", signal="cpu", hits=1):
    row = {"alert_id": aid, "node_id": "device:x", "rule": rule, "signal": signal,
           "first_seen_ms": start, "last_seen_ms": last or start,
           "status": "open" if closed is None else "cleared", "hits": hits,
           "latest": {"value": 20}, "evidence_ids": [f"e-{aid}-{start}"], "peak_score": 10}
    if closed is not None:
        row["cleared_at_ms"] = closed
    return row


class RollingMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = RollingNodeMemory("case", "device:x", 0, 1000)

    def test_duplicate_delivery_is_idempotent_and_different_nodes_cannot_mix(self):
        t, a = transition("a", "open", 10), snapshot("a", 10)
        gid = self.memory.sync([t], [a], 20)[0]
        self.assertEqual(self.memory.sync([t], [a], 20), [])
        self.assertEqual(self.memory.project(gid)["version"], 1)
        other = deepcopy(t)
        other["node_id"] = "device:y"
        with self.assertRaises(ValueError):
            self.memory.sync([other], [a], 20)
        with self.assertRaises(ValueError):
            self.memory.sync([transition("future", "open", 50)], [], 20)

    def test_two_rules_same_signal_do_not_double_activation_and_one_clear_does_not_clear_all(self):
        rule = "metric.std_up"
        changes = [transition("a", "open", 10), transition("b", "open", 10, rule)]
        alerts = [snapshot("a", 10), snapshot("b", 10, rule=rule)]
        gid = self.memory.sync(changes, alerts, 20)[0]
        first = self.memory.project(gid)
        self.assertEqual(first["signal_activation_cycles"], 1)
        self.assertEqual(first["rule_episodes"], 2)
        self.memory.sync([transition("a", "cleared", 30)],
                         [snapshot("a", 10, closed=30), alerts[1]], 40)
        result = self.memory.project(gid)
        self.assertEqual(result["status"], "triggered")
        self.assertEqual(result["active_rules"], [rule])

    def test_clear_reopen_retains_earlier_interval_and_stable_id(self):
        gid = self.memory.sync([transition("a", "open", 10)], [snapshot("a", 10)], 20)[0]
        old = self.memory.freeze(gid)
        self.memory.sync([transition("a", "cleared", 30)], [snapshot("a", 10, closed=30)], 40)
        self.memory.sync([transition("b", "open", 50)],
                         [snapshot("a", 10, closed=30), snapshot("b", 50)], 60)
        result = self.memory.project(gid)
        self.assertEqual(result["signal_activation_cycles"], 2)
        self.assertEqual(len(result["state_changes"]), 3)
        self.assertEqual(old["fact"]["signal_activation_cycles"], 1)
        self.assertEqual(len(self.memory.records), 1)

    def test_late_model_reply_cannot_mark_new_evidence_processed(self):
        gid = self.memory.sync([transition("a", "open", 10)], [snapshot("a", 10)], 20)[0]
        old = self.memory.freeze(gid)
        self.memory.sync([], [snapshot("a", 10, last=25, hits=2)], 30)
        result = self.memory.complete(old["request_id"],
                                      {"observation_id": gid, "text": "当前仍有规则触发"})
        self.assertFalse(result["applies_to_current"])
        self.assertEqual(self.memory.project(gid)["model_status"], "pending")
        new = self.memory.freeze(gid)
        self.memory.complete(new["request_id"], {"observation_id": gid, "text": "目前仍有规则触发"})
        self.assertNotIn("目前仍有规则触发", str(memory_messages(self.memory.freeze(gid))))

    def test_incremental_view_replaces_same_record_not_appends_every_hit(self):
        gid = self.memory.sync([transition("a", "open", 10)], [snapshot("a", 10)], 20)[0]
        versions = {gid: 1}
        self.assertEqual(self.memory.changes_since(versions), [])
        self.memory.sync([], [snapshot("a", 10, last=25, hits=8)], 30)
        delta = self.memory.changes_since(versions)
        self.assertEqual(len(delta), 1)
        self.assertEqual(delta[0]["rule_hits"], 8)
        self.assertEqual(delta[0]["transition_count"], 1)

    def test_same_time_different_signals_stay_separate(self):
        self.memory.sync([transition("a", "open", 10), transition("b", "open", 10, signal="disk")],
                         [snapshot("a", 10), snapshot("b", 10, signal="disk")], 20)
        self.assertEqual(len(self.memory.records), 2)

    def test_absence_of_alert_is_not_model_processed_or_normal(self):
        self.memory.receive_counts([{"node_id": "device:x", "modality": "trace", "available_at_ms": 10}], 20)
        self.memory.sync([], [], 20)
        overview = self.memory.overview()
        self.assertEqual(overview["raw_rows_received"], {"trace": 1})
        self.assertEqual(overview["observation_groups"], 0)
        self.assertIn("不代表正常", overview["note"])

    def test_recurrence_must_not_be_omitted_by_model(self):
        self.memory.sync([transition("a", "open", 10), transition("a", "cleared", 20),
                          transition("b", "open", 30)],
                         [snapshot("a", 10, closed=20), snapshot("b", 30)], 40)
        gid = next(iter(self.memory.records))
        req = self.memory.freeze(gid)
        with self.assertRaises(ValueError):
            self.memory.complete(req["request_id"], {"observation_id": gid, "text": "当前仍异常"})
        self.memory.complete(req["request_id"], {"observation_id": gid, "text": "再次出现，目前仍有规则触发"})
        self.memory.complete(req["request_id"], {"observation_id": gid, "text": "已触发2次，目前仍有规则触发"})
        prompt = memory_messages(req)[-1]["content"]
        self.assertIn("解除后再次触发", prompt)
        self.assertIn("目前最后已知状态", prompt)

    def test_qualitative_note_cannot_rewrite_program_timestamps(self):
        gid = self.memory.sync([transition("a", "open", 10)], [snapshot("a", 10)], 20)[0]
        request = self.memory.freeze(gid)
        with self.assertRaises(ValueError):
            validate_memory_reply(request, {"observation_id": gid, "text": "当前18:20仍触发"},
                                  qualitative_only=True)
        validate_memory_reply(request, {"observation_id": gid, "text": "当前仍有规则触发"},
                              qualitative_only=True)
        self.assertEqual(request["fact"]["state_changes"], self.memory.project(gid)["state_changes"])


if __name__ == "__main__":
    unittest.main()
