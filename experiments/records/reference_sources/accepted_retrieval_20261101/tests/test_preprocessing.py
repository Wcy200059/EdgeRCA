"""Filtering must be explicit, traceable and distinct from leaving data unread."""

from copy import deepcopy
import json
import unittest

from edge_rca.preprocessing import build_preprocessor_prompt, validate_preprocessor_response
from edge_rca.prompts import build_coordinator_prompt


def packet():
    return {"node_id": "device:Tomcat01", "window_start_ms": 0, "cutoff_ms": 60_000,
            "observations": [{"observation_id": key, "description": "程序保存的观测"}
                             for key in ("o1", "o2", "o3", "o4")],
            "previous_findings": [], "scope": {}, "limitations": []}


def response():
    return {"groups": [{"observation_ids": ["o1", "o2"], "summary": "两次连接拒绝。",
                        "priority": "immediate"}],
            "retain_local": ["o3"], "defer": ["o4"], "questions": []}


class PreprocessingTests(unittest.TestCase):
    def test_grouping_keeps_both_sources_and_distinguishes_local_from_unread(self):
        source, reply = packet(), response()
        original = deepcopy(source)
        checked = validate_preprocessor_response(reply, source)
        self.assertEqual(checked["groups"][0]["observation_ids"], ["o1", "o2"])
        self.assertEqual(checked["retain_local"], ["o3"])
        self.assertEqual(checked["defer"], ["o4"])
        checked["groups"][0]["summary"] = "Changed output copy"
        self.assertEqual(reply["groups"][0]["summary"], "两次连接拒绝。")
        self.assertEqual(source, original)

    def test_silent_omission_is_not_a_successful_filter(self):
        reply = response()
        reply["defer"] = []
        with self.assertRaisesRegex(ValueError, "no disposition"):
            validate_preprocessor_response(reply, packet())

    def test_unknown_or_double_assigned_observations_are_rejected(self):
        replies = []
        unknown = response()
        unknown["defer"] = ["invented"]
        replies.append(unknown)
        duplicate = response()
        duplicate["retain_local"] = ["o1", "o3"]
        replies.append(duplicate)
        duplicate_group = response()
        duplicate_group["groups"][0]["observation_ids"] = ["o1", "o1", "o2"]
        replies.append(duplicate_group)
        for reply in replies:
            with self.subTest(reply=reply), self.assertRaisesRegex(ValueError, "duplicate|unknown"):
                validate_preprocessor_response(reply, packet())

    def test_program_required_alert_cannot_be_silenced_or_delayed(self):
        source = packet()
        source["scope"]["must_report_ids"] = ["o3"]
        reply = response()
        with self.assertRaisesRegex(ValueError, "mandatory"):
            validate_preprocessor_response(reply, source)
        reply["retain_local"] = []
        reply["groups"].append({"observation_ids": ["o3"], "summary": "严重异常。", "priority": "routine"})
        with self.assertRaisesRegex(ValueError, "mandatory"):
            validate_preprocessor_response(reply, source)
        reply["groups"][-1]["priority"] = "immediate"
        self.assertEqual(validate_preprocessor_response(reply, source), reply)

    def test_node_can_receive_a_scoped_follow_up_request_without_extra_labels(self):
        source = packet()
        source["command"] = {"command_id": "c1", "type": "summarize_window",
                             "target_node": source["node_id"], "start_ms": 0,
                             "end_ms": 50_000, "focus": "核对连接异常前后的调用变化"}
        prompt = build_preprocessor_prompt(source)
        restored = json.loads(prompt["user_prompt"].split("\n", 1)[1])
        self.assertEqual(restored, source)
        source["ground_truth"] = "do not pass"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            build_preprocessor_prompt(source)

    def test_coordinator_supports_explicitly_advertised_command_capabilities(self):
        source = {"cutoff_ms": 60_000, "node_reports": [], "observed_links": [],
                  "existing_incidents": [], "window_start_ms": 0,
                  "known_nodes": ["device:Tomcat01"],
                  "available_commands": ["fetch_evidence", "summarize_window", "compare_signal"]}
        prompt = build_coordinator_prompt(source)
        restored = json.loads(prompt["user_prompt"].split("\n", 1)[1])
        self.assertEqual(restored["available_commands"], source["available_commands"])
        self.assertIn("未提供可用命令时commands必须为空", prompt["system_prompt"])

    def test_empty_page_and_explicit_all_local_page_are_valid(self):
        empty = packet()
        empty["observations"] = []
        reply = {"groups": [], "retain_local": [], "defer": [], "questions": []}
        self.assertEqual(validate_preprocessor_response(reply, empty), reply)
        reply["retain_local"] = ["o1", "o2", "o3", "o4"]
        self.assertEqual(validate_preprocessor_response(reply, packet()), reply)

    def test_protocol_validation_does_not_claim_to_verify_summary_semantics(self):
        reply = response()
        reply["groups"][0]["summary"] = "这段说明还需要语义核查。"
        self.assertEqual(validate_preprocessor_response(reply, packet()), reply)


if __name__ == "__main__":
    unittest.main()
