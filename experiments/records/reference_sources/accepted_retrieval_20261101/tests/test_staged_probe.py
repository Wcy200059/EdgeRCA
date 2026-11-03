import unittest

from edge_rca.staged_probe import brief_facts, call_messages, decode_summary, factual_history, required_ids, summary_messages, validate_findings


class StagedProbeTests(unittest.TestCase):
    def test_latest_complete_minute_is_visible_without_dropping_older_intervals(self):
        whole = {"id": "t1", "kind": "latency_over_analysis_window", "reference":
                 {"count": 0, "p50": None, "p95": None, "max": None},
                 "current": {"count": 100, "p50": 0, "p95": 9, "max": 100}, "slow_calls": []}
        facts = [whole] + [{"id": f"t{i}", "kind": "minute_latency", "start": f"18:0{i-2}",
                 "end": f"18:0{i-1}", "partial": i == 4, "count": 20, "p50": 0, "p95": 5, "max": 20}
                 for i in (2, 3, 4)]
        result = brief_facts({"scope": {}, "observed_links": [], "facts": facts})
        self.assertEqual([f["id"] for f in result["facts"]], ["t1", "t3", "t4", "t2"])
        self.assertEqual([f["id"] for f in facts], ["t1", "t2", "t3", "t4"])
        bundle = {"context": {"alerts": [{"rule": "trace.latency"}]},
                  "tools": {"get_metrics": [], "get_logs": [], "get_calls": facts}}
        self.assertEqual(required_ids(bundle, "get_calls"), ["t1", "t3"])
        facts[1]["end"] = "23:59"
        facts[2]["end"] = "00:00"
        facts[3]["end"] = "00:01"
        self.assertEqual(required_ids(bundle, "get_calls"), ["t1", "t3"])
        self.assertEqual(brief_facts({"scope": {}, "observed_links": [], "facts": facts})["facts"][1]["id"], "t3")

    def test_bad_generated_claim_does_not_enter_next_history(self):
        value = {"request_id": "r1", "scope": {"cutoff": "18:01"}, "observed_links": [],
            "facts": [{"id": "t1", "kind": "minute_latency", "start": "18:00", "end": "18:01",
                       "partial": False, "count": 2, "p50": 100, "p95": 200, "max": 200}]}
        reply = {"findings": [{"ids": ["t1"], "text": "错误断言：系统恢复，延迟只有1毫秒"}]}
        history = factual_history(value, reply)
        self.assertNotIn("系统恢复", str(history))
        self.assertIn("P95=200", str(history))
        self.assertIn("18:01", str(history))
        self.assertNotIn("'id': 't1'", str(history))

    def test_bare_list_is_compatible_but_bad_citations_are_not_repaired(self):
        raw = '```json\n[{"id":"m1","text":"当前CPU为30%。"}]\n```<|im_end|>'
        checked = validate_findings(decode_summary(raw), [{"id": "m1"}], ["m1"])
        self.assertTrue(checked["wrapped_findings_list"])
        self.assertEqual(checked["reply"]["findings"][0]["ids"], ["m1"])
        with self.assertRaises(ValueError):
            validate_findings(decode_summary(raw.replace('"m1"', '"other"')), [{"id": "m1"}], ["m1"])

    def test_action_and_summary_protocols_do_not_compete(self):
        messages = call_messages("get_calls", "r2")
        self.assertNotIn("findings", str(messages))
        self.assertIn("get_calls", messages[-1]["content"])
        value = {"node_id": "device:test", "request_id": "r2", "scope": {"cutoff": "18:02"},
                 "facts": [{"id": "t1", "kind": "missing", "note": "没有记录"}], "observed_links": []}
        messages = summary_messages(value, [], [], [], "get_calls")
        self.assertIn("JSON数组", messages[0]["content"])
        self.assertIn("18:02", messages[1]["content"])
        self.assertNotIn("调用工具", messages[1]["content"])

    def test_missing_optional_questions_normalizes_but_extra_fields_fail(self):
        reply = {"findings": [{"id": "t1", "text": "参考段无样本。"}]}
        checked = validate_findings(reply, [{"id": "t1"}], ["t1"])
        self.assertTrue(checked["defaulted_empty_questions"])
        self.assertEqual(checked["reply"]["questions"], [])
        with self.assertRaises(ValueError):
            validate_findings({**reply, "root_cause": "db"}, [{"id": "t1"}], [])

    def test_single_id_spelling_normalizes_without_changing_meaning(self):
        reply = {"findings": [{"id": "m1", "text": "CPU为30%，无参考段。"}], "questions": []}
        checked = validate_findings(reply, [{"id": "m1"}], [])
        self.assertEqual(checked["reply"]["findings"][0]["ids"], ["m1"])
        self.assertEqual(checked["normalized_single_id_count"], 1)
        self.assertIn("id", reply["findings"][0])

    def test_omissions_stay_unsummarized_instead_of_being_marked_normal(self):
        reply = {"findings": [{"ids": ["t1"], "text": "当前窗口有慢调用，位置尚未确定。"}], "questions": []}
        result = validate_findings(reply, [{"id": "t1"}, {"id": "t2"}], ["t1"])
        self.assertEqual(result["not_summarized_ids"], ["t2"])
        self.assertFalse(result["semantics_verified"])

    def test_unknown_or_duplicate_or_missing_mandatory_citations_rejected(self):
        for ids in (["other"], ["t1", "t1"], ["t2"]):
            with self.assertRaises(ValueError):
                validate_findings({"findings": [{"ids": ids, "text": "有调用。"}], "questions": []},
                                  [{"id": "t1"}, {"id": "t2"}], ["t1"])

    def test_latency_text_keeps_partial_window_and_does_not_change_numbers(self):
        value = {"scope": {}, "observed_links": [], "facts": [{"id": "t4", "kind": "minute_latency",
                 "start": "18:01:00", "end": "18:01:10", "partial": True,
                 "count": 100, "p50": 0, "p95": 218, "max": 10017}]}
        text = brief_facts(value)["facts"][0]["observation"]
        for expected in ("未满一分钟", "18:01:10", "10017", "P95=218", "100次"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
