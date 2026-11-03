from copy import deepcopy
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from edge_rca.coordination import ask, make_view, validate, validate_sections
from edge_rca.closed_loop import tools as run_tools
from edge_rca.followup import NodeRequestBroker, answer, build_packet, case_id_for, request_data_key, request_key
from edge_rca.node_inquiry import run_node
from edge_rca.node_reports import ReportInbox


START = 1614852000000
CPU = "OSLinux-CPU_CPU_CPUCpuUtil"


class FakeNodeModel:
    def invoke(self, messages, tools, maximum, phase):
        payload = json.JSONDecoder().raw_decode(messages[-1]["content"].split("\n", 1)[1])[0]
        return json.dumps({"groups": [{"observation_ids": payload["allowed_ids"],
                           "summary": "已查看工具观测，需结合其他节点判断。", "priority": "routine"}],
                           "retain_local": [], "defer": [], "questions": []}, ensure_ascii=False)


class ScriptedNodeModel:
    """Contract fixture, not evidence that a real model finds the right gaps."""
    def __init__(self, actions):
        self.actions, self.calls = actions, []

    def invoke(self, messages, tools, maximum, phase):
        payload = json.JSONDecoder().raw_decode(messages[-1]["content"].split("\n", 1)[1])[0]
        index = len(self.calls)
        self.calls.append(deepcopy(messages))
        extra = self.actions[index](payload) if index < len(self.actions) else {}
        return json.dumps({"groups": [{"observation_ids": payload["allowed_ids"],
                           "summary": "已整理本地观测，全局原因仍需核实。", "priority": "routine"}],
                           "retain_local": [], "defer": [], "questions": [], **extra}, ensure_ascii=False)


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.cutoff = START + 600000
        self.events = []

        def event(eid, node, modality, at, available=None, **raw):
            return {"event_id": eid, "node_id": node, "source_id": node.split(":", 1)[1],
                    "modality": modality, "observed_at_ms": START + at,
                    "available_at_ms": START + (at if available is None else available), "raw": raw}

        self.events = [event("old", "device:MG01", "metric", 60000, kpi_name=CPU, value=30),
                       event("cur", "device:MG01", "metric", 180000, kpi_name=CPU, value=80),
                       event("later", "device:MG01", "metric", 480000, kpi_name=CPU, value=99999),
                       event("unavailable", "device:MG01", "trace", 180000, available=620000, duration=440000),
                       event("early_call", "device:MG01", "trace", 160000, available=160020, duration=20),
                       event("call", "device:MG01", "trace", 190000, available=190020, duration=20),
                       event("prior_log", "device:MG01", "log", 50000, value="old ERROR", log_name="app"),
                       event("log", "device:MG01", "log", 185000, value="ERROR refused", log_name="app"),
                       event("other", "device:Tomcat04", "metric", 180000, kpi_name=CPU, value=777777)]
        nodes = []
        for index, node in enumerate(("device:MG01", "device:Tomcat04")):
            filename = f"{index}.jsonl"
            entries = sorted((e for e in self.events if e["node_id"] == node), key=lambda e: (e["available_at_ms"], e["event_id"]))
            (self.prepared / filename).write_text("".join(json.dumps(e) + "\n" for e in entries))
            nodes.append({"node_id": node, "source_id": node.split(":", 1)[1], "path": filename})
        (self.prepared / "manifest.json").write_text(json.dumps({"schema_version": 1,
            "window": {"start_ms": START, "end_ms": START + 1800000}, "nodes": nodes}))
        self.case = case_id_for(self.prepared)
        self.inbox = ReportInbox()
        self.inbox.receive({"case_id": self.case, "node_id": "device:Tomcat04", "window_start_ms": START,
                           "cutoff_ms": self.cutoff, "groups": [{"source_ids": ["f1"], "summary": "发现长调用。",
                               "cpu_facts": [{"id": "f1", "slow_calls": [{"start": "18:03:00", "end": "18:03:01"}]}]}],
                           "tool_card_catalog": {"f1": {"id": "f1", "tool_fact": "调用发生在18:03:00"}},
                           "retain_local": [], "defer": [], "cpu_only_metric_ids": []})
        self.view, self.binding = make_view(self.inbox, self.case, self.cutoff, ["device:Tomcat04"])
        self.request = {"node_id": "device:MG01", "start": "18:02:00", "end": "18:04:00",
                        "tools": ["get_metrics", "get_logs", "get_calls"], "metrics": [CPU],
                        "focus": "核查同一时段是否存在本地资源或调用变化。", "basis_ids": ["e1"]}
        self.reply = {"hypotheses": [{"id": "h1", "assessment": "unresolved", "component": None,
                       "reason": None, "time": "18:03:00", "time_evidence_id": "e1",
                       "support_ids": ["e1"], "counter_ids": [], "explanation": "尚需下游对照。"}],
                      "requests": [self.request]}

    def broker(self):
        broker = NodeRequestBroker(self.prepared, self.case, self.cutoff)
        self.addCleanup(broker.close)
        return broker

    def test_decision_uses_short_bound_ids_and_actual_times(self):
        self.assertEqual(list(self.binding), ["e1"])
        self.assertEqual(validate(self.reply, self.view), self.reply)
        wrong = deepcopy(self.reply)
        wrong["hypotheses"][0]["time"] = "18:00:00"
        with self.assertRaisesRegex(ValueError, "time_mark"):
            validate(wrong, self.view)

    def test_coordinator_overview_keeps_reports_but_requires_loading_facts(self):
        before = deepcopy(self.inbox.history)
        view, binding = make_view(self.inbox, self.case, self.cutoff, ["device:Tomcat04"], include_report_details=False)
        self.assertEqual(view["evidence"], [])
        self.assertEqual(binding, {})
        self.assertEqual(view["node_summaries"][0]["available_fact_count"], 1)
        self.assertFalse(view["node_summaries"][0]["details_loaded"])
        self.assertEqual(self.inbox.history, before)
        self.assertNotIn(self.view["evidence"][0]["tool_fact"], json.dumps(view, ensure_ascii=False))
        with self.assertRaisesRegex(ValueError, "current evidence"):
            validate(self.reply, view)
        request = {**self.request, "basis_ids": [], "tools": ["list_metrics"], "metrics": []}
        validate({"hypotheses": [], "requests": [request]}, view)
        too_broad = {**request, "tools": ["get_metrics"]}
        with self.assertRaisesRegex(ValueError, "exact metric"):
            validate({"hypotheses": [], "requests": [too_broad]}, view)

    def test_coordinator_receives_latest_node_questions(self):
        self.inbox.history[0]["questions"] = ["需要MG核查同期变化。"]
        view, _ = make_view(self.inbox, self.case, self.cutoff, ["device:Tomcat04"])
        self.assertEqual(view["node_questions"][0]["question_unverified"], "需要MG核查同期变化。")

    def test_retry_has_previous_output_and_specific_time_error(self):
        bad = deepcopy(self.reply)
        bad["hypotheses"][0]["time"] = "18:00:00"
        good = deepcopy(self.reply)

        class Model:
            def __init__(self):
                self.prompts = []

            def invoke(self, prompt, *_):
                self.prompts.append(deepcopy(prompt))
                return json.dumps(bad if len(self.prompts) == 1 else good)

        model = Model()
        result, attempts = ask(model, self.view, "test")
        self.assertEqual(result, good)
        self.assertEqual([a["valid"] for a in attempts], [False, True])
        self.assertEqual(json.loads(model.prompts[1][-2]["content"]), bad)
        self.assertIn("h1", model.prompts[1][-1]["content"])

    def test_bad_hypothesis_does_not_block_valid_bounded_request(self):
        reply = deepcopy(self.reply)
        reply["hypotheses"][0]["time"] = "18:00:00"
        bad_request = deepcopy(self.request)
        bad_request["end"] = "18:11:00"
        reply["requests"].append(bad_request)
        accepted, rejected = validate_sections(reply, self.view)
        self.assertEqual(accepted, {"hypotheses": [], "requests": [self.request]})
        self.assertEqual([r["section"] for r in rejected], ["hypotheses", "requests"])
        self.assertEqual(rejected[0]["item"], reply["hypotheses"][0])
        validate(accepted, self.view)

    def test_planner_can_continue_without_repeating_an_actionable_model_call(self):
        reply = deepcopy(self.reply)
        reply["hypotheses"][0]["time"] = "18:00:00"

        class Model:
            def __init__(self):
                self.calls = 0

            def invoke(self, *_):
                self.calls += 1
                return json.dumps(reply)

        model = Model()
        accepted, attempts = ask(model, self.view, "plan", allow_partial_requests=True)
        self.assertEqual(model.calls, 1)
        self.assertEqual(accepted["requests"], [self.request])
        self.assertFalse(attempts[-1]["valid"])
        self.assertTrue(attempts[-1]["read_only_requests_usable"])
        self.assertEqual(len(attempts[-1]["rejected_sections"]), 1)

    def test_unresolved_without_citations_is_allowed_but_cannot_claim_time(self):
        reply = {"hypotheses": [{"id": "h1", "assessment": "unresolved", "component": "Tomcat04",
                 "reason": None, "time": None, "time_evidence_id": None,
                 "support_ids": [], "counter_ids": [], "explanation": "尚无节点明细，需补查。"}],
                 "requests": []}
        self.assertEqual(validate(reply, self.view), reply)
        reply["hypotheses"][0]["time"] = "18:03:00"
        with self.assertRaisesRegex(ValueError, "null time"):
            validate(reply, self.view)

    def test_overview_pseudo_id_in_request_is_cleaned_not_blocking_the_query(self):
        reply = deepcopy(self.reply)
        reply["requests"][0]["basis_ids"] = ["e1", "node:device:Tomcat01"]
        checked = validate(reply, self.view)
        self.assertEqual(checked["requests"][0]["basis_ids"], ["e1"])
        invented = deepcopy(self.reply)
        invented["requests"][0]["basis_ids"] = ["e999"]
        with self.assertRaisesRegex(ValueError, "current evidence"):
            validate(invented, self.view)

    def test_data_key_is_stable_across_checkpoints_while_request_key_is_not(self):
        later = deepcopy(self.view)
        later["cutoff_ms"] = START + 1200000
        self.assertEqual(request_data_key(self.request, self.view),
                         request_data_key(self.request, later))
        self.assertNotEqual(request_key(self.request, self.view),
                            request_key(self.request, later))

    def test_key_numbers_from_node_facts_reach_overview_summary(self):
        cutoff = START + 1800000
        report = {"case_id": "case", "node_id": "device:Redis02", "window_start_ms": START,
                  "cutoff_ms": cutoff, "overview_text": "内存高位，单位未确认。",
                  "groups": [{"source_ids": ["f1"], "summary": "内存告警。",
                              "cpu_facts": [{"id": "f1", "kind": "alarm_history", "signal": "MEMORY",
                                             "state": "triggered", "timeline": [["18:10:00", "triggered"]],
                                             "first_trigger_values": {"value": 98.0, "baseline_median": 23.0}}]}],
                  "tool_card_catalog": {"f1": {"id": "f1", "tool_fact": "内存告警。"}},
                  "retain_local": [], "defer": [], "cpu_only_metric_ids": []}
        inbox = ReportInbox()
        inbox.receive(report)
        view, _ = make_view(inbox, "case", cutoff, None, include_report_details=False)
        summary = next(s for s in view["node_summaries"] if s["node_id"] == "device:Redis02")
        self.assertTrue(any("98" in item and "23" in item for item in summary["key_numbers"]))
        self.assertIn("不是可直接引用的证据", summary["key_numbers_note"])

    def test_unsupported_candidate_is_distinct_from_supported_diagnosis(self):
        reply = deepcopy(self.reply)
        reply["hypotheses"][0].update(assessment="not_supported", reason="network latency")
        self.assertEqual(validate(reply, self.view), reply)
        reply["hypotheses"][0]["assessment"] = "supported"
        with self.assertRaisesRegex(ValueError, "component and reason"):
            validate(reply, self.view)

    def test_wrong_node_future_range_or_arbitrary_tool_rejected(self):
        for key, value in (("node_id", "device:Unknown"), ("end", "18:11:00"), ("tools", ["exec_python"]),
                           ("tools", ["get_logs", "get_logs"]), ("metrics", [CPU, CPU])):
            reply = deepcopy(self.reply)
            reply["requests"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate(reply, self.view)

    def test_reversed_or_empty_range_rejected(self):
        for start, end in (("18:04:00", "18:02:00"), ("18:02:00", "18:02:00")):
            self.reply["requests"][0].update(start=start, end=end)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                validate(self.reply, self.view)

    def test_request_can_end_at_cutoff_but_not_read_after_it(self):
        self.reply["requests"][0]["end"] = "18:10:00"
        validate(self.reply, self.view)

    def test_wrong_case_rejected_before_data_access(self):
        with self.assertRaisesRegex(ValueError, "case"):
            NodeRequestBroker(self.prepared, "other", self.cutoff)

    def test_tools_honor_node_historical_end_and_trace_availability(self):
        packet = self.broker().execute(self.request, self.view, self.root / "q")
        metric = next(f for f in packet["facts"] if "metric" in f)
        self.assertEqual(metric["current"]["max"], 80)
        self.assertEqual(metric["reference"]["median"], 30)
        serialized = json.dumps(packet)
        for forbidden in ("99999", "777777", "440000", "unavailable", "prior_log"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(packet["tools_executed"], ["get_metrics", "get_logs", "get_calls"])

    def test_requested_logs_are_current_only_not_counted_with_reference(self):
        packet = self.broker().execute(self.request, self.view, self.root / "q")
        logs = [f for f in packet["facts"] if f.get("kind") == "log"]
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["count"], 1)
        self.assertEqual(logs[0]["first"], "18:03:05.000")

    def test_non_minute_aligned_query_keeps_first_partial_minute(self):
        self.request.update(start="18:02:30", tools=["get_calls"], metrics=[])
        packet = self.broker().execute(self.request, self.view, self.root / "q")
        buckets = [f for f in packet["facts"] if f.get("kind") == "minute_latency"]
        total = next(f for f in packet["facts"] if f.get("kind") == "latency_over_analysis_window")
        self.assertEqual(buckets[0]["start"], "18:02:30.000")
        self.assertEqual(buckets[0]["count"], 2)
        self.assertEqual(sum(f["count"] for f in buckets), total["current"]["count"])

    def test_duplicate_request_reuses_audit_without_overwriting_it(self):
        broker = self.broker()
        a = broker.execute(self.request, self.view, self.root / "q")
        b = broker.execute(self.request, self.view, self.root / "q")
        self.assertEqual(a, b)
        other_order = deepcopy(self.request)
        other_order["tools"].reverse()
        self.assertEqual(request_key(other_order, self.view), request_key(self.request, self.view))

    def test_query_identity_ignores_wording_and_clock_precision(self):
        other = deepcopy(self.request)
        other.update(start="18:02:00.000", end="18:04:00.000", focus="换一种措辞继续查。", basis_ids=[])
        self.assertEqual(request_key(other, self.view), request_key(self.request, self.view))
        newer = {**self.view, "cutoff_ms": self.cutoff + 60000}
        self.assertNotEqual(request_key(other, newer), request_key(other, self.view))

    def test_cached_data_keeps_the_current_question(self):
        broker = self.broker()
        before = broker.execute(self.request, self.view, self.root / "q")
        other = {**self.request, "focus": "这次关注是否存在反证。"}
        after = broker.execute(other, self.view, self.root / "unused")
        self.assertEqual(before["facts"], after["facts"])
        self.assertEqual(after["request"]["focus"], other["focus"])
        self.assertFalse((self.root / "unused").exists())

    def test_unknown_named_metric_is_reported_as_missing_not_other_metrics(self):
        self.request["metrics"] = ["nonexistent"]
        packet = self.broker().execute(self.request, self.view, self.root / "q")
        self.assertEqual(packet["missing_metrics"], ["nonexistent"])
        self.assertFalse(any("metric" in f for f in packet["facts"]))

    def test_duplicate_model_questions_do_not_consume_the_execution_budget(self):
        plan = self.root / "plan"
        plan.mkdir()
        other = deepcopy(self.request)
        other.update(tools=["get_logs"], metrics=[])
        reply = deepcopy(self.reply)
        reply["requests"] = [self.request, deepcopy(self.request), other]
        (plan / "decision.json").write_text(json.dumps({"reply": reply}))
        (plan / "input.json").write_text(json.dumps({"view": self.view}))
        output = self.root / "tools"
        run_tools(Namespace(plan=plan, prepared=self.prepared, output=output, max_requests=2))
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(manifest["files"]), 2)
        self.assertEqual(manifest["duplicate_requests"], 1)
        self.assertEqual(manifest["pending_requests"], [])

    def test_empty_metric_list_checks_all_metrics_without_top_k(self):
        self.request["metrics"] = []
        packet = self.broker().execute(self.request, self.view, self.root / "q")
        self.assertEqual([f["metric"] for f in packet["facts"] if "metric" in f], [CPU])

    def test_metric_catalog_returns_names_without_expanding_statistics(self):
        request = {**self.request, "tools": ["list_metrics"], "metrics": []}
        packet = self.broker().execute(request, self.view, self.root / "q")
        self.assertEqual(packet["tools_executed"], ["list_metrics"])
        self.assertEqual(len(packet["facts"]), 1)
        self.assertEqual(packet["facts"][0]["names"], [CPU])
        self.assertNotIn("current", packet["facts"][0])
        self.assertEqual(packet["available_metric_names"], [CPU])

    def test_catalog_filters_are_literal_name_fragments_not_regex(self):
        request = {**self.request, "tools": ["list_metrics"], "metrics": ["cpu"]}
        broker = self.broker()
        matched = broker.execute(request, self.view, self.root / "q1")
        self.assertEqual(matched["facts"][0]["names"], [CPU])
        request["metrics"] = ["CPU.*"]
        empty = broker.execute(request, self.view, self.root / "q2")
        self.assertEqual(empty["facts"][0]["names"], [])
        self.assertEqual(empty["facts"][0]["total_available"], 1)
        self.assertFalse(empty["missing_metrics"])

    def test_metric_catalog_is_paged_without_dropping_names(self):
        broker, seed = self.node_seed()
        cpu = json.loads((self.root / "seed" / "cpu.json").read_text())
        template = next(o for o in cpu["observations"] if o["modality"] == "metric" and "name" in o)
        cpu["observations"] = [{**deepcopy(template), "name": f"metric_{i}", "observation_id": f"m{i}"} for i in range(37)]
        request = {**self.request, "tools": ["list_metrics"], "metrics": []}
        packet = build_packet(cpu, {"links": []}, request, self.case, self.cutoff, "qcatalog")
        self.assertEqual([len(f["names"]) for f in packet["facts"]], [16, 16, 5])
        self.assertEqual([name for f in packet["facts"] for name in f["names"]], [f"metric_{i}" for i in range(37)])

    def test_selected_tool_does_not_silently_execute_other_tools(self):
        self.request.update(tools=["get_logs"], metrics=[])
        packet = self.broker().execute(self.request, self.view, self.root / "q")
        self.assertEqual(packet["tools_executed"], ["get_logs"])
        self.assertEqual({f["kind"] for f in packet["facts"]}, {"log"})

    def test_tool_node_coordinator_roundtrip_keeps_original_report(self):
        before = deepcopy(self.inbox.history)
        packet = self.broker().execute(self.request, self.view, self.root / "q")
        answer_dir = self.root / "answer"
        answer_dir.mkdir()
        result = answer(FakeNodeModel(), packet, answer_dir)
        self.assertEqual(result["node_model_status"], "complete")
        view, binding = make_view(self.inbox, self.case, self.cutoff, ["device:Tomcat04"], [result],
                                  self.reply["hypotheses"], [self.request])
        self.assertGreater(len(view["evidence"]), len(self.view["evidence"]))
        self.assertEqual(self.inbox.history, before)
        self.assertEqual(view["pending_requests"], [self.request])
        self.assertTrue(any(b["origin"] == "followup" for b in binding.values()))
        self.assertEqual(view["followup_status"][0]["node_id"], "device:MG01")
        self.assertEqual(view["followup_status"][0]["available_metric_names"], [])
        self.assertEqual(result["available_metric_names"], [CPU])
        notes = [n for n in view["node_summaries"] if n["origin"] == "followup"]
        self.assertEqual(len(notes), len(result["groups"]))
        self.assertTrue(all(i in binding for n in notes for i in n["basis_ids"]))
        self.assertEqual(notes[0]["text_unverified"], result["groups"][0]["summary"])

    def test_node_summary_cannot_add_an_uncited_fact(self):
        bad = {"case_id": self.case, "cutoff_ms": self.cutoff,
               "node_id": "device:MG01", "request_id": "q1", "facts": [],
               "groups": [{"observation_ids": ["invented"], "summary": "unverified"}]}
        with self.assertRaisesRegex(ValueError, "absent"):
            make_view(self.inbox, self.case, self.cutoff, ["device:Tomcat04"], [bad])

    def test_other_case_or_future_supplement_cannot_enter_coordinator(self):
        bad = {"case_id": "other", "cutoff_ms": self.cutoff}
        with self.assertRaisesRegex(ValueError, "another case"):
            make_view(self.inbox, self.case, self.cutoff, ["device:Tomcat04"], [bad])
        bad = {"case_id": self.case, "cutoff_ms": self.cutoff + 1}
        with self.assertRaisesRegex(ValueError, "future"):
            make_view(self.inbox, self.case, self.cutoff, ["device:Tomcat04"], [bad])

    def node_seed(self):
        broker = self.broker()
        request = {**self.request, "tools": ["get_metrics"], "metrics": [CPU]}
        return broker, broker.execute(request, self.view, self.root / "seed")

    @staticmethod
    def local_query(payload, tools=None):
        return {"start": "18:02:00", "end": "18:04:00", "tools": tools or ["get_logs"],
                "metrics": [], "focus": "核实资源变化同期的其他实际观测。", "basis_ids": payload["allowed_ids"][:1]}

    def test_local_node_can_request_tools_and_read_new_evidence(self):
        broker, seed = self.node_seed()
        model = ScriptedNodeModel([lambda p: {"requests": [self.local_query(p)]}])
        out = self.root / "local"
        state = run_node(model, broker, seed, out, max_self_requests=2)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(state["executed_self_requests"], 1)
        self.assertEqual(state["stop_reason"], "no_more_queries")
        final = json.loads((out / state["files"][-1]).read_text())["answer"]
        self.assertTrue(any("refused" in f["tool_fact"] for f in final["facts"]))
        self.assertEqual(final["node_id"], seed["node_id"])
        self.assertNotIn("99999", json.dumps(final))
        self.assertNotIn("777777", json.dumps(final))

    def test_initial_node_snapshot_can_start_local_inquiry(self):
        broker, seed = self.node_seed()
        seed.pop("request")
        model = ScriptedNodeModel([lambda p: {"requests": [self.local_query(p)]}])
        state = run_node(model, broker, seed, self.root / "local")
        self.assertEqual(state["executed_self_requests"], 1)

    def test_local_node_can_discover_names_then_request_specific_metrics(self):
        broker = self.broker()
        request = {**self.request, "tools": ["get_logs"], "metrics": []}
        seed = broker.execute(request, self.view, self.root / "seed")
        def choose_metric(payload):
            request = self.local_query(payload, ["get_metrics"])
            request["metrics"] = [CPU]
            return {"requests": [request]}
        model = ScriptedNodeModel([lambda p: {"requests": [self.local_query(p, ["list_metrics"])]}, choose_metric])
        out = self.root / "local"
        state = run_node(model, broker, seed, out, max_self_requests=2)
        self.assertEqual(state["executed_self_requests"], 2)
        final = json.loads((out / state["files"][-1]).read_text())["answer"]
        self.assertTrue(any(CPU in f["tool_fact"] and "80" in f["tool_fact"] for f in final["facts"]))
        view, _ = make_view(self.inbox, self.case, self.cutoff, ["device:Tomcat04"], [final], include_report_details=False)
        self.assertEqual(len(view["evidence"]), len(final["facts"]))
        self.assertTrue(view["node_summaries"])

    def test_reworded_local_queries_stop_without_repeated_model_calls(self):
        broker, seed = self.node_seed()
        def repeat(payload):
            request = self.local_query(payload)
            request.update(focus="措辞不同，再问同一个数据问题。", start="18:02:00.000", end="18:04:00.000")
            return {"requests": [request]}
        model = ScriptedNodeModel([lambda p: {"requests": [self.local_query(p)]}, repeat])
        state = run_node(model, broker, seed, self.root / "local", max_self_requests=5)
        self.assertEqual(state["executed_self_requests"], 1)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(state["skipped_queries"]), 1)

    def test_local_budget_and_peer_questions_reach_coordinator(self):
        broker, seed = self.node_seed()
        model = ScriptedNodeModel([
            lambda p: {"requests": [self.local_query(p)]},
            lambda p: {"requests": [self.local_query(p, ["get_calls"])], "questions": ["需要关联IG节点核查同期观测。"]}])
        out = self.root / "local"
        state = run_node(model, broker, seed, out, max_self_requests=1)
        self.assertEqual(state["stop_reason"], "budget_reached")
        self.assertEqual(len(state["pending_requests"]), 1)
        self.assertEqual(len(state["questions"]), 1)
        answers = [json.loads((out / f).read_text())["answer"] for f in state["files"]]
        view, _ = make_view(self.inbox, self.case, self.cutoff, ["device:Tomcat04"], answers,
                            node_pending=state["pending_requests"])
        self.assertEqual(view["node_questions"][0]["question_unverified"], state["questions"][0]["question"])
        pending = view["pending_node_requests"][0]
        self.assertEqual(pending["status"], "not_executed")
        self.assertTrue(all(i.startswith("e") for i in pending["request"]["basis_ids"]))
        validate({"hypotheses": [], "requests": [pending["request"]]}, view)

    def test_bad_local_query_preserves_report_and_does_not_run_tools(self):
        for edits in ({"node_id": "device:Tomcat04"}, {"end": "18:11:00"}, {"tools": ["exec_python"]}):
            with self.subTest(edits=edits):
                local_tmp = self.root / ("bad" + str(len(list(self.root.iterdir()))))
                broker = self.broker()
                request = {**self.request, "tools": ["get_metrics"], "metrics": [CPU]}
                seed = broker.execute(request, self.view, local_tmp / "seed")
                def action(payload):
                    q = self.local_query(payload)
                    q.update(edits)
                    return {"requests": [q]}
                model = ScriptedNodeModel([action])
                state = run_node(model, broker, seed, local_tmp / "worker")
                self.assertEqual(state["executed_self_requests"], 0)
                self.assertEqual(len(state["request_errors"]), 1)
                self.assertEqual(len(state["files"]), 1)

    def test_tool_failure_keeps_the_unfinished_query(self):
        broker, seed = self.node_seed()
        model = ScriptedNodeModel([lambda p: {"requests": [self.local_query(p)]}])
        out = self.root / "local"
        with patch.object(broker, "execute", side_effect=RuntimeError("unavailable")):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                run_node(model, broker, seed, out)
        state = json.loads((out / "manifest.json").read_text())
        self.assertEqual(state["stop_reason"], "failed")
        self.assertEqual(len(state["pending_requests"]), 1)

    def test_model_failure_keeps_the_active_snapshot(self):
        broker, seed = self.node_seed()
        model = ScriptedNodeModel([])
        out = self.root / "local"
        with patch.object(model, "invoke", side_effect=RuntimeError("out of memory")):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                run_node(model, broker, seed, out)
        state = json.loads((out / "manifest.json").read_text())
        self.assertEqual(state["stop_reason"], "failed")
        self.assertTrue((out / state["active_step"]["directory"] / "input.json").exists())


if __name__ == "__main__":
    unittest.main()
