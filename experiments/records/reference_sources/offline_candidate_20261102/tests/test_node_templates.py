from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from edge_rca import node_templates as templates
from edge_rca.case_runtime import assemble_report
from edge_rca.coordination import event_times, make_view, model_view, model_input
from edge_rca.node_inquiry import run_node
from edge_rca.node_reports import ReportInbox


START = 1614852000000
MEMORY = "OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc"


def metric(identifier="f1", low=23, high=98, median=23, count=14):
    return {"id": identifier, "metric": MEMORY, "meaning": "内存占用比例，可能含缓存", "unit": "%",
            "reference": {"count": 0, "median": None},
            "current": {"count": count, "min": low, "max": high, "median": median},
            "early": {"count": 4, "median": 23}, "late": {"count": 4, "median": 11},
            "early_late_change": {"direction": "down"},
            "extremes": [{"time_ms": START + 600000, "value": high}],
            "change": {"direction": "unknown"},
            "quality": {"invalid": 0, "conflicts": 0, "counter_resets": 0}}


def packet(facts):
    return {"case_id": "fixture", "node_id": "device:Redis02", "window_start_ms": START,
            "cutoff_ms": START + 1200000, "request_id": "initial",
            "scope": {"start": "18:00:00", "end": "18:20:00"}, "facts": facts,
            "catalog": {}, "limitations": [], "cpu_only_metric_ids": []}


class Selector:
    """Deliberately downplays every observation; it is not a real LLM."""
    def __init__(self, invalid=False):
        self.invalid = invalid
        self.calls = []

    def invoke(self, messages, tools, max_tokens, phase):
        self.calls.append(deepcopy(messages))
        if self.invalid:
            return '{"items":[],"request_id":null}'
        data = json.JSONDecoder().raw_decode(messages[-1]["content"])[0]
        return json.dumps({"items": [{"id": r["id"], "reading": "uncertain", "use": "background", "related_ids": []}
                                     for r in data["forms"]], "request_id": None})


class NoQueryBroker:
    case_id = "fixture"
    cutoff_ms = START + 1200000

    def execute(self, *args):
        raise AssertionError("fixture selected no query")


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def report(self, model=None):
        initial = packet([metric(), metric("f2", 98, 98, 98)])
        model = model or Selector()
        local = run_node(model, NoQueryBroker(), initial, self.root / "worker", node_mode="template")
        answer = json.loads((self.root / "worker/step000/node_answer.json").read_text())["answer"]
        report = assemble_report(initial, local, self.root / "worker", templates.overview(answer["template_rows"]))
        return report, model

    def test_screen_keeps_a_resource_percentage_ahead_of_a_disk_ratio(self):
        memory = metric("m1", 23, 98, 23)
        disk = {"id": "d1", "metric": "OSLinux-LOCALDISK-sda_DSKBps", "meaning": "磁盘速率",
                "unit": None, "reference": {"count": 0, "median": None},
                "current": {"count": 20, "min": 0.0, "max": 29.0, "median": 0.01},
                "early": {"count": 4, "median": 0.0}, "late": {"count": 4, "median": 0.0},
                "early_late_change": {"direction": "unknown"},
                "extremes": [{"time_ms": START + 900000, "value": 29.0}],
                "change": {"direction": "unknown"}, "quality": {"invalid": 0, "conflicts": 0, "counter_resets": 0}}
        p = packet([memory, disk])
        forms = [templates.form(f, p) for f in [memory, disk]]
        chosen = templates.screen_rows(forms, limit=2)
        subjects = [row["subject"] for row in chosen]
        self.assertIn(MEMORY, subjects)
        self.assertLessEqual(len(chosen), 2)

    def test_screen_limit_keeps_all_rows_but_asks_only_the_screen(self):
        facts = [metric(f"m{i}", 23, 98, 23) for i in range(5)]
        for index, fact in enumerate(facts):
            fact["metric"] = MEMORY + f"_{index}"
        p = packet(facts)
        model = Selector()
        local = run_node(model, NoQueryBroker(), p, self.root / "screened", node_mode="template", screen_limit=2)
        answer = json.loads((self.root / "screened/step000/node_answer.json").read_text())["answer"]
        self.assertEqual(answer["screening"]["model_candidates"], 2)
        self.assertGreaterEqual(answer["screening"]["retained_by_rule"], 3)
        self.assertEqual(len(answer["template_rows"]), 5)
        self.assertEqual(len(model.calls), 1)

    def test_static_high_is_distinct_from_a_transient_middle_peak(self):
        self.assertEqual(templates.classify(metric("f1", 98, 98, 98)), ("stable_samples", False))
        self.assertEqual(templates.classify(metric("f2", 11, 98, 23)), ("variation_observed", True))
        # Relative increases from tiny percentages do not become severe simply
        # because their ratio is large. Nothing is deleted from local storage.
        self.assertEqual(templates.classify(metric("f3", .02, .10, .03)), ("small_variation", False))
        prior = metric("f4", 98, 98, 98)
        prior["reference"] = {"count": 5, "median": 23}
        self.assertEqual(templates.classify(prior), ("variation_observed", True))

    def test_insufficient_samples_and_unknown_units_are_not_normalized(self):
        f = metric(count=1)
        f["unit"] = "source_unit_unverified"
        row = templates.form(f, packet([f]))
        self.assertEqual(row["pattern"], "insufficient_data")
        self.assertEqual(row["fields"]["unit"], "source_unit_unverified")
        self.assertNotIn("GB", templates.evidence_text(row))
        self.assertEqual(templates.classify({"kind": "http_access", "status": 200}), ("observation_only", False))
        self.assertEqual(templates.classify({"kind": "log", "status": "available", "explicit_error": True}),
                         ("error_observed", True))

    def test_model_cannot_rewrite_values_or_invent_other_node_facts(self):
        rows = [templates.form(metric(), packet([metric()]))]
        good = {"items": [{"id": "f1", "reading": "variable", "use": "report", "related_ids": []}], "request_id": None}
        for change in ("number", "foreign", "duplicate", "omission", "request"):
            bad = deepcopy(good)
            if change == "number": bad["items"][0]["value"] = 1
            elif change == "foreign": bad["items"][0]["related_ids"] = ["other-node-f1"]
            elif change == "duplicate": bad["items"] *= 2
            elif change == "omission": bad["items"] = []
            else: bad["request_id"] = "made-up-tool"
            with self.assertRaises(ValueError): templates.validate(bad, rows, [])

    def test_filled_sentences_use_observed_values_and_reject_wrong_direction(self):
        p = packet([metric()]); row = templates.form(p["facts"][0], p)
        self.assertEqual(row["template"], "metric_change")
        self.assertIn("middle_peak", row["reading_options"])
        good = {"items": [{"id": "f1", "reading": "middle_peak", "use": "report", "related_ids": []}], "request_id": None}
        templates.validate(good, [row], [])
        text = templates.statement(row, good["items"][0])
        for fragment in ("device:Redis02", "23", "98", "11", "18:10:00.000", "中途出现高值"):
            self.assertIn(fragment, text)
        stable = templates.form(metric(low=98, high=98, median=98), p)
        with self.assertRaisesRegex(ValueError, "reading"):
            templates.validate(good, [stable], [])
        minute = templates.form({"id": "f2", "kind": "minute_latency", "start": "18:01:00", "end": "18:02:00",
                                 "count": 10, "p95": 120, "max": 800, "p50": 20}, p)
        text = templates.statement(minute, {"reading": "tail", "use": "report"})
        self.assertIn("P95=120ms", text)
        self.assertIn("最大=800ms", text)

    def test_downplayed_change_still_reaches_coordinator_with_exact_numbers_and_time(self):
        report, model = self.report()
        inbox = ReportInbox()
        inbox.receive(report)
        view, binding = make_view(inbox, "fixture", START + 1200000, None, include_report_details=False)
        self.assertEqual(view["report_detail_mode"], "template_facts")
        self.assertEqual(len(view["evidence"]), 1)
        fact = view["evidence"][0]
        self.assertEqual(fact["node_id"], "device:Redis02")
        for value in ('"n,min,max,median":[14,23,98,23]', '18:10:00.000'):
            self.assertIn(value, fact["tool_fact"])
        self.assertIn("18:10:00.000", fact["time_marks"])
        self.assertEqual(binding[fact["id"]]["origin"], "node_template")
        self.assertEqual(report["groups"][0]["priority"], "immediate")
        self.assertEqual(report["template_rows"][0]["model_selection"]["use"], "background")
        self.assertEqual(len(report["retain_local"]), 1)
        self.assertEqual(len(model.calls), 1)  # no second prose overview
        self.assertNotIn("scoring_points", json.dumps(model.calls))
        compact = model_view(view)
        packed = compact["evidence"]["device:Redis02"][0]
        self.assertEqual(packed[0], fact["id"])
        self.assertEqual(packed[2], fact["time_marks"])
        self.assertEqual(packed[1][0:4], ["metric", MEMORY, "variation_observed", "%"])
        self.assertEqual(packed[1][4], [14, 23, 98, 23])
        self.assertEqual(packed[1][5], [["18:10:00.000", 98]])
        self.assertIsInstance(view["evidence"], list)

    def test_global_table_packing_preserves_values_and_unknown_formats(self):
        from edge_rca.template_tables import pack_fact
        fact = {"node_id": "cluster:observations", "tool_fact":
                'business:test；成功率%：30点，中位数100.0，范围97.53—100.0；极值观测[{"time":"18:03:00.000","value":97.53}]'}
        self.assertEqual(pack_fact(fact), ["business", "business:test", [
            ["成功率%", 30, 100., 97.53, 100., [["18:03:00.000", 97.53]]]]])
        fact["tool_fact"] += "；新字段不能丢掉"
        self.assertEqual(pack_fact(fact), fact["tool_fact"])
        fact["tool_fact"] = "实际父子ID匹配：device:a → device:b；子调用4次，P95=1053.0ms，最大=10017.0ms；其中2对时间区间不满足包含关系。"
        self.assertEqual(pack_fact(fact), ["call_link", "device:a", "device:b", 4, 1053., 10017., 2])

    def test_context_pages_preserve_protected_fact_and_do_not_mutate_audit(self):
        report, _ = self.report()
        inbox = ReportInbox(); inbox.receive(report)
        view, _ = make_view(inbox, "fixture", START + 1200000, None, include_report_details=False)
        extra = deepcopy(view["evidence"][0]); extra["id"] = "e2"
        extra["template_payload"].update(protected=False, subject="other_metric")
        view["evidence"].append(extra)
        view["node_summaries"][0]["loaded_fact_count"] += 1
        class Counter:
            max_context = 3500
            def count_input_tokens(self, messages):
                payload = json.loads(messages[-1]["content"])
                return 1000 if sum(len(v) for v in payload["evidence"].values()) > 1 else 10
        _, shown = model_input(Counter(), view, "fixture", 2000)
        self.assertEqual([f["id"] for f in shown["evidence"]], ["e1"])
        self.assertEqual(len(view["evidence"]), 2)
        self.assertEqual(shown["context_omissions"]["observations"][0]["subject"], "other_metric")

    def test_invalid_model_output_remains_unprocessed_but_cpu_fact_survives(self):
        report, model = self.report(Selector(invalid=True))
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(report["defer"]), 1)
        self.assertFalse(report["groups"])
        inbox = ReportInbox()
        inbox.receive(report)
        view, _ = make_view(inbox, "fixture", START + 1200000, None, include_report_details=False)
        self.assertEqual(len(view["evidence"]), 1)
        self.assertEqual(view["node_summaries"][0]["model_unprocessed_count"], 1)
        self.assertEqual(view["node_summaries"][0]["rule_retained_count"], 1)

    def test_stable_node_skips_model_and_stays_queryable(self):
        p = packet([metric(low=98, high=98, median=98)])
        model = Selector()
        result = templates.answer(model, p, self.root)
        self.assertEqual(model.calls, [])
        self.assertEqual(result["screening"]["retained_by_rule"], 1)
        self.assertEqual(len(result["facts"]), 1)
        self.assertEqual(result["groups"], [])
        self.assertFalse(result["template_rows"][0]["model_valid"])
        self.assertEqual(templates.board_rows({"template_rows": result["template_rows"]}), [])
        p["request"] = {}  # explicitly requested counterevidence must be inspected
        templates.answer(model, p, self.root)
        self.assertEqual(len(model.calls), 1)

    def test_unchanged_selection_reused_but_new_value_is_reconsidered(self):
        p = packet([metric()]); model = Selector()
        first = templates.answer(model, p, self.root)
        p["previous_template_decisions"] = first["template_rows"]
        p["facts"][0]["current"]["count"] += 1
        second = templates.answer(model, p, self.root)
        self.assertEqual(second["screening"]["unchanged_reused"], 1)
        self.assertEqual(len(model.calls), 1)
        p["facts"][0]["current"]["max"] = 99
        third = templates.answer(model, p, self.root)
        self.assertEqual(third["screening"]["unchanged_reused"], 0)
        self.assertEqual(len(model.calls), 2)

    def test_template_cannot_include_future_extremes_or_future_reports(self):
        f = metric()
        f["extremes"][0]["time_ms"] = START + 1800000
        with self.assertRaisesRegex(ValueError, "visible"):
            templates.form(f, packet([f]))
        report, _ = self.report()
        inbox = ReportInbox()
        inbox.receive(report)
        future = deepcopy(report)
        future["cutoff_ms"] += 600000
        future["template_rows"][0]["fields"]["current"]["max"] = 123456
        inbox.receive(future)
        view, _ = make_view(inbox, "fixture", START + 1200000, None, include_report_details=False)
        self.assertNotIn("123456", json.dumps(view))

    def test_extreme_timestamps_are_citable_without_future_values(self):
        f = metric()
        f["extremes"].append({"time_ms": START + 1800000, "value": 999})
        self.assertEqual(event_times(f, START, START + 1200000), ["18:10:00.000"])

    def test_query_choices_stay_local_bounded_and_optional(self):
        trace = {"id": "f2", "kind": "observed_connection", "from": "device:Redis02", "to": "device:MG01"}
        p = packet([metric(), trace])
        rows = [templates.form(f, p) for f in p["facts"]]
        options = templates.query_options(p, rows)
        self.assertEqual(len(options), 1)
        q = options[0]["request"]
        self.assertEqual(q["node_id"], p["node_id"])
        self.assertEqual(q["tools"], ["get_calls"])
        self.assertEqual((q["start"], q["end"]), ("18:08:00.000", "18:12:00.000"))
        good = {"items": [{"id": r["id"], "reading": "uncertain", "use": "uncertain", "related_ids": []} for r in rows], "request_id": None}
        self.assertIsNone(templates.validate(good, rows, options)["request_id"])


if __name__ == "__main__":
    unittest.main()
