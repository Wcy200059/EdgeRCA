"""Synthetic online-trigger checks; no answers, models, APIs or GPU needed."""

from copy import deepcopy
import unittest

from edge_rca.alerts import AlertConfig, AlertEngine


def event(identifier, at_ms, *, node="device:Tomcat01", modality="metric",
          observed_ms=None, **raw):
    return {
        "event_id": identifier,
        "node_id": node,
        "source_id": node.split(":", 1)[1],
        "modality": modality,
        "available_at_ms": at_ms,
        "observed_at_ms": at_ms if observed_ms is None else observed_ms,
        "raw": raw,
    }


def metric(identifier, at_ms, value, *, node="device:Tomcat01", kpi="test_gauge"):
    return event(identifier, at_ms, node=node, kpi_name=kpi, value=value)


def span(identifier, available_ms, duration, *, node="device:Tomcat01"):
    return event(identifier, available_ms, node=node, modality="trace",
                 observed_ms=available_ms - duration,
                 timestamp=str(available_ms - duration), duration=str(duration),
                 trace_id=f"trace-{identifier}", span_id=f"span-{identifier}",
                 parent_id="parent")


def log(identifier, at_ms, text, *, name="application", node="device:Tomcat01"):
    return event(identifier, at_ms, node=node, modality="log", log_name=name,
                 value=text, log_id=identifier)


class AlertEngineTests(unittest.TestCase):
    def test_return_to_equal_baseline_is_unchanged_on_clear(self):
        for rule, baseline in (("metric.relative_down", [10]*5), ("metric.std_down", [10,11,12,13,14])):
            engine = self.engine()
            for i,v in enumerate(baseline):
                engine.feed(metric(f"b{i}",i*1000,v))
            engine.feed(metric("low",5000,0))
            engine.feed(metric("back",6000,10 if rule == "metric.relative_down" else 12))
            clear = next(t for t in engine.transitions if t["rule"] == rule and t["action"] == "cleared")
            self.assertEqual(clear["details"]["direction"], "unchanged")

    def config(self, **overrides):
        settings = {
            "min_history": 5, "history_size": 30,
            "relative_change": .5, "mad_factor": 6,
            "window_ms": 60_000, "trace_min_calls": 20,
            "trace_history_windows": 3, "trace_fixed_ms": 1000,
            "trace_extreme_ms": 5000, "trace_factor": 3,
            "trace_min_increase_ms": 100, "gc_pause_ms": 1000,
            "cpu_high_percent": 90, "task_window_ms": 10_000,
        }
        settings.update(overrides)
        return AlertConfig(**settings)

    def engine(self, **overrides):
        return AlertEngine(start_ms=0, config=self.config(**overrides))

    def matching(self, engine, rule):
        return [item for item in engine.alerts.values() if item["rule"] == rule]

    def prime(self, engine, *, node="device:Tomcat01", value=100, kpi="test_gauge"):
        for index in range(5):
            engine.feed(metric(f"{node}-{kpi}-base-{index}", (index + 1) * 1000,
                               str(value), node=node, kpi=kpi))

    def test_relative_baseline_uses_only_prior_observations(self):
        engine = self.engine()
        self.prime(engine)
        self.assertEqual(self.matching(engine, "metric.relative_up"), [])
        engine.feed(metric("jump", 6000, "200"))
        alarms = self.matching(engine, "metric.relative_up")
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0]["first_seen_ms"], 6000)
        self.assertIn("jump", alarms[0]["evidence_ids"])
        self.assertEqual(alarms[0]["status"], "open")
        # Later large values must not retrospectively change the first trigger.
        saved_open = deepcopy(engine.transitions[0])
        engine.feed(metric("later-extreme", 7000, "10000"))
        self.assertEqual(engine.transitions[0], saved_open)

    def test_cold_start_does_not_invent_a_relative_baseline(self):
        engine = self.engine()
        for index, value in enumerate((10, 20, 40, 80, 160)):
            engine.feed(metric(f"cold-{index}", index * 1000, str(value)))
        self.assertEqual(self.matching(engine, "metric.relative_up"), [])
        self.assertEqual(self.matching(engine, "metric.relative_down"), [])

    def test_prior_three_sigma_catches_small_but_unusual_change(self):
        engine = self.engine()
        for index, value in enumerate((99, 100, 101, 100, 100)):
            engine.feed(metric(f"sigma-base-{index}", (index + 1) * 1000, value))
        engine.feed(metric("small-jump", 6000, 105))
        self.assertEqual(self.matching(engine, "metric.relative_up"), [])
        alert = self.matching(engine, "metric.std_up")[0]
        self.assertEqual(alert["latest"]["baseline_mean"], 100)
        self.assertEqual(alert["evidence_ids"], ["small-jump"])
        self.assertEqual(alert["latest"]["history_samples"], 5)
        first = deepcopy(engine.transitions)
        engine.feed(metric("much-later", 7000, 9000))
        self.assertEqual(engine.transitions[:len(first)], first)

    def test_flat_history_does_not_turn_tiny_noise_into_infinite_sigma(self):
        engine = self.engine()
        self.prime(engine)
        engine.feed(metric("rounding", 6000, 100.000001))
        self.assertEqual(self.matching(engine, "metric.std_up"), [])
        self.assertEqual(self.matching(engine, "metric.std_down"), [])

    def test_missing_and_nonfinite_values_are_not_filled_with_zero(self):
        engine = self.engine()
        self.prime(engine)
        for index, missing in enumerate(("", "NaN", "nan", "inf", None)):
            engine.feed(metric(f"missing-{index}", 6000 + index * 1000, missing))
        self.assertEqual(self.matching(engine, "metric.relative_down"), [])
        self.assertEqual(self.matching(engine, "metric.relative_up"), [])
        engine.feed(metric("real-fall", 12_000, "10"))
        self.assertEqual(len(self.matching(engine, "metric.relative_down")), 1)

    def test_cpu_absolute_threshold_has_an_explicit_unit_aware_whitelist(self):
        engine = self.engine()
        for index, kpi in enumerate(("JVM-CPU_CPULoad", "HeapMemoryUsage",
                                     "Container-CpuPercent", "untyped_value")):
            engine.feed(metric(f"unknown-unit-{index}", index * 1000, "9500", kpi=kpi))
        self.assertEqual(self.matching(engine, "metric.cpu_high"), [])
        engine.feed(metric("known-cpu", 5000, "95", kpi="OSLinux-CPU_CPU_CPUCpuUtil"))
        self.assertEqual(len(self.matching(engine, "metric.cpu_high")), 1)

    def test_kpis_and_device_identities_do_not_share_baselines(self):
        engine = self.engine()
        self.prime(engine, node="device:Tomcat01")
        engine.feed(metric("other-node-first", 6000, "10000", node="device:Tomcat02"))
        engine.feed(metric("other-kpi-first", 7000, "10000", kpi="different_gauge"))
        self.assertEqual(self.matching(engine, "metric.relative_up"), [])
        engine.feed(metric("same-series-jump", 8000, "200"))
        alarms = self.matching(engine, "metric.relative_up")
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0]["node_id"], "device:Tomcat01")

    def test_known_counter_uses_rate_and_reset_does_not_invent_a_negative_value(self):
        engine = self.engine()
        name = "redis-Redis_6379_Redis  (total_commands_processed)"
        for index in range(7):
            engine.feed(metric(f"count-{index}", (index + 1) * 1000,
                               str(index * 100), kpi=name))
        engine.feed(metric("reset", 8000, "10", kpi=name))
        engine.feed(metric("after-reset", 9000, "110", kpi=name))
        self.assertEqual(engine.alerts, {})
        engine.feed(metric("rate-jump", 10_000, "2110", kpi=name))
        alert = self.matching(engine, "metric.relative_up")[0]
        self.assertEqual(alert["latest"]["kind"], "rate_per_second")
        self.assertEqual(alert["latest"]["value"], 2000)

    def test_repeated_hits_merge_and_recovery_allows_a_new_incident(self):
        engine = self.engine()
        self.prime(engine)
        engine.feed(metric("jump-1", 6000, "200"))
        engine.feed(metric("jump-2", 7000, "210"))
        alarms = self.matching(engine, "metric.relative_up")
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0]["hits"], 2)
        self.assertEqual(alarms[0]["last_seen_ms"], 7000)
        engine.feed(metric("recovered", 8000, "100"))
        self.assertEqual(alarms[0]["status"], "cleared")
        engine.feed(metric("jump-again", 9000, "200"))
        alarms = self.matching(engine, "metric.relative_up")
        self.assertEqual(len(alarms), 2)
        self.assertEqual(len({alarm["alert_id"] for alarm in alarms}), 2)
        self.assertEqual(sum(alarm["status"] == "open" for alarm in alarms), 1)

    def test_same_rule_on_other_devices_has_a_separate_incident(self):
        engine = self.engine()
        engine.feed(log("oom-a", 1000, "java.lang.OutOfMemoryError: Java heap space"))
        engine.feed(log("oom-b", 2000, "java.lang.OutOfMemoryError: Java heap space",
                        node="device:Tomcat02"))
        alarms = self.matching(engine, "log.oom")
        self.assertEqual(len(alarms), 2)
        self.assertEqual({item["node_id"] for item in alarms},
                         {"device:Tomcat01", "device:Tomcat02"})

    def test_normal_allocation_failure_gc_is_not_an_error(self):
        engine = self.engine()
        text = ("27649.153: [GC (Allocation Failure) "
                "[ParNew: 848530K->9675K(943744K), 0.0493539 secs] "
                "1724017K->885221K(4089472K), 0.0495582 secs] "
                "[Times: user=0.11 sys=0.00, real=0.05 secs]")
        engine.feed(log("normal-gc", 1000, text, name="gc"))
        self.assertEqual(self.matching(engine, "log.error"), [])
        self.assertEqual(self.matching(engine, "log.gc_pause"), [])
        self.assertEqual(self.matching(engine, "log.oom"), [])

    def test_long_outer_gc_pause_triggers_even_if_inner_collection_is_short(self):
        engine = self.engine()
        text = ("[GC (Allocation Failure) [ParNew: 100K->20K(200K), 0.02 secs] "
                "200K->100K(400K), 1.50 secs] [Times: user=0.1 sys=0.0, real=1.50 secs]")
        engine.feed(log("long-gc", 1000, text, name="gc"))
        alarms = self.matching(engine, "log.gc_pause")
        self.assertEqual(len(alarms), 1)
        self.assertIn("long-gc", alarms[0]["evidence_ids"])
        self.assertEqual(self.matching(engine, "log.error"), [])

    def test_trace_extreme_waits_for_completion_and_fixed_window_closure(self):
        engine = self.engine()
        engine.advance_to(8000)
        self.assertEqual(engine.alerts, {})
        engine.feed(span("extreme-completed", 9000, 8000))
        engine.advance_to(59_999)
        self.assertEqual(self.matching(engine, "trace.extreme"), [])
        engine.advance_to(60_000)
        alarms = self.matching(engine, "trace.extreme")
        self.assertEqual(len(alarms), 1)
        self.assertIn("extreme-completed", alarms[0]["evidence_ids"])
        self.assertTrue(all(item["at_ms"] >= 60_000 for item in engine.transitions))

    def test_trace_fixed_latency_requires_enough_completed_calls(self):
        engine = self.engine()
        for index in range(19):
            engine.feed(span(f"first-{index}", 2000 + index * 100, 1200))
        engine.advance_to(60_000)
        self.assertEqual(self.matching(engine, "trace.latency"), [])
        for index in range(20):
            engine.feed(span(f"second-{index}", 62_000 + index * 100, 1200))
        engine.advance_to(120_000)
        self.assertEqual(len(self.matching(engine, "trace.latency")), 1)

    def test_trace_relative_latency_compares_with_previous_closed_windows(self):
        engine = self.engine()
        for window in range(3):
            for index in range(20):
                engine.feed(span(f"baseline-{window}-{index}",
                                 window * 60_000 + 1000 + index * 100, 50))
            engine.advance_to((window + 1) * 60_000)
        self.assertEqual(self.matching(engine, "trace.latency"), [])
        for index in range(20):
            engine.feed(span(f"changed-{index}", 181_000 + index * 100, 500))
        engine.advance_to(240_000)
        self.assertEqual(len(self.matching(engine, "trace.latency")), 1)

    def test_sparse_trace_window_does_not_prove_a_previous_alarm_cleared(self):
        engine = self.engine()
        for index in range(20):
            engine.feed(span(f"slow-{index}", 2000 + index * 100, 1200))
        engine.advance_to(60_000)
        alert = self.matching(engine, "trace.latency")[0]
        engine.feed(span("one-short-call", 61_000, 1))
        engine.advance_to(120_000)
        self.assertEqual(alert["status"], "open")

    def test_java_exception_and_http_5xx_are_explicit_log_triggers(self):
        engine = self.engine()
        engine.feed(log("exception", 1000, "java.lang.NullPointerException: request failed"))
        engine.feed(log("http-error", 2000, '"GET / HTTP/1.1" 503 0'))
        self.assertEqual({a["signal"] for a in self.matching(engine, "log.error")},
                         {"explicit_error", "http_5xx"})

    def test_feeding_and_alerts_do_not_modify_raw_records(self):
        engine = self.engine()
        original = log("preserve", 1000, "java.lang.OutOfMemoryError: heap space")
        original["raw"]["extra_field"] = "001.000"
        before = deepcopy(original)
        engine.feed(original)
        engine.advance_to(60_000)
        self.assertEqual(original, before)

    def test_a_new_error_kind_wakes_the_node_but_identical_repeats_do_not(self):
        engine = self.engine()
        engine.feed(log("first", 1000, "ERROR transient request failed"))
        engine.advance_to(10_000)
        engine.feed(log("repeat", 11_000, "ERROR transient request failed"))
        engine.feed(log("different-error", 12_000, "java.net.ConnectException: Connection refused"))
        engine.advance_to(20_000)
        engine.feed(log("same-exception", 21_000, "java.net.ConnectException: Connection refused"))
        engine.advance_to(30_000)
        self.assertEqual([item["action"] for item in engine.transitions], ["open", "updated"])
        self.assertEqual(len(engine.tasks), 2)
        self.assertEqual(engine.tasks[-1]["transitions"][0]["evidence_ids"], ["different-error"])

    def test_task_snapshots_contain_no_later_transitions_and_remain_unchanged(self):
        engine = self.engine()
        engine.feed(log("first", 1000, "java.lang.OutOfMemoryError: heap space"))
        engine.advance_to(10_000)
        self.assertTrue(engine.tasks)
        snapshots = deepcopy(engine.tasks)
        engine.feed(log("later", 12_000, "java.lang.OutOfMemoryError: heap space",
                        node="device:Tomcat02"))
        engine.finish(30_000)
        self.assertEqual(engine.tasks[:len(snapshots)], snapshots)
        for task in engine.tasks:
            for transition in task["transitions"]:
                self.assertLessEqual(transition["at_ms"], task["ready_at_ms"])
                self.assertEqual(transition["node_id"], task["node_id"])

    def test_delivery_batch_size_does_not_change_alerts_or_tasks(self):
        events = [metric(f"baseline-{index}", 1000 + index * 1000, str(value))
                  for index, value in enumerate((99, 100, 101, 100, 100))]
        events += [metric("up", 6000, "200"), metric("up-again", 7000, "210"),
                   metric("recovery", 8000, "100"),
                   log("oom", 12_000, "java.lang.OutOfMemoryError: heap space"),
                   span("extreme", 19_000, 8000)]
        events += [span(f"slow-{index}", 62_000 + index * 100, 1200)
                   for index in range(20)]
        events.sort(key=lambda item: (item["available_at_ms"], item["event_id"]))
        outcomes = []
        for step in (1000, 10_000, 60_000):
            engine = self.engine()
            cursor = 0
            for clock in range(step, 120_000 + 1, step):
                while cursor < len(events) and events[cursor]["available_at_ms"] < clock:
                    engine.feed(deepcopy(events[cursor]))
                    cursor += 1
                engine.advance_to(clock)
            engine.finish(120_000)
            self.assertEqual(cursor, len(events))
            self.assertTrue(self.matching(engine, "metric.std_up"))
            outcomes.append((deepcopy(engine.alerts), deepcopy(engine.transitions),
                             deepcopy(engine.tasks)))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0], outcomes[2])


if __name__ == "__main__":
    unittest.main()
