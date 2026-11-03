"""Small, causal CPU monitors; alerts are investigation hints, not RCA answers."""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
import re
from statistics import median, fmean, pstdev


@dataclass(frozen=True)
class AlertConfig:
    min_history: int = 5
    history_size: int = 30
    relative_change: float = 0.5
    mad_factor: float = 6.0
    std_factor: float = 3.0
    window_ms: int = 60_000
    trace_min_calls: int = 20
    trace_history_windows: int = 3
    trace_fixed_ms: float = 1000
    trace_extreme_ms: float = 5000
    trace_factor: float = 3.0
    trace_min_increase_ms: float = 100
    gc_pause_ms: float = 1000
    cpu_high_percent: float = 90
    memory_high_percent: float = 90
    business_success_low_percent: float = 95
    task_window_ms: int = 10_000

    def __post_init__(self):
        for name in ("min_history", "history_size", "window_ms", "trace_min_calls",
                     "trace_history_windows", "task_window_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.history_size < self.min_history:
            raise ValueError("history_size must cover min_history")
        for name in ("relative_change", "mad_factor", "std_factor", "trace_fixed_ms", "trace_extreme_ms",
                     "trace_factor", "trace_min_increase_ms", "gc_pause_ms", "cpu_high_percent",
                     "memory_high_percent", "business_success_low_percent"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


CPU_PERCENT = "OSLinux-CPU_CPU_CPUCpuUtil"
MEMORY_PERCENT = "OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc"
# Only known accumulated counters are converted to rates. 'Counter' in an
# arbitrary name does not establish its semantics (e.g. active sessions).
COUNTER = re.compile(
    r"\((?:total_commands_processed|total_connections_received|used_cpu_(?:sys|user)(?:_children)?)\)$"
    r"|(?:RequestCountRequestInfo|ErrorCountRequestInfo)$"
    r"|MySQL_3306_Bytes (?:Received|Sent)$", re.I
)
GC_PAUSE = re.compile(r",\s*(\d+(?:\.\d+)?)\s*secs\](?:\s*\[Times:.*)?\s*$", re.S)
OOM = re.compile(r"\bOutOfMemoryError\b|Java heap space|GC overhead limit exceeded|Cannot allocate memory|\bOOM\b", re.I)
ERROR = re.compile(r"\b(?:error|fatal|crash|exception|fail|failed|failure)\b"
                   r"|\b[\w.$]+(?:Exception|Error)\b", re.I)
HTTP_5XX = re.compile(r'HTTP/\d(?:\.\d)?["\s]+(5\d\d)\b')
EXCEPTION_CLASS = re.compile(r"\b[\w.$]+(?:Exception|Error)\b")


def _error_context(text):
    exception = EXCEPTION_CLASS.search(text)
    if exception:
        return exception.group(0)
    # Preserve error codes and wording; mask only obvious volatile identifiers.
    normalized = re.sub(r"\b\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?", "<TIME>", text)
    normalized = re.sub(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b", "<UUID>", normalized)
    normalized = re.sub(r"\b0x[0-9a-fA-F]+\b", "<POINTER>", normalized)
    return hashlib.sha256(" ".join(normalized.split()).encode()).hexdigest()[:20]


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


class AlertEngine:
    """Process individual arrivals; transport batch size never defines a rule.

    Fixed minute windows use completed-call arrival time. Model tasks merge only
    transitions for the same node in fixed short windows; all raw data remains
    in ReplaySession's inbox. No language model is called here.
    """

    def __init__(self, start_ms: int, config: AlertConfig | None = None):
        self.config = config or AlertConfig()
        self.clock_ms = start_ms
        self.alerts: dict[str, dict] = {}
        self.transitions: list[dict] = []
        self.tasks: list[dict] = []
        self.stats = Counter()
        self._series = {}
        self._active = {}
        self._episodes = Counter()
        self._trace_windows = {}
        self._trace_history = {}
        self._task_windows = {}
        self._log_contexts = {}
        self._finished = False

    def _transition(self, alert, action, at_ms, details, evidence_ids):
        transition = {
            "action": action, "at_ms": at_ms, "alert_id": alert["alert_id"],
            "node_id": alert["node_id"], "rule": alert["rule"],
            "signal": alert["signal"], "details": deepcopy(details),
            "evidence_ids": list(evidence_ids),
        }
        self.transitions.append(transition)
        self.stats[f"transitions_{action}"] += 1
        bucket_end = (at_ms // self.config.task_window_ms + 1) * self.config.task_window_ms
        self._task_windows.setdefault((bucket_end, alert["node_id"]), []).append(deepcopy(transition))

    def _condition(self, node, rule, signal, at_ms, hit, details, evidence_ids, score=1.0):
        key = node, rule, signal
        active_id = self._active.get(key)
        if not hit:
            if active_id is not None:
                alert = self.alerts[active_id]
                alert["status"] = "cleared"
                alert["cleared_at_ms"] = at_ms
                self._transition(alert, "cleared", at_ms, details, evidence_ids)
                del self._active[key]
            return
        self.stats["rule_hits"] += 1
        if active_id is None:
            self._episodes[key] += 1
            identifier = hashlib.sha256(
                repr((key, at_ms, self._episodes[key])).encode()
            ).hexdigest()[:20]
            alert = {
                "alert_id": identifier, "node_id": node, "rule": rule, "signal": signal,
                "status": "open", "first_seen_ms": at_ms, "last_seen_ms": at_ms,
                "hits": 1, "evidence_ids": list(dict.fromkeys(evidence_ids)),
                "latest": deepcopy(details), "peak_score": float(score),
            }
            self.alerts[identifier] = alert
            self._active[key] = identifier
            if rule == "log.error":
                self._log_contexts[identifier] = {details.get("error_context")}
            self._transition(alert, "open", at_ms, details, evidence_ids)
            return
        alert = self.alerts[active_id]
        previous_peak = alert["peak_score"]
        alert["last_seen_ms"] = at_ms
        alert["hits"] += 1
        alert["latest"] = deepcopy(details)
        alert["latest_evidence_ids"] = list(evidence_ids)
        alert["peak_score"] = max(previous_peak, float(score))
        # Ordinary repeated hits update the same alert without new model work.
        if score >= max(previous_peak * 1.5, previous_peak + 1e-9):
            self._transition(alert, "escalated", at_ms, details, evidence_ids)
        elif rule == "log.error":
            contexts = self._log_contexts.setdefault(active_id, set())
            context = details.get("error_context")
            if context not in contexts:
                contexts.add(context)
                self._transition(alert, "updated", at_ms, details, evidence_ids)

    def _sample(self, event, signal, value, counter=False):
        key = event["node_id"], signal
        state = self._series.setdefault(key, {
            "history": deque(maxlen=self.config.history_size), "last_at": None,
            "previous_raw": None, "samples": 0,
        })
        observed = event["observed_at_ms"]
        if state["last_at"] is not None and observed <= state["last_at"]:
            self.stats["metric_duplicate_sample_times"] += 1
            return
        previous_at, previous_raw = state["last_at"], state["previous_raw"]
        state["last_at"], state["previous_raw"] = observed, value
        state["samples"] += 1
        raw_value = value
        if counter:
            if previous_at is None or value < previous_raw:
                self.stats["counter_warmup_or_reset"] += 1
                return
            value = (value - previous_raw) / ((observed - previous_at) / 1000)
        history = state["history"]
        details = {"value": value, "raw_value": raw_value, "observed_at_ms": observed,
                   "history_samples": len(history), "kind": "rate_per_second" if counter else "observation"}
        at = event["available_at_ms"]
        node = event["node_id"]
        evidence = [event["event_id"]]
        if signal == CPU_PERCENT and 0 <= value <= 100:
            self._condition(node, "metric.cpu_high", signal, at,
                            value >= self.config.cpu_high_percent,
                            {**details, "threshold": self.config.cpu_high_percent, "unit": "%"},
                            evidence, value)
        if signal == MEMORY_PERCENT and 0 <= value <= 100:
            self._condition(node, "metric.memory_high", signal, at,
                            value >= self.config.memory_high_percent,
                            {**details, "threshold": self.config.memory_high_percent, "unit": "%",
                             "note": "Occupancy may include cache; investigate memory pressure before diagnosis"},
                            evidence, value)
        if signal == "sr" and event["modality"] == "business_metric" and 0 <= value <= 100:
            self._condition(node, "business.success_low", signal, at,
                            value < self.config.business_success_low_percent,
                            {**details, "threshold": self.config.business_success_low_percent,
                             "unit": "%"}, evidence, 100-value)
        if len(history) < self.config.min_history:
            self.stats["relative_warmup_observations"] += 1
            history.append(value)
            return
        center = median(history)
        mad = median(abs(item - center) for item in history)
        threshold = max(self.config.mad_factor * 1.4826 * mad,
                        self.config.relative_change * abs(center), 1e-8)
        diff = value - center
        details.update({"baseline_median": center, "baseline_mad": mad,
                        "change_threshold": threshold, "direction": "up" if diff > 0 else "down" if diff < 0 else "unchanged"})
        up, down = diff > threshold, diff < -threshold
        self._condition(node, "metric.relative_up", signal, at, up,
                        details, evidence, abs(diff))
        self._condition(node, "metric.relative_down", signal, at, down,
                        details, evidence, abs(diff))
        # Borrow OpsAgent's 3-sigma idea, but use prior observations only. The
        # original whole-window statistics would reveal later samples online.
        mean, std = fmean(history), pstdev(history)
        sigma_diff = value - mean
        sigma_threshold = self.config.std_factor * std
        sigma_hit = std > 1e-8 and abs(sigma_diff) > sigma_threshold
        sigma_details = {**details, "baseline_mean": mean, "baseline_std": std,
                         "change_threshold": sigma_threshold,
                         "direction": "up" if sigma_diff > 0 else "down" if sigma_diff < 0 else "unchanged",
                         "note": "Prior-sample 3-sigma candidate; not a root-cause verdict"}
        self._condition(node, "metric.std_up", signal, at,
                        sigma_hit and sigma_diff > 0, sigma_details, evidence, abs(sigma_diff))
        self._condition(node, "metric.std_down", signal, at,
                        sigma_hit and sigma_diff < 0, sigma_details, evidence, abs(sigma_diff))
        # Freeze an alerted baseline rather than learning the spike as normal.
        if not (up or down or sigma_hit):
            history.append(value)

    def _log(self, event):
        raw, node, at = event["raw"], event["node_id"], event["available_at_ms"]
        text = raw.get("value", "")
        evidence = [event["event_id"]]
        gc = raw.get("log_name", "").lower() == "gc" or "[GC" in text or "[Full GC" in text
        details = {"log_name": raw.get("log_name", ""), "observed_at_ms": event["observed_at_ms"]}
        if OOM.search(text):
            self._condition(node, "log.oom", "out_of_memory", at, True, details, evidence)
        elif not gc:
            status = HTTP_5XX.search(text)
            if status:
                self._condition(node, "log.error", "http_5xx", at, True,
                                {**details, "http_status": int(status.group(1)),
                                 "error_context": f"http_status:{status.group(1)}"}, evidence)
            elif ERROR.search(text):
                self._condition(node, "log.error", "explicit_error", at, True,
                                {**details, "error_context": _error_context(text)}, evidence)
        if gc:
            pause = GC_PAUSE.search(text)
            if pause:
                pause_ms = float(pause.group(1)) * 1000
                self._condition(node, "log.gc_pause", "gc_pause_ms", at,
                                pause_ms >= self.config.gc_pause_ms,
                                {**details, "pause_ms": pause_ms,
                                 "threshold_ms": self.config.gc_pause_ms}, evidence, pause_ms)
            else:
                self.stats["gc_lines_without_complete_pause"] += 1

    def _trace(self, event):
        duration = _number(event["raw"].get("duration"))
        if duration is None or duration < 0:
            self.stats["invalid_numeric_observations"] += 1
            return
        end = (event["available_at_ms"] // self.config.window_ms + 1) * self.config.window_ms
        bucket = self._trace_windows.setdefault((end, event["node_id"]), {
            "durations": [], "max_event_id": None, "max_duration": -1,
        })
        bucket["durations"].append(duration)
        if duration > bucket["max_duration"]:
            bucket["max_duration"] = duration
            bucket["max_event_id"] = event["event_id"]

    def _finish_trace(self, end, node, bucket, at_ms=None):
        durations = sorted(bucket["durations"])
        count = len(durations)
        p95 = durations[max(0, math.ceil(count * .95)-1)]
        history = self._trace_history.setdefault(node, deque(maxlen=self.config.history_size))
        baseline = median(history) if history else None
        relative = (count >= self.config.trace_min_calls
                    and len(history) >= self.config.trace_history_windows
                    and p95 > max(baseline * self.config.trace_factor,
                                  baseline + self.config.trace_min_increase_ms))
        fixed = count >= self.config.trace_min_calls and p95 >= self.config.trace_fixed_ms
        hit = relative or fixed
        maximum = bucket["max_duration"]
        at = end if at_ms is None else at_ms
        details = {"window_start_ms": end-self.config.window_ms,
                   "window_end_ms": at, "partial_window": at != end,
                   "calls": count, "p95_ms": p95, "max_ms": maximum,
                   "baseline_p95_median_ms": baseline,
                   "baseline_windows": len(history),
                   "fixed_threshold_ms": self.config.trace_fixed_ms,
                   "rule_basis": "past_relative" if relative else "fixed_threshold"}
        evidence = [bucket["max_event_id"]]
        if count >= self.config.trace_min_calls:
            self._condition(node, "trace.latency", "completed_call_p95", at,
                            hit, details, evidence, p95)
        else:
            self.stats["trace_windows_insufficient_samples"] += 1
        self._condition(node, "trace.extreme", "maximum_call_duration", at,
                        maximum >= self.config.trace_extreme_ms,
                        {**details, "threshold_ms": self.config.trace_extreme_ms}, evidence, maximum)
        if not hit and count >= self.config.trace_min_calls:
            history.append(p95)
        self.stats["trace_windows_evaluated"] += 1

    def _make_task(self, end, node, transitions, ready=None):
        at = end if ready is None else ready
        identifier = hashlib.sha256(repr((node, end)).encode()).hexdigest()[:20]
        self.tasks.append({
            "task_id": identifier, "node_id": node, "ready_at_ms": at,
            "window_end_ms": end, "partial_window": at != end,
            "transitions": deepcopy(transitions),
            "checks_to_run": ["read_new_observations", "update_local_record",
                              "report_key_changes"],
            "data_access": {"node_id": node, "available_before_ms": at},
            "note": "Local summary request only; no model called and no root cause assigned",
        })

    def _roll_to(self, target):
        for end, node in sorted(key for key in self._trace_windows if key[0] <= target):
            self._finish_trace(end, node, self._trace_windows.pop((end, node)))
        for end, node in sorted(key for key in self._task_windows if key[0] <= target):
            self._make_task(end, node, self._task_windows.pop((end, node)))

    def feed(self, event):
        if self._finished:
            raise RuntimeError("Alert engine is finished")
        at = event["available_at_ms"]
        if at < self.clock_ms:
            raise ValueError("Alert events must arrive in nondecreasing availability order")
        self._roll_to(at)
        self.clock_ms = at
        self.stats["observations_seen"] += 1
        modality = event["modality"]
        self.stats[f"observations_{modality}"] += 1
        if modality == "metric":
            value = _number(event["raw"].get("value"))
            if value is None:
                self.stats["invalid_numeric_observations"] += 1
            else:
                name = event["raw"].get("kpi_name")
                if not name:
                    raise ValueError("Metric kpi_name is required")
                self._sample(event, name, value, bool(COUNTER.search(name)))
        elif modality == "business_metric":
            for name in ("rr", "sr", "cnt", "mrt"):
                value = _number(event["raw"].get(name))
                if value is None:
                    self.stats["invalid_numeric_observations"] += 1
                else:
                    self._sample(event, name, value)
        elif modality == "log":
            self._log(event)
        elif modality == "trace":
            self._trace(event)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def advance_to(self, clock_ms):
        if self._finished:
            raise RuntimeError("Alert engine is finished")
        if clock_ms < self.clock_ms:
            raise ValueError("Alert clock cannot move backwards")
        self._roll_to(clock_ms)
        self.clock_ms = clock_ms

    def finish(self, end_ms):
        self.advance_to(end_ms)
        for end, node in sorted(self._trace_windows):
            self._finish_trace(end, node, self._trace_windows[(end, node)], at_ms=end_ms)
        self._trace_windows.clear()
        for end, node in sorted(self._task_windows):
            self._make_task(end, node, self._task_windows[(end, node)], ready=end_ms)
        self._task_windows.clear()
        self.stats["metric_series"] = len(self._series)
        self.stats["metric_series_without_relative_baseline"] = sum(
            len(state["history"]) < self.config.min_history for state in self._series.values())
        self.stats["alerts_total"] = len(self.alerts)
        self.stats["tasks_total"] = len(self.tasks)
        self._finished = True
