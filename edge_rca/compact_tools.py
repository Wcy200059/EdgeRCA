"""Bound, read-only SOP tools for the first tiny-node experiment.

CPU summaries stay intact. This view selects alerted/changed metrics plus named
controls, compresses logs and latency, and exposes its omissions as CPU-only.
There is no top-K ranking, source deletion, free code execution or label access.
"""

from __future__ import annotations

from copy import deepcopy

from .alerts import CPU_PERCENT, MEMORY_PERCENT
from .data import format_time


TOOL_ORDER = ("get_metrics", "get_logs", "get_calls")
CONTROL_SUFFIXES = ("CPUUserTime", "CPUSysTime", "CPUWio", "CPUidleutil",
                    "NoCacheMemPerc", "JVMMemoryUsedPercent", "JVMFreeMemory",
                    "JVMUsedMemory", "CurrentThreadsBusyThreadInfo")


def _rounded(value):
    # Six significant digits for presentation; exact values remain in CPU output.
    return float(f"{value:.6g}") if isinstance(value, float) else value


def _when(ms):
    return format_time(ms).split("T", 1)[1].split("+", 1)[0] if ms is not None else None


def _stats(value, fields):
    return {key: _rounded(value[key]) for key in fields}


def build_compact_tools(result, graph):
    if graph["cutoff_ms"] != result["cutoff_ms"]:
        raise ValueError("graph and local statistics must share the same cutoff")
    node = result["node_id"]
    observations = result["observations"]
    alerts = [f for f in observations if f["modality"] == "alert"]
    alerted_names = {f["signal"] for f in alerts}
    records, catalog, cpu_only = {name: [] for name in TOOL_ORDER}, {}, []

    def add(tool, fact, value):
        prefix = {"get_metrics": "m", "get_logs": "l", "get_calls": "t"}[tool]
        local_id = f"{prefix}{len(records[tool]) + 1}"
        records[tool].append({"id": local_id, **value})
        catalog[local_id] = {"observation_id": fact["observation_id"],
                             "evidence_ids": deepcopy(fact.get("evidence_ids", [])),
                             "cpu_fact": deepcopy(fact)}

    for f in observations:
        modality = f["modality"]
        if f["status"] != "available":
            target = {"metric": "get_metrics", "log": "get_logs", "trace": "get_calls"}.get(modality)
            if target:
                add(target, f, {"kind": "missing", "note": "本查询范围无已到达记录，不能判正常。"})
            continue
        if modality == "metric":
            ratio = f["change"]["ratio"]
            changed = (f["comparison_status"] == "enough_samples_for_description"
                       and ratio is not None and (ratio >= 1.5 or 0 <= ratio <= .5))
            quality = f["invalid_count"] or f["conflicting_record_count"] or f["counter_reset_count"]
            control = f["name"] in (CPU_PERCENT, MEMORY_PERCENT) or f["name"].endswith(CONTROL_SUFFIXES)
            if not (f["name"] in alerted_names or changed or quality or control):
                cpu_only.append(f["observation_id"])
                continue
            add("get_metrics", f, {
                "metric": f["name"], "meaning": f["semantics"].get("meaning", f["semantics"]["kind"]),
                "unit": f["semantics"]["analysis_unit"],
                "reference": _stats(f["reference"], ("count", "median")),
                "current": _stats(f["current"], ("count", "median", "min", "max")),
                "early": _stats(f["early"], ("count", "median")),
                "late": _stats(f["late"], ("count", "median")),
                "early_late_change": {k: _rounded(v) for k, v in f["early_late_change"].items()},
                "extremes": [{"time_ms": e["time_ms"], "value": _rounded(e["value"])}
                             for e in f.get("extremes", [])],
                "change": {k: _rounded(v) for k, v in f["change"].items()},
                "quality": {"invalid": f["invalid_count"], "conflicts": f["conflicting_record_count"],
                            "counter_resets": f["counter_reset_count"], "comparison": f["comparison_status"]},
                "selected_because": [name for name, yes in (("alert", f["name"] in alerted_names),
                    ("window_change", changed), ("control", control), ("quality", bool(quality))) if yes]})
        elif modality == "log":
            details = f["details"]
            label = ({k: details.get(k) for k in ("method", "path", "status", "requester")}
                     if f["category"] == "http_access" else details)
            add("get_logs", f, {"kind": f["category"], "details": label,
                "reference_count": f["reference_count"], "current_count": f["current_count"],
                "first_time": _when(f["first_observed_ms"]), "last_time": _when(f["last_observed_ms"]),
                "explicit_error": f["has_explicit_error"],
                "gc_pause_max_ms": _rounded(f["gc_pause_ms"]["max"])})
        elif modality == "trace":
            if "p95_change" in f:
                add("get_calls", f, {"kind": "latency_over_analysis_window", "unit": "ms",
                    "reference": _stats(f["reference"], ("count", "p50", "p95", "max")),
                    "current": _stats(f["current"], ("count", "p50", "p95", "max")),
                    "slow_calls": [{"start": _when(e["start_ms"]), "end": _when(e["completed_ms"]),
                                    "duration_ms": e["duration_ms"]} for e in f["slow_examples"]],
                    "invalid_count": f["invalid_count"]})
            else:
                add("get_calls", f, {"kind": "minute_latency", "start": _when(f["start_ms"]),
                    "end": _when(f["end_ms"]), "partial": f["partial_minute"],
                    "unit": "ms", **_stats(f, ("count", "p50", "p95", "max"))})
    links = []
    for link in graph["links"]:
        if node in (link["source_node"], link["target_node"]):
            links.append({"link_id": link["link_id"], "from": link["source_node"], "to": link["target_node"],
                          "child_latency_ms": {key: _stats(value, ("calls", "p95", "max"))
                                               for key, value in link["child_span_latency_ms"].items()}})
    context = {"node_id": node, "request_scope": {
        "reference": [_when(result["scope"]["reference"][k]) for k in ("start_ms", "end_ms")],
        "current": [_when(result["scope"]["current"][k]) for k in ("start_ms", "end_ms")],
        "cutoff": _when(result["cutoff_ms"])},
        "alerts": [{"action": f["action"], "rule": f["rule"], "signal": f["signal"],
                    "time": _when(f["at_ms"]),
                    "values": {k: _rounded(v) for k, v in f["details"].items()
                               if k in ("calls", "p95_ms", "max_ms", "value", "baseline_median", "threshold_ms")}}
                   for f in alerts],
        "coverage": {"cpu_facts": len(observations), "tool_facts": len(catalog),
                     "cpu_only_metrics": len(cpu_only), "model_unread_cpu_only": True},
        "limitations": ["参考段不保证正常；没有日志或没有告警都不能证明正常。",
                         "清除只表示规则不再满足，不证明故障恢复；原始证据仍在。",
                         "调用耗时可帮助比较，不能等同网络延迟或确定根因；父子耗时不能相加。"]}
    return {"context": context, "tools": records, "links": links, "catalog": catalog,
            "cpu_only_observation_ids": cpu_only,
            "all_cpu_observation_ids": [f["observation_id"] for f in observations]}


def tool_schema(name):
    if name not in TOOL_ORDER:
        raise ValueError("unknown SOP tool")
    descriptions = {"get_metrics": "读取本次告警相关指标及正常对照的CPU统计。",
                    "get_logs": "读取本节点日志分组、次数与GC情况。",
                    "get_calls": "读取调用延迟、时间变化和已观察连接。"}
    return {"type": "function", "function": {"name": name, "description": descriptions[name],
        "parameters": {"type": "object", "properties": {
            "request_id": {"type": "string", "description": "复制当前请求编号"}},
            "required": ["request_id"], "additionalProperties": False}}}


class BoundToolSession:
    """Read-only tool dispatcher bound to one immutable node/cutoff request."""
    def __init__(self, request_id, bundle):
        self.request_id = request_id
        self.bundle = deepcopy(bundle)
        self.next_index = 0

    @property
    def next_tool(self):
        return TOOL_ORDER[self.next_index] if self.next_index < len(TOOL_ORDER) else None

    def execute(self, name, arguments):
        if name != self.next_tool or name is None:
            raise ValueError("tool out of SOP order or not allowed")
        if arguments != {"request_id": self.request_id}:
            raise ValueError("tool arguments must contain only the bound request_id")
        self.next_index += 1
        return {"request_id": self.request_id, "node_id": self.bundle["context"]["node_id"],
                "scope": self.bundle["context"]["request_scope"],
                "facts": deepcopy(self.bundle["tools"][name]),
                "observed_links": deepcopy(self.bundle["links"]) if name == "get_calls" else [],
                "note": "记录ID用于本次引用；CPU原文索引保留。未展示指标不是正常或已被模型检查。"}


def response_packet(bundle):
    """Short tool IDs map back to immutable CPU observations after validation."""
    facts = [f for name in TOOL_ORDER for f in bundle["tools"][name]]
    mandatory = []
    for f in facts:
        if "alert" in f.get("selected_because", []) or f.get("explicit_error"):
            mandatory.append(f["id"])
        if f.get("kind") == "latency_over_analysis_window" and any(
                a["rule"].startswith("trace.") for a in bundle["context"]["alerts"]):
            mandatory.append(f["id"])
    return {"observations": [{"observation_id": f["id"]} for f in facts],
            "scope": {"must_report_ids": mandatory}}


class PendingNodeUpdates:
    """Single owner queues task receipts while a frozen request is in flight.

    Coalescing preserves open/clear/reopen and every transition's evidence. It
    reduces queued requests, not observations or phenomena.
    """
    def __init__(self, node_id):
        self.node_id = node_id
        self.pending = []
        self.in_flight = None
        self.seen_task_ids = set()
        self.last_ready_ms = -1
        self.episode_anchors = {}

    def offer(self, task):
        if task["node_id"] != self.node_id:
            raise ValueError("wrong node")
        if task["task_id"] in self.seen_task_ids:
            return False
        if task["ready_at_ms"] < self.last_ready_ms:
            raise ValueError("tasks must arrive in order")
        anchors = dict(self.episode_anchors)
        for transition in task["transitions"]:
            key = transition["alert_id"]
            if transition["action"] == "open":
                anchors.setdefault(key, transition["details"].get("window_start_ms", transition["at_ms"]))
            elif key not in anchors:
                raise ValueError("update without original open event")
        self.episode_anchors = anchors
        self.last_ready_ms = task["ready_at_ms"]
        self.seen_task_ids.add(task["task_id"])
        self.pending.append(deepcopy(task))
        return True

    def begin(self):
        if self.in_flight is not None:
            raise RuntimeError("request already in flight")
        if not self.pending:
            return None
        tasks, self.pending = self.pending, []
        merged = deepcopy(tasks[-1])
        merged["task_id"] = "+".join(t["task_id"] for t in tasks)
        merged["transitions"] = [t for task in tasks for t in task["transitions"]]
        merged["coalesced_task_ids"] = [t["task_id"] for t in tasks]
        self.in_flight = merged
        return deepcopy(merged)

    def finish(self, task_id):
        if self.in_flight is None or self.in_flight["task_id"] != task_id:
            raise ValueError("completion does not match in-flight request")
        self.in_flight = None
        # Newly offered tasks are deliberately not acknowledged or cleared.
