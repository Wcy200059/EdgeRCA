"""Fixed CPU SOP over an arrived node prefix. No labels, model or generated code.

Source IDs for every group live in the audit index; model pages contain exact
statistics and representative source IDs. Pagination never drops observations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import re
from statistics import median

from .alerts import COUNTER, CPU_PERCENT, ERROR, GC_PAUSE, MEMORY_PERCENT, OOM
from .data import format_time
from .tool_stats import change, distribution, identifier, number, period, unique


HTTP = re.compile(r'(?P<method>GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(?P<path>\S+)\s+HTTP/[\d.]+["\s]+(?P<status>\d{3})\b')
GC_REASON = re.compile(r'\[(?:Full )?GC\s*\(([^)]+)\)')
TIMESTAMP = re.compile(r'\b\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?')
UUID = re.compile(r'\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b')


def _time_extent(rows):
    return {"first_observed_ms": min(e["observed_at_ms"] for e in rows),
            "last_observed_ms": max(e["observed_at_ms"] for e in rows),
            "first_available_ms": min(e["available_at_ms"] for e in rows),
            "last_available_ms": max(e["available_at_ms"] for e in rows)}


def _metric_semantics(name):
    if COUNTER.search(name):
        return {"kind": "known_cumulative_counter", "analysis_unit": "raw_units/second"}
    if name == CPU_PERCENT:
        return {"kind": "gauge", "analysis_unit": "%", "meaning": "OS CPU使用率"}
    if name == MEMORY_PERCENT:
        return {"kind": "gauge", "analysis_unit": "%", "meaning": "内存占用比例，可能包含缓存"}
    if "idle" in name.lower():
        return {"kind": "observed_value", "analysis_unit": "source_unit_unverified",
                "meaning": "CPU空闲相关值；下降表示空闲减少，不能写成CPU使用率下降"}
    if any(word in name for word in ("FreeMem", "FreeMemory", "AvailableSpace")):
        return {"kind": "observed_value", "analysis_unit": "source_unit_unverified",
                "meaning": "可用/剩余量；下降不是用量下降"}
    return {"kind": "semantics_unverified", "analysis_unit": "source_unit_unverified",
            "meaning": "只描述原始数值，未确认累计/瞬时含义，不直接断言故障"}


def metric_summary(name, rows, reference_start, anchor, cutoff, field="value"):
    rows = sorted(rows, key=lambda e: (e["observed_at_ms"], e["event_id"]))
    semantics = _metric_semantics(name)
    if field != "value":
        semantics = {"kind": "business_sample", "analysis_unit": {
            "rr": "%", "sr": "%", "cnt": "requests_per_source_sample", "mrt": "ms"}[field]}
    timed = defaultdict(list)
    invalid_ids = []
    for row in rows:
        value = number(row["raw"].get(field))
        if value is None:
            invalid_ids.append(row["event_id"])
        else:
            timed[row["observed_at_ms"]].append((row, value))
    samples, conflicts = [], []
    for timestamp, entries in sorted(timed.items()):
        if len({value for _, value in entries}) != 1:
            conflicts.extend(e["event_id"] for e, _ in entries)
        else:
            samples.append((timestamp, entries[0][1], [e["event_id"] for e, _ in entries]))
    cadence = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
    values = samples
    resets, reset_count = [], 0
    if semantics["kind"] == "known_cumulative_counter":
        values = []
        for before, after in zip(samples, samples[1:]):
            # Conflicting/invalid intermediate samples break a rate interval.
            intervening = [r for r in rows if before[0] < r["observed_at_ms"] < after[0]]
            if intervening:
                continue
            if after[1] < before[1]:
                resets.extend(before[2] + after[2])
                reset_count += 1
                continue
            values.append((after[0], (after[1] - before[1]) * 1000 / (after[0] - before[0]),
                           before[2] + after[2]))
    ref = [s for s in values if reference_start <= s[0] < anchor]
    current = [s for s in values if anchor <= s[0] < cutoff]
    before, after = distribution([s[1] for s in ref]), distribution([s[1] for s in current])
    representatives = []
    for subset in (ref, current):
        if subset:
            representatives += [subset[0], subset[-1], min(subset, key=lambda s: s[1]),
                                max(subset, key=lambda s: s[1])]
    observed_change = change(before["median"], after["median"])
    # A whole-window reference is often empty at the first checkpoint because
    # the window itself starts at replay start. Give the model a within-window
    # comparison as well: first 5 minutes vs last 5 minutes of the queried range.
    # It is a description, not a changepoint or a normal baseline.
    early, late = distribution([]), distribution([])
    early_late_change = change(None, None)
    if values:
        first_time, last_time = values[0][0], values[-1][0]
        early = distribution([s[1] for s in values if s[0] < first_time + 300_000])
        late = distribution([s[1] for s in values if s[0] >= last_time - 300_000])
        early_late_change = change(early["median"], late["median"])
    return {
        "name": name, "semantics": semantics, "record_count": len(rows),
        "sample_count": len(samples), "invalid_count": len(invalid_ids),
        "conflicting_record_count": len(conflicts), "counter_reset_count": reset_count,
        "reference": before, "current": after, "change": observed_change,
        "early": early, "late": late, "early_late_change": early_late_change,
        "early_late_window_ms": 300_000,
        "comparison_status": "enough_samples_for_description" if len(ref) >= 5 and len(current) >= 2 else "limited_history",
        "sample_interval_median_ms": median(cadence) if cadence else None,
        "sample_interval_max_ms": max(cadence) if cadence else None,
        "current_first_last": change(current[0][1], current[-1][1]) if len(current) >= 2 else change(None, None),
        "extremes": [{"time_ms": s[0], "value": s[1], "evidence_ids": s[2][:2]}
                     for s in ([min(current, key=lambda s: s[1]), max(current, key=lambda s: s[1])] if current else [])],
        "evidence_ids": unique([eid for s in representatives for eid in s[2][:2]]
                               + invalid_ids[:1] + conflicts[:2] + resets[:2]),
        "quality_note": "相邻值与窗口变化是观测，不自动等于趋势或异常；参考段不保证正常。",
    }


def _log_key(row):
    raw = row["raw"]
    text, name = str(raw.get("value", "")), str(raw.get("log_name", ""))
    oom = bool(OOM.search(text))
    gc = name.lower() == "gc" or "[GC" in text or "[Full GC" in text
    match = HTTP.search(text)
    if oom:
        category, details = "oom_text", {"text_pattern": UUID.sub("<UUID>", TIMESTAMP.sub("<TIME>", text))}
    elif gc:
        cause = GC_REASON.search(text)
        category, details = "gc", {"gc_kind": "full" if "[Full GC" in text else "young_or_unspecified",
                                   "gc_reason": cause.group(1) if cause else "unspecified"}
    elif match:
        category, details = "http_access", match.groupdict()
        details["requester"] = text[:match.start()].strip().strip('"')
        # Status, path (including instance suffixes), requester and method remain
        # distinct. Sizes/durations varying per access do not produce new groups.
    else:
        category = "error_text" if ERROR.search(text) else "other_text"
        details = {"text_pattern": " ".join(UUID.sub("<UUID>", TIMESTAMP.sub("<TIME>", text)).split())}
    return (name, category, json.dumps(details, sort_keys=True)), details


def log_summaries(rows, reference_start, anchor, cutoff):
    groups = defaultdict(list)
    descriptions = {}
    for row in rows:
        key, details = _log_key(row)
        groups[key].append(row)
        descriptions[key] = details
    results = []
    for key, entries in sorted(groups.items()):
        entries.sort(key=lambda e: (e["observed_at_ms"], e["event_id"]))
        before = sum(reference_start <= e["available_at_ms"] < anchor for e in entries)
        after = sum(anchor <= e["available_at_ms"] < cutoff for e in entries)
        ref_seconds, cur_seconds = (anchor - reference_start) / 1000, (cutoff - anchor) / 1000
        pauses = [(e, GC_PAUSE.search(str(e["raw"].get("value", "")))) for e in entries]
        pauses = [(e, float(m.group(1)) * 1000) for e, m in pauses if m]
        reps = [entries[0], entries[-1]]
        if pauses:
            reps.append(max(pauses, key=lambda pair: pair[1])[0])
        details = descriptions[key]
        # Unknown long text is preserved fully in the archive, and explicitly
        # excerpted for the model; it is not silently described as fully read.
        if "text_pattern" in details and len(details["text_pattern"]) > 400:
            details = dict(details, text_pattern=details["text_pattern"][:400], pattern_excerpted=True)
        result = {"log_name": key[0], "category": key[1], "signature": identifier(*key),
                  "details": details, "count": len(entries), **_time_extent(entries),
                  "reference_count": before, "current_count": after,
                  "reference_per_second": before / ref_seconds if ref_seconds else None,
                  "current_per_second": after / cur_seconds if cur_seconds else None,
                  "has_explicit_error": key[1] in ("oom_text", "error_text") or
                                        (key[1] == "http_access" and int(details["status"]) >= 500),
                  "gc_pause_ms": distribution([value for _, value in pauses]),
                  "evidence_ids": unique(e["event_id"] for e in reps),
                  "note": "普通GC不是OOM；HTTP错误/错误词是观测，不直接确定根因。"}
        results.append((result, [e["event_id"] for e in entries]))
    return results


def trace_summary(rows, reference_start, anchor, cutoff):
    valid, invalid = [], []
    for e in rows:
        value = number(e["raw"].get("duration"))
        (valid if value is not None and value >= 0 else invalid).append((e, value))
    periods = {}
    for name, low, high in (("reference", reference_start, anchor), ("current", anchor, cutoff)):
        entries = [(e, d) for e, d in valid if low <= e["available_at_ms"] < high]
        periods[name] = distribution([d for _, d in entries])
        periods[name]["completed_calls_per_second"] = len(entries) * 1000 / (high - low) if high > low else None
        periods[name]["p95_low_sample"] = len(entries) < 20
    buckets = []
    for low in range(reference_start, cutoff, 60_000):
        high = min(cutoff, low + 60_000)
        entries = [(e, d) for e, d in valid if low <= e["available_at_ms"] < high]
        buckets.append({**period(low, high), **distribution([d for _, d in entries]),
                        "partial_minute": high - low < 60_000})
    slow = []
    for low, high in ((reference_start, anchor), (anchor, cutoff)):
        entries = [(e, d) for e, d in valid if low <= e["available_at_ms"] < high]
        if entries:
            e, duration = max(entries, key=lambda item: item[1])
            slow.append({"event_id": e["event_id"], "start_ms": e["observed_at_ms"],
                         "completed_ms": e["available_at_ms"], "duration_ms": duration,
                         "trace_id": e["raw"].get("trace_id"), "span_id": e["raw"].get("span_id")})
    return {"unit": "ms", "window_basis": "completion_availability", "quantile_method": "nearest_rank",
            "valid_count": len(valid), "invalid_count": len(invalid), **periods,
            "p95_change": change(periods["reference"]["p95"], periods["current"]["p95"]),
            "minute_buckets": buckets, "slow_examples": slow,
            "evidence_ids": unique([e["event_id"] for e, _ in valid[:1] + valid[-1:]]
                                   + [e["event_id"] for e in slow] + [e["event_id"] for e, _ in invalid[:1]]),
            "note": "只统计已完成调用；调用减少也可能是尚未完成。耗时不是网络延迟，不等于失败。"}


class FixedSOP:
    def __init__(self, session):
        self.session = session

    def build_task(self, task, episode_anchors, previous_findings=()):
        """Connect an actual alert task to tools at the task's original cutoff.

        ``episode_anchors`` is maintained from first-open transitions. For a
        latency-window alert use that window's start, not its evaluation time.
        Requiring it prevents escalations from silently rolling the baseline.
        """
        node = task["node_id"]
        access = task["data_access"]
        if access.get("node_id") != node:
            raise ValueError("task data_access belongs to a different node")
        transitions = task["transitions"]
        if not transitions:
            raise ValueError("alert task requires transitions")
        anchors = []
        for transition in transitions:
            key = transition["alert_id"]
            if key not in episode_anchors:
                raise ValueError("missing original episode anchor")
            anchor = episode_anchors[key]
            if isinstance(anchor, bool) or not isinstance(anchor, int) or anchor > transition["at_ms"]:
                raise ValueError("invalid original episode anchor")
            anchors.append(max(self.session.start_ms, anchor))
        result = self.build(node, access["available_before_ms"], min(anchors),
                            transitions, previous_findings)
        result["scope"].update(task_id=task["task_id"], anchor_basis="original_alert_episode",
                               task_prepared_not_executed=True)
        return result

    def build(self, node_id, cutoff_ms, anchor_ms, transitions=(), previous_findings=()):
        """All statistics use the same cutoff and a frozen caller-owned anchor.

        The caller keeps this anchor for updates of the same incident. Choosing
        a new rolling cutoff must not silently redefine the reference interval.
        """
        if (any(isinstance(t, bool) or not isinstance(t, int) for t in (cutoff_ms, anchor_ms))
                or not self.session.start_ms <= anchor_ms <= cutoff_ms <= self.session.clock_ms):
            raise ValueError("invalid or future SOP time range")
        start = max(self.session.start_ms, anchor_ms - 300_000)
        prefix = self.session.query(node_id, limit=None, available_before_ms=cutoff_ms)
        by_id = {e["event_id"]: e for e in prefix}
        rows = [e for e in prefix if e["available_at_ms"] >= start]
        observations, evidence_index, mandatory = [], {}, []

        def add(modality, key, fact, source_ids):
            oid = identifier(node_id, modality, key, cutoff_ms)
            if any(eid not in by_id for eid in source_ids + fact.get("evidence_ids", [])):
                raise ValueError("tool references unavailable or nonlocal evidence")
            observations.append({"observation_id": oid, "modality": modality,
                                 "status": "available", **fact})
            evidence_index[oid] = unique(source_ids)
            return oid

        # The alert itself remains visible even if it falls before a narrowed
        # statistical scope; no future/cross-node alert is accepted.
        for i, item in enumerate(transitions):
            if item["node_id"] != node_id or item["at_ms"] > cutoff_ms:
                raise ValueError("alert does not belong to this causal node task")
            fact = {key: deepcopy(item[key]) for key in ("action", "rule", "signal", "at_ms", "details", "evidence_ids")}
            mandatory.append(add("alert", i, fact, list(item["evidence_ids"])))

        metrics = defaultdict(list)
        for e in rows:
            if e["modality"] == "metric":
                name = e["raw"].get("kpi_name")
                if not name:
                    raise ValueError("metric missing kpi_name")
                metrics[(name, "value")].append(e)
            elif e["modality"] == "business_metric":
                for name in ("rr", "sr", "cnt", "mrt"):
                    metrics[(name, name)].append(e)
        for (name, field), entries in sorted(metrics.items()):
            fact = metric_summary(name, entries, start, anchor_ms, cutoff_ms, field)
            add("business_metric" if field != "value" else "metric", name, fact,
                [e["event_id"] for e in entries])
        logs = [e for e in rows if e["modality"] == "log"]
        for fact, ids in log_summaries(logs, start, anchor_ms, cutoff_ms):
            add("log", fact["signature"], fact, ids)
        traces = [e for e in rows if e["modality"] == "trace"]
        if traces:
            fact = trace_summary(traces, start, anchor_ms, cutoff_ms)
            # Minute buckets are separate facts, not an ever-growing model row.
            buckets = fact.pop("minute_buckets")
            add("trace", "latency", fact, [e["event_id"] for e in traces])
            for bucket in buckets:
                entries = [e for e in traces if bucket["start_ms"] <= e["available_at_ms"] < bucket["end_ms"]]
                reps = sorted(entries, key=lambda e: number(e["raw"].get("duration")) or 0)
                fact = {"kind": "latency_minute", **bucket, "unit": "ms",
                        "evidence_ids": [e["event_id"] for e in reps[-1:]]}
                add("trace", bucket["start_ms"], fact, [e["event_id"] for e in entries])
        modalities = ("business_metric",) if node_id.startswith("business:") else ("metric", "log", "trace")
        counts = Counter(e["modality"] for e in rows)
        for modality in modalities:
            if not counts[modality]:
                add(modality, "absence", {"status": "no_arrived_records_in_scope", "evidence_ids": [],
                    "note": "所查范围无已到达记录，不代表正常或永久没有该模态。"}, [])
        # Time alignment is explicit observation time, not an inferred causal chain.
        timeline = []
        for fact in observations:
            ids = fact.get("evidence_ids", [])
            if ids:
                timeline.append({"observation_id": fact["observation_id"],
                                 **_time_extent([by_id[eid] for eid in ids]),
                                 "extent_basis": "representative_records"})
        timeline.sort(key=lambda item: (item["first_observed_ms"], item["observation_id"]))
        cited = unique(eid for fact in observations for eid in fact.get("evidence_ids", []))
        return {"node_id": node_id, "cutoff_ms": cutoff_ms, "window_start_ms": self.session.start_ms,
                "steps": ["prepare_context", "summarize_metrics", "summarize_logs", "summarize_traces", "align_evidence"],
                "scope": {"reference": period(start, anchor_ms), "current": period(anchor_ms, cutoff_ms),
                          "rows_considered": len(rows), "prefix_rows": len(prefix),
                          "counts_by_modality": dict(counts), "must_report_ids": mandatory,
                          "raw_archive_preserved": True},
                "limitations": ["参考时段不保证正常；只做描述性统计，不自动确诊。",
                                 "工具已整理不等于模型已检查；CNN尚未接入。",
                                 "来源完整保存在inbox；代表原文可能截取，未展示内容不代表不存在。"],
                "observations": observations, "previous_findings": deepcopy(list(previous_findings)),
                "timeline": timeline, "evidence_index": evidence_index,
                "representatives": {eid: deepcopy(by_id[eid]) for eid in cited}}


def model_pages(result, max_observations=8, max_characters=7000):
    """Bound pages by actual encoded characters; retain explicit overflow.

    Full raw evidence and source membership remain in the CPU result, not copied
    into each prompt. Oversized single facts are marked, never silently clipped.
    """
    if max_observations < 1 or max_characters < 500:
        raise ValueError("invalid page budget")
    pages, group = [], []
    base = {key: deepcopy(result[key]) for key in
            ("node_id", "cutoff_ms", "window_start_ms", "previous_findings", "scope", "limitations")}

    def packet(items):
        p = deepcopy(base)
        p["observations"] = deepcopy(items)
        ids = {item["observation_id"] for item in items}
        p["scope"]["must_report_ids"] = [i for i in base["scope"]["must_report_ids"] if i in ids]
        for item in p["observations"]:
            item["source_examples"] = []
            for eid in item.get("evidence_ids", [])[:2]:
                e = result["representatives"][eid]
                raw = e["raw"]
                if e["modality"] == "log":
                    text = str(raw.get("value", ""))
                    raw = {"log_name": raw.get("log_name"), "value": text[:320],
                           "excerpted": len(text) > 320, "original_characters": len(text)}
                item["source_examples"].append({"event_id": eid,
                    "time": format_time(e["observed_at_ms"]), "raw": raw})
        return p

    def size(p):
        return len(json.dumps(p, ensure_ascii=False, separators=(",", ":")))

    for item in result["observations"]:
        candidate = packet(group + [item])
        if group and (len(group) >= max_observations or size(candidate) > max_characters - 220):
            pages.append(packet(group))
            group = []
        group.append(item)
    if group:
        pages.append(packet(group))
    for i, p in enumerate(pages):
        p["scope"].update(page_number=i + 1, page_count=len(pages), model_processing_status="not_started")
        p["scope"]["oversized_single_page"] = size(p) > max_characters - 45
    return pages
