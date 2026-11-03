"""Read-only node follow-ups over a case-bound, arrived telemetry prefix."""

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .compact_tools import _rounded, _when
from .coordinator import occurrence_ms
from .coordination import REQUEST_TOOLS, event_times, validate as validate_decision
from .investigation import FixedSOP, log_summaries, trace_summary
from .node_probe import decode_object, write_json
from .node_reports import messages, validate as validate_node
from .note_audit import review_note
from .report_cards import card
from .report_probe import report_pages
from .replay import ReplaySession
from .trace_index import TraceIndex


def case_id_for(prepared):
    return hashlib.sha256(str(Path(prepared).resolve()).encode()).hexdigest()[:16]


def request_key(request, view):
    bound = {"case_id": view["case_id"], "cutoff_ms": view["cutoff_ms"],
             "request": {k: deepcopy(request[k]) for k in ("node_id", "start", "end", "tools", "metrics")}}
    # Focus/citations explain a request; changing the wording must not rerun
    # the same data query. Different cutoffs still represent new data.
    bound["request"]["tools"] = sorted(bound["request"]["tools"])
    bound["request"]["metrics"] = sorted(bound["request"]["metrics"])
    bound["request"]["start"] = occurrence_ms(request["start"], view["window_start_ms"], view["cutoff_ms"])
    bound["request"]["end"] = occurrence_ms(request["end"], view["window_start_ms"], view["cutoff_ms"] + 1)
    return "q" + hashlib.sha256(json.dumps(bound, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def request_data_key(request, view):
    """Identify the queried data range across checkpoints (no cutoff in the key).

    The same node/range/tool/metrics returns the same data at a later checkpoint,
    so a carried pending request can reuse the earlier supplement instead of
    running the tools and the 4B node again.
    """
    bound = {"case_id": view["case_id"],
             "request": {k: deepcopy(request[k]) for k in ("node_id", "start", "end", "tools", "metrics")}}
    bound["request"]["tools"] = sorted(bound["request"]["tools"])
    bound["request"]["metrics"] = sorted(bound["request"]["metrics"])
    bound["request"]["start"] = occurrence_ms(request["start"], view["window_start_ms"], view["cutoff_ms"])
    bound["request"]["end"] = occurrence_ms(request["end"], view["window_start_ms"], view["cutoff_ms"] + 1)
    return "d" + hashlib.sha256(json.dumps(bound, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


class NodeRequestBroker:
    def __init__(self, prepared, case_id, cutoff_ms):
        if case_id != case_id_for(prepared):
            raise ValueError("follow-up data does not belong to this case")
        self.case_id, self.cutoff_ms = case_id, cutoff_ms
        self.session = ReplaySession(prepared)
        self.graph = TraceIndex()
        try:
            batch = self.session.advance_to(cutoff_ms)
            self.graph.ingest(batch["events"], cutoff_ms)
        except Exception:
            self.close()
            raise
        self.completed = {}

    def close(self):
        self.graph.close()
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def execute(self, request, view, output):
        validate_decision({"hypotheses": [], "requests": [request]}, view)
        if (view["case_id"] != self.case_id or view["cutoff_ms"] != self.cutoff_ms
                or view["window_start_ms"] != self.session.start_ms):
            raise ValueError("request has wrong case, window or cutoff")
        rid = request_key(request, view)
        if rid in self.completed:
            cached = deepcopy(self.completed[rid])
            cached["request"] = deepcopy(request)
            return cached
        start = occurrence_ms(request["start"], self.session.start_ms, self.cutoff_ms)
        end = occurrence_ms(request["end"], self.session.start_ms, self.cutoff_ms + 1)
        node = request["node_id"]
        # FixedSOP queries one node with an explicit historical cutoff even
        # when the receiver has already ingested later records.
        cpu = FixedSOP(self.session).build(node, end, start)
        if "get_logs" in request["tools"]:
            rows = [e for e in self.session.query(node, "log", limit=None, available_before_ms=end)
                    if e["available_at_ms"] >= start]
            cpu["followup_log_observations"] = [{"modality": "log", "status": "available",
                **f, "all_source_ids": ids} for f, ids in log_summaries(rows, start, start, end)]
            if not rows:
                cpu["followup_log_observations"] = [{"modality": "log", "status": "no_arrived_records_in_scope"}]
        if "get_calls" in request["tools"]:
            rows = [e for e in self.session.query(node, "trace", limit=None, available_before_ms=end)
                    if e["available_at_ms"] >= start]
            # Rebucket at the requested start. Filtering pre-existing minute
            # buckets could lose the first partial minute of a precise query.
            cpu["followup_trace_minutes"] = trace_summary(rows, start, start, end)["minute_buckets"] if rows else []
            for bucket in cpu["followup_trace_minutes"]:
                bucket["all_source_ids"] = [e["event_id"] for e in rows
                    if bucket["start_ms"] <= e["available_at_ms"] < bucket["end_ms"]]
        graph = self.graph.snapshot(end, max(self.session.start_ms, start - 300000), start)
        packet = build_packet(cpu, graph, request, self.case_id, self.cutoff_ms, rid)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "request.json", {"request_id": rid, "request": request,
                   "case_id": self.case_id, "decision_cutoff_ms": self.cutoff_ms})
        write_json(output / "cpu.json", cpu)
        write_json(output / "links.json", graph)
        write_json(output / "packet.json", packet)
        packet["audit_directory"] = str(output.resolve())
        self.completed[rid] = deepcopy(packet)
        return packet


def build_packet(cpu, graph, request, case_id, decision_cutoff, rid):
    facts, catalog, metric_names, missing_names = [], {}, [], []
    tools = [tool for tool in REQUEST_TOOLS if tool in request["tools"]]

    def add(value, sources):
        identifier = f"f{len(facts) + 1}"
        facts.append({"id": identifier, **deepcopy(value)})
        catalog[identifier] = deepcopy(sources)

    observations = cpu["observations"]
    metrics = [f for f in observations if f["modality"] == "metric" and "name" in f]
    if "list_metrics" in tools or "get_metrics" in tools:
        metric_names = [f["name"] for f in metrics]
    if "list_metrics" in tools:
        catalog_names = [name for name in metric_names if not request["metrics"] or
                         any(part.casefold() in name.casefold() for part in request["metrics"])]
        parts = [catalog_names[i:i + 16] for i in range(0, len(catalog_names), 16)] or [[]]
        for index, names in enumerate(parts):
            add({"kind": "metric_catalog", "names": names, "part": index + 1, "parts": len(parts),
                 "name_filters": request["metrics"], "total_available": len(metric_names)},
                {"scope": deepcopy(cpu["scope"]), "metric_observation_ids":
                 [f["observation_id"] for f in metrics if f["name"] in names]})
    if "get_metrics" in tools:
        missing_names = sorted(set(request["metrics"]) - set(metric_names))
        for f in metrics:
            if request["metrics"] and f["name"] not in request["metrics"]:
                continue
            add({"metric": f["name"], "meaning": f["semantics"].get("meaning", f["semantics"]["kind"]),
                 "unit": f["semantics"]["analysis_unit"],
                 "reference": {k: _rounded(f["reference"][k]) for k in ("count", "median")},
                 "current": {k: _rounded(f["current"][k]) for k in ("count", "median", "min", "max")},
                 "early": {k: _rounded(f["early"][k]) for k in ("count", "median")},
                 "late": {k: _rounded(f["late"][k]) for k in ("count", "median")},
                 "early_late_change": {k: _rounded(v) for k, v in f["early_late_change"].items()},
                 "extremes": [{"time_ms": e["time_ms"], "value": _rounded(e["value"])}
                              for e in f.get("extremes", [])],
                 "change": f["change"], "quality": {"invalid": f["invalid_count"],
                    "conflicts": f["conflicting_record_count"], "counter_resets": f["counter_reset_count"]}}, f)
        if not metrics or missing_names:
            add({"kind": "missing", "modality": "metric"}, {"missing_requested_names": missing_names})
    if "get_logs" in tools:
        access = defaultdict(list)
        for f in cpu.get("followup_log_observations", observations):
            if f["modality"] != "log":
                continue
            if f["status"] != "available":
                add({"kind": "log", "status": f["status"]}, f)
            elif f["category"] == "http_access" and not f["has_explicit_error"]:
                access[f["details"]["status"]].append(f)
            else:
                add({"kind": "log", "status": "available", "category": f["category"],
                     "details": f["details"], "count": f["current_count"],
                     "first": _when(f["first_observed_ms"]), "last": _when(f["last_observed_ms"]),
                     "explicit_error": f["has_explicit_error"], "gc_pause_max_ms": f["gc_pause_ms"]["max"]}, f)
        for status, items in sorted(access.items()):
            add({"kind": "http_access", "status": status, "count": sum(f["current_count"] for f in items),
                 "first": _when(min(f["first_observed_ms"] for f in items)),
                 "last": _when(max(f["last_observed_ms"] for f in items)), "template_count": len(items)}, items)
    if "get_calls" in tools:
        for f in observations:
            if f["modality"] != "trace":
                continue
            if f["status"] != "available":
                add({"kind": "missing", "modality": "trace"}, f)
            elif "p95_change" in f:
                add({"kind": "latency_over_analysis_window", "unit": "ms",
                     "reference": {k: f["reference"][k] for k in ("count", "p50", "p95", "max")},
                     "current": {k: f["current"][k] for k in ("count", "p50", "p95", "max")},
                     "slow_calls": [{"start": _when(e["start_ms"]), "end": _when(e["completed_ms"]),
                                     "duration_ms": e["duration_ms"]} for e in f["slow_examples"]
                                    if start_of(cpu) <= e["completed_ms"] < cpu["cutoff_ms"]],
                     "invalid_count": f["invalid_count"]}, f)
            elif "followup_trace_minutes" not in cpu and f["start_ms"] >= start_of(cpu):
                add({"kind": "minute_latency", "start": _when(f["start_ms"]), "end": _when(f["end_ms"]),
                     "partial": f["partial_minute"], **{k: f[k] for k in ("count", "p50", "p95", "max")}}, f)
        for f in cpu.get("followup_trace_minutes", []):
            add({"kind": "minute_latency", "start": _when(f["start_ms"]), "end": _when(f["end_ms"]),
                 "partial": f["partial_minute"], **{k: f[k] for k in ("count", "p50", "p95", "max")}},
                f)
        for link in graph["links"]:
            stats = link["child_span_latency_ms"]["current"]
            if request["node_id"] in (link["source_node"], link["target_node"]) and stats["calls"]:
                add({"kind": "observed_connection", "from": link["source_node"], "to": link["target_node"],
                     "child_latency_ms": {k: stats[k] for k in ("calls", "p95", "max")},
                     "interval_mismatch_count": link["interval_mismatch_count"],
                     "mismatch_scope": "all_arrived_prefix"}, link)
    scope = {"start": _when(start_of(cpu)), "end": _when(cpu["cutoff_ms"])}
    return {"case_id": case_id, "node_id": cpu["node_id"], "request_id": rid,
            "window_start_ms": cpu["window_start_ms"], "cutoff_ms": decision_cutoff,
            "query_end_ms": cpu["cutoff_ms"], "scope": scope, "requested_scope": deepcopy(cpu["scope"]),
            "request": deepcopy(request), "tools_executed": tools, "facts": facts, "catalog": catalog,
            "available_metric_names": metric_names, "missing_metrics": missing_names,
            "limitations": ["只检查请求节点、指定范围与最多前5分钟对照；参考段不保证正常。",
                            "请求之外的指标尚未检查；原始记录和工具完整结果保留。",
                            "没有相应字段或记录不能证明设备正常；慢调用不是网络RTT。"], "labels_used": False}


def start_of(cpu):
    return cpu["scope"]["current"]["start_ms"]


NODE_QUERIES = """本节点可以提出可执行补查，不必等待协调者发现所有缺口。
主流程仍是理解、合并并汇报已有事实。若现有材料不足以解释一个具体变化，可额外输出requests（每页最多1项）；没有需要就用空列表，不强制补查。
只查本节点，node_id由程序绑定。每项包含start、end、tools、metrics、focus、basis_ids。
start/end只写HH:MM:SS，可在本题已经到达的时间窗内查较早或更窄的时段；basis_ids引用本页f编号。
tools可选list_metrics（仅查实际指标名称目录，metrics可用["CPU","MEMORY"]按名称包含任一片段筛选；空列表列全部目录）、get_metrics（指标范围/中位数与最多前5分钟对照，不是完整曲线）、get_logs（日志分类/次数/出现时间与错误类型，不是任意全文检索）、get_calls（延迟/逐分钟统计/实际连接，不是失败率或网络RTT）。get_metrics的metrics须填写少量已有完整名称，不能为空；名称未知时先list_metrics，不能猜造名称。
focus说明缺少什么、查询能帮助区分什么。其他页还没读不表示数据缺失；已有范围的相同查询不重复，未观察到的现象不能预设存在。
涉及其他节点、工具不支持或仍缺依据时，写到questions交给协调者。不自行宣布全局根因。
格式沿用groups/retain_local/defer/questions，可额外加："requests":[{"start":"__START__","end":"__END__","tools":["get_logs"],"metrics":[],"focus":"核查资源变化同期是否有明确错误；查不到则保留未知","basis_ids":["__FACT__"]}]。例子仅示范格式，不是当前任务。"""


def node_request_view(packet, facts=None):
    return {"case_id": packet["case_id"], "window_start_ms": packet["window_start_ms"],
            "cutoff_ms": packet["cutoff_ms"], "node_status": [{"node_id": packet["node_id"]}],
            "system_context": {}, "query_policy": {"require_named_metrics": True},
            "evidence": [{"id": f["id"]} for f in (packet["facts"] if facts is None else facts)]}


def parse_node_reply(parsed, facts, packet):
    """Keep a useful report even when a proposed extra query is invalid."""
    if not isinstance(parsed, dict):
        raise ValueError("node reply needs an object")
    value = deepcopy(parsed)
    proposed = value.pop("requests", [])
    report = validate_node(value, facts)
    queries, errors = [], []
    if not isinstance(proposed, list) or len(proposed) > 1:
        return report, [], [{"item": proposed, "error": "at most one local request per page"}]
    required = {"start", "end", "tools", "metrics", "focus", "basis_ids"}
    for request in proposed:
        try:
            if not isinstance(request, dict) or set(request) != required:
                raise ValueError("local request needs start/end/tools/metrics/focus/basis_ids; node is bound by program")
            bound = {"node_id": packet["node_id"], **request}
            validate_decision({"hypotheses": [], "requests": [bound]}, node_request_view(packet, facts))
        except (ValueError, TypeError, KeyError) as error:
            errors.append({"item": deepcopy(request), "error": str(error)})
        else:
            queries.append(bound)
    return report, queries, errors


def answer(model, packet, output, *, allow_self_requests=False):
    results, groups, pending, queries, errors, questions = [], [], [], [], [], []
    facts_by_id = {f["id"]: f for f in packet["facts"]}
    for index, facts in enumerate(report_pages(packet, max_facts=5, max_chars=5000)):
        base = messages(packet, facts, readable=True)
        focus = packet.get("request", {}).get("focus", "汇总本节点实际现象、对照和未知，必要时提出本地补查。")
        source = "这是协调者补查。仅回答此问题：" if "request" in packet else "这是本节点持续整理任务："
        base[-1]["content"] += "\n" + source + focus
        base[-1]["content"] += "\n说明尽量不超过100字。统计数值由程序附上，优先概括相关现象与不能确定的部分。"
        if allow_self_requests:
            example = NODE_QUERIES.replace("__START__", packet["scope"]["start"]).replace("__END__", packet["scope"]["end"])
            example = example.replace("__FACT__", facts[0]["id"])
            base[-1]["content"] += "\n" + example
            base[-1]["content"] += "\n本题可查时间窗：" + _when(packet["window_start_ms"]) + "—" + _when(packet["cutoff_ms"])
            # The directory prevents treating another input page as absent.
            base[-1]["content"] += "\n全部事实目录（正文分在各页，不等于本页都已检查）：" + json.dumps([
                {"id": f["id"], "type": f.get("metric", f.get("signal", f.get("kind", "unknown")))}
                for f in packet["facts"]], ensure_ascii=False, separators=(",", ":"))
            base[-1]["content"] += "\n本轮已查过的查询：" + json.dumps(packet.get("already_checked", []), ensure_ascii=False)
        prompt, reply, attempts = deepcopy(base), None, []
        for attempt in range(2):
            try:
                raw = model.invoke(prompt, None, 1000, f"node_followup/{packet['request_id']}/{index}/{attempt}")
                parsed = decode_object(raw)
                if allow_self_requests:
                    reply, page_queries, page_errors = parse_node_reply(parsed, facts, packet)
                else:
                    reply, page_queries, page_errors = validate_node(parsed, facts), [], []
                warnings = [w for g in reply["groups"] for w in review_note(g["summary"], [facts_by_id[i] for i in g["observation_ids"]])]
                if warnings:
                    raise ValueError("工具事实核对：" + "; ".join(warnings))
            except (ValueError, KeyError, TypeError) as error:
                reply = None
                attempts.append({"valid": False, "error": str(error)})
                prompt = deepcopy(base)
                prompt[-1]["content"] += "\n修订：" + str(error)
                continue
            attempts.append({"valid": True})
            break
        if reply is None:
            pending += [f["id"] for f in facts]
        else:
            groups += reply["groups"]
            pending += reply["defer"]
            queries += page_queries
            errors += page_errors
            questions += [q for q in reply["questions"] if q not in questions]
        results.append({"page": index, "reply": reply, "attempts": attempts,
                        "self_requests": page_queries if reply is not None else [],
                        "request_errors": page_errors if reply is not None else []})
    cards = []
    for f in packet["facts"]:
        value = card(f, packet)
        value["time_marks"] = event_times(f, packet["window_start_ms"], packet["cutoff_ms"])
        if f.get("kind") == "missing":
            if f.get("modality") == "metric" and packet.get("missing_metrics"):
                value["tool_fact"] = "所请求的这些指标名在已到达记录中未找到：" + ", ".join(packet["missing_metrics"]) + "。不表示其他指标缺失或正常。"
        cards.append(value)
    result = {k: deepcopy(packet[k]) for k in ("case_id", "node_id", "request_id", "cutoff_ms")}
    result.update(requested_scope=deepcopy(packet.get("requested_scope", packet["scope"])),
                  tools_executed=deepcopy(packet.get("tools_executed", [])),
                  missing_metrics=deepcopy(packet.get("missing_metrics", [])),
                  available_metric_names=deepcopy(packet.get("available_metric_names", [])))
    result.update(facts=cards, groups=groups, pending_fact_ids=pending,
                  self_requests=queries, request_errors=errors, questions=questions,
                  metric_catalog_in_facts=any(f.get("kind") == "metric_catalog" for f in packet["facts"]),
                  node_model_status="complete" if not pending else "incomplete", labels_used=False,
                  note="工具事实完整返回；模型说明单独保留，未通过检查的页面仍待处理。")
    write_json(Path(output) / "node_answer.json", {"answer": result, "pages": results})
    return result
