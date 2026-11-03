"""Constrained node findings: program-filled slots plus a small model choice.

The node does not write a free-form report. It fills one of a small set of
templates. Numbers, times, units and source fact IDs are copied from the CPU
tool packet by the program; the 4B only selects a template's status, direction,
candidate/ reason and gap. A template that does not apply is marked by the
program, never forced. "No anomaly" and "no candidate" are explicit answers,
not guesses. Findings are short, citable tool facts; the coordinator can use
them without loading the full node report.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .bank_context import public_bank_context
from .compact_tools import _when
from .node_probe import decode_object, write_json


TEMPLATE_IDS = ("anomaly", "candidate", "log", "calls", "coverage")
TEMPLATE_LABELS = {"anomaly": "异常数值", "candidate": "根因候选", "log": "日志",
                   "calls": "调用", "coverage": "覆盖/无发现"}
ANOMALY_STATUSES = ("filled", "insufficient", "no_finding")
CANDIDATE_STATUSES = ("supported", "unresolved", "no_candidate")
EVIDENCE_STATUSES = ("filled", "insufficient", "no_finding")
COVERAGE_STATUSES = ("no_finding", "insufficient")
DIRECTIONS = ("up", "down", "unchanged", "unknown")

SYSTEM = """你是边缘节点填表员。程序已把数值、时间、单位、告警状态、证据编号和候选清单算好并放在输入里。你只做判断和选择，不重抄、不计算、不编造数值。
必须为每一个applicable_templates中的模板输出恰好一条finding；不要输出not_applicable_templates中的模板。状态只能使用模板规定值。候选允许为空：没有足够依据时用unresolved或no_candidate，不许为了填满而猜一个组件。key_evidence_id必须从该模板的evidence_options里选择；support_ids/counter_ids也只能用这些编号。没有异常就填no_finding；数据不够就填insufficient，并在note写清缺什么。coverage模板必须写检查了什么、缺哪些模态、不能据此得出什么结论。
只输出JSON：{"findings":[...],"unexpected":"可选，最多200字，不是证据"}。
每个finding必须包含template、status、note；其他字段按模板给。note不超过200字。不要输出推理过程，不要输出模板以外的字段。"""


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fmt(value):
    return "未知" if value is None else f"{value:.6g}" if isinstance(value, (int, float)) else str(value)


def _unit(unit):
    return "%" if unit == "%" else ""


def _anomaly_score(signal, value, baseline, unit):
    severity = 100.0 if unit == "%" and _number(value) and value >= 80 else 0.0
    if _number(value) and _number(baseline) and abs(baseline) > 1e-9:
        return severity + min(abs(value - baseline) / abs(baseline), 10.0)
    return severity + (0.5 if _number(value) else 0.0)


def _alarm_anomaly(fact):
    first = fact.get("first_trigger_values") or {}
    value, baseline = first.get("value"), first.get("baseline_median")
    if not _number(value):
        return None
    timeline = fact.get("timeline") or []
    return {
        "signal": fact.get("signal", "未知"),
        "value": value,
        "baseline": baseline,
        "unit": first.get("unit"),
        "time": timeline[0][0] if timeline else None,
        "state": fact.get("state", "未知"),
        "evidence": [fact["id"]],
        "score": _anomaly_score(fact.get("signal"), value, baseline, first.get("unit")),
    }


def _metric_anomaly(fact):
    current = fact.get("current") or {}
    median = current.get("median")
    extremes = [e for e in fact.get("extremes", []) if _number(e.get("value"))]
    if not _number(median) or not extremes:
        return None
    extreme = max(extremes, key=lambda e: abs(e["value"] - median))
    unit = fact.get("unit")
    return {
        "signal": fact.get("metric", "未知"),
        "value": extreme["value"],
        "baseline": median,
        "unit": unit,
        "time": _when(extreme.get("time_ms")),
        "state": "observed",
        "evidence": [fact["id"]],
        "score": _anomaly_score(fact.get("metric"), extreme["value"], median, unit),
    }


def _log_summary(fact):
    if fact.get("kind") == "http_access":
        return {"id": fact["id"], "kind": "http_access", "status": fact.get("status"),
                "count": fact.get("count"), "first": fact.get("first"), "last": fact.get("last"),
                "explicit_error": False, "templates": fact.get("template_count")}
    return {"id": fact["id"], "kind": "log", "status": fact.get("status"),
            "category": fact.get("category"), "count": fact.get("count"),
            "first": fact.get("first"), "last": fact.get("last"),
            "explicit_error": fact.get("explicit_error"),
            "gc_pause_max_ms": fact.get("gc_pause_max_ms")}


def _call_summary(fact):
    if fact.get("kind") == "observed_connection":
        stats = fact.get("child_latency_ms") or {}
        return {"id": fact["id"], "kind": "observed_connection", "from": fact.get("from"),
                "to": fact.get("to"), "calls": stats.get("calls"),
                "p95": stats.get("p95"), "max": stats.get("max"),
                "mismatch": fact.get("interval_mismatch_count")}
    current = fact.get("current") or {}
    return {"id": fact["id"], "kind": fact.get("kind"), "count": current.get("count", fact.get("count")),
            "p50": current.get("p50", fact.get("p50")), "p95": current.get("p95", fact.get("p95")),
            "max": current.get("max", fact.get("max")), "partial": fact.get("partial")}


def build_context(packet):
    """Build a small, program-filled form. No template is forced to have a cause."""
    context = public_bank_context()
    facts = packet.get("facts", [])
    scope = packet.get("scope", {})
    raw_rows = scope.get("raw_rows") or scope.get("counts_by_modality") or {}
    alarms = [f for f in facts if f.get("kind") == "alarm_history"]
    metrics = [f for f in facts if "metric" in f]
    logs = [f for f in facts if f.get("kind") in ("log", "http_access")]
    calls = [f for f in facts if f.get("kind") in
             ("latency_over_analysis_window", "minute_latency", "observed_connection")]

    anomalies, seen = [], {}
    for fact in alarms:
        item = _alarm_anomaly(fact)
        if item and (item["signal"] not in seen or item["score"] > seen[item["signal"]]["score"]):
            seen[item["signal"]] = item
    for fact in metrics:
        item = _metric_anomaly(fact)
        if item and (item["signal"] not in seen or item["score"] > seen[item["signal"]]["score"]):
            seen[item["signal"]] = item
    anomalies = sorted(seen.values(), key=lambda item: item["score"], reverse=True)[:3]

    log_available = bool(raw_rows.get("log")) or any(
        f.get("status") != "no_arrived_records_in_scope" for f in logs if f.get("kind") == "log") \
        or any(f.get("kind") == "http_access" for f in logs)
    calls_available = bool(raw_rows.get("trace")) or bool(calls)
    anomaly_available = bool(anomalies)
    candidate_available = bool(alarms or anomalies)
    missing = [name for name, key in (("日志", "log"), ("调用", "trace"))
               if not raw_rows.get(key)]
    if not log_available and "日志" not in missing:
        missing.append("日志")
    if not calls_available and "调用" not in missing:
        missing.append("调用")

    applicable = ["coverage"]
    if anomaly_available:
        applicable.insert(0, "anomaly")
    if candidate_available:
        applicable.insert(1 if "anomaly" in applicable else 0, "candidate")
    if log_available:
        applicable.insert(len(applicable) - 1, "log")
    if calls_available:
        applicable.insert(len(applicable) - 1, "calls")
    not_applicable = []
    if not anomaly_available:
        not_applicable.append({"template": "anomaly", "reason": "没有可用的指标/告警数值"})
    if not candidate_available:
        not_applicable.append({"template": "candidate", "reason": "没有异常证据，不能要求候选"})
    if not log_available:
        not_applicable.append({"template": "log", "reason": "本时段没有已到达日志"})
    if not calls_available:
        not_applicable.append({"template": "calls", "reason": "本时段没有已到达调用"})

    candidate_evidence = list(dict.fromkeys(
        [item["evidence"][0] for item in anomalies] + [f["id"] for f in alarms[:10]]))
    templates = []
    for name in applicable:
        if name == "anomaly":
            templates.append({"template": name, "label": TEMPLATE_LABELS[name],
                              "statuses": list(ANOMALY_STATUSES),
                              "evidence_options": [item["evidence"][0] for item in anomalies],
                              "anomalies": anomalies})
        elif name == "candidate":
            templates.append({"template": name, "label": TEMPLATE_LABELS[name],
                              "statuses": list(CANDIDATE_STATUSES),
                              "candidate_components": context["root_cause_components"],
                              "candidate_reasons": context["root_cause_reasons"],
                              "time_candidates": [item["time"] for item in anomalies if item.get("time")],
                              "evidence_options": candidate_evidence})
        elif name == "log":
            templates.append({"template": name, "label": TEMPLATE_LABELS[name],
                              "statuses": list(EVIDENCE_STATUSES),
                              "evidence_options": [f["id"] for f in logs if f.get("status") != "no_arrived_records_in_scope"],
                              "logs": [_log_summary(f) for f in logs
                                       if f.get("status") != "no_arrived_records_in_scope"]})
        elif name == "calls":
            templates.append({"template": name, "label": TEMPLATE_LABELS[name],
                              "statuses": list(EVIDENCE_STATUSES),
                              "evidence_options": [f["id"] for f in calls],
                              "calls": [_call_summary(f) for f in calls]})
        else:
            templates.append({"template": name, "label": TEMPLATE_LABELS[name],
                              "statuses": list(COVERAGE_STATUSES),
                              "checked": [label for label, key in (("指标", "metric"), ("日志", "log"), ("调用", "trace"))
                                          if raw_rows.get(key)],
                              "missing": missing,
                              "raw_rows": raw_rows,
                              "alarm_groups": scope.get("alarm_groups"),
                              "cpu_only_metric_count": scope.get("cpu_only_metric_count")})
    return {"node_id": packet.get("node_id"), "case_id": packet.get("case_id"),
            "window_start_ms": packet.get("window_start_ms"), "cutoff_ms": packet.get("cutoff_ms"),
            "scope": {"start": scope.get("start"), "end": scope.get("end")},
            "applicable_templates": applicable, "not_applicable_templates": not_applicable,
            "templates": templates,
            "output_schema": {
                "findings": [
                    {"template": "anomaly|log|calls", "status": "filled|insufficient|no_finding",
                     "key_evidence_id": "从evidence_options选", "direction": "up|down|unchanged|unknown",
                     "note": "不超过200字"},
                    {"template": "candidate", "status": "supported|unresolved|no_candidate",
                     "component": "公开候选或null", "reason": "公开原因或null",
                     "time": "从time_candidates选或null", "support_ids": [], "counter_ids": [],
                     "note": "不超过200字"},
                    {"template": "coverage", "status": "no_finding|insufficient",
                     "missing": "从missing选", "cannot_conclude": "不能据此得出什么结论",
                     "note": "不超过200字"}],
                "unexpected": "可选自由文本，不是证据"}}


def _evidence_ids(value, allowed):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("support/counter must be lists of fact IDs")
    if len(value) != len(set(value)) or any(item not in allowed for item in value):
        raise ValueError("support/counter IDs must come from the given evidence options")
    return value


def validate_findings(reply, context):
    if not isinstance(reply, dict) or not set(reply) <= {"findings", "unexpected"} or "findings" not in reply:
        raise ValueError("reply needs findings and may contain unexpected")
    unexpected = reply.get("unexpected")
    if unexpected is not None and (not isinstance(unexpected, str) or len(unexpected) > 200):
        raise ValueError("unexpected must be a short string")
    findings = reply["findings"]
    if not isinstance(findings, list):
        raise ValueError("findings must be a list")
    specs = {item["template"]: item for item in context["templates"]}
    if len(findings) != len(specs):
        raise ValueError("one finding is required for every applicable template")
    normalized, seen = [], set()
    for finding in findings:
        if not isinstance(finding, dict) or not isinstance(finding.get("template"), str):
            raise ValueError("each finding needs a template")
        template = finding["template"]
        if template not in specs or template in seen:
            raise ValueError("duplicate or not-applicable template")
        seen.add(template)
        spec = specs[template]
        status = finding.get("status")
        if status not in spec["statuses"]:
            raise ValueError(f"{template}: invalid status")
        note = finding.get("note")
        if not isinstance(note, str) or not 1 <= len(note) <= 200:
            raise ValueError(f"{template}: note must contain 1..200 characters")
        item = {"template": template, "status": status, "note": note}
        if template in ("anomaly", "log", "calls"):
            evidence = finding.get("key_evidence_id")
            if status in ("filled", "insufficient"):
                if evidence not in spec["evidence_options"]:
                    raise ValueError(f"{template}: choose key_evidence_id from evidence_options")
            elif evidence is not None and evidence not in spec["evidence_options"]:
                raise ValueError(f"{template}: invalid key_evidence_id")
            item["key_evidence_id"] = evidence
            if template == "anomaly":
                direction = finding.get("direction", "unknown")
                if direction not in DIRECTIONS:
                    raise ValueError("anomaly: invalid direction")
                item["direction"] = direction
        elif template == "candidate":
            component, reason = finding.get("component"), finding.get("reason")
            if component is not None and component not in spec["candidate_components"]:
                raise ValueError("candidate: component must use a public candidate or null")
            if reason is not None and reason not in spec["candidate_reasons"]:
                raise ValueError("candidate: reason must use a public reason or null")
            support = _evidence_ids(finding.get("support_ids", []), spec["evidence_options"])
            counter = _evidence_ids(finding.get("counter_ids", []), spec["evidence_options"])
            if set(support) & set(counter):
                raise ValueError("candidate: support and counter must differ")
            time = finding.get("time")
            if time is not None and time not in spec["time_candidates"]:
                raise ValueError("candidate: time must come from time_candidates or be null")
            if status == "supported":
                if component is None or reason is None or not support:
                    raise ValueError("candidate: supported needs component, reason and support_ids")
            elif status == "no_candidate":
                if component is not None or reason is not None or support or counter or time is not None:
                    raise ValueError("candidate: no_candidate must be null with no support/time")
            else:  # unresolved: a candidate may be named, but no claim of proof
                if time is not None:
                    raise ValueError("candidate: unresolved must leave time null")
            item.update(component=component, reason=reason, time=time,
                        support_ids=support, counter_ids=counter)
        else:  # coverage
            missing = finding.get("missing", [])
            if not isinstance(missing, list) or any(item not in spec["missing"] for item in missing):
                raise ValueError("coverage: missing must come from the program list")
            cannot = finding.get("cannot_conclude")
            if not isinstance(cannot, str) or not 1 <= len(cannot) <= 120:
                raise ValueError("coverage: cannot_conclude must be 1..120 characters")
            item.update(missing=missing, cannot_conclude=cannot)
        normalized.append(item)
    if seen != set(specs):
        raise ValueError("not every applicable template was answered")
    return {"findings": normalized, "unexpected": unexpected}


def _selected_anomaly(context, evidence_id):
    for template in context["templates"]:
        if template["template"] == "anomaly":
            for item in template["anomalies"]:
                if item["evidence"][0] == evidence_id:
                    return item
    return None


def _selected_log(context, evidence_id):
    for template in context["templates"]:
        if template["template"] == "log":
            for item in template["logs"]:
                if item["id"] == evidence_id:
                    return item
    return None


def _selected_call(context, evidence_id):
    for template in context["templates"]:
        if template["template"] == "calls":
            for item in template["calls"]:
                if item["id"] == evidence_id:
                    return item
    return None


def finding_text(context, finding):
    node = context["node_id"].split(":", 1)[-1]
    template = finding["template"]
    if template == "anomaly":
        item = _selected_anomaly(context, finding.get("key_evidence_id"))
        if item is None:
            return f"{node} 异常模板：未选择有效观测；模型状态{finding['status']}；{finding['note']}"
        return (f"{node} 异常：{item['signal']} 当前{_fmt(item['value'])}{_unit(item.get('unit'))}，"
                f"基线{_fmt(item.get('baseline'))}{_unit(item.get('unit'))}，时间{item.get('time') or '未知'}，"
                f"规则状态{item.get('state')}；程序证据{finding.get('key_evidence_id') or '无'}。"
                f"模型判断（未核实）：{finding['note']}")
    if template == "candidate":
        return (f"{node} 候选（{finding['status']}）：{finding.get('component') or '未知'}"
                f" / {finding.get('reason') or '未知'}；时间{finding.get('time') or '未知'}；"
                f"支持{finding.get('support_ids') or []}；反证{finding.get('counter_ids') or []}；"
                f"缺口/判断（未核实）：{finding['note']}")
    if template == "log":
        item = _selected_log(context, finding.get("key_evidence_id"))
        if item is None:
            return f"{node} 日志模板：未选择有效日志；模型状态{finding['status']}；{finding['note']}"
        return (f"{node} 日志：{item.get('category') or item.get('kind')} 共{item.get('count')}条，"
                f"{item.get('first')}—{item.get('last')}，明确错误{item.get('explicit_error')}；"
                f"模型判断（未核实）：{finding['note']}")
    if template == "calls":
        item = _selected_call(context, finding.get("key_evidence_id"))
        if item is None:
            return f"{node} 调用模板：未选择有效调用；模型状态{finding['status']}；{finding['note']}"
        return (f"{node} 调用：{item.get('kind')} 调用数{item.get('count') or item.get('calls')}，"
                f"P95={item.get('p95')}，最大={item.get('max')}；模型判断（未核实）：{finding['note']}")
    spec = next((item for item in context["templates"] if item["template"] == "coverage"), {})
    return (f"{node} 覆盖：检查{spec.get('checked') or []}；缺失{spec.get('missing') or []}；"
            f"未触发告警{finding.get('status') == 'no_finding'}；不能据此说正常；"
            f"模型结论（未核实）：{finding.get('cannot_conclude')}；{finding['note']}")


def build_report(context, reply):
    findings = reply["findings"]
    facts, groups, catalog = [], [], {}
    for index, finding in enumerate(findings, start=1):
        identifier = f"n{index}"
        text = finding_text(context, finding)
        time_marks = []
        if finding["template"] == "candidate" and finding.get("time"):
            time_marks.append(finding["time"])
        if finding["template"] == "anomaly":
            item = _selected_anomaly(context, finding.get("key_evidence_id"))
            if item and item.get("time"):
                time_marks.append(item["time"])
        fact = {"id": identifier, "tool_fact": text, "time_marks": time_marks,
                "template": finding["template"], "finding_status": finding["status"],
                "model_note_unverified": finding["note"]}
        facts.append(fact)
        catalog[identifier] = {"id": identifier, "tool_fact": text,
                               "source_fact_ids": [finding.get("key_evidence_id")] if finding.get("key_evidence_id") else []}
        groups.append({"source_ids": [identifier], "summary": text,
                       "cpu_facts": [{**deepcopy(finding), "id": identifier}]})
    overview_parts = [finding_text(context, finding) for finding in findings
                      if finding["template"] in ("anomaly", "candidate", "coverage")]
    overview = "；".join(overview_parts)
    if len(overview) > 700:
        overview = overview[:697] + "…"
    if reply.get("unexpected"):
        overview += "；意外线索（未核实）：" + reply["unexpected"]
    return {
        "case_id": context.get("case_id"), "node_id": context.get("node_id"),
        "window_start_ms": context.get("window_start_ms"), "cutoff_ms": context.get("cutoff_ms"),
        "scope": deepcopy(context.get("scope", {})),
        "findings": deepcopy(findings), "finding_facts": facts,
        "groups": groups, "tool_card_catalog": catalog,
        "retain_local": [], "defer": [], "questions": [],
        "overview_text": overview, "template_mode": True,
        "template_failure": None, "unexpected_unverified": reply.get("unexpected"),
        "cpu_only_metric_ids": [], "prose_semantics_verified": False,
    }


def failure_report(packet, error):
    context = build_context(packet)
    finding = {"template": "coverage", "status": "insufficient",
               "missing": next((t["missing"] for t in context["templates"]
                                if t["template"] == "coverage"), []),
               "cannot_conclude": "节点填表失败，不能用该节点作结论",
               "note": f"填表未通过校验：{str(error)[:150]}"}
    reply = {"findings": [finding], "unexpected": None}
    report = build_report(context, reply)
    report["template_failure"] = str(error)
    report["findings"] = []
    report["finding_facts"] = []
    report["groups"] = []
    report["tool_card_catalog"] = {}
    return report


def run_node_template(model, packet, output, max_new_tokens=900):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context = build_context(packet)
    write_json(output / "template_input.json", context)
    base = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False, separators=(",", ":"))}]
    prompt = deepcopy(base)
    attempts = []
    for attempt in range(2):
        try:
            raw = model.invoke(prompt, None, max_new_tokens,
                               f"template/{packet.get('request_id', packet.get('node_id'))}/{attempt}")
            reply = validate_findings(decode_object(raw), context)
        except (ValueError, KeyError, TypeError) as error:
            attempts.append({"valid": False, "error": str(error)})
            prompt = deepcopy(base)
            prompt[-1]["content"] += "\n请修正：" + str(error)
            continue
        attempts.append({"valid": True})
        report = build_report(context, reply)
        report["attempts"] = attempts
        write_json(output / "template_answer.json", {"reply": reply, "attempts": attempts})
        write_json(output / "report.json", report)
        return report
    report = failure_report(packet, attempts[-1]["error"] if attempts else "unknown error")
    report["attempts"] = attempts
    write_json(output / "template_answer.json", {"reply": None, "attempts": attempts})
    write_json(output / "report.json", report)
    return report
