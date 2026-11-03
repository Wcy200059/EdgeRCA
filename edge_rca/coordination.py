"""Coordinator decisions and bounded follow-up requests for the local loop.

Evidence IDs are short in model inputs and bound to full provenance by the
program. Hypothesis assessment is distinct from the name of a candidate cause.
"""

from copy import deepcopy
import json
import math

from .compact_tools import _when
from .coordinator import make_packet, matches_time_mark, occurrence_ms
from .node_probe import decode_object
from .note_audit import review_note

REQUEST_TOOLS = ("list_metrics", "get_metrics", "get_logs", "get_calls")


SYSTEM = """你是Bank集群协调者。边缘小模型整理设备观测，你综合判断，并可请求节点用固定工具补查。
public_task若存在，包含公开题目和题目明确的故障数量。调查必须覆盖题目要求；双故障题需分别寻找两件故障的证据，不能按单故障调查。hypotheses是竞争假设，可以多于最终故障数量；早期证据尚未到达时保留未知，不为凑数编造故障。expected_failures为null表示题目未明确数量，不等于只有一个。不要把两个组件或原因合并进一个字符串。
按顺序处理：先读各节点实际现象与未处理范围；比较多个候选的支持、反证及缺口；再决定是否已有依据，或需要补查。告警数量不是故障数量。没有上报、没有日志、没有观测连接都不等于正常。不能仅凭长调用认定网络故障，普通GC不等于OOM。
分析顺序不要按变化幅度排。先列本时段最早/最晚的异常，再看实际父子调用边、业务影响和同一时段的资源变化；候选必须说明机制和时间先后（如资源高→下游延迟→业务失败），并列出反证。小变化若处在因果链上也要保留；变化大但无时间、边或业务支持，不能当根因。某节点没有展示的观测不等于正常，也不等于无关，需要时定向补查。candidate_coverage按公开候选组件列出可用模态数量，是覆盖索引，不是根因排序；形成候选前逐一检查，不能因为某组件/模态的数字大就忽略其他组件。candidate_slate保留4B明确report/uncertain/defer的本地观测，每组件最多一条，是候选索引不是答案；程序不再用protected/规则覆盖4B的background决定；必须逐项考虑，不能因为不是首屏就忽略。
evidence是已加载的工具事实。node_summaries是节点模型的归纳建议，按basis_ids查回工具事实；它们未经完全核实，不能增加工具事实没有的数值或因果。node_relations_unverified是4B建议一起查看的观测，不代表因果；其中未加载的ID只供按需读取，不能直接引用。report_detail_mode为overview_only时，初始节点报告的明细尚未加载，应先按信息缺口请求工具，不能只凭总览作已证实诊断；这种查询允许basis_ids为空。摘要与工具事实矛盾时保留矛盾，不沿用错误解释。P95不是最大值，总调用数不是慢调用数；未知单位不能换算。
hypotheses列待比较的候选，assessment选supported（有支持）、unresolved（仍待核实）、not_supported（证据不支持）。解释认为不支持时不能填supported；证据不足时不能把候选当作诊断结果。component、reason来自公开候选，也可null。support_ids与counter_ids只引用本次evidence的短id。supported必须有support_ids；not_supported必须有counter_ids；unresolved可以没有引用，此时time和time_evidence_id必须为null，并在explanation写清缺什么、希望补查什么。不要因为证据不完美就删掉所有候选；先保留待核实的候选，再发补查。supported必须有support_ids；not_supported必须有counter_ids；unresolved可以没有引用，此时time和time_evidence_id必须为null，并在explanation写清缺什么、希望补查什么。不要因为证据不完美就删掉所有假设；先保留待核实的候选，再发补查。
time不知道就填null。若填写，必须从某条支持证据的time_marks原样选取，并用time_evidence_id引用它。窗口开始、报告时间不是默认故障时间。时间标记只是观测发生点，也需解释为何与候选相关。
requests是发给节点的补查：先说明缺少什么，以及结果会帮助区分什么可能性，再选择tool_capabilities里实际可回答的工具。node_id选已知节点，start/end使用HH:MM:SS或带毫秒形式，限定在已到达时间窗内；tools只选list_metrics、get_metrics、get_logs、get_calls。指标名不清楚时先list_metrics，其metrics可填名称包含片段如["CPU","MEMORY"]（匹配任一片段，非正则），空列表列全部名称；get_metrics的metrics须选少量完整指标名。如果问题是要确认数值或变化趋势，优先用get_metrics加你已知的完整指标名；一次requests可以同时包含list_metrics和get_metrics；只剩一次补查预算时，先取数值，不要只查名称目录。为使补查聚焦，优先查具体几分钟与少量相关指标；需要整窗仍可请求。focus写清要核实的现象，不把猜测写成指令结论。basis_ids只能引用本次evidence的e编号；node_summaries和总览是模型归纳，不是证据，不能写成node:device:XXX；如果请求只是由总览触发，basis_ids留空。节点不写代码，数据不存在时会如实返回。已有结果能回答的问题不重复查，没有可执行的补查就保留未知，不凑请求。
参考公开bank常用指标名：OSLinux-CPU_CPU_CPUCpuUtil是OS CPU使用率；OSLinux-OSLinux_MEMORY_MEMORY_MEMUsedMemPerc是内存占比。JVM与磁盘指标先依据已知名称，未知不要杜撰。
最多3个requests，按优先级排列。pending_requests是预算内尚未执行的中央请求；pending_node_requests是节点预算未执行完的本地请求。node_summaries汇总每节点已加载/未加载范围和节点模型的待核实概述；evidence_directory只索引协调者已收录的证据，未收录的节点观测需定向补查。哪些观测重要由你按时间、实际连接和机制判断，程序不按变化倍率替你排序。node_questions是节点提出的跨节点或能力缺口，需要你判断是否继续查，不能把问题当成事实。已经完成的查询不重复；followup_status写明工具和节点模型分别完成到哪一步。若请求的指标名不存在，参考available_metric_names中的真实名称示例；metric_catalog_evidence_id指向完整目录，可用read_pages查看。片段相似不代表指标等价，名称目录不等于已经查值；也可用list_metrics按片段查目录。补查返回也可能不支持原假设，必须允许修改判断。previous_hypotheses保留旧候选和解释，但旧support_ids、counter_ids、time已被清空，必须按当前evidence重新核对和引用，不能照抄旧编号或旧时间。previous_hypotheses也是候选账本：上一轮提出、本轮没有反证的候选不能静默删除；要么保留并更新assessment，要么在explanation说明撤回理由。
只输出JSON：顶层hypotheses和requests。hypotheses每项严格包含id、assessment、component、reason、time、time_evidence_id、support_ids、counter_ids、explanation。requests每项严格包含node_id、start、end、tools、metrics、focus、basis_ids。explanation不超过180字，focus不超过120字。没有足够依据时允许保留未知，不凑故障数量；有明确缺口时提出具体补查。
类型示例（仅示范格式，内容须根据实际证据填写）：{"hypotheses":[{"id":"h1","assessment":"unresolved","component":null,"reason":null,"time":null,"time_evidence_id":null,"support_ids":[],"counter_ids":[],"explanation":"尚无节点明细；说明缺什么以及需要补查什么"}],"requests":[]}。
id必须是字符串如"h1"；证据引用必须原样使用"e12"，不能写成12或"12"。unresolved可以没有引用，此时time必须为null；解释提到哪个e编号就引用对应编号。请求start/end只写时分秒，如"18:02:30"，不能写日期或时区，可参照request_window。"""


def event_times(fact, start_ms, cutoff_ms):
    candidates = [t for t, _ in fact.get("timeline", [])]
    for sample in fact.get("slow_calls", []):
        candidates += [sample["start"], sample["end"]]
    candidates += [fact[k] for k in ("first", "last") if fact.get(k)]
    candidates += [_when(item["time_ms"]) for item in fact.get("extremes", [])
                   if isinstance(item.get("time_ms"), (int, float))
                   and not isinstance(item["time_ms"], bool)
                   and start_ms <= item["time_ms"] < cutoff_ms]
    result = []
    for text in candidates:
        try:
            occurrence_ms(text, start_ms, cutoff_ms)
        except ValueError:
            continue
        if text not in result:
            result.append(text)
    return result


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _fmt(value):
    return "未知" if value is None else f"{value:.6g}" if isinstance(value, (int, float)) else str(value)


def key_numbers(report, limit=2):
    """Program-selected numeric leads from the node's own facts.

    The 4B overview can omit the decisive number. These leads stay visibly
    derived from tool facts, but they are not citable evidence: the coordinator
    must load the corresponding detail through a bounded follow-up.
    """
    groups = [fact for group in report.get("groups", []) for fact in group.get("cpu_facts", [])]
    metric_by_name = {fact["metric"]: fact for fact in groups
                      if "metric" in fact and _number((fact.get("current") or {}).get("median"))}
    candidates = []
    for fact in groups:
        kind = fact.get("kind")
        if kind == "alarm_history":
            first = fact.get("first_trigger_values") or {}
            value, baseline = first.get("value"), first.get("baseline_median")
            if not _number(value):
                continue
            timeline = fact.get("timeline") or []
            when = timeline[0][0] if timeline else "未知"
            signal = fact.get("signal", "信号")
            baseline_label = "此前中位数"
            metric = metric_by_name.get(signal)
            if not _number(baseline) and metric:
                # The alarm may not carry its own baseline; the same node's
                # whole-window metric card usually does. Use it as a lead.
                baseline = metric["current"]["median"]
                baseline_label = "整窗中位数"
            unit = first.get("unit") or (metric.get("unit") if metric else None)
            if unit is None and any(word in signal for word in ("Perc", "Percent", "CpuUtil")):
                unit = "%"
            severity = 100.0 if unit == "%" and value >= 80 else 0.0
            if _number(baseline) and abs(baseline) > 1e-9:
                score = severity + min(abs(value - baseline) / abs(baseline), 10.0)
            else:
                score = severity + (1.0 if _number(baseline) else 0.5)
            detail = f"当前{_fmt(value)}" + ("%" if unit == "%" else "")
            if _number(baseline):
                detail += f"，{baseline_label}{_fmt(baseline)}" + ("%" if unit == "%" else "")
            else:
                detail += "，缺少对照"
            candidates.append((score, f"{signal}首次触发{when}：{detail}"))
        elif "metric" in fact:
            current = fact.get("current") or {}
            median = current.get("median")
            extremes = [e for e in fact.get("extremes", []) if _number(e.get("value"))]
            if not _number(median) or not extremes:
                continue
            extreme = max(extremes, key=lambda e: abs(e["value"] - median))
            severity = 100.0 if fact.get("unit") == "%" and extreme["value"] >= 80 else 0.0
            if abs(median) > 1e-9:
                score = severity + min(abs(extreme["value"] - median) / abs(median), 10.0)
            else:
                score = severity + 1.0
            candidates.append((score, f"{fact['metric']}极值{_when(extreme.get('time_ms'))}="
                                      f"{_fmt(extreme['value'])}，整窗中位数{_fmt(median)}"))
    result, seen = [], set()
    for _, text in sorted(candidates, key=lambda item: item[0], reverse=True):
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def candidate_coverage(reports, packet):
    """Per-candidate modality coverage; an index, not a root-cause ranking."""
    statuses = {item["node_id"]: item["status"] for item in packet.get("node_status", [])}
    table = []
    for component in packet["system_context"]["root_cause_components"]:
        node = f"device:{component}"
        report = reports.get(node)
        facts = report.get("template_rows", []) if report else []
        counts = {"metric": 0, "log": 0, "call": 0, "alert": 0, "network": 0, "resource": 0, "reported": 0}
        for row in facts:
            template = row.get("template")
            if template == "log_event":
                counts["log"] += 1
            elif template in ("connection", "call"):
                counts["call"] += 1
            elif template in ("rule_notice", "alert"):
                counts["alert"] += 1
            else:
                counts["metric"] += 1
            if (row.get("model_selection") or {}).get("use") == "report":
                counts["reported"] += 1
            text = (str(row.get("subject", "")) + " " + str((row.get("fields") or {}).get("metric", ""))).lower()
            if any(token in text for token in ("network", "net_", "net-", "rx", "tx", "packet", "drop", "latency", "tcp", "udp")):
                counts["network"] += 1
            if any(token in text for token in ("cpu", "mem", "dsk", "disk", "filesystem")):
                counts["resource"] += 1
        table.append({"component": component, "node_id": node,
                      "report_status": statuses.get(node, "not_in_inventory"),
                      "available_fact_count": len(facts), **counts})
    return table


def candidate_slate(reports, source_to_short, cutoff_ms, candidates=None, limit=14):
    """Model-flagged/protected local observations, one per candidate component.

    This is not a root-cause list and does not rank by magnitude. It preserves
    what the 4B actually asked to report or marked uncertain, so the 9B does
    not have to rediscover it from a 580-row archive.
    """
    priority = {"report": 0, "uncertain": 1, "defer": 2, "unprocessed": 3, "background": 4}
    allowed = {f"device:{item}" for item in candidates} if candidates is not None else None
    slate = []
    for node, report in reports.items():
        if allowed is not None and node not in allowed:
            continue
        rows = report.get("template_rows", [])
        eligible = [r for r in rows if
                    (r.get("model_selection") or {}).get("use") in ("report", "uncertain")
                    or r.get("catalogue_action") == "defer"]
        if not eligible:
            continue
        eligible.sort(key=lambda r: (priority.get((r.get("model_selection") or {}).get("use"), 9),
                                     not r.get("protected"), r.get("subject", "")))
        row = eligible[0]
        source = f"{node}@{report.get('cutoff_ms', cutoff_ms)}/{row['id']}"
        short = source_to_short.get(source)
        if not short:
            continue
        slate.append({"component": node.split(":")[-1], "node": node,
                      "subject": row.get("subject"), "pattern": row.get("pattern"),
                      "use": (row.get("model_selection") or {}).get("use"),
                      "protected": bool(row.get("protected")),
                      "evidence_id": short, "fact": json.dumps(row.get("fields", {}), ensure_ascii=False)[:260]})
        if len(slate) >= limit:
            break
    return slate


def make_view(inbox, case_id, cutoff_ms, nodes, supplements=(), previous=(), pending=(), node_pending=(), *, include_report_details=True):
    packet = make_packet(inbox, case_id, cutoff_ms, nodes)
    raw, latest_reports = {}, {}
    for report in inbox.history:
        if report["case_id"] != case_id or report["cutoff_ms"] > cutoff_ms:
            continue
        node = report["node_id"]
        if node not in latest_reports or latest_reports[node]["cutoff_ms"] < report["cutoff_ms"]:
            latest_reports[node] = report
        for group in report["groups"]:
            for fact in group.get("cpu_facts", []):
                raw[f"{report['node_id']}@{report['cutoff_ms']}/{fact['id']}"] = fact
    binding = {}
    evidence = []

    def add(source_id, node, text, times, origin):
        short = f"e{len(evidence) + 1}"
        binding[short] = {"source_id": source_id, "node_id": node, "origin": origin}
        evidence.append({"id": short, "node_id": node, "tool_fact": text, "time_marks": times})

    if include_report_details:
        for item in packet["evidence"]:
            add(item["id"], item["node_id"], item["tool_fact"],
                event_times(raw.get(item["id"], {}), packet["window_start_ms"], cutoff_ms), "node_report")
    else:
        # The template path forwards the numerical slots themselves. It does
        # not ask a language model to restate them in a short overview first.
        from .node_templates import board_rows, evidence_fields, evidence_text
        visible_nodes = {s["node_id"] for s in packet["node_status"] if s["status"] == "received"}
        for node, report in latest_reports.items():
            if report.get("node_mode") not in ("template", "catalogue") or node not in visible_nodes:
                continue
            rows = report.get("template_rows", []) if report.get("node_mode") == "catalogue" else board_rows(report)
            if report.get("node_mode") == "catalogue":
                # Keep targets of a reported/uncertain 4B relation addressable,
                # even if the target itself was marked background. They are
                # not automatically loaded and a relation is not evidence.
                relation_targets = {target for row in rows
                    if (row.get("model_selection") or {}).get("use") in ("report", "uncertain")
                    for target in (row.get("model_selection") or {}).get("related_ids", [])}
                # The 9B sees only what the 4B put on the table. Program
                # protected/rule flags no longer override a 4B background
                # decision; they remain visible in the node report and can be
                # raised as a 4B recheck, not silently injected into 9B.
                rows = [row for row in rows if
                        (row.get("model_selection") or {}).get("use") in ("report", "uncertain")
                        or row.get("catalogue_action") == "defer"
                        or row["id"] in relation_targets]
            for row in rows:
                source = f"{node}@{report['cutoff_ms']}/{row['id']}"
                add(source, node, evidence_text(row),
                    event_times(row["raw_fact"], packet["window_start_ms"], cutoff_ms), "node_template")
                evidence[-1]["template_payload"] = {"subject": row["subject"], "pattern": row["pattern"],
                                                    "fields": evidence_fields(row), "protected": row["protected"]}
                if report.get("node_mode") == "catalogue":
                    evidence[-1]["detail_fields"] = deepcopy(row["fields"])
                    evidence[-1]["initial_display"] = ((row.get("model_selection") or {}).get("use") in ("report", "uncertain")
                        or row.get("catalogue_action") == "defer")
                    evidence[-1]["protected"] = row["protected"]
    source_to_short = {entry["source_id"]: key for key, entry in binding.items()}
    summaries = [{"node_id": group["node_id"], "origin": "node_report",
                  "priority": group.get("priority", "routine"),
                  "basis_ids": [source_to_short[i] for i in group["source_ids"] if i in source_to_short],
                  "available_fact_count": len(group["source_ids"]),
                  "details_loaded": include_report_details,
                  "text_unverified": group["node_note_unverified"], "note_warnings": group["note_warnings"]}
                 for group in packet["node_groups"]]
    if not include_report_details:
        # Case workers provide one hierarchical overview per node. Keep the
        # complete groups in the inbox; do not concatenate every page into 9B.
        compact = []
        for status in packet["node_status"]:
            if status["status"] != "received":
                continue
            report = latest_reports[status["node_id"]]
            own = [s for s in summaries if s["node_id"] == status["node_id"]]
            if report.get("node_mode") in ("template", "catalogue"):
                rows = report.get("template_rows", [])
                namespace = f"{report['node_id']}@{report['cutoff_ms']}/"
                relations = []
                interpretations = []
                for row in rows:
                    origin = source_to_short.get(namespace + row["id"])
                    pick = row.get("model_selection") or {}
                    if origin and pick.get("reading"):
                        from .node_templates import READINGS
                        interpretations.append({"basis_id": origin, "reading": READINGS[pick["reading"]],
                                                "use": pick["use"]})
                        if "interpretation" in pick:
                            interpretations[-1]["interpretation"] = deepcopy(pick["interpretation"])
                    targets = [source_to_short[namespace + i] for i in (row.get("model_selection") or {}).get("related_ids", [])
                               if namespace + i in source_to_short]
                    if origin and targets:
                        relations.append({"from_id": origin, "related_ids": targets})
                compact.append({"node_id": status["node_id"], "origin": "node_template",
                                "basis_ids": [e["id"] for e in evidence if e["node_id"] == status["node_id"]],
                                "details_loaded": True,
                                "text_unverified": report["overview_text"],
                                "note": "数值表是程序事实；模型只选择相关性。未传入的观测仍可按指标、日志、调用工具补查。",
                                "available_fact_count": len(rows),
                                "loaded_fact_count": len([e for e in evidence if e["node_id"] == status["node_id"]]),
                                "unloaded_fact_count": len(rows) - len([e for e in evidence if e["node_id"] == status["node_id"]]),
                                "loading_policy": "只展示变化/错误候选，首屏每节点最多3条；无变化不凑报告。其余完整表留在节点，可补查，不等于正常。",
                                "protected_count": sum(r["protected"] for r in rows),
                                "background_count": sum((r.get("model_selection") or {}).get("use") == "background" for r in rows),
                                "catalogue_unprocessed_count": sum(r.get("catalogue_action") == "unprocessed" for r in rows),
                                "model_unprocessed_count": sum(not r["model_valid"] and r.get("handling") != "retained_by_rule_not_model_checked" for r in rows),
                                "rule_retained_count": sum(r.get("handling") == "retained_by_rule_not_model_checked" for r in rows),
                                "relations_unverified": relations,
                                "filled_interpretations_unverified": interpretations,
                                "note_warnings": []})
                continue
            if "overview_text" not in report:
                compact.extend(own)
                continue
            facts = [f for g in report["groups"] for f in g.get("cpu_facts", [])]
            warnings = review_note(report["overview_text"], facts)
            compact.append({"node_id": status["node_id"], "origin": "node_overview",
                            "basis_ids": [], "details_loaded": False,
                            "available_fact_count": sum(s["available_fact_count"] for s in own),
                            "available_group_count": len(own),
                            "key_numbers": key_numbers(report),
                            "key_numbers_note": "程序从节点事实挑出的最强数值线索；用于决定补查哪个节点，不是可直接引用的证据。",
                            "text_unverified": None if warnings else report["overview_text"], "note_warnings": warnings})
        summaries = compact
    received_nodes = {s["node_id"] for s in packet["node_status"] if s["status"] == "received"}
    node_questions = []
    for node, report in latest_reports.items():
        if node not in received_nodes:
            continue
        questions = ([report["question_overview"]] if not include_report_details and report.get("question_overview")
                     else report.get("questions", []))
        node_questions.extend({"node_id": node, "request_id": None, "question_unverified": q} for q in questions)
    followup_status = []
    updates = {}
    for answer in supplements:
        if answer["case_id"] != case_id or answer["cutoff_ms"] > cutoff_ms:
            raise ValueError("supplement belongs to another case or future cutoff")
        for fact in answer["facts"]:
            add(f"{answer['request_id']}/{fact['id']}", answer["node_id"], fact["tool_fact"],
                fact["time_marks"], "followup")
            for key in ("template_payload", "detail_fields"):
                if key in fact:
                    evidence[-1][key] = deepcopy(fact[key])
            evidence[-1]["observation_scope"] = deepcopy(answer.get("requested_scope"))
            evidence[-1]["origin"] = "followup"
        catalog_id = None
        names = answer.get("available_metric_names", [])
        if answer.get("missing_metrics") and names and not answer.get("metric_catalog_in_facts"):
            # The tool already returned this directory. Keep the complete
            # names addressable without making another node/tool call; the
            # directory contains no numerical observation or causal claim.
            add(f"{answer['request_id']}/__available_metric_names__", answer["node_id"],
                "补查返回的真实指标名称目录；仅供选择名称，不代表已经查值。", [], "metric_catalog")
            evidence[-1]["detail_fields"] = {"available_metric_names": deepcopy(names)}
            evidence[-1]["origin"] = "metric_catalog"
            evidence[-1]["initial_display"] = False
            catalog_id = evidence[-1]["id"]
        source_to_short = {entry["source_id"]: key for key, entry in binding.items()}
        if answer.get("question_response"):
            response = deepcopy(answer["question_response"])
            for key in ("observation_ids", "reported_ids", "unresolved_ids"):
                response[key] = [source_to_short[f"{answer['request_id']}/{i}"] for i in response[key]]
            for item in response["interpretations_unverified"]:
                item["id"] = source_to_short[f"{answer['request_id']}/{item['id']}"]
            response["node_id"] = answer["node_id"]
            key = json.dumps([answer["node_id"], response["question"], response["checked_scope"], response["tools"]], sort_keys=True)
            updates[key] = response
        for group in answer.get("groups", []):
            ids = [f"{answer['request_id']}/{i}" for i in group["observation_ids"]]
            if any(i not in source_to_short for i in ids):
                raise ValueError("node summary cites a fact absent from its supplement")
            summaries.append({"node_id": answer["node_id"], "origin": "followup",
                              "basis_ids": [source_to_short[i] for i in ids],
                              "text_unverified": group["summary"]})
        followup_status.append({k: deepcopy(answer[k]) for k in
            ("request_id", "node_id", "requested_scope", "tools_executed", "node_model_status", "pending_fact_ids", "missing_metrics")})
        # A loaded catalog already carries its names. Do not duplicate hundreds
        # of them again in status; metadata remains in the saved node answer.
        followup_status[-1]["available_metric_names"] = (deepcopy(answer.get("available_metric_names", []))
            if answer["missing_metrics"] and not answer.get("metric_catalog_in_facts") else [])
        if catalog_id:
            followup_status[-1]["metric_catalog_evidence_id"] = catalog_id
        followup_status[-1]["metric_catalog_in_facts"] = answer.get("metric_catalog_in_facts", False)
        followup_status[-1]["local_query_errors"] = deepcopy(answer.get("request_errors", []))
        node_questions.extend({"node_id": answer["node_id"], "request_id": answer["request_id"],
                               "question_unverified": q} for q in answer.get("questions", []))
    # The node's synthesis is useful input, but remains separately attributed.
    # It can never replace facts or become a standalone evidence citation.
    view = {k: deepcopy(packet[k]) for k in
            ("case_id", "window_start_ms", "cutoff_ms", "scope", "system_context", "node_status")}
    pending_nodes = []
    source_to_short = {entry["source_id"]: key for key, entry in binding.items()}
    for task in node_pending:
        request = deepcopy(task["request"])
        full = [task["parent_request_id"] + "/" + i for i in request.get("basis_ids", [])]
        mapped = [source_to_short[i] for i in full if i in source_to_short]
        citation_unavailable = len(mapped) != len(full)
        request["basis_ids"] = mapped
        try:
            validate({"hypotheses": [], "requests": [request]}, {**view, "evidence": evidence})
        except (ValueError, TypeError, KeyError) as error:
            pending_nodes.append({"node_id": task["node_id"], "status": "invalid_request",
                                  "request": request, "error": str(error)})
            continue
        entry = {"status": "not_executed", "request": request}
        if citation_unavailable:
            entry["status"] = "not_executed_citation_unavailable"
            entry["basis_note"] = "原引用不在当前视图，保留查询范围；执行前由程序重新校验。"
        pending_nodes.append(entry)
    view.update(evidence=evidence, node_summaries=summaries, followup_status=followup_status,
                node_questions=node_questions, pending_node_requests=pending_nodes,
                candidate_coverage=candidate_coverage(latest_reports, packet),
                candidate_slate=candidate_slate(latest_reports, source_to_short, cutoff_ms,
                                                packet.get("system_context", {}).get("root_cause_components")),
                report_detail_mode="full_reports" if include_report_details else "overview_only",
                query_policy={"require_named_metrics": not include_report_details},
                previous_hypotheses=deepcopy(list(previous)), pending_requests=deepcopy(list(pending)), labels_used=False)
    if latest_reports and all(r.get("node_mode") in ("template", "catalogue") for r in latest_reports.values()):
        view["report_detail_mode"] = "template_facts"
    if latest_reports and all(r.get("node_mode") == "catalogue" for r in latest_reports.values()):
        view["evidence_memory"] = True
        view["investigation_updates"] = list(updates.values())
        if all(r.get("reply_schema") == "node_slots_v2" for r in latest_reports.values()):
            view["node_reply_schema"] = "node_slots_v2"
    view["request_window"] = {k: packet["scope"][k].split("T")[-1].split("+")[0] for k in ("start", "end")}
    view["tool_capabilities"] = {
        "list_metrics": "只返回实际指标名称；metrics可按名称包含任一片段筛选（忽略大小写、非正则），空列表列全部名称；包括此前最多5分钟参考段，目录不表示已分析或正常。",
        "get_metrics": "返回指定指标的样本数、范围、中位数、此前最多5分钟对照与实际指标名称目录；不返回完整曲线，不能仅凭范围判断趋势或泄漏。",
        "get_logs": "返回指定时段的日志类别、次数、首次/最近时间、错误类型及GC暂停；原文留在节点，未知长文本可能只是片段，不等于全文搜索已覆盖。",
        "get_calls": "返回调用量、P50/P95/最大值、逐分钟统计、长调用样本及实际父子连接；Bank无调用失败字段，不能提供失败率或网络RTT。",
    }
    return view, binding


def validate(reply, view):
    required = {"hypotheses", "requests"}
    optional = {"read_pages"} if view.get("evidence_memory") else set()
    if not isinstance(reply, dict) or not required <= set(reply) or set(reply) - required - optional:
        raise ValueError("reply needs hypotheses and requests")
    if "read_pages" in reply:
        from .evidence_memory import validate_page_requests
        validate_page_requests(reply["read_pages"], view)
    refs = {e["id"]: e for e in view["evidence"]}
    known = {s["node_id"] for s in view["node_status"]}
    context = view["system_context"]

    def cite(ids, required=False, allow_catalog=False):
        if (not isinstance(ids, list) or (required and not ids)
                or any(not isinstance(i, str) or i not in refs for i in ids) or len(ids) != len(set(ids))):
            raise ValueError("use unique e IDs from current evidence; node overviews are not evidence")
        if not allow_catalog and any(refs[i].get("origin") == "metric_catalog" for i in ids):
            raise ValueError("metric catalog is an index, not evidence for or against a hypothesis")

    def text(s, max_chars):
        if not isinstance(s, str) or not s.strip() or len(s) > max_chars:
            raise ValueError("explanation/focus missing or too long")

    if not isinstance(reply["hypotheses"], list):
        raise ValueError("hypotheses must be a list")
    seen = set()
    for h in reply["hypotheses"]:
        if not isinstance(h, dict) or set(h) != {"id", "assessment", "component", "reason", "time", "time_evidence_id", "support_ids", "counter_ids", "explanation"}:
            raise ValueError("invalid hypothesis fields")
        if not isinstance(h["id"], str) or not h["id"] or h["id"] in seen:
            raise ValueError("hypothesis IDs must be unique and nonempty")
        seen.add(h["id"])
        if h["assessment"] not in ("supported", "unresolved", "not_supported"):
            raise ValueError("assessment must be supported, unresolved or not_supported")
        for key, choices in (("component", "root_cause_components"), ("reason", "root_cause_reasons")):
            if h[key] is not None and h[key] not in context[choices]:
                raise ValueError(f"{key} must use a public candidate or null")
        if h["assessment"] == "supported":
            if h["component"] is None or h["reason"] is None:
                raise ValueError("supported requires component and reason")
            if not h["support_ids"]:
                raise ValueError("supported requires supporting evidence")
        elif h["assessment"] == "not_supported":
            if not (h["support_ids"] or h["counter_ids"]):
                raise ValueError("not_supported requires supporting or opposing evidence")
        elif not h["support_ids"] and not h["counter_ids"]:
            # An unresolved candidate may be kept without citations, but then it
            # must not claim an occurrence time. The next step is a bounded query.
            if h["time"] is not None or h["time_evidence_id"] is not None:
                raise ValueError("unresolved without citations needs null time and time_evidence_id")
        cite(h["support_ids"])
        cite(h["counter_ids"])
        if set(h["support_ids"]) & set(h["counter_ids"]):
            raise ValueError("support and counter IDs must differ")
        if h["time"] is None:
            if h["time_evidence_id"] is not None:
                raise ValueError("unknown time needs null time_evidence_id")
        else:
            occurrence_ms(h["time"], view["window_start_ms"], view["cutoff_ms"])
            source = h["time_evidence_id"]
            if source not in h["support_ids"] or not matches_time_mark(
                    h["time"], refs[source]["time_marks"], view["window_start_ms"], view["cutoff_ms"]):
                raise ValueError(f"{h['id']}: time {h['time']} requires {source} in support_ids and an exact time_mark match; use null for time and time_evidence_id if unknown")
        text(h["explanation"], 320)
    if not isinstance(reply["requests"], list) or len(reply["requests"]) > 3:
        raise ValueError("requests must have at most 3 entries")
    for request in reply["requests"]:
        if not isinstance(request, dict) or set(request) != {"node_id", "start", "end", "tools", "metrics", "focus", "basis_ids"}:
            raise ValueError("invalid request fields")
        if request["node_id"] not in known:
            raise ValueError("request target is not a known node")
        start = occurrence_ms(request["start"], view["window_start_ms"], view["cutoff_ms"])
        end = occurrence_ms(request["end"], view["window_start_ms"], view["cutoff_ms"] + 1)
        if start >= end:
            raise ValueError("request range must be nonempty and already arrived")
        tools = request["tools"]
        if (not isinstance(tools, list) or not tools
                or any(not isinstance(t, str) or t not in REQUEST_TOOLS for t in tools)
                or len(tools) != len(set(tools))):
            raise ValueError("only list_metrics, get_metrics, get_logs, get_calls allowed")
        names = request["metrics"]
        if (not isinstance(names, list) or any(not isinstance(n, str) or not n.strip() for n in names)
                or len(names) != len(set(names)) or (names and not {"get_metrics", "list_metrics"}.intersection(tools))):
            raise ValueError("metrics must list unique names and require get_metrics or list_metrics")
        if view.get("query_policy", {}).get("require_named_metrics") and "get_metrics" in tools and not names:
            raise ValueError("select exact metric names for get_metrics; use list_metrics first if unknown")
        if not isinstance(request["basis_ids"], list):
            raise ValueError("basis_ids must be a list")
        # A node overview is a lead, not evidence. Models sometimes put an
        # overview pseudo-ID here; drop that advisory ID so a concrete read-only
        # query is not blocked by bookkeeping. Invented evidence IDs still fail.
        cleaned = []
        for basis in request["basis_ids"]:
            if isinstance(basis, str) and basis in refs:
                cleaned.append(basis)
            elif isinstance(basis, str) and basis.startswith(("node:", "node_summary:", "device:")):
                continue
            else:
                raise ValueError("basis_ids must use e IDs from current evidence")
        request["basis_ids"] = cleaned
        cite(request["basis_ids"], allow_catalog=True)
        text(request["focus"], 200)
    return deepcopy(reply)


def validate_sections(reply, view):
    """Validate read-only requests independently from diagnostic hypotheses.

    A bad occurrence time is not a reason to suppress an otherwise valid data
    request. Rejected sections stay explicit for repair and cannot become facts.
    """
    optional = {"read_pages"} if view.get("evidence_memory") else set()
    if not isinstance(reply, dict) or not {"hypotheses", "requests"} <= set(reply) or set(reply) - {"hypotheses", "requests"} - optional:
        raise ValueError("reply needs hypotheses and requests")
    if not isinstance(reply["hypotheses"], list) or not isinstance(reply["requests"], list) or len(reply["requests"]) > 3:
        raise ValueError("hypotheses/requests need lists, at most three requests")
    accepted, rejected = {"hypotheses": [], "requests": []}, []
    if "read_pages" in reply:
        try:
            from .evidence_memory import validate_page_requests
            validate_page_requests(reply["read_pages"], view)
            accepted["read_pages"] = deepcopy(reply["read_pages"])
        except (ValueError, TypeError, KeyError) as error:
            rejected.append({"section": "read_pages", "index": 0, "error": str(error)})
    for section in ("hypotheses", "requests"):
        for index, item in enumerate(reply[section]):
            candidate = deepcopy(accepted)
            candidate[section].append(item)
            try:
                validate(candidate, view)
            except (ValueError, TypeError, KeyError) as error:
                rejected.append({"section": section, "index": index, "item": deepcopy(item), "error": str(error)})
            else:
                accepted[section].append(deepcopy(item))
    return accepted, rejected


def model_view(view):
    """Pack repeated table keys for template mode without removing fact rows."""
    if view.get("report_detail_mode") != "template_facts":
        return view
    value = deepcopy(view)
    from .template_tables import legend, pack_fact
    by_node = {}
    for fact in value["evidence"]:
        by_node.setdefault(fact["node_id"], []).append([fact["id"], pack_fact(fact), fact["time_marks"]])
    value["evidence"] = by_node
    value["table_schemas"] = legend()
    value["evidence_columns"] = ["可引用的e编号", "工具事实", "实际观测时间（不自动等于故障时间）"]
    columns = ["node_id", "status", "report_through", "lag_seconds", "reported", "retained_local", "deferred", "cpu_only_metrics"]
    value["node_status_columns"] = columns
    value["node_status"] = [[n.get(k) for k in columns] for n in value["node_status"]]
    summaries = []
    for node in value["node_summaries"]:
        if node.get("origin") != "node_template":
            summaries.append(node)
            continue
        summaries.append({"node_id": node["node_id"],
                          "available,loaded,unloaded,model_pending,rule_retained": [node.get(k, 0) for k in (
                              "available_fact_count", "loaded_fact_count", "unloaded_fact_count", "model_unprocessed_count", "rule_retained_count")],
                          "relations_unverified": node["relations_unverified"],
                          "filled_interpretations_unverified": node.get("filled_interpretations_unverified", [])})
    value["node_summaries"] = summaries
    value["template_legend"] = ("evidence按节点分组，行字段见evidence_columns。数值由程序提供，可直接引用；"
        "节点相关性选择未证明因果。只展示变化/错误，首屏每节点最多3条；没有展示不等于正常。其余完整表留在节点，可用指标目录和指标/日志/调用工具补查。"
        "稳定采样不证明正常；内存占比可能含缓存。before为可用此前对照（不保证正常），first5/last5为查询范围前/后最多5分钟，可能重叠。"
        "min/max不证明方向，first_alarm是历史首次告警；extremes是采样时刻。unknown单位不可换算。")
    return value


def model_input(model, view, system, output_tokens, public_question=None):
    """Fit whole fact cards, never truncate a prompt or a numerical row.

    Initial low-priority pages may be unloaded to make room for requested
    details. Protected changes and already cited evidence stay loaded. The
    complete original view stays on disk; unloaded names remain discoverable.
    """
    if view.get("evidence_memory"):
        from .evidence_memory import fit
        return fit(model, view, system, output_tokens, public_question)
    visible = deepcopy(view)
    omitted = []
    cited = set()
    for h in view.get("previous_hypotheses", []):
        cited.update(h.get("support_ids", []) + h.get("counter_ids", []))
    for r in view.get("pending_requests", []):
        cited.update(r.get("basis_ids", []))
    while True:
        payload = model_view(visible)
        if public_question is not None:
            payload = {"public_question": public_question, "observations": payload}
        prompt = [{"role": "system", "content": system},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]
        if (view.get("report_detail_mode") != "template_facts" or not hasattr(model, "count_input_tokens")
                or model.count_input_tokens(prompt) + output_tokens + 700 <= model.max_context):
            return prompt, visible
        removable = [e for e in visible["evidence"] if e.get("template_payload")
                     and not e["template_payload"].get("protected") and e["id"] not in cited]
        if not removable:
            raise ValueError("context budget exhausted by requested/protected evidence; narrow the query; no prompt truncation")
        counts = {}
        for e in removable:
            counts[e["node_id"]] = counts.get(e["node_id"], 0) + 1
        # Remove the lowest-ranked last row from the fullest initial page.
        item = max(reversed(removable), key=lambda e: counts[e["node_id"]])
        visible["evidence"].remove(item)
        omitted.append({"node": item["node_id"], "subject": item["template_payload"]["subject"]})
        visible["context_omissions"] = {"reason": "为已请求的明细和输出预留空间，以下首屏候选暂未加载；仍可定向补查，不能当作正常。",
                                        "observations": omitted}
        for summary in visible["node_summaries"]:
            if summary.get("origin") == "node_template" and summary["node_id"] == item["node_id"]:
                summary["loaded_fact_count"] -= 1
                summary["unloaded_fact_count"] += 1
                summary["basis_ids"] = [i for i in summary["basis_ids"] if i != item["id"]]
                summary["relations_unverified"] = [r for r in summary["relations_unverified"]
                    if r["from_id"] != item["id"] and item["id"] not in r["related_ids"]]
                summary["filled_interpretations_unverified"] = [r for r in summary.get("filled_interpretations_unverified", [])
                                                                if r["basis_id"] != item["id"]]


GLOBAL_FILTER_SYSTEM = """你是9B全局信息过滤器。现在只做一件事：从程序汇总的全局调用/业务观测中，筛出与本故障题相关的观测。
你没有最终裁决权，不要输出根因、故障组件或最终时间。
输入global_evidence是程序从trace/业务表算出的全局事实，每条有e编号；node_summaries是各设备4B节点的摘要，只用于判断相关性。
每个全局ID必须明确保留或拒绝；拒绝要写原因。优先保留：
- 涉及候选组件或可疑组件的调用/业务边；
- 与本时段异常时间接近的调用/业务变化；
- 能区分资源故障、传播链和正常背景的证据。
不要按数值大小排序；不要保留所有调用，也不要因为一条边数值大就认定根因。
只输出JSON：{"ids":["e12","e18"],"notes":"筛选依据；最多200字"}。
ids必须来自global_evidence的真实e编号；找不到相关项时ids为空，并说明原因。"""


def is_global_evidence(item):
    return (item.get("node_id") == "cluster:observations"
            or item.get("origin") in ("global", "business"))


def global_filter(model, view, phase):
    """A model-only relevance filter over program-computed global observations."""
    globals_ = [e for e in view.get("evidence", []) if is_global_evidence(e)]
    if not globals_:
        return {"ids": [], "notes": "no global evidence", "attempts": [], "fallback": False}
    filtered = deepcopy(view)
    filtered["evidence"] = globals_ + [e for e in view.get("evidence", [])
                                      if not is_global_evidence(e)][:0]
    base, packed = model_input(model, filtered, GLOBAL_FILTER_SYSTEM, 900,
                               public_question=view.get("public_task"))
    prompt, attempts = deepcopy(base), []
    ids = [e.get("id") for e in globals_]
    for attempt in range(2):
        raw = None
        try:
            raw = model.invoke(prompt, None, 900, f"{phase}/global_filter/{attempt}")
            value = decode_object(raw)
            if set(value) - {"ids", "notes"}:
                raise ValueError("only ids and notes are allowed")
            picked = value.get("ids")
            if not isinstance(picked, list) or any(not isinstance(i, str) for i in picked):
                raise ValueError("ids must be a list of e IDs")
            if len(picked) != len(set(picked)) or not set(picked) <= set(ids):
                raise ValueError("ids must be unique IDs from global evidence")
            return {"ids": picked, "notes": str(value.get("notes", ""))[:400],
                    "attempts": attempts + [{"valid": True}], "fallback": False,
                    "available": len(globals_)}
        except (ValueError, TypeError, KeyError) as error:
            attempts.append({"valid": False, "error": str(error)})
            prompt = deepcopy(base)
            if raw is not None:
                prompt.append({"role": "assistant", "content": raw})
            prompt.append({"role": "user", "content": "只输出JSON ids/notes；ids必须是global_evidence中的真实e编号且不重复。修正：" + str(error)})
    # No silent drop on failure: keep all global evidence and flag it.
    return {"ids": ids, "notes": "filter invalid; retained all global evidence", "attempts": attempts,
            "fallback": True, "available": len(globals_)}


def apply_global_filter(view, triage):
    if not triage or triage.get("fallback"):
        return view
    keep = set(triage.get("ids", []))
    value = deepcopy(view)
    value["evidence"] = [e for e in value.get("evidence", [])
                         if (not is_global_evidence(e)) or e.get("id") in keep]
    value["global_filter"] = {"ids": sorted(keep), "notes": triage.get("notes", ""),
                              "available": triage.get("available")}
    return value


def ask(model, view, phase, *, allow_partial_requests=False):
    instruction = SYSTEM
    if view.get("node_reply_schema") == "node_slots_v2":
        instruction += ('\n节点详细填写表中的interpretation含meaning、can_answer、cannot_answer、next_check。'
                        '这是节点的待核实意见，不是新增事实。按原问题分别处理已能回答、反对猜测、尚不能回答的部分；'
                        '不能把整份回复只概括成证据不足，也不能把缺日志当作没有故障。'
                        '工具不能完全回答的问题仍保留在假设的缺口或补查中；不把同一份统计说成完整曲线。'
                        '引用和时间仍只能来自当前已加载工具证据，节点解释不能替代它。')
    if view.get("evidence_memory"):
        instruction += ('\n当前使用观测目录与分页。允许顶层额外输出read_pages列表，如["e17:0"]或["directory:1"]。'
                        '目录不是证据，需要的详情优先翻页，不必重复调用原始数据工具。各项时间只能引用本页展示的time_marks。'
                        '补查结果按question/checked_scope/observations/interpretations/limitations组织；区别相关观测与真正回答了问题，'
                        '工具不支持的内容保留未知。候选账本应更新支持、反证和缺口，不复制过去所有措辞。')
    base, view = model_input(model, view, instruction, 2000)
    prompt, attempts = deepcopy(base), []
    for attempt in range(2):
        raw, parsed = None, None
        try:
            raw = model.invoke(prompt, None, 2000, f"{phase}/{attempt}")
            parsed = decode_object(raw)
            reply = validate(parsed, view)
        except (ValueError, KeyError, TypeError) as error:
            entry = {"valid": False, "error": str(error)}
            if parsed is not None:
                try:
                    accepted, rejected = validate_sections(parsed, view)
                except (ValueError, TypeError, KeyError):
                    accepted, rejected = None, []
                entry["rejected_sections"] = rejected
                if allow_partial_requests and accepted and (accepted["requests"] or accepted.get("read_pages")):
                    entry["read_only_requests_usable"] = True
                    attempts.append(entry)
                    return accepted, attempts
            attempts.append(entry)
            prompt = deepcopy(base)
            if raw is not None and view.get("report_detail_mode") != "template_facts":
                prompt.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False) if parsed is not None else raw})
            feedback = ("; ".join(f"{r['section']}[{r['index']}]: {r['error']}" for r in entry.get("rejected_sections", [])) or str(error))[:700]
            prompt.append({"role": "user", "content": "请修订上面的JSON，重新输出完整对象。校验错误：" + feedback
                           + '。同时检查id是"h1"这类字符串，引用是["e12"]而非[12]，请求时间只写HH:MM:SS；未知发生时间填null。不得虚构缺失依据。'})
            continue
        attempts.append({"valid": True, **({"input_budget": view["input_budget"]} if "input_budget" in view else {})})
        return reply, attempts
    return None, attempts
