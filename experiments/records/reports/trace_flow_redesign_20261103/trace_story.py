"""Build an auditable, ordered walkthrough from saved model calls and replies."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def calls_at(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def payload_from_call(call: dict) -> dict:
    for message in call.get("messages", []):
        if message.get("role") == "user":
            try:
                value = json.JSONDecoder().raw_decode(message.get("content", ""))[0]
                if isinstance(value, dict):
                    return value
            except (ValueError, TypeError):
                continue
    return {}


def visible_evidence(call: dict, binding: dict) -> list[dict]:
    """Only literal user-payload facts or literal final-system excerpts are visible."""
    facts = {}
    for item in payload_from_call(call).get("evidence", []):
        if isinstance(item, list) and len(item) >= 4:
            identifier, node, fact, times = item[:4]
        elif isinstance(item, dict):
            identifier, node, fact, times = (item.get(k) for k in ("id", "node_id", "tool_fact", "time_marks"))
        else:
            continue
        origin = binding.get(identifier, {})
        source = origin.get("source_id", "")
        facts[identifier] = {"id": identifier, "node": str(node or "?").split(":")[-1],
                             "fact": fact, "time_marks": times or [], "shown_in": "user",
                             "source_id": source, "origin": origin.get("origin", "unknown"),
                             "row_id": source.split("/")[-1] if origin.get("origin") == "node_template" else None}
    # The final prompt may also show evidence excerpts in its system message.
    for message in call.get("messages", []):
        text = message.get("content", "")
        if message.get("role") != "system" or "再输出：" not in text:
            continue
        try:
            candidates = json.JSONDecoder().raw_decode(text.split("再输出：", 1)[1])[0]
        except (ValueError, TypeError):
            continue
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            for brief in candidate.get("evidence_brief", []):
                identifier = brief.get("id")
                if not identifier:
                    continue
                if identifier in facts:
                    facts[identifier].setdefault("system_briefs", []).append({
                        "fact": brief.get("fact", ""), "time_marks": brief.get("times", [])})
                    continue
                origin = binding.get(identifier, {})
                source = origin.get("source_id", "")
                facts[identifier] = {"id": identifier, "node": str(origin.get("node_id") or "?").split(":")[-1],
                                     "fact": brief.get("fact", ""), "time_marks": brief.get("times", []),
                                     "shown_in": "system_brief", "source_id": source,
                                     "origin": origin.get("origin", "unknown"),
                                     "row_id": source.split("/")[-1] if origin.get("origin") == "node_template" else None}
    return list(facts.values())


def snapshot(case: Path, call: dict, saved: dict, key: str) -> dict:
    payload = payload_from_call(call)
    return {"key": key, "phase": call.get("phase"), "input_tokens": call.get("input_tokens"),
            "output_tokens": call.get("output_tokens"), "seconds": call.get("seconds"),
            "raw_reply": call.get("raw_reply", ""), "input": payload,
            "evidence": visible_evidence(call, saved.get("binding", {})),
            "archive_count": len(saved.get("view", {}).get("evidence", [])),
            "omissions": payload.get("context_omissions", {}),
            "source_file": "coordinator_1800/calls.jsonl", "call_number": call.get("call")}


def metric_series(case: Path, nodes: list[dict]) -> None:
    """Find original samples by exact component and metric; do not invent lineage."""
    wanted = defaultdict(set)
    for node in nodes:
        eligible = [r for r in node["template_rows"] if r.get("template") == "metric_change"]
        eligible.sort(key=lambda r: (r.get("use") != "report", not r.get("protected"), not r.get("expanded")))
        for row in eligible[:4]:
            wanted["device:" + node["node"]].add(row["subject"])
    path = case / "frames/inbox.sqlite"
    if not path.exists():
        return
    series, totals = defaultdict(list), Counter()
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        for node, stamp, text in connection.execute(
                "SELECT node_id,observed_at_ms,payload FROM events WHERE modality='metric' ORDER BY observed_at_ms"):
            if node not in wanted:
                continue
            event = json.loads(text)
            raw = event.get("raw", {})
            metric = raw.get("kpi_name")
            if metric not in wanted[node]:
                continue
            key = (node.split(":")[-1], metric)
            totals[key] += 1
            if len(series[key]) >= 180:
                continue
            try:
                value = float(raw["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            series[key].append({"observed_ms": stamp, "value": value, "raw": raw,
                                "event_id": event.get("event_id"), "source_file": event.get("source_file"),
                                "source_row": event.get("source_row")})
    finally:
        connection.close()
    for node in nodes:
        for row in node["template_rows"]:
            key = (node["node"], row["subject"])
            if key in series:
                row["series"] = series[key]
                row["series_total"] = totals[key]


def enrich_trace(case: Path, data: dict) -> dict:
    base = case / "coordinator_1800"
    node_journals = defaultdict(list)
    for call in calls_at(case / "nodes_1800/calls.jsonl"):
        payload = payload_from_call(call)
        node = str(payload.get("node", "")).split(":")[-1]
        node_journals[node].append({"phase": call.get("phase"), "input": payload,
                                    "raw_reply": call.get("raw_reply", ""), "seconds": call.get("seconds"),
                                    "source_file": "nodes_1800/calls.jsonl", "call_number": call.get("call")})
    for node in data["nodes"]:
        node["call_log"] = node_journals[node["node"]]
        node["calls"] = len(node["call_log"])
        by_request = defaultdict(lambda: {"catalogue": set(), "expanded": set()})
        for call in node["call_log"]:
            parts = (call.get("phase") or "").split("/")
            if len(parts) < 2:
                continue
            by_request[parts[1]]["catalogue"].update(item["id"] for item in call["input"].get("catalogue", []))
            by_request[parts[1]]["expanded"].update(item["id"] for item in call["input"].get("forms", []))
        node_dir = case / f"nodes_1800/device_{node['node']}"
        manifest = read_json(node_dir / "manifest.json", {})
        step_requests = {}
        for filename in manifest.get("files", []):
            match = re.match(r"step(\d+)/", filename)
            if match:
                answer = read_json(node_dir / filename, {}).get("answer", {})
                step_requests[int(match.group(1))] = answer.get("request_id")
        report = read_json(node_dir / "report.json", {})
        originals = {row["id"]: row for row in report.get("template_rows", [])}
        counts = Counter()
        for row in node["template_rows"]:
            original = originals.get(row["id"], {})
            step = re.match(r"^s(\d+)_(.*)$", row["id"])
            local_id = step.group(2) if step else row["id"]
            request_id = step_requests.get(int(step.group(1))) if step else None
            viewed = by_request.get(request_id) if request_id else None
            row.update(fields=original.get("fields", {}), template=original.get("template"),
                       meaning=original.get("fields", {}).get("meaning"),
                       statement=original.get("filled_statement", ""),
                       selection=original.get("model_selection"),
                       catalogue_seen=local_id in viewed["catalogue"] if viewed is not None else None,
                       expanded=local_id in viewed["expanded"] if viewed is not None else None)
            if row["catalogue_seen"]:
                counts["catalogue_seen"] += 1
            if row["expanded"]:
                counts["expanded"] += 1
            if row.get("selection"):
                counts["filled"] += 1
            if row.get("use") == "report":
                counts["model_reported"] += 1
            if row.get("use") == "uncertain":
                counts["uncertain"] += 1
            if row.get("use") == "background" and row["catalogue_seen"]:
                counts["background"] += 1
            if row["catalogue_seen"] is False:
                counts["not_shown"] += 1
            if row["catalogue_seen"] is None:
                counts["visibility_unknown"] += 1
        node["processing"] = dict(counts)
        # Four-B batch decisions: the call log is the authoritative record of
        # what was shown and what the model chose on each catalogue page.
        batches = []
        for call in node["call_log"]:
            phase = call.get("phase") or ""
            if not phase.startswith("node_catalogue/"):
                continue
            payload = call.get("input") or {}
            reply = call.get("raw_reply") or ""
            text = str(reply).strip()
            # Small models often wrap JSON in a Markdown fence and append
            # end-of-generation tokens. Parse the decision object so the page
            # does not label a valid 4B choice as an invalid call.
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
            text = re.sub(r"\s*```.*$", "", text, flags=re.S)
            text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            try:
                parsed = json.loads(text)
            except (TypeError, json.JSONDecodeError):
                match = re.search(r"\{.*\}", text, flags=re.S)
                try:
                    parsed = json.loads(match.group(0)) if match else None
                except (TypeError, json.JSONDecodeError):
                    parsed = None
            batches.append({
                "page": payload.get("page"), "pages": payload.get("pages"),
                "shown_ids": [item.get("id") for item in payload.get("catalogue", [])],
                "coverage": payload.get("coverage"),
                "raw_reply": reply,
                "parsed": parsed if isinstance(parsed, dict) else None,
                "valid": isinstance(parsed, dict),
            })
        node["batch_pages"] = batches
        node["batch_count"] = len(batches)
        node["processing"]["batch_pages"] = len(batches)
        node["processing"]["detail_fill_skipped"] = True
    metric_series(case, data["nodes"])

    calls = calls_at(base / "calls.jsonl")
    rounds = []
    for decision in data["central"]["decisions"]:
        index = int(decision["round"])
        matching = [call for call in calls if re.match(rf"case/\d+/round{index}/\d+$", call.get("phase", ""))]
        saved = read_json(base / f"input_{index}.json", {})
        snapshots = [snapshot(case, call, saved, f"round-{index}-attempt-{i}") for i, call in enumerate(matching)]
        rounds.append({**decision, "index": index, "snapshots": snapshots,
                       "snapshot": snapshots[-1] if snapshots else None,
                       "file": f"coordinator_1800/decision_{index}.json"})
    last = read_json(base / "last_input.json", {})
    finals = [snapshot(case, call, last, f"final-attempt-{i}") for i, call in enumerate(calls)
              if call.get("phase", "").startswith("final_commit/")]
    executed = []
    for directory in sorted(base.glob("node_followup_*")):
        index = int(directory.name.split("_")[-1])
        manifest = read_json(directory / "manifest.json", {})
        answers = [read_json(directory / file, {}).get("answer", {}) for file in manifest.get("files", [])]
        if not answers:
            continue
        request_id = answers[0].get("request_id")
        query = read_json(base / "queries" / str(request_id) / "request.json", {})
        packet = read_json(base / "queries" / str(request_id) / "packet.json", {})
        next_round = next((r for r in rounds if r["index"] == index + 1), None)
        next_evidence = (next_round.get("snapshot") or {}).get("evidence", []) if next_round else []
        returned = [e["id"] for e in next_evidence if e.get("source_id", "").startswith(str(request_id) + "/")]
        executed.append({"round": index, "request_id": request_id, "request": query.get("request", packet.get("request", {})),
                         "node": str(answers[0].get("node_id", "?")).split(":")[-1],
                         "answers": answers, "loaded_in_next_round": returned,
                         "tools_executed": packet.get("tools_executed", []),
                         "source_file": f"coordinator_1800/queries/{request_id}/request.json",
                         "answer_file": f"coordinator_1800/{directory.name}/{manifest.get('files', [''])[0]}",
                         "calls": [{"phase": c.get("phase"), "input": payload_from_call(c),
                                    "raw_reply": c.get("raw_reply", ""), "seconds": c.get("seconds")}
                                   for c in calls_at(base / f"node_calls_{index}.jsonl")]})
    events = [{"kind": "raw", "stage": 0, "title": "半小时数据归集", "actor": "程序", "summary": "按组件保存指标、日志与调用，业务数据单独保留。"},
              {"kind": "facts", "stage": 1, "title": "程序计算本地观测", "actor": "固定工具", "summary": "将原始采样整理为统计、事件和变化时间，供节点查看。"},
              {"kind": "nodes", "stage": 2, "title": "4B节点整理与上报", "actor": "各节点4B", "summary": "查看目录、选择展开、填写解释，区分上报、暂留与未处理。"}]
    for item in rounds:
        index = item["index"]
        events.append({"kind": "decision", "stage": 3, "round": index, "title": f"第{index + 1}轮中央判断", "actor": "9B协调者", "summary": "比较本轮实际载入的证据，提出假设和补查请求。"})
        query = next((q for q in executed if q["round"] == index), None)
        if query:
            events.extend([{"kind": "request", "stage": 4, "round": index, "title": f"下发补查 · {query['node']}", "actor": "9B → 节点", "summary": query["request"].get("focus", "")},
                           {"kind": "return", "stage": 4, "round": index, "title": f"补查返回 · {query['node']}", "actor": "节点 → 9B", "summary": "固定工具查询，节点整理返回；追加为独立证据。"}])
        pages = read_json(base / f"page_request_{index}.json", None)
        if pages:
            events.append({"kind": "pages", "stage": 4, "round": index, "title": "加载已有证据页", "actor": "程序", "summary": "按9B请求加载存档详情，不增加一次节点模型调用。", "pages": pages})
    total_facts = sum(n.get("facts", 0) for n in data["nodes"])
    total_pages = sum(n.get("batch_count", 0) for n in data["nodes"])
    total_reported = sum(1 for n in data["nodes"] for r in n.get("template_rows", []) if r.get("use") == "report")
    total_uncertain = sum(1 for n in data["nodes"] for r in n.get("template_rows", []) if r.get("use") == "uncertain")
    total_background = sum(1 for n in data["nodes"] for r in n.get("template_rows", []) if r.get("action") == "background")
    events.insert(2, {"kind": "batch", "stage": 2, "title": "4B分批决策", "actor": "18个4B节点",
                      "actor_type": "model", "summary": f"{total_facts}项节点观测按{total_pages}个目录页全部展示；4B逐页决定上报/待查/背景。",
                      "batches": total_pages, "facts": total_facts})
    events.insert(3, {"kind": "assembly", "stage": 2, "title": "程序装配节点报告", "actor": "程序",
                      "actor_type": "program", "summary": f"程序汇总18份节点报告：{total_reported}项上报、{total_uncertain}项待查、{total_background}项背景；建立短证据ID和候选索引。",
                      "reported": total_reported, "uncertain": total_uncertain, "background": total_background})
    # Rebuild the static node title wording for the new decision-only workflow.
    for event in events:
        if event.get("kind") == "nodes":
            event["summary"] = "4B逐页查看本节点全部观测，决定上报、待查或背景；程序随后装配节点报告。"
    # Global filter stage, if saved.
    gf = data.get("central", {}).get("global_filter") or {}
    if gf:
        before = (gf.get("before") or {}).get("count")
        after = (gf.get("after") or {}).get("count")
        events.insert(4, {"kind": "global_filter", "stage": 3, "title": "9B过滤全局/业务统计", "actor": "9B协调者",
                          "actor_type": "model", "summary": f"程序提供{before if before is not None else '?'}条全局调用/业务事实；9B只做相关性过滤，保留{after if after is not None else '?'}条；不做根因裁决。",
                          "filter": gf, "before": before, "after": after, "fallback": (gf.get("filter") or {}).get("fallback")})
    events.append({"kind": "final", "stage": 5, "title": "冻结答案，再独立评分", "actor": "9B / 独立评价", "summary": "先保存模型答案，评分程序随后读取标准答案。"})
    # Put model/program stages in the order they really happened:
    # raw -> facts -> 4B batches -> node reports -> program assembly -> 9B global filter
    # -> 9B hypotheses/requests -> final.
    extracted = {kind: next((e for e in events if e.get("kind") == kind), None)
                 for kind in ("batch", "assembly", "global_filter")}
    events = [e for e in events if e.get("kind") not in extracted]
    nodes_index = next((i for i, e in enumerate(events) if e.get("kind") == "nodes"), len(events))
    if extracted["batch"]:
        events.insert(nodes_index, extracted["batch"])
        nodes_index += 1
    insert_at = nodes_index + 1
    if extracted["assembly"]:
        events.insert(insert_at, extracted["assembly"])
    decision_index = next((i for i, e in enumerate(events) if e.get("kind") == "decision"), len(events))
    if extracted["global_filter"]:
        events.insert(decision_index, extracted["global_filter"])
    data["story"] = {"events": events, "rounds": rounds, "executed": executed, "final_snapshots": finals,
                     "pending_requests": data["central"].get("complete", {}).get("pending_requests", []),
                     "visibility_source": "actual model messages in calls.jsonl; archived input views are not automatically visible"}
    initial = read_json(base / "input_0.json", {}).get("view", {})
    digest = hashlib.sha256()
    for path in sorted((case / "source/edge_rca").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    data["case"].update(window_start_ms=initial.get("window_start_ms"), cutoff_ms=initial.get("cutoff_ms"),
                         source_sha256=digest.hexdigest() if (case / "source/edge_rca").exists() else None)
    data["flow"].update(catalogue_seen=sum(n["processing"].get("catalogue_seen", 0) for n in data["nodes"]),
                        expanded=sum(n["processing"].get("expanded", 0) for n in data["nodes"]),
                        batch_pages=sum(n.get("batch_count", 0) for n in data["nodes"]),
                        executed_queries=len(executed),
                        initial_visible=len((rounds[0].get("snapshot") or {}).get("evidence", [])) if rounds else None)
    for event in data["story"]["events"]:
        kind = event.get("kind")
        if not event.get("actor_type"):
            event["actor_type"] = (
                "program" if kind in ("raw", "facts", "assembly", "pages") else
                "model_4b" if kind in ("nodes", "batch") else
                "model_9b" if kind in ("decision", "global_filter") else
                "mixed" if kind in ("request", "return", "final") else "program")
    data["schema_version"] = 2
    data["division"] = {
        "program": [
            {"step": "数据归集", "detail": f"{data['raw']['total']} 条原始记录按组件/模态保存，业务流独立保留。"},
            {"step": "事实计算", "detail": f"程序把原始采样整理为 {data['flow']['facts']} 项节点观测。"},
            {"step": "分批目录", "detail": f"程序把全部观测切成 {total_pages} 页，每页最多12项，确保全部展示给4B。"},
            {"step": "报告装配", "detail": f"程序把18份节点报告装配为中央视图；建立短eID、候选索引和全局索引。"},
            {"step": "9B输入", "detail": "程序把节点摘要、可引用证据、程序全局统计和过滤结果装入9B输入；不替9B裁决。"},
            {"step": "评分", "detail": "答案冻结后，独立评分程序读取标准答案；标签不进入模型输入。"},
        ],
        "model": [
            {"step": "4B分批决策", "detail": f"4B逐页决定上报/待查/背景；本案例 {total_reported} 项上报、{total_uncertain} 项待查、{total_background} 项背景。"},
            {"step": "9B全局过滤", "detail": "9B只判断哪些程序全局调用/业务事实与本故障相关，不输出根因。"},
            {"step": "9B假设与补查", "detail": "9B比较节点材料，提出竞争假设和有限补查请求。"},
            {"step": "9B最终裁决", "detail": "9B在冻结证据上输出最终故障解释；程序随后独立评分。"},
        ],
        "handoff": {
            "nodes": len(data["nodes"]),
            "node_summaries": len(data["central"].get("node_status", [])),
            "global_before": (gf.get("before") or {}).get("count"),
            "global_after": (gf.get("after") or {}).get("count"),
            "central_evidence": data["flow"].get("evidence"),
            "hypotheses": data["flow"].get("hypotheses"),
            "requests": data["flow"].get("requests"),
        },
    }
    return data
