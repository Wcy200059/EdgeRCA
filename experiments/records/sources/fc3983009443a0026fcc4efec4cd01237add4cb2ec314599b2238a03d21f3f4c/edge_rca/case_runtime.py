"""One Bank case: causal replay, all node workers, coordinator, and final answer.

CPU replay runs in a separate process and publishes immutable prefix frames.
Local 4B and 9B phases share one GPU sequentially. This is accelerated replay
with queued inference, not a claim about real-time latency or physical devices.
No labels or scorer are imported here; case_eval is a separate final process.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import gc
import hashlib
import json
from pathlib import Path
import time

from .alerts import AlertEngine
from .bank_context import public_bank_context
from .coordination import apply_global_filter, ask, global_filter, make_view, model_input
from .coordinator import matches_time_mark, occurrence_ms
from .data import format_time
from .followup import NodeRequestBroker, request_data_key, request_key
from .investigation import FixedSOP
from .node_inquiry import run_node
from .node_probe import LocalModel, decode_object
from .node_reports import ReportInbox, build_snapshot
from .query import parse_failure_count, parse_question_window
from .replay import ReplaySession
from .report_cards import card
from .report_probe import report_pages
from .rolling_memory import RollingNodeMemory
from .trace_index import TraceIndex


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def clock(ms):
    return format_time(ms).split("T")[1].split("+")[0]


def public_question(path, row):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for index, item in enumerate(csv.DictReader(handle)):
            if index == row:
                return item["instruction"]
    raise ValueError("public question row not found")


def global_observations(session, sop, links):
    """CPU business observations and actual links, all bounded by the prefix."""
    facts = []
    for node in session.manifest["nodes"]:
        if not node["node_id"].startswith("business:"):
            continue
        cpu = sop.build(node["node_id"], session.clock_ms, session.start_ms)
        text, marks = [], []
        for obs in cpu["observations"]:
            if "name" not in obs:
                continue
            s = obs["current"]
            label = {"cnt": "每源采样请求数", "sr": "成功率%", "rr": "响应率%", "mrt": "平均响应耗时ms"}[obs["name"]]
            extremes = [{"time": clock(e["time_ms"]), "value": e["value"]} for e in obs["extremes"]]
            text.append(f"{label}：{s['count']}点，中位数{s['median']}，范围{s['min']}—{s['max']}；极值观测{json.dumps(extremes,ensure_ascii=False)}")
            marks += [e["time"] for e in extremes]
        facts.append({"id": f"b{len(facts) + 1}", "tool_fact": node["node_id"] + "；" + "；".join(text)
                      + "。业务级观测，不能强行归属设备；中位数不是总计，极值时刻不是已证实故障时间。",
                      "time_marks": list(dict.fromkeys(marks))})
    for edge in links["links"]:
        s = edge["child_span_latency_ms"]["current"]
        facts.append({"id": f"b{len(facts) + 1}",
                      "tool_fact": f"实际父子ID匹配：{edge['source_node']} → {edge['target_node']}；子调用{s['calls']}次，"
                      f"P95={s['p95']}ms，最大={s['max']}ms；其中{edge['interval_mismatch_count']}对时间区间不满足包含关系。"
                      "调用耗时不是网络RTT；不能相加父子耗时、确定故障传播方向或认为图完整。",
                      "time_marks": []})
    return {"case_id": hashlib.sha256(str(session.prepared_dir).encode()).hexdigest()[:16],
            "node_id": "cluster:observations", "request_id": "global_" + str(session.clock_ms),
            "cutoff_ms": session.clock_ms, "requested_scope": {"start": clock(session.start_ms), "end": clock(session.clock_ms)},
            "tools_executed": ["business_statistics", "observed_connections"], "facts": facts, "groups": [],
            "node_model_status": "cpu_only", "pending_fact_ids": [], "missing_metrics": [],
            "available_metric_names": [], "questions": [], "labels_used": False}


def prepare(prepared, output, query_file, row=0, checkpoints=(600, 1200, 1800), expected_nodes=18):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    instruction = public_question(query_file, row)
    window = parse_question_window(instruction)
    save(output / "public_question.json", {"instruction": instruction, "query_file": str(Path(query_file).resolve()),
                                          "row": row, "labels_used": False})
    started = time.monotonic()
    cursor = 0
    with ReplaySession(prepared, output / "inbox.sqlite") as session, TraceIndex() as graph:
        if tuple(x[:19].replace("T", " ") for x in (format_time(session.start_ms), format_time(session.end_ms))) != window:
            raise ValueError("prepared data does not match the public question window")
        devices = sorted(n["node_id"] for n in session.manifest["nodes"] if n["node_id"].startswith("device:"))
        if expected_nodes and len(devices) != expected_nodes:
            raise ValueError(f"expected {expected_nodes} device nodes, found {len(devices)}")
        points = {session.start_ms + seconds * 1000: seconds for seconds in checkpoints}
        if any(t <= session.start_ms or t > session.end_ms for t in points) or session.end_ms not in points:
            raise ValueError("checkpoints must be visible and include the window end")
        case = hashlib.sha256(str(session.prepared_dir).encode()).hexdigest()[:16]
        memories = {n["node_id"]: RollingNodeMemory(case, n["node_id"], session.start_ms, session.end_ms)
                    for n in session.manifest["nodes"]}
        engine, completed = AlertEngine(session.start_ms), []
        for batch in session.iter_ticks(10000):
            cutoff = batch["clock_ms"]
            graph.ingest(batch["events"], cutoff)
            by_node = defaultdict(list)
            for event in batch["events"]:
                by_node[event["node_id"]].append(event)
                engine.feed(event)
            for node, events in by_node.items():
                memories[node].receive_counts(events, cutoff)
            engine.advance_to(cutoff)
            changes, alerts = defaultdict(list), defaultdict(list)
            for change in engine.transitions[cursor:]:
                changes[change["node_id"]].append(change)
            cursor = len(engine.transitions)
            for alert in engine.alerts.values():
                alerts[alert["node_id"]].append(alert)
            for node, memory in memories.items():
                memory.sync(changes[node], alerts[node], cutoff)
            if cutoff not in points:
                continue
            frame = output / f"c{points[cutoff]}"
            frame.mkdir()
            sop = FixedSOP(session)
            links = graph.snapshot(cutoff, session.start_ms, session.start_ms)
            entries = []
            for node in devices:
                cpu = sop.build(node, cutoff, session.start_ms)
                packet = build_snapshot(memories[node].export(), cpu, links)
                packet.update(case_id=case, request_id=node.replace(":", "_") + "_" + str(cutoff))
                filename = node.replace(":", "_") + ".json"
                save(frame / filename, packet)
                entries.append({"node_id": node, "file": filename, "facts": len(packet["facts"]),
                                "pages": len(report_pages(packet, max_facts=5, max_chars=5000))})
            globals_ = global_observations(session, sop, links)
            save(frame / "global_observations.json", globals_)
            frame_info = {"case_id": case, "window_start_ms": session.start_ms, "cutoff_ms": cutoff,
                          "seconds": points[cutoff], "nodes": entries, "delivered_rows": session.total_delivered,
                          "counts_by_modality": deepcopy(session.delivered_counts), "labels_used": False,
                          "note": "不可变历史前缀；CPU回放不等待GPU，模型后续只读取其任务截止时间。"}
            save(frame / "ready.json", frame_info)
            completed.append(points[cutoff])
            print(json.dumps({"prepared_checkpoint": points[cutoff], "nodes": len(entries),
                              "facts": sum(e["facts"] for e in entries), "rows": session.total_delivered}, ensure_ascii=False), flush=True)
        if set(completed) != set(checkpoints):
            raise RuntimeError("not all causal checkpoints were produced")
        if sum(sum(m.rows.values()) for m in memories.values()) != session.total_delivered:
            raise RuntimeError("raw record counts do not match node inboxes")
        save(output / "complete.json", {"case_id": case, "device_nodes": len(devices), "checkpoints": completed,
             "delivered_rows": session.total_delivered, "seconds": time.monotonic() - started, "labels_used": False})


OVERVIEW_SYSTEM = """你是边缘节点汇总员。输入是本节点分组后的观测说明，可能有措辞错误，不能当成已证实根因。
将所有组组织成一段不超过160字的总览。总览必须写出本节点最强的一个量化变化：对象、时间、当前值与此前中位数或基线；如果最强线索来自告警首次触发值，就写触发值和基线。然后再说明先后关系、反证或未知。不能只写“多次告警”“数值波动”“单位不明”而不给数。
不能只摘最严重一项，也不必重抄所有数值；不把GC说成OOM、慢调用说成网络故障，不能编造正常基线。旧总览仅供保持连续，需要以本次观测修订。不要确定全局根因，不把未汇总或没日志说成正常。只输出JSON：{"summary":"简短总览"}。"""


def node_overview(model, texts, prior, output, prefix):
    # Hierarchical synthesis processes all notes; a long node is not silently
    # truncated to a fixed number of groups.
    groups, current = [], []
    for text in texts or ["模型没有有效分组，详细工具事实和待处理项仍保留。"]:
        if current and sum(len(s) for s in current) + len(text) > 2400:
            groups.append(current)
            current = []
        current.append(text)
    if current:
        groups.append(current)
    notes = []
    for index, part in enumerate(groups):
        prompt = [{"role": "system", "content": OVERVIEW_SYSTEM},
                  {"role": "user", "content": json.dumps({"observations": part, "previous_overview_unverified": prior}, ensure_ascii=False)}]
        reply, attempts = None, []
        for attempt in range(2):
            try:
                value = decode_object(model.invoke(prompt, None, 400, prefix + f"/overview{index}/{attempt}"))
                if set(value) != {"summary"} or not isinstance(value["summary"], str) or not 1 <= len(value["summary"]) <= 220:
                    raise ValueError("return only summary, 1..220 characters")
                reply = value["summary"]
            except (ValueError, TypeError, KeyError) as error:
                attempts.append({"valid": False, "error": str(error)})
                prompt[-1]["content"] += "\n请修正：" + str(error)
                continue
            attempts.append({"valid": True})
            break
        notes.append(reply or "本组未生成合法总览；明细仍可补查，不能认定正常。")
        save(output / f"overview_{index}.json", {"summary": reply, "attempts": attempts, "prose_verified": False})
    if len(notes) > 1:
        merged = output / "overview_merge"
        merged.mkdir()
        return node_overview(model, notes, prior, merged, prefix + "/merge")
    return notes[0]


def assemble_report(seed, local, directory, overview):
    groups, cards, local_ids, deferred, questions = [], {}, [], [], []
    template_rows = []
    rename_by_parent = {}
    for index, filename in enumerate(local["files"]):
        answer_file = directory / filename
        reply = json.loads(answer_file.read_text())["answer"]
        packet = json.loads((answer_file.parent / "input.json").read_text())
        raw = {f["id"]: f for f in packet["facts"]}
        rename = {key: f"s{index}_{key}" for key in raw}
        rename_by_parent[packet["request_id"]] = rename
        for key, fact in raw.items():
            cards[rename[key]] = {"id": rename[key], "tool_fact": card(fact, packet)["tool_fact"]}
        used = set()
        for group in reply["groups"]:
            used.update(group["observation_ids"])
            groups.append({"source_ids": [rename[i] for i in group["observation_ids"]],
                           "summary": group["summary"],
                           "priority": group.get("priority", "routine"),
                           "cpu_facts": [{**deepcopy(raw[i]), "id": rename[i]} for i in group["observation_ids"]]})
        for row in reply.get("template_rows", []):
            value = deepcopy(row)
            value["id"] = rename[row["id"]]
            value["raw_fact"] = {**deepcopy(raw[row["id"]]), "id": value["id"]}
            if value["model_selection"] is not None:
                value["model_selection"]["id"] = value["id"]
                value["model_selection"]["related_ids"] = [rename[i] for i in value["model_selection"]["related_ids"]]
            template_rows.append(value)
        pending = set(reply["pending_fact_ids"])
        deferred += [rename[i] for i in pending]
        local_ids += [rename[i] for i in set(raw) - used - pending]
        questions += [q for q in reply["questions"] if q not in questions]
    # A node can exhaust its local budget. Keep the concrete request in the
    # report, but translate its page-local f IDs to the report's s IDs so the
    # coordinator view can bind the same facts without rerunning the node.
    pending_queries = []
    for task in local["pending_requests"]:
        request = deepcopy(task["request"])
        mapping = rename_by_parent.get(task.get("parent_request_id"), {})
        request["basis_ids"] = [mapping[i] for i in request.get("basis_ids", []) if i in mapping]
        pending_queries.append({"node_id": seed["node_id"],
                                "parent_request_id": f"{seed['node_id']}@{seed['cutoff_ms']}",
                                "request": request})
    return {"case_id": seed["case_id"], "node_id": seed["node_id"], "window_start_ms": seed["window_start_ms"],
            "cutoff_ms": seed["cutoff_ms"], "groups": groups, "tool_card_catalog": cards,
            "retain_local": local_ids, "defer": deferred, "cpu_only_metric_ids": seed.get("cpu_only_metric_ids", []),
            "questions": questions, "overview_text": overview, "scope": deepcopy(seed["scope"]),
            "question_overview": (f"本节点有{len(questions)}条未解问题、{len(local['pending_requests'])}个未完成本地查询；"
                                  "总览已归纳缺口，详细问题保留在节点记录，可定向补查。")
                                  if questions or local["pending_requests"] else None,
            "prose_semantics_verified": False, "pending_self_queries": pending_queries,
            "node_mode": local.get("node_mode", "prose"), "template_rows": template_rows,
            **({"reply_schema": local["reply_schema"]} if "reply_schema" in local else {})}


def nodes(root, prepared, seconds, model_path, max_self_requests=2, node_mode="prose", screen_limit=None,
          detailed=False, decisions_only=False):
    if detailed and node_mode != "catalogue":
        raise ValueError("detailed node replies require catalogue mode")
    root = Path(root)
    frame = root / "frames" / f"c{seconds}"
    manifest = json.loads((frame / "ready.json").read_text())
    output = root / f"nodes_{seconds}"
    output.mkdir(exist_ok=False)
    previous = {}
    for prior in (600, 1200, 1800):
        p = root / f"nodes_{prior}" / "reports.json"
        if prior < seconds and p.exists():
            previous = {r["node_id"]: r for r in json.loads(p.read_text())["reports"]}
    with NodeRequestBroker(prepared, manifest["case_id"], manifest["cutoff_ms"]) as broker:
        model = LocalModel(model_path, output / "calls.jsonl", max_context=12288)
        reports = []
        for index, entry in enumerate(manifest["nodes"]):
            packet = json.loads((frame / entry["file"]).read_text())
            if node_mode in ("template", "catalogue"):
                old = previous.get(entry["node_id"])
                if old and (old["case_id"] != packet["case_id"] or old["cutoff_ms"] >= packet["cutoff_ms"]):
                    raise ValueError("template history must belong to this case and an earlier cutoff")
                packet["previous_template_decisions"] = (old or {}).get("template_rows", [])
            directory = output / entry["node_id"].replace(":", "_")
            local = run_node(model, broker, packet, directory, max_self_requests=max_self_requests,
                             node_mode=node_mode, screen_limit=screen_limit, detailed=detailed,
                             decisions_only=decisions_only)
            texts = []
            template_rows = []
            for filename in local["files"]:
                reply = json.loads((directory / filename).read_text())["answer"]
                texts += [g["summary"] for g in reply["groups"]]
                texts += ["未解问题：" + q for q in reply["questions"]]
                template_rows += reply.get("template_rows", [])
            texts += ["未完成本地补查：" + q["request"]["focus"] for q in local["pending_requests"]]
            if node_mode in ("template", "catalogue"):
                from .node_templates import overview as template_overview
                overview = template_overview(template_rows)
            else:
                overview = node_overview(model, texts, previous.get(entry["node_id"], {}).get("overview_text"),
                                         directory, packet["request_id"])
            report = assemble_report(packet, local, directory, overview)
            save(directory / "report.json", report)
            reports.append(report)
            save(output / "progress.json", {"seconds": seconds, "completed": index + 1, "total": len(manifest["nodes"]),
                 "node": entry["node_id"], "model_calls": len(model.calls), "labels_used": False})
            print(json.dumps({"node_checkpoint": seconds, "completed": index + 1, "total": len(manifest["nodes"]),
                              "node": entry["node_id"], "calls": len(model.calls)}, ensure_ascii=False), flush=True)
        save(output / "reports.json", {"case_id": manifest["case_id"], "cutoff_ms": manifest["cutoff_ms"],
             "reports": reports, "model_calls": len(model.calls), "peak_gpu_gib": model.torch.cuda.max_memory_allocated()/1024**3,
             "labels_used": False, "node_mode": node_mode,
             **({"reply_schema": "node_slots_v2"} if detailed else {})})


def release_model(model):
    torch = model.torch
    del model.model
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.memory_allocated() > 512 * 1024**2:
        raise RuntimeError("model memory was not released before the next model")


FINAL_SYSTEM = """你是Bank集群的最终裁决者。依据当前各节点观测和已加载工具证据，回答给定公开题目。
总览与先前假设是待核实的线索，不能代替证据；节点目录中的观测是否重要由你判断，不能只按数值倍率。选出最有依据的组件和原因，使用公开候选原文。故障数量按公开题目要求，不按告警条数或候选假设条数决定。
如果previous_hypotheses中有多个supported或有证据的候选，输出前必须逐项比较直接证据、时间先后、实际连接、业务影响和反证；不能默认选最先出现、最先支持或最熟悉的候选。资源背景、普通GC、磁盘高变化本身不是根因；只有直接异常和因果链更完整的候选优先。高内存但没有OOM日志不能否定内存证据；缺日志也不能否定已有指标。
不要因为证据不完美就一律null：先比较候选；如果只有间接证据，也给出相对最有依据的候选，引用间接证据，并在explanation说明不确定性。只有所有候选都没有任何线索时才null。业务成功率100%不能单独排除资源型根因。
previous_hypotheses可能包含中间轮次暂时没提的候选；如果它仍有支持证据，最终答案要重新比较，不能因为上一轮没提就忽略。时间未知填null；若填写必须从某条evidence.time_marks原样选一个真实观测时刻，并引用它，不要用窗口开始代替。
只输出JSON：{"faults":[{"component":null,"reason":null,"time":null,"evidence_ids":[],"explanation":"用不超过120字说明该故障的支持、反证或不足"}]}。
faults每项是一件故障的完整对象，同一件故障的组件、原因、时间和引用放在一起；不同故障分别填写，不能把两个组件挤进一个字符串，不能复制同一故障凑数量。未确定的字段填null。evidence_ids只引用当前输入已有的e编号。"""


GLOBAL_LIMITS = ("业务级观测不能强行归属设备；中位数不是总计，极值时刻不是已证实故障时间。"
                 "调用耗时不是网络RTT；不能相加父子耗时、确定故障传播方向或认为图完整。")


def compact_globals(observation):
    value = deepcopy(observation)
    suffixes = ("。业务级观测，不能强行归属设备；中位数不是总计，极值时刻不是已证实故障时间。",
                "调用耗时不是网络RTT；不能相加父子耗时、确定故障传播方向或认为图完整。")
    for fact in value["facts"]:
        for suffix in suffixes:
            fact["tool_fact"] = fact["tool_fact"].removesuffix(suffix)
    return value


def latest_reports(inbox, case, cutoff):
    latest = {}
    for report in inbox.history:
        if report["case_id"] != case or report["cutoff_ms"] > cutoff:
            continue
        node = report["node_id"]
        if node not in latest or latest[node]["cutoff_ms"] < report["cutoff_ms"]:
            latest[node] = report
    return latest


def pending_node_requests(inbox, case, cutoff):
    """Carry a node's budget-exhausted local requests into the coordinator view."""
    tasks, seen = [], set()
    for report in latest_reports(inbox, case, cutoff).values():
        for task in report.get("pending_self_queries", []):
            request = task["request"]
            key = (task["node_id"], request["start"], request["end"],
                   tuple(sorted(request["tools"])), tuple(sorted(request["metrics"])), request["focus"])
            if key in seen:
                continue
            seen.add(key)
            tasks.append(deepcopy(task))
    return tasks


def coordinator_state(root, seconds):
    """Load the latest earlier checkpoint's coordinator memory."""
    for prior in (1800, 1200, 600):
        if prior >= seconds:
            continue
        path = root / f"coordinator_{prior}" / "state.json"
        if path.exists():
            return json.loads(path.read_text())
    return {}


def case_view(inbox, case, cutoff, supplements, previous, pending, node_pending=(), *, instruction=None):
    view, binding = make_view(inbox, case, cutoff, None, supplements, previous, pending, node_pending,
                              include_report_details=False)
    view["scope"]["global_interpretation_limits"] = GLOBAL_LIMITS
    view["scope"]["business_streams"] = "业务观测已独立接入；没有可靠设备映射，不强行归属节点。"
    view["scope"]["scope_note"] = "当前题的可见历史；组件根因由中央比较，不按节点报告数量确定。"
    if instruction is not None:
        view["public_task"] = {"instruction": instruction,
                               "expected_failures": parse_failure_count(instruction)}
    return view, binding


def prioritize_requests(pending, fresh, view, seen=()):
    """Current model priorities precede old work; no unexecuted work is lost.

    This is not FIFO: reselecting an existing request promotes it and replaces
    its outdated focus/citations. New evidence can therefore change the plan.
    """
    chosen, keys = [], set(seen)
    for item in [*fresh, *pending]:
        key = request_key(item, view)
        if key not in keys:
            chosen.append(deepcopy(item))
            keys.add(key)
    return chosen


def merge_requested_pages(previous, new):
    """Keep earlier loaded pages visible; a later page request must not erase them."""
    return list(dict.fromkeys([*previous, *new]))


def merge_hypotheses(previous, current):
    """Keep a candidate ledger across rounds without inventing new support.

    A model may temporarily stop mentioning a correct candidate (as the 142409
    run did with Redis02). Dropping it from the next view silently would lose
    the hypothesis; keep the last version keyed by public component/reason.
    """
    ledger = {(item.get("component"), item.get("reason")): deepcopy(item)
              for item in previous}
    for item in current:
        ledger[(item.get("component"), item.get("reason"))] = deepcopy(item)
    return list(ledger.values())


def coordinate(root, prepared, seconds, coordinator_model, node_model, max_queries=2, final=False,
               node_mode="prose", detailed=False, max_candidates=1):
    if detailed and node_mode != "catalogue":
        raise ValueError("detailed node replies require catalogue mode")
    if isinstance(max_candidates, bool) or max_candidates not in (1, 2, 3):
        raise ValueError("max_candidates must be 1, 2 or 3")
    root = Path(root)
    instruction = json.loads((root / "frames" / "public_question.json").read_text())["instruction"]
    # Validate explicit task constraints before loading a model. No labels read.
    parse_failure_count(instruction)
    output = root / f"coordinator_{seconds}"
    output.mkdir(parents=True, exist_ok=False)
    inbox = ReportInbox()
    for prior in (600, 1200, 1800):
        p = root / f"nodes_{prior}" / "reports.json"
        if prior <= seconds and p.exists():
            for report in json.loads(p.read_text())["reports"]:
                inbox.receive(report)
    cases = {r["case_id"] for r in inbox.history}
    if len(cases) != 1:
        raise ValueError("need one isolated case")
    if {r.get("node_mode", "prose") for r in inbox.history} != {node_mode}:
        raise ValueError("coordinator and node reports must use the same node_mode")
    expected_schema = "node_slots_v2" if detailed else "node_slots_v1"
    if {r.get("reply_schema", "node_slots_v1") for r in inbox.history} != {expected_schema}:
        raise ValueError("coordinator and node reports must use the same reply schema")
    case = cases.pop()
    frame = root / "frames" / f"c{seconds}"
    cutoff = json.loads((frame / "ready.json").read_text())["cutoff_ms"]
    state = coordinator_state(root, seconds)
    prior_supplements = [answer for answer in state.get("supplements", [])
                         if not str(answer.get("request_id", "")).startswith("global_")]
    supplements = prior_supplements + [compact_globals(
        json.loads((frame / "global_observations.json").read_text()))]
    previous = []
    for hypothesis in state.get("previous", []):
        carried = deepcopy(hypothesis)
        # Keep the candidate and explanation, but old short evidence IDs do not
        # bind to the new view. The coordinator must cite current e IDs again.
        carried["support_ids"] = []
        carried["counter_ids"] = []
        carried["time"] = None
        carried["time_evidence_id"] = None
        previous.append(carried)
    pending = []
    for request in state.get("pending", []):
        carried = deepcopy(request)
        # Short evidence IDs are bound to the old view; keep the concrete query
        # but let the new view decide whether it still has an evidence basis.
        carried["basis_ids"] = []
        pending.append(carried)
    executed = deepcopy(state.get("executed", {}))
    node_pending = pending_node_requests(inbox, case, cutoff)
    seen, rounds = set(), []
    requested_pages, seen_pages = [], set()
    executed_now, reused_now = 0, 0
    global_triage = None
    model = LocalModel(coordinator_model, output / "calls.jsonl", max_context=20480)
    try:
        with NodeRequestBroker(prepared, case, cutoff) as broker:
            for index in range(max_queries + 1):
                view, binding = case_view(inbox, case, cutoff, supplements, previous, pending, node_pending,
                                          instruction=instruction)
                if global_triage is None:
                    global_triage = global_filter(model, view, f"case/{seconds}/global")
                    save(output / "global_filter.json", {"case_id": case, "filter": global_triage,
                         "labels_used": False, "stage": "global_relevance_only"})
                view = apply_global_filter(view, global_triage)
                view["requested_pages"] = requested_pages
                save(output / f"input_{index}.json", {"view": view, "binding": binding})
                reply, attempts = ask(model, view, f"case/{seconds}/round{index}", allow_partial_requests=True)
                save(output / f"decision_{index}.json", {"reply": reply, "attempts": attempts, "labels_used": False})
                rounds.append({"round": index, "reply_valid": reply is not None, "evidence": len(view["evidence"])})
                if reply is None:
                    break
                previous = merge_hypotheses(previous, reply["hypotheses"])
                pending = prioritize_requests(pending, reply["requests"], view, seen)
                new_pages = [p for p in reply.get("read_pages", []) if p not in seen_pages]
                if index < max_queries and new_pages:
                    requested_pages = merge_requested_pages(requested_pages, new_pages)
                    seen_pages.update(new_pages)
                    save(output / f"page_request_{index}.json", {"read_pages": new_pages,
                         "note": "下一轮只加载已有观测的所选页，不重跑模型或原始工具；仍占一轮调查预算。"})
                    continue
                if index >= max_queries or not pending:
                    break
                request = pending.pop(0)
                key = request_key(request, view)
                if key in seen:
                    continue
                seen.add(key)
                data_key = request_data_key(request, view)
                if data_key in executed:
                    request_id = executed[data_key]
                    if not any(answer.get("request_id") == request_id for answer in supplements):
                        raise ValueError("coordinator state references a missing supplement")
                    reused_now += 1
                    continue
                packet = broker.execute(request, view, output / "queries" / key)
                save(output / "progress.json", {"phase": "node_followup", "round": index, "node": request["node_id"]})
                release_model(model)
                del model
                worker = LocalModel(node_model, output / f"node_calls_{index}.jsonl", max_context=12288)
                directory = output / f"node_followup_{index}"
                local = run_node(worker, broker, packet, directory, max_self_requests=2,
                                 node_mode=node_mode, detailed=detailed)
                answers = [json.loads((directory / filename).read_text())["answer"] for filename in local["files"]]
                supplements.extend(answers)
                if answers:
                    executed[data_key] = answers[0]["request_id"]
                executed_now += 1
                release_model(worker)
                del worker
                model = LocalModel(coordinator_model, output / "calls.jsonl", max_context=20480)
    finally:
        state = {"case_id": case, "cutoff_ms": cutoff,
                 "supplements": deepcopy(supplements), "previous": deepcopy(previous),
                 "pending": [{**deepcopy(item), "basis_ids": []} for item in pending],
                 "executed": deepcopy(executed)}
        save(output / "state.json", state)
    view, binding = case_view(inbox, case, cutoff, supplements, previous, pending, node_pending,
                              instruction=instruction)
    view = apply_global_filter(view, global_triage)
    view["requested_pages"] = requested_pages
    save(output / "last_input.json", {"view": view, "binding": binding})
    if final:
        final_answer(model, view, instruction, output, max_candidates=max_candidates)
    save(output / "complete.json", {"case_id": case, "cutoff_ms": cutoff, "rounds": rounds,
         "device_nodes_received": sum(n["status"] == "received" for n in view["node_status"]),
         "executed_central_queries": executed_now, "reused_central_queries": reused_now,
         "pending_requests": pending,
         "peak_gpu_gib": model.torch.cuda.max_memory_allocated()/1024**3,
         "final_commit": final, "labels_used": False, "rca_evaluated": False, "node_mode": node_mode,
         "global_filter": global_triage,
         **({"node_reply_schema": "node_slots_v2"} if detailed else {}),
         **({"max_candidates": max_candidates} if max_candidates > 1 else {})})
    print(json.dumps({"coordinator_checkpoint": seconds, "queries": executed_now,
                      "reused": reused_now, "final_commit": final}, ensure_ascii=False), flush=True)


def validate_final_answers(value, view, expected_failures):
    if set(value) != {"faults"} or not isinstance(value["faults"], list) or not value["faults"]:
        raise ValueError("use one object with a nonempty faults list")
    faults = value["faults"]
    if expected_failures is not None and len(faults) != expected_failures:
        raise ValueError(f"public question requires exactly {expected_failures} fault objects")
    refs = {e["id"]: e for e in view["evidence"]}
    seen = set()
    for fault in faults:
        if not isinstance(fault, dict) or set(fault) != {"component", "reason", "time", "evidence_ids", "explanation"}:
            raise ValueError("each fault needs exactly component/reason/time/evidence_ids/explanation")
        for key, choices in (("component", "root_cause_components"), ("reason", "root_cause_reasons")):
            if fault[key] is not None and (not isinstance(fault[key], str)
                                          or fault[key] not in view["system_context"][choices]):
                raise ValueError(f"{key} must be a public candidate or null")
        ids = fault["evidence_ids"]
        if not isinstance(ids, list) or any(not isinstance(i, str) or i not in refs for i in ids):
            raise ValueError("evidence_ids must use current e IDs")
        if any(refs[i].get("origin") == "metric_catalog" for i in ids):
            raise ValueError("metric catalog is an index, not evidence for a diagnosis")
        if any(fault[key] is not None for key in ("component", "reason", "time")) and not ids:
            raise ValueError("each prediction requires its own cited evidence")
        if fault["time"] is not None:
            if not isinstance(fault["time"], str):
                raise ValueError("time must be a cited time_mark or null")
            occurrence_ms(fault["time"], view["window_start_ms"], view["cutoff_ms"])
            if not any(matches_time_mark(fault["time"], refs[i]["time_marks"],
                                        view["window_start_ms"], view["cutoff_ms"]) for i in ids):
                raise ValueError("time must match this fault's cited time_mark, otherwise null")
        if not isinstance(fault["explanation"], str) or not fault["explanation"].strip():
            raise ValueError("each explanation must be a nonempty string")
        identity = tuple(fault[k] for k in ("component", "reason", "time"))
        if any(v is not None for v in identity):
            if identity in seen:
                raise ValueError("do not duplicate a fault to satisfy the requested count")
            seen.add(identity)
    return value


def supported_fault_seeds(view, expected_failures):
    """Turn coordinator-supported hypotheses into final-answer slots.

    This is not a programmatic diagnosis: the component/reason come from the
    model's own hypothesis ledger. It prevents a valid supported candidate from
    disappearing when the final generator emits an all-null placeholder.
    """
    refs = {e["id"]: e for e in view.get("evidence", [])}
    seeds, seen = [], set()
    for hypothesis in view.get("previous_hypotheses", []):
        if hypothesis.get("assessment") != "supported":
            continue
        component, reason = hypothesis.get("component"), hypothesis.get("reason")
        if component is None and reason is None:
            continue
        ids = [i for i in hypothesis.get("support_ids", []) if i in refs]
        times = [hypothesis.get("time")] if hypothesis.get("time") else []
        for identifier in ids:
            times.extend(refs[identifier].get("time_marks", []))
        time = next((t for t in times if any(t in refs[i].get("time_marks", []) for i in ids)), None)
        key = (component, reason, time)
        if key in seen:
            continue
        seen.add(key)
        brief = []
        for identifier in ids[:3]:
            fact = refs[identifier]
            brief.append({"id": identifier, "node": fact.get("node_id"),
                          "fact": str(fact.get("tool_fact", ""))[:320],
                          "times": fact.get("time_marks", [])[:4]})
        seeds.append({"component": component, "reason": reason, "time": time,
                      "evidence_ids": ids[:8], "evidence_brief": brief,
                      "explanation": (hypothesis.get("explanation") or "协调者账本中已支持的候选。")[:220]})
        if len(seeds) >= (expected_failures or 1):
            break
    return seeds


def _seed_fault(seed):
    return {key: deepcopy(seed.get(key)) for key in
            ("component", "reason", "time", "evidence_ids", "explanation")}


def retain_valid_fields_with_unknown_time(value, view, expected_failures):
    """After retries, remove unsupported times without inventing other fields."""
    candidate = deepcopy(value)
    if not isinstance(candidate, dict) or set(candidate) != {"faults"} or not isinstance(candidate["faults"], list):
        raise ValueError("partial recovery requires a faults object")
    refs = {fact["id"]: fact for fact in view["evidence"]}
    repairs = []
    for index, fault in enumerate(candidate["faults"]):
        if not isinstance(fault, dict):
            raise ValueError("partial recovery requires fault objects")
        when = fault.get("time")
        if when is None:
            continue
        if not isinstance(when, str):
            raise ValueError("partial recovery requires a string time or null")
        ids = fault.get("evidence_ids")
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(identifier, str) or identifier not in refs for identifier in ids)):
            raise ValueError("partial recovery cannot invent or replace citations")
        try:
            matched = any(matches_time_mark(
                when, refs[identifier]["time_marks"], view["window_start_ms"], view["cutoff_ms"]) for identifier in ids)
        except (ValueError, TypeError):
            matched = False
        if not matched:
            repairs.append({"fault_index": index, "field": "time", "original_value": when,
                            "new_value": None, "reason": "No matching currently displayed timestamp in this fault's cited evidence."})
            fault["time"] = None
    if not repairs:
        raise ValueError("no unsupported time fields to remove")
    validate_final_answers(candidate, view, expected_failures)
    return candidate, repairs


def register_displayed_briefs(view, candidates, original_refs):
    """Register only source excerpts and times actually rendered in the system prompt.

    The existing final prompt contains candidate evidence excerpts in addition
    to the paged user payload. Packing may evict their user-payload copies;
    that does not erase the literal excerpts still visible to the model.
    No other field, timestamp or historical citation becomes loaded here.
    """
    visible = deepcopy(view)
    refs = {fact["id"]: fact for fact in visible["evidence"]}
    registered = []
    for candidate in candidates:
        for brief in candidate["evidence_brief"]:
            identifier = brief["id"]
            source = original_refs[identifier]
            if identifier not in refs:
                fact = {"id": identifier, "node_id": source.get("node_id"),
                        "tool_fact": brief["fact"], "time_marks": list(brief["times"]),
                        "display_origin": "final_system_evidence_brief"}
                visible["evidence"].append(fact)
                refs[identifier] = fact
            else:
                refs[identifier]["time_marks"] = list(dict.fromkeys(
                    [*refs[identifier]["time_marks"], *brief["times"]]))
            registered.append({"id": identifier, "fact": brief["fact"], "times": brief["times"]})
    visible["displayed_system_briefs"] = registered
    return visible


def final_answer(model, view, instruction, output, *, max_candidates=1):
    if isinstance(max_candidates, bool) or max_candidates not in (1, 2, 3):
        raise ValueError("max_candidates must be 1, 2 or 3")
    count = parse_failure_count(instruction)
    constraint = (f"\n本题明确涉及{count}个故障，faults必须恰有{count}项。缺少某件故障的证据时，保留其位置、未知字段填null并说明不足。"
                  if count is not None else "\n题目未明确故障数量，不默认只有一个；只列出当前有依据的故障，无法确认时返回一项未知并说明不足。")
    output_tokens = 700 * (count or 1) * max_candidates
    if max_candidates > 1:
        from .ranked_answers import INSTRUCTION
        constraint += INSTRUCTION + f"\n本次含首选最多{max_candidates}组候选，最多{max_candidates - 1}组备选。"
    # Do not privilege "supported" slots in the final prompt: the 9B's own
    # earlier assessment can be wrong (row 1 marked Redis02 unresolved while
    # choosing Mysql02). Show every evidence-backed hypothesis equally; the
    # supported seeds are only a fallback for an all-null final answer.
    candidates = []
    refs = {e["id"]: e for e in view.get("evidence", [])}
    for hypothesis in view.get("previous_hypotheses", []):
        if not hypothesis.get("component") and not hypothesis.get("reason"):
            continue
        ids = [i for i in hypothesis.get("support_ids", []) if i in refs][:3]
        brief = [{"id": i, "fact": str(refs[i].get("tool_fact", ""))[:260],
                  "times": refs[i].get("time_marks", [])[:4]} for i in ids]
        candidates.append({"component": hypothesis.get("component"),
                           "reason": hypothesis.get("reason"),
                           "assessment": hypothesis.get("assessment"),
                           "time": hypothesis.get("time"),
                           "evidence_brief": brief,
                           "explanation": (hypothesis.get("explanation") or "")[:220]})
    if candidates:
        constraint += ("\n以下是依据当前证据形成的全部候选，不是答案，也不能按assessment或出现顺序选；"
                       "请逐项比较直接证据、时间、反证和业务/调用影响后再输出："
                       + json.dumps(candidates, ensure_ascii=False))
    archive_view = deepcopy(view)
    base, view = model_input(model, view, FINAL_SYSTEM + constraint, output_tokens, instruction)
    view = register_displayed_briefs(view, candidates, refs)
    seeds = supported_fault_seeds(view, count or 1)
    save(output / "final_input.json", {"view": view, "labels_used": False})
    prompt, attempts, result, alternatives = deepcopy(base), [], None, []
    parsed_replies, field_repairs = [], []
    for attempt in range(2):
        try:
            value = decode_object(model.invoke(prompt, None, output_tokens, f"final_commit/{attempt}"))
            parsed_replies.append(deepcopy(value))
            if max_candidates > 1:
                if not isinstance(value, dict) or set(value) - {"faults", "alternatives"}:
                    raise ValueError("ranked output permits faults and alternatives only")
                result = validate_final_answers({"faults": value["faults"]}, view, count)
                alternatives = value.get("alternatives", [])
            else:
                result = validate_final_answers(value, view, count)
        except (ValueError, KeyError, TypeError) as error:
            attempts.append({"valid": False, "error": str(error)})
            if attempt == 0 and parsed_replies:
                # A stale citation can name a real fact evicted by packing.
                # Load at most three existing pages before the already-budgeted
                # retry. Never fabricate an ID or query a different case/window.
                loaded = {fact["id"] for fact in view["evidence"]}
                parsed = parsed_replies[-1]
                faults = parsed.get("faults", []) if isinstance(parsed, dict) else []
                missing = []
                for fault in faults if isinstance(faults, list) else []:
                    ids = fault.get("evidence_ids", []) if isinstance(fault, dict) else []
                    for identifier in ids if isinstance(ids, list) else []:
                        if (isinstance(identifier, str) and identifier in refs
                                and identifier not in loaded and identifier not in missing):
                            missing.append(identifier)
                if missing:
                    requested = archive_view.get("requested_pages", [])
                    archive_view["requested_pages"] = list(dict.fromkeys(
                        [*requested, *(identifier + ":0" for identifier in missing[:3])]))
                    base, view = model_input(model, archive_view, FINAL_SYSTEM + constraint, output_tokens, instruction)
                    view = register_displayed_briefs(view, candidates, refs)
                    seeds = supported_fault_seeds(view, count or 1)
                    attempts[-1]["requested_existing_evidence"] = missing[:3]
                    attempts[-1]["note"] = "重新加载本题已有证据页；是否支持原结论仍须核对。"
                    save(output / "final_input.json", {"view": view, "labels_used": False})
            prompt = deepcopy(base)
            prompt[-1]["content"] += "\n请修正：" + str(error)
            continue
        attempts.append({"valid": True})
        break
    if result is None and seeds and len(seeds) >= (count or 1):
        fallback = {"faults": [_seed_fault(seed) for seed in seeds[:count or len(seeds)]]}
        try:
            result = validate_final_answers(fallback, view, count)
            attempts.append({"valid": True, "source": "coordinator_seed_fallback"})
        except (ValueError, KeyError, TypeError) as error:
            attempts.append({"valid": False, "source": "coordinator_seed_fallback", "error": str(error)})
    elif result is not None and seeds:
        merged = False
        for index, fault in enumerate(result["faults"]):
            if index >= len(seeds):
                break
            if all(not fault[key] for key in ("component", "reason", "time", "evidence_ids")):
                result["faults"][index] = _seed_fault(seeds[index])
                merged = True
        if merged:
            try:
                result = validate_final_answers(result, view, count)
                attempts.append({"valid": True, "source": "merged_supported_seed"})
            except (ValueError, KeyError, TypeError) as error:
                result = None
                attempts.append({"valid": False, "source": "merged_supported_seed", "error": str(error)})
    if result is None and max_candidates == 1:
        for parsed in reversed(parsed_replies):
            try:
                result, field_repairs = retain_valid_fields_with_unknown_time(parsed, view, count)
            except (ValueError, KeyError, TypeError):
                continue
            attempts.append({"valid": True, "source": "retained_fields_with_unknown_time"})
            break
    save(output / "final_answer.json", {"reply": result, "attempts": attempts, "labels_used": False,
         "cutoff_ms": view["cutoff_ms"], "model_judgment_verified": False,
         "expected_failures": count, "answer_schema": "faults_v1", "field_repairs": field_repairs,
         "warnings": [{"fault_index": index, "kind": "explanation_length",
                       "characters": len(fault["explanation"]), "requested_maximum": 220}
                      for index, fault in enumerate(result["faults"] if result else [])
                      if len(fault["explanation"]) > 220]})
    predictions = []
    for fault in (result["faults"] if result else [None] * (count or 1)):
        prediction = {"root cause occurrence datetime": "", "root cause component": "", "root cause reason": ""}
        if fault:
            prediction.update({"root cause component": fault["component"] or "", "root cause reason": fault["reason"] or ""})
            if fault["time"]:
                ms = occurrence_ms(fault["time"], view["window_start_ms"], view["cutoff_ms"])
                prediction["root cause occurrence datetime"] = format_time(ms)[:19].replace("T", " ")
        predictions.append(prediction)
    save(output / "prediction.json", predictions)
    if max_candidates > 1:
        from .ranked_answers import build
        ranked = build(result, alternatives, view, count, max_candidates, validate_final_answers)
        save(output / "ranked_predictions.json", ranked)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="phase", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--query-file", type=Path, required=True)
    p.add_argument("--row", type=int, default=0)
    p.add_argument("--checkpoints", default="600,1200,1800",
                   help="Comma-separated visible checkpoints; use 1800 for a fast single-snapshot run")
    p.add_argument("--expected-nodes", type=int, default=18,
                   help="Expected device-node count; 0 accepts the per-window inventory")
    for name in ("nodes", "coordinate"):
        p = subs.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--prepared", type=Path, required=True)
        p.add_argument("--seconds", type=int, choices=(600,1200,1800), required=True)
        p.add_argument("--node-model", type=Path, required=True)
        p.add_argument("--node-mode", choices=("prose", "template", "catalogue"), default="catalogue")
        p.add_argument("--screen-limit", type=int, default=None, help="Template mode: only screen this many rows per node for the 4B")
        p.add_argument("--detailed-node-replies", action="store_true",
                       help="Catalogue mode: fill meaning, partial answers, unknowns and next checks")
        p.add_argument("--decisions-only", action="store_true",
                       help="Catalogue batch: page decisions only; skip per-row detailed fill")
        if name == "coordinate":
            p.add_argument("--coordinator-model", type=Path, required=True)
            p.add_argument("--max-queries", type=int, default=2)
            p.add_argument("--final", action="store_true")
            p.add_argument("--max-candidates", type=int, choices=(1, 2, 3), default=1,
                           help="Save up to this many complete alternatives separately from the final prediction")
    args = parser.parse_args()
    try:
        if args.phase == "prepare":
            try:
                checkpoints = tuple(int(item) for item in args.checkpoints.split(","))
                if (not checkpoints or len(set(checkpoints)) != len(checkpoints)
                        or any(item <= 0 for item in checkpoints)):
                    raise ValueError
            except ValueError:
                parser.error("--checkpoints must be unique positive integers")
            prepare(args.prepared, args.output, args.query_file, args.row, checkpoints, args.expected_nodes)
        elif args.phase == "nodes":
            nodes(args.root, args.prepared, args.seconds, args.node_model, node_mode=args.node_mode,
                  screen_limit=args.screen_limit, detailed=args.detailed_node_replies,
                  decisions_only=args.decisions_only)
        else:
            coordinate(args.root, args.prepared, args.seconds, args.coordinator_model, args.node_model,
                       args.max_queries, args.final, node_mode=args.node_mode,
                       detailed=args.detailed_node_replies, max_candidates=args.max_candidates)
    except Exception as error:
        directory = args.output if args.phase == "prepare" else args.root
        save(directory / f"{args.phase}_failure.json", {"error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
