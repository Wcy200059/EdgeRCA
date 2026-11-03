"""Compare local models on identical real, multi-observation node prefixes."""

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from .alerts import AlertEngine
from .investigation import FixedSOP
from .node_probe import LocalModel, decode_object, write_json
from .node_reports import build_snapshot, pages, messages, validate, ReportInbox
from .report_cards import card, fact_view
from .replay import ReplaySession
from .rolling_memory import RollingNodeMemory
from .trace_index import TraceIndex


def prepare(source, prepared, output):
    output.mkdir(parents=True, exist_ok=False)
    requests = []

    def save(memory, cpu, graph, probe):
        packet = build_snapshot(memory, cpu, graph)
        packet["case_id"] = memory["overview"]["case_id"]
        packet["request_id"] = packet["node_id"].replace(":", "_") + "_" + str(packet["cutoff_ms"])
        planned = pages(packet)
        write_json(output / (packet["request_id"] + ".json"), packet)
        requests.append({"file": packet["request_id"] + ".json", "node_id": packet["node_id"],
                         "cutoff_ms": packet["cutoff_ms"], "facts": len(packet["facts"]),
                         "pages": len(planned), "compare": probe})

    # Earlier snapshot is rebuilt from a real prefix, never cut from final stats.
    node = "device:Tomcat04"
    with ReplaySession(prepared) as session, TraceIndex() as graph:
        case_id = hashlib.sha256(str(prepared.resolve()).encode()).hexdigest()[:16]
        memory = RollingNodeMemory(case_id, node, session.start_ms, session.end_ms)
        engine, cursor = AlertEngine(session.start_ms), 0
        cutoff = session.start_ms + 10 * 60_000
        for batch in session.iter_ticks(10_000):
            graph.ingest(batch["events"], batch["clock_ms"])
            for event in batch["events"]:
                engine.feed(event)
            memory.receive_counts([e for e in batch["events"] if e["node_id"] == node], batch["clock_ms"])
            engine.advance_to(batch["clock_ms"])
            memory.sync([t for t in engine.transitions[cursor:] if t["node_id"] == node],
                        [a for a in engine.alerts.values() if a["node_id"] == node], batch["clock_ms"])
            cursor = len(engine.transitions)
            if batch["clock_ms"] == cutoff:
                save(memory.export(), FixedSOP(session).build(node, cutoff, session.start_ms),
                     graph.snapshot(cutoff, session.start_ms, session.start_ms), True)
                break
    graph = json.loads((source / "final_links.json").read_text())
    for path in sorted((source / "nodes").glob("device_*.json")):
        memory = json.loads(path.read_text())
        cpu = json.loads((source / "final_cpu" / path.name).read_text())
        save(memory, cpu, graph, memory["overview"]["node_id"] in
             {"device:Tomcat04", "device:Mysql02", "device:dockerA2"})
    write_json(output / "manifest.json", {"source": str(source.resolve()), "labels_used": False,
        "requests": requests, "comparison_note": "应用节点10/30分钟、密集数据库节点、无告警但有数据的容器节点；按工作负荷选取，不按答案。"})
    print(json.dumps(requests, ensure_ascii=False), flush=True)


def source_numbers_check(reply, facts):
    """Weak numeric provenance check, not a proof that the prose is correct."""
    import re
    catalog = {f["id"]: f for f in facts}
    failures = []
    for group in reply["groups"]:
        supported = json.dumps([catalog[i] for i in group["observation_ids"]], ensure_ascii=False)
        # Match complete decimal values; arbitrary arithmetic/rounding is not verified.
        present = set(re.findall(r"\d+(?:\.\d+)?", supported))
        extras = set(re.findall(r"\d+(?:\.\d+)?", group["summary"])) - present
        if extras:
            failures.append({"ids": group["observation_ids"], "unverified_numbers": sorted(extras)})
    return failures


def report_pages(packet, facts=None, max_facts=8, max_chars=7200):
    facts = packet["facts"] if facts is None else facts
    by_id = {f["id"]: f for f in facts}
    visible = [fact_view(f, packet) for f in facts]
    return [[by_id[f["id"]] for f in chunk]
            for chunk in pages({"facts": visible}, max_facts, max_chars)]


def run(inputs, output, model_path, all_nodes=False, coordinator=False):
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    for file in Path(__file__).parent.glob("*.py"):
        (output / "source" / file.name).write_bytes(file.read_bytes())
    manifest = json.loads((inputs / "manifest.json").read_text())
    if manifest["labels_used"]:
        raise ValueError("labels are not node inputs")
    print(json.dumps({"loading_model": str(model_path), "inputs": str(inputs)}, ensure_ascii=False), flush=True)
    model = LocalModel(model_path, output / "calls.jsonl", max_context=16384)
    inbox, results = ReportInbox(), []
    started = time.monotonic()

    def ask(packet, facts, phase, merge=False):
        prompt = messages(packet, facts, merge=merge, readable=True)
        tries = []
        for attempt in range(2):
            try:
                raw = model.invoke(prompt, None, 1500, phase + f"/{attempt}")
                reply = validate(decode_object(raw), facts)
            except (ValueError, KeyError, TypeError) as error:
                tries.append({"valid": False, "error": str(error)})
                prompt = messages(packet, facts, merge=merge, readable=True)
                prompt[-1]["content"] += "\n请修正上次的格式错误：" + str(error)
                continue
            tries.append({"valid": True})
            return reply, tries, source_numbers_check(reply, facts)
        return None, tries, []

    for entry in manifest["requests"]:
        if not all_nodes and not entry["compare"]:
            continue
        packet = json.loads((inputs / entry["file"]).read_text())
        if packet["labels_used"]:
            raise ValueError("label-bearing snapshot")
        write_json(output / entry["file"], packet)
        group_candidates, local, deferred, stages = [], set(), set(), []
        for index, facts in enumerate(report_pages(packet)):
            reply, attempts, numeric_flags = ask(packet, facts, packet["request_id"] + f"/page{index}")
            stages.append({"phase": "page", "index": index, "attempts": attempts,
                           "reply": reply, "numeric_review_flags": numeric_flags})
            if reply is None:
                deferred.update(f["id"] for f in facts)
                continue
            local.update(reply["retain_local"])
            deferred.update(reply["defer"])
            for g in reply["groups"]:
                group_candidates.append({"id": f"g{len(group_candidates) + 1}",
                    "summary": g["summary"], "priority": g["priority"],
                    "source_ids": g["observation_ids"]})
        # One cross-page synthesis pass, itself paged if needed. No forced cap:
        # groups that cannot safely be merged remain separate in the report.
        catalog = {f["id"]: f for f in packet["facts"]}
        final_groups = []
        candidates = []
        for g in group_candidates:
            # Keep complete facts at synthesis: the old field whitelist lost
            # connection counts/durations, example times and log pause values.
            candidates.append({**g, "cpu_facts": [catalog[i] for i in g["source_ids"]]})
        candidate_map = {c["id"]: c for c in candidates}
        for index, chunk in enumerate(report_pages(packet, candidates, max_facts=12, max_chars=11000)):
            reply, attempts, numeric_flags = ask(packet, chunk, packet["request_id"] + f"/merge{index}", True)
            stages.append({"phase": "merge", "index": index, "attempts": attempts,
                           "reply": reply, "numeric_review_flags": numeric_flags})
            if reply is None:
                deferred.update(i for c in chunk for i in c["source_ids"])
                continue
            for key, destination in (("retain_local", local), ("defer", deferred)):
                destination.update(i for c in reply[key] for i in candidate_map[c]["source_ids"])
            for g in reply["groups"]:
                ids = [i for c in g["observation_ids"] for i in candidate_map[c]["source_ids"]]
                final_groups.append({"summary": g["summary"], "priority": g["priority"],
                                     "source_ids": ids, "cpu_facts": [catalog[i] for i in ids],
                                     "tool_cards": [card(catalog[i], packet) for i in ids]})
        reported = [i for g in final_groups for i in g["source_ids"]]
        if (len(reported) != len(set(reported)) or set(reported) & local or set(reported) & deferred or
                local & deferred or set(reported) | local | deferred != set(catalog)):
            raise ValueError("final disposition lost or duplicated an observation")
        report = {"case_id": packet["case_id"], "node_id": packet["node_id"],
            "window_start_ms": packet["window_start_ms"],
            "cutoff_ms": packet["cutoff_ms"], "scope": packet["scope"], "groups": final_groups,
            "retain_local": sorted(local), "defer": sorted(deferred),
            "cpu_only_metric_ids": packet["cpu_only_metric_ids"],
            "source_snapshot": entry["file"], "prose_semantics_verified": False,
            "tool_card_catalog": {f["id"]: card(f, packet) for f in packet["facts"]},
            "note": "每份报告替换本节点旧版本；CPU事实可查，模型说明仍需内容核验。"}
        inbox.receive(report)
        write_json(output / (packet["request_id"] + "_report.json"), {"report": report, "stages": stages})
        result = {"node_id": packet["node_id"], "cutoff_ms": packet["cutoff_ms"],
                  "input_facts": len(catalog), "page_groups": len(candidates), "report_groups": len(final_groups),
                  "reported_facts": len(reported), "retained_local": len(local), "deferred": len(deferred),
                  "failed_stages": sum(s["reply"] is None for s in stages),
                  "numeric_review_flags": sum(len(s["numeric_review_flags"]) for s in stages)}
        results.append(result)
        write_json(output / "progress.json", results)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    write_json(output / "coordinator_inbox.json", {"latest_reports": list(inbox.latest.values()),
        "report_history": inbox.history, "coordinator_model_executed": False})
    if coordinator:
        from .coordinator import run_probe
        run_probe(model, inbox, output, nodes=("device:Tomcat04", "device:dockerA2"))
    write_json(output / "summary.json", {"model": str(model_path), "results": results,
        "model_calls": len(model.calls), "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_gib": model.torch.cuda.max_memory_allocated() / 1024**3,
        "labels_used": False, "prose_semantics_verified": False,
        "coordinator_requested": coordinator,
        "input_sha256": hashlib.sha256((inputs / "manifest.json").read_bytes()).hexdigest()})
    lines = ["# 多观测节点报告实测", "", "以下是模型原文，尚未完整语义核验；不是RCA诊断结论。", ""]
    for report in inbox.history:
        lines += [f"## {report['node_id']} · 截至 {report['scope']['end']}", ""]
        for g in report["groups"]:
            lines += [f"- 模型说明（待核实）：{g['summary']} （引用：{', '.join(g['source_ids'])}）"]
            lines += [f"  - 工具事实 [{c['id']}]：{c['tool_fact']}" for c in g["tool_cards"]]
        lines += ["", f"留本地 {len(report['retain_local'])} 项；未处理 {len(report['defer'])} 项。", ""]
    (output / "reports.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("source", "prepared", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("run")
    for name in ("inputs", "output", "model"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--all-nodes", action="store_true")
    p.add_argument("--coordinator", action="store_true", help="probe coordinator on Tomcat04/dockerA2 reports; not full RCA")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.prepared, args.output)
    else:
        run(args.inputs, args.output, args.model, args.all_nodes, args.coordinator)
