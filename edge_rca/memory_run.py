"""Replay one Bank window into local cumulative memories; no labels or LLM.

Snapshots are written while replay advances. Final full-window tool statistics
and the call graph are explicitly end-of-window artifacts, never fed backwards.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

from .alerts import AlertEngine
from .investigation import FixedSOP
from .node_probe import write_json
from .replay import ReplaySession
from .rolling_memory import RollingNodeMemory
from .trace_index import TraceIndex


def directory_text(directory):
    """Small navigation table; detailed evidence is fetched by stable node ID."""
    lines = ["# 节点持续记录目录", "",
             "这是定位明细的入口，不能仅凭告警数量判断根因。完整事实与来源在节点明细中。", "",
             "| 节点 | 收到指标/日志/调用记录 | 信号组 | 仍触发 | 多次出现 | 模型待处理组 |",
             "|---|---:|---:|---:|---:|---:|"]
    for n in directory["node_reports"]:
        rows = n["raw_rows_received"]
        counts = "/".join(str(rows.get(k, 0)) for k in ("metric", "log", "trace"))
        lines.append(f"| [{n['node_id']}]({n['detail_file']}) | {counts} | "
                     f"{n['observation_groups']} | {n['last_known_triggered_groups']} | "
                     f"{n['groups_with_multiple_activations']} | {n['model_pending_groups']} |")
    lines.extend(["", "业务流另列于coordinator_directory.json；没有设备映射时不猜测其所属节点。",
                  "模型尚未完成的组明确显示为待处理；无告警不代表正常。"])
    return "\n".join(lines) + "\n"


def run(prepared, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for part in ("nodes", "final_cpu", "source"):
        (output / part).mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        (output / "source" / path.name).write_bytes(path.read_bytes())
    started = time.monotonic()
    frozen, last_status = [], None
    versions, transition_cursor = defaultdict(dict), 0
    checks, upserts = 0, 0
    with ReplaySession(prepared, output / "inbox.sqlite") as session, TraceIndex() as graph:
        case_id = hashlib.sha256(str(Path(prepared).resolve()).encode()).hexdigest()[:16]
        memories = {n["node_id"]: RollingNodeMemory(case_id, n["node_id"], session.start_ms, session.end_ms)
                    for n in session.manifest["nodes"]}
        engine = AlertEngine(session.start_ms)
        journal = (output / "updates.jsonl").open("w", encoding="utf-8")

        def update(cutoff):
            nonlocal transition_cursor, checks, upserts, last_status
            changes, alerts = defaultdict(list), defaultdict(list)
            for t in engine.transitions[transition_cursor:]:
                changes[t["node_id"]].append(t)
            for a in engine.alerts.values():
                alerts[a["node_id"]].append(a)
            transition_cursor = len(engine.transitions)
            for node, memory in memories.items():
                dirty = memory.sync(changes[node], alerts[node], cutoff)
                delta = memory.changes_since(versions[node])
                if delta:
                    journal.write(json.dumps({"cutoff_ms": cutoff, "node_id": node,
                        "upsert_records": delta}, ensure_ascii=False) + "\n")
                    versions[node].update({r["observation_id"]: r["version"] for r in delta})
                    upserts += len(delta)
                checks += len(dirty)
                # Real sequential lifecycle examples, chosen by rule/state only.
                if node == "device:Tomcat04":
                    gid = memory.group_id("trace.latency", "completed_call_p95")
                    if gid in memory.records:
                        fact = memory.project(gid)
                        signature = (fact["status"], fact["signal_activation_cycles"])
                        if signature != last_status and len(frozen) < 4:
                            frozen.append(memory.freeze(gid))
                            last_status = signature

        try:
            for batch in session.iter_ticks(10_000):
                graph.ingest(batch["events"], batch["clock_ms"])
                by_node = defaultdict(list)
                for event in batch["events"]:
                    by_node[event["node_id"]].append(event)
                    engine.feed(event)
                for node, rows in by_node.items():
                    memories[node].receive_counts(rows, batch["clock_ms"])
                engine.advance_to(batch["clock_ms"])
                update(batch["clock_ms"])
            engine.finish(session.end_ms)
            update(session.end_ms)
        finally:
            journal.close()
        target = memories["device:Tomcat04"]
        gid = target.group_id("trace.latency", "completed_call_p95")
        if gid in target.records:
            frozen.append(target.freeze(gid))
        write_json(output / "model_requests.json", {"requests": frozen, "labels_used": False,
            "note": "真实时间前缀冻结；最终CPU工具结果不得用于较早请求。"})
        summaries, all_transitions, all_episodes = [], set(), set()
        sop = FixedSOP(session)
        for node, memory in memories.items():
            exported = memory.export()
            all_transitions.update(exported["transitions"])
            all_episodes.update(a for r in memory.records.values() for a in r["episodes"])
            overview = exported["overview"]
            overview["detail_file"] = "nodes/" + node.replace(":", "_") + ".json"
            overview["full_tool_file"] = "final_cpu/" + node.replace(":", "_") + ".json"
            summaries.append(overview)
            write_json(output / overview["detail_file"], exported)
            result = sop.build(node, session.end_ms, session.start_ms)
            write_json(output / overview["full_tool_file"], result)
        directory = {
            "window_start_ms": session.start_ms, "cutoff_ms": session.end_ms,
            "node_reports": [x for x in summaries if x["node_id"].startswith("device:")],
            "business_reports": [x for x in summaries if x["node_id"].startswith("business:")],
            "note": "小目录可定位明细，不能把仅有数量的目录当成足够诊断的证据。协调者尚未运行。"}
        write_json(output / "coordinator_directory.json", directory)
        (output / "coordinator_index.md").write_text(directory_text(directory), encoding="utf-8")
        links = graph.snapshot(session.end_ms, session.start_ms, session.start_ms)
        write_json(output / "final_links.json", links)
        expected_rows = session.total_delivered
        assert sum(sum(m.rows.values()) for m in memories.values()) == expected_rows
        assert len(all_transitions) == len(engine.transitions)
        assert all_episodes == set(engine.alerts)
        devices = [m for n, m in memories.items() if n.startswith("device:")]
        summary = {"prepared": str(Path(prepared).resolve()), "delivered_rows": expected_rows,
            "device_nodes": len(devices), "business_streams": len(memories) - len(devices),
            "device_rule_episodes": sum(len(r["episodes"]) for m in devices for r in m.records.values()),
            "device_transitions": sum(len(m.transitions) for m in devices),
            "device_observation_groups": sum(len(m.records) for m in devices),
            "device_groups_with_recurrence": sum(m.overview()["groups_with_multiple_activations"] for m in devices),
            "all_transition_count": len(all_transitions), "all_episode_count": len(all_episodes),
            "versioned_updates": checks, "upsert_records": upserts,
            "model_requests_frozen": len(frozen), "model_calls": 0, "labels_used": False,
            "integrity_checks_passed": True, "elapsed_seconds": time.monotonic() - started,
            "limits": ["本次是CPU持续记录与输入准备；未让模型处理全部半小时。",
                       "同一信号可含多条规则或错误变体；明细仍保留，不等于同一根因。",
                       "最终完整工具统计包含未触发告警的数据，尚未全部交给模型。"]}
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.prepared, args.output)
