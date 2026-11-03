"""Audit one full causal window on every node and prepare representative pages.

No labels, model calls or root-cause scores. Required means associated with an
observed alert or a declared test control, not a proven root-cause observation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from .alarm_pages import choose_representatives, enrich_alert_facts, plan_pages
from .alerts import AlertEngine, GC_PAUSE
from .compact_tools import build_compact_tools
from .data import format_time
from .investigation import FixedSOP
from .node_probe import write_json
from .replay import ReplaySession
from .trace_index import TraceIndex


def inspect_task(result, bundle, plan, transitions, expected_source_ids):
    used = set(e for ids in result["evidence_index"].values() for e in ids)
    cpu = {f["observation_id"] for f in result["observations"]}
    accounted = {f["observation_id"] for f in bundle["catalog"].values()} | set(bundle["cpu_only_observation_ids"])
    alarms = [fact for facts in bundle["tools"].values() for fact in facts if fact.get("kind") == "alert_evaluation"]
    planned = {i for p in plan["pages"] for i in p["mandatory_ids"]}
    errors = []
    if used != set(expected_source_ids):
        errors.append("raw_to_cpu_source_membership")
    if accounted != cpu:
        errors.append("cpu_to_compact_accounting")
    if len(alarms) != len(transitions) or not {f["id"] for f in alarms} <= planned:
        errors.append("alert_snapshot_not_planned")
    return {"raw_rows": len(expected_source_ids), "cpu_facts": len(cpu),
        "compact_facts": len(bundle["catalog"]), "cpu_only_facts": len(bundle["cpu_only_observation_ids"]),
        "transitions": len(transitions), "required_facts": len(plan["required_ids"]),
        "planned_pages": len(plan["pages"]), "deferred_compact_facts": len(plan["deferred_ids"]),
        "oversized_pages": sum(p["oversized_single_fact"] for p in plan["pages"]), "errors": errors}


def run(prepared, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    started = time.monotonic()
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    rows_by_node, log_info = defaultdict(Counter), Counter()
    audits, candidates, sources, anchors = [], [], {}, {}
    with ReplaySession(prepared) as session, TraceIndex() as index:
        engine, sop = AlertEngine(session.start_ms), FixedSOP(session)
        for batch in session.iter_ticks(10_000):
            index.ingest(batch["events"], batch["clock_ms"])
            for event in batch["events"]:
                rows_by_node[event["node_id"]][event["modality"]] += 1
                if event["modality"] == "log":
                    raw = event["raw"]
                    text = raw.get("value", "")
                    if raw.get("log_name", "").lower() == "gc" or "[GC" in text or "[Full GC" in text:
                        log_info["gc_lines"] += 1
                        pause = GC_PAUSE.search(text)
                        if pause and float(pause.group(1)) * 1000 >= engine.config.gc_pause_ms:
                            log_info["gc_above_configured_pause"] += 1
                    if "OutOfMemoryError" in text or "Java heap space" in text:
                        log_info["explicit_oom_literal_lines"] += 1
                engine.feed(event)
            engine.advance_to(batch["clock_ms"])
        # Flush final partial windows/tasks as well as the last transport batch.
        engine.finish(session.end_ms)
        for t in engine.transitions:
            if t["action"] == "open":
                anchors[t["alert_id"]] = t["details"].get("window_start_ms", t["at_ms"])
        tasks = [t for t in engine.tasks if t["node_id"].startswith("device:")]
        tasks.sort(key=lambda t: (t["ready_at_ms"], t["node_id"], t["task_id"]))
        gc_control_added = False

        def prepare_request(rid, task=None, node=None):
            result = sop.build_task(task, anchors) if task else sop.build(node, session.end_ms, session.start_ms)
            # Graph is unnecessary for accounting; selected model inputs receive
            # a real historical graph below, never an inferred or future graph.
            bundle = enrich_alert_facts(result, build_compact_tools(result,
                {"cutoff_ms": result["cutoff_ms"], "links": []}))
            extra = []
            if not task:
                extra = [next(f["id"] for facts in bundle["tools"].values() for f in facts)]
            plan = plan_pages(bundle, rid, extra_required=extra)
            return result, bundle, plan

        with (output / "task_coverage.jsonl").open("w", encoding="utf-8") as journal:
            for n, task in enumerate(tasks, 1):
                rid = f"r{n:04}"
                result, bundle, plan = prepare_request(rid, task)
                expected = {e["event_id"] for e in session.query(task["node_id"], limit=None,
                    available_before_ms=result["cutoff_ms"]) if e["available_at_ms"] >= result["scope"]["reference"]["start_ms"]}
                audit = {"request_id": rid, "node_id": task["node_id"], "task_id": task["task_id"],
                         "cutoff_ms": result["cutoff_ms"], **inspect_task(result, bundle, plan, task["transitions"], expected)}
                audits.append(audit)
                journal.write(json.dumps(audit, ensure_ascii=False) + "\n")
                sources[rid] = {"task": task, "extra_required": []}
                phase = min(2, (result["cutoff_ms"] - session.start_ms) * 3 // (session.end_ms - session.start_ms))
                for page in plan["pages"]:
                    page["features"].append(f"window_third:{phase}")
                    if len(task["transitions"]) >= 20:
                        page["features"].append("dense_update")
                    candidates.append({"request_id": rid, **page})
                if not gc_control_added:
                    gc = next((f for f in bundle["tools"]["get_logs"] if f.get("kind") == "gc"), None)
                    if gc:
                        control = plan_pages(bundle, rid, extra_required=[gc["id"]])
                        page = next(p for p in control["pages"] if gc["id"] in p["mandatory_ids"])
                        page = dict(page, page_id=rid + "-gc-control", features=page["features"] + ["ordinary_gc_control"])
                        candidates.append({"request_id": rid, **page})
                        gc_control_added = True
                if n % 25 == 0:
                    print(json.dumps({"tasks_checked": n, "total": len(tasks), "errors": sum(bool(a["errors"]) for a in audits)}), flush=True)
            alerted_nodes = {t["node_id"] for t in tasks}
            controls = []
            for node in session.manifest["nodes"]:
                node_id = node["node_id"]
                if node["kind"] != "device" or node_id in alerted_nodes:
                    continue
                rid = f"r{len(tasks) + len(controls) + 1:04}"
                result, bundle, plan = prepare_request(rid, node=node_id)
                page = plan["pages"][0]
                page["features"].append("no_alert_control")
                candidates.append({"request_id": rid, **page})
                sources[rid] = {"node": node_id, "extra_required": page["mandatory_ids"]}
                controls.append({"request_id": rid, "node_id": node_id, "no_alert_does_not_mean_healthy": True})

        chosen = choose_representatives(candidates)
        requests = {}
        for c in chosen:
            rid = c["request_id"]
            if rid in requests:
                continue
            source = sources[rid]
            result, _, _ = prepare_request(rid, task=source.get("task"), node=source.get("node"))
            graph = index.snapshot(result["cutoff_ms"], result["scope"]["reference"]["start_ms"], result["scope"]["current"]["start_ms"])
            bundle = enrich_alert_facts(result, build_compact_tools(result, graph))
            request = {"request_id": rid, "node_id": result["node_id"], "cutoff_ms": result["cutoff_ms"],
                       "bundle": bundle, "selected_pages": [x for x in chosen if x["request_id"] == rid],
                       "control": "no_alert_control" if "node" in source else None}
            write_json(output / "inputs" / f"{rid}.json", request)
            requests[rid] = request
        write_json(output / "inputs" / "manifest.json", {"requests": [
            {"request_id": rid, "path": f"{rid}.json"} for rid in requests],
            "selected_pages": len(chosen), "labels_used": False, "purpose": "representative_alert_summary_checks"})
        node_tasks = Counter(t["node_id"] for t in engine.tasks)
        nodes = [{"node_id": n["node_id"], "kind": n["kind"], "rows": dict(rows_by_node[n["node_id"]]),
                  "tasks": node_tasks[n["node_id"]]} for n in session.manifest["nodes"]]
        summary = {"prepared": str(Path(prepared).resolve()), "start": format_time(session.start_ms),
            "end": format_time(session.end_ms), "full_window_replayed": True,
            "delivered_records": session.total_delivered, "delivered_by_modality": session.delivered_counts,
            "nodes": nodes, "device_tasks_audited": len(tasks), "business_tasks_cpu_only": len(engine.tasks) - len(tasks),
            "alert_episodes_not_faults": len(engine.alerts), "rules": dict(Counter(a["rule"] for a in engine.alerts.values())),
            "actions": dict(Counter(t["action"] for t in engine.transitions)), "engine_stats": dict(engine.stats),
            "rule_config": asdict(engine.config), "raw_log_characteristics": dict(log_info),
            "max_transitions_per_task": max((a["transitions"] for a in audits), default=0),
            "old_two_facts_limit_insufficient_tasks": sum(a["required_facts"] > 2 for a in audits),
            "required_fact_appearances": sum(a["required_facts"] for a in audits),
            "planned_pages": sum(a["planned_pages"] for a in audits),
            "coverage_error_tasks": sum(bool(a["errors"]) for a in audits),
            "oversized_pages": sum(a["oversized_pages"] for a in audits),
            "representative_pages": len(chosen), "representative_features": sorted(set().union(*(set(c["features"]) for c in chosen))),
            "no_alert_controls": controls, "model_calls": 0, "labels_used": False,
            "elapsed_seconds": time.monotonic() - started, "source_sha256": source_hashes,
            "limits": ["只审查一题半小时，不是Bank全数据或最终RCA正确率。",
                       "覆盖已触发告警不代表检测到了全部真实故障；无告警不代表正常。",
                       "部分非告警指标仍是CPU-only；没有将所有原始记录送进模型。"]}
        write_json(output / "summary.json", summary)
        write_json(output / "alerts.json", list(engine.alerts.values()))
        write_json(output / "tasks.json", engine.tasks)
        print(json.dumps({k: summary[k] for k in ("device_tasks_audited", "coverage_error_tasks", "planned_pages", "representative_pages", "elapsed_seconds")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.prepared, args.output)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
