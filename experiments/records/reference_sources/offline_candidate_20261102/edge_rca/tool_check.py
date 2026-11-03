"""CPU acceptance replay for the fixed tools; not a model worker/runtime."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

from .alerts import AlertEngine
from .data import format_time
from .investigation import FixedSOP, model_pages
from .preprocessing import build_preprocessor_prompt
from .replay import ReplaySession
from .trace_index import TraceIndex


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_tool_check(args):
    offsets = sorted(set(int(v) for v in args.checkpoints_seconds.split(",")))
    if not offsets or offsets[0] <= 0:
        raise ValueError("checkpoints must be positive integer seconds")
    prepared, output = Path(args.prepared).resolve(), Path(args.output).resolve()
    manifest = json.loads((prepared / "manifest.json").read_text())
    start, end = manifest["window"]["start_ms"], manifest["window"]["end_ms"]
    checkpoints = [start + offset * 1000 for offset in offsets]
    if checkpoints[-1] > end:
        raise ValueError("checkpoint exceeds replay window")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    snapshots = []
    try:
        with ReplaySession(prepared, output / "inbox.sqlite") as session, TraceIndex(output / "trace_index.sqlite") as index:
            engine, sop = AlertEngine(start), FixedSOP(session)
            for cutoff in checkpoints:
                while session.clock_ms < cutoff:
                    batch = session.advance_to(min(session.clock_ms + 10_000, cutoff))
                    index.ingest(batch["events"], batch["clock_ms"])
                    for event in batch["events"]:
                        engine.feed(event)
                    engine.advance_to(batch["clock_ms"])
                directory = output / str((cutoff - start) // 1000)
                directory.mkdir()
                # This is an explicit rolling probe, not a live incident update.
                # A runtime task will supply and retain its own incident anchor.
                anchor = max(start, cutoff - 60_000)
                reference_start = max(start, anchor - 300_000)
                graph = index.snapshot(cutoff, reference_start, anchor)
                _write(directory / "call_links.json", graph)
                nodes = []
                for node in session.manifest["nodes"]:
                    node_id = node["node_id"]
                    result = sop.build(node_id, cutoff, anchor)
                    pages = model_pages(result)
                    # Same path/IDs irrespective of manifest future record counts.
                    target = directory / (node_id.replace(":", "_") + ".json")
                    _write(target, result)
                    _write(target.with_suffix(".pages.json"), pages)
                    for page in pages:
                        build_preprocessor_prompt(page)
                    inspected = set(eid for ids in result["evidence_index"].values() for eid in ids)
                    expected = {e["event_id"] for e in session.query(node_id, limit=None, available_before_ms=cutoff)
                                if e["available_at_ms"] >= reference_start}
                    if inspected != expected:
                        raise RuntimeError(f"SOP coverage mismatch for {node_id}")
                    node_links = [link["link_id"] for link in graph["links"]
                                  if node_id in (link["source_node"], link["target_node"])]
                    nodes.append({"node_id": node_id, "kind": node["kind"],
                                  "rows_checked": len(inspected), "facts": len(result["observations"]),
                                  "pages_prepared": len(pages), "observed_link_ids": node_links,
                                  "oversized_pages": sum(p["scope"]["oversized_single_page"] for p in pages),
                                  "model_processing_status": "not_started"})
                # Prepare up to three real alert tasks as interface examples.
                # They retain their own historical cutoff, not this probe time.
                anchors = {}
                for transition in engine.transitions:
                    if transition["action"] == "open":
                        anchors.setdefault(transition["alert_id"], transition["details"].get(
                            "window_start_ms", transition["at_ms"]))
                previews, selected_nodes = [], set()
                for task in reversed(engine.tasks):
                    node_id = task["node_id"]
                    if not node_id.startswith("device:") or node_id in selected_nodes:
                        continue
                    selected_nodes.add(node_id)
                    task_result = sop.build_task(task, anchors)
                    task_pages = model_pages(task_result)
                    for page in task_pages:
                        build_preprocessor_prompt(page)
                    _write(directory / f"task_{task['task_id']}.json", task_result)
                    _write(directory / f"task_{task['task_id']}.pages.json", task_pages)
                    previews.append({"task_id": task["task_id"], "node_id": node_id,
                                     "original_cutoff_ms": task_result["cutoff_ms"],
                                     "anchor_ms": task_result["scope"]["current"]["start_ms"],
                                     "pages_prepared": len(task_pages), "model_calls": 0})
                    if len(previews) == 3:
                        break
                _write(directory / "task_previews.json", previews)
                _write(directory / "node_status.json", nodes)
                snapshot = {"cutoff_ms": cutoff, "time": format_time(cutoff),
                            "probe_reference_start_ms": reference_start, "probe_anchor_ms": anchor,
                            "delivered": session.total_delivered, "graph_counts": graph["counts"],
                            "node_count": len(nodes), "pages": sum(n["pages_prepared"] for n in nodes),
                            "oversized_pages": sum(n["oversized_pages"] for n in nodes),
                            "source_coverage_exact": True}
                snapshots.append(snapshot)
                print(json.dumps(snapshot, ensure_ascii=False), flush=True)
            engine.finish(session.clock_ms)
            _write(output / "alerts.json", list(engine.alerts.values()))
            _write(output / "tasks.json", engine.tasks)
            complete = session.clock_ms == end
            if complete and session.total_delivered != manifest["totals"]["deliverable"]:
                raise RuntimeError("full replay count mismatch")
            summary = {"status": "complete", "purpose": "fixed_CPU_tools_acceptance_only",
                       "prepared": str(prepared), "snapshots": snapshots,
                       "delivered_records": session.total_delivered,
                       "delivered_by_modality": session.delivered_counts,
                       "full_window_replayed": complete, "alert_count": len(engine.alerts),
                       "task_count": len(engine.tasks), "model_calls": 0, "cnn_loaded": False,
                       "elapsed_seconds": time.monotonic() - started,
                       "nodes_by_kind": dict(Counter(n["kind"] for n in manifest["nodes"])),
                       "limitations": ["在检查点整理所有节点用于核对工具覆盖；不是生产告警调度策略。",
                                        "没有运行模型、协调者或后台工作队列，没有诊断正确率。",
                                        "检查采用最近一分钟对比此前最多五分钟；持续事件需保留其触发锚点。"]}
            _write(output / "summary.json", summary)
            return summary
    except BaseException as error:
        _write(output / "failure.json", {"error": str(error), "snapshots_completed": snapshots})
        raise
