"""Prepare and replay a Bank window without loading any language model."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

from .data import format_time, prepare_bank


ROOT = Path(__file__).resolve().parents[1]


def _window(args):
    if args.start is not None or args.end is not None:
        if args.start is None or args.end is None:
            raise ValueError("Provide both --start and --end")
        return args.start, args.end, None
    from .query import read_query_window

    return read_query_window(args.query_file, args.query_row)


def _prepare(args):
    start, end, source = _window(args)
    manifest = prepare_bank(args.telemetry_root, start, end, args.output, source)
    print(json.dumps({"output": str(args.output.resolve()), "window": manifest["window"],
                      "nodes": len(manifest["nodes"]), "totals": manifest["totals"]},
                     ensure_ascii=False, indent=2))


def _replay(args):
    from .replay import ReplaySession

    if (not math.isfinite(args.step_seconds) or not math.isfinite(args.speed)
            or round(args.step_seconds * 1000) <= 0 or args.speed < 0):
        raise ValueError("Step must be positive; speed must be nonnegative (0 = no waiting)")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    expected = manifest["totals"]["deliverable"]
    total = 0
    by_modality = Counter()
    by_node = Counter()
    ticks = 0
    started = time.monotonic()
    try:
        with ReplaySession(args.prepared, inbox_path=output / "inbox.sqlite") as session:
            with (output / "ticks.jsonl").open("w", encoding="utf-8") as ticks_file:
                for batch in session.iter_ticks(round(args.step_seconds * 1000)):
                    ticks += 1
                    events = batch["events"]
                    current_nodes = Counter()
                    current_modalities = Counter()
                    for event in events:
                        if not event["available_at_ms"] < batch["clock_ms"]:
                            raise RuntimeError("Future observation was delivered")
                        by_node[event["node_id"]] += 1
                        by_modality[event["modality"]] += 1
                        current_nodes[event["node_id"]] += 1
                        current_modalities[event["modality"]] += 1
                    total += len(events)
                    receipt = {
                        "clock_ms": batch["clock_ms"],
                        "clock": format_time(batch["clock_ms"]),
                        "delivered": len(events), "cumulative": total,
                        "by_node": dict(current_nodes), "by_modality": dict(current_modalities),
                    }
                    ticks_file.write(json.dumps(receipt, ensure_ascii=False) + "\n")
                    ticks_file.flush()
                    print(f"{receipt['clock']}  received={len(events)}  cumulative={total}",
                          flush=True)
                    if args.speed:
                        simulated = (batch["clock_ms"] - manifest["window"]["start_ms"]) / 1000
                        remaining = simulated / args.speed - (time.monotonic() - started)
                        while remaining > 0:
                            time.sleep(min(remaining, 1.0))
                            remaining = simulated / args.speed - (time.monotonic() - started)
            if total != expected:
                raise RuntimeError(f"Delivery count mismatch: {total} != {expected}")
            examples = []
            for node in manifest["nodes"]:
                examples.extend(session.query(node["node_id"], limit=1))
            (output / "received_examples.json").write_text(
                json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            summary = {
                "status": "complete", "prepared": str(args.prepared.resolve()),
                "window": manifest["window"], "clock_ms": session.clock_ms,
                "step_seconds": args.step_seconds, "ticks": ticks,
                "prepared_records": manifest["totals"]["prepared"],
                "delivered_records": total, "pending_records": manifest["totals"]["pending"],
                "by_modality": dict(by_modality), "by_node": dict(by_node),
                "checks": {"delivery_count_matches": total == expected,
                           "no_future_records_delivered": True,
                           "raw_data_filtered_by_anomaly": False,
                           "models_loaded": False, "labels_used": False},
                "wall_seconds": round(time.monotonic() - started, 3),
                "inbox": str(output / "inbox.sqlite"),
            }
            (output / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException as error:
        (output / "failure.json").write_text(
            json.dumps({"status": "failed", "error": str(error),
                        "delivered_before_failure": total}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Split and sort raw Bank data")
    prepare.add_argument("--telemetry-root", type=Path,
                         default=ROOT.parent / "OpenRCA/dataset/Bank/telemetry")
    prepare.add_argument("--query-file", type=Path,
                         default=ROOT.parent / "OpenRCA/dataset/Bank/query.csv")
    prepare.add_argument("--query-row", type=int, default=0, help="Zero-based row, time interval only")
    prepare.add_argument("--start", help="Explicit local datetime instead of query interval")
    prepare.add_argument("--end", help="Exclusive end datetime")
    prepare.add_argument("--output", required=True, type=Path)
    prepare.set_defaults(run=_prepare)
    replay = commands.add_parser("replay", help="Advance one shared clock and populate node inboxes")
    replay.add_argument("--prepared", required=True, type=Path)
    replay.add_argument("--output", required=True, type=Path)
    replay.add_argument("--step-seconds", type=float, default=10)
    replay.add_argument("--speed", type=float, default=0,
                        help="0 = fast simulation; otherwise simulated seconds per wall second")
    replay.set_defaults(run=_replay)
    alerts = commands.add_parser("alerts", help="Replay through CPU alert rules; no models")
    alerts.add_argument("--prepared", required=True, type=Path)
    alerts.add_argument("--output", required=True, type=Path)
    alerts.add_argument("--step-seconds", type=float, default=10)
    alerts.add_argument("--config", type=Path, help="JSON AlertConfig overrides")
    from .alert_run import run_alerts

    alerts.set_defaults(run=run_alerts)
    tool_check = commands.add_parser("tool-check", help="Validate fixed CPU tools and trace latency on replay; no model")
    tool_check.add_argument("--prepared", required=True, type=Path)
    tool_check.add_argument("--output", required=True, type=Path)
    tool_check.add_argument("--checkpoints-seconds", default="120,600,1800")
    from .tool_check import run_tool_check

    tool_check.set_defaults(run=run_tool_check)
    node_demo = commands.add_parser("node-demo", help="Export synthetic node inputs/prompts and test state; no model")
    node_demo.add_argument("--output", required=True, type=Path)
    from .node_demo import run_node_demo

    node_demo.set_defaults(run=lambda args: run_node_demo(args.output))
    args = parser.parse_args()
    if getattr(args, "query_row", 0) < 0:
        parser.error("Query row must be nonnegative")
    args.run(args)


if __name__ == "__main__":
    main()
