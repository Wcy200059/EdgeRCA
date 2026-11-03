"""Build a read-only JSON trace for the EdgeRCA node/coordinator dashboard.

Reads one completed offline case directory (frames, node reports, coordinator
decisions, final answer, evaluation) and writes a compact JSON file.  It never
calls a model, reads labels into any model input, or mutates a case.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .trace_story import enrich_trace
    from .trace_pipeline import enrich_pipeline
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.trace_story import enrich_trace
    from tools.trace_pipeline import enrich_pipeline


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return default


def short(value, limit=420):
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "…"
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def node_short(node: str) -> str:
    return node.split(":", 1)[-1]


def iso_ms(ms):
    try:
        from datetime import datetime, timezone, timedelta
        tz = timezone(timedelta(hours=8))
        return datetime.fromtimestamp(ms / 1000, tz).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return ""


def raw_overview(case: Path) -> dict:
    """Aggregate the raw inbox without scanning the 100–500MB payload per node."""
    db = case / "frames" / "inbox.sqlite"
    result = {"total": 0, "by_node": {}, "by_modality": {}, "by_node_modality": {},
              "by_minute": [], "samples": [], "time_min": None, "time_max": None}
    if not db.exists():
        return result
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        result["total"] = cur.execute("select count(*) from events").fetchone()[0]
        result["time_min"], result["time_max"] = cur.execute(
            "select min(observed_at_ms), max(observed_at_ms) from events").fetchone()
        by_minute = defaultdict(lambda: defaultdict(int))
        for node, modality, minute, count in cur.execute(
                "select node_id, modality, (observed_at_ms/60000)*60000 as m, count(*) "
                "from events group by node_id, modality, m"):
            result["by_node"][node] = result["by_node"].get(node, 0) + count
            result["by_modality"][modality] = result["by_modality"].get(modality, 0) + count
            result["by_node_modality"].setdefault(node, {})[modality] = \
                result["by_node_modality"].get(node, {}).get(modality, 0) + count
            by_minute[minute][node] += count
        result["by_minute"] = [
            {"minute": int(m), "time": iso_ms(int(m)), "count": sum(v.values()), "by_node": dict(v)}
            for m, v in sorted(by_minute.items())]
        # One sequential scan; parse payload only for the first three samples per node/modality.
        seen = Counter()
        for node, modality, observed, payload in cur.execute(
                "select node_id, modality, observed_at_ms, payload from events order by observed_at_ms"):
            key = (node, modality)
            if seen[key] >= 3:
                continue
            seen[key] += 1
            try:
                event = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                event = {}
            raw = event.get("raw", event)
            raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            result["samples"].append({
                "node": node_short(node), "node_id": event.get("node_id", node), "modality": modality,
                "observed_ms": observed, "observed": iso_ms(observed),
                "source_file": event.get("source_file"),
                "source_row": event.get("source_row"),
                "event_id": event.get("event_id"),
                "raw": short(raw, 360), "raw_truncated": len(raw_text) > 360,
            })
    finally:
        con.close()
    return result

def node_frames(case: Path) -> dict:
    out = {}
    for path in sorted((case / "frames" / "c1800").glob("device_*.json")):
        data = read_json(path, {}) or {}
        node = node_short(data.get("node_id", path.stem))
        facts = data.get("facts") or []
        kinds = Counter(f.get("kind", "?") for f in facts)
        samples = []
        for fact in facts[:6]:
            samples.append({
                "id": fact.get("id"), "kind": fact.get("kind"), "signal": fact.get("signal"),
                "state": fact.get("state"), "timeline": fact.get("timeline", [])[:4],
                "whole": fact.get("whole_prefix"), "first": fact.get("first_trigger_values"),
                "meaning": short(fact.get("meaning"), 180), "unit": fact.get("metric_unit"),
            })
        out[node] = {
            "node_id": data.get("node_id", f"device:{node}"),
            "facts": len(facts), "kinds": dict(kinds),
            "scope": data.get("scope", {}), "samples": samples,
            "limitations": data.get("limitations", [])[:3],
            "cpu_only_metric_count": data.get("scope", {}).get("cpu_only_metric_count"),
        }
    return out


def node_reports(case: Path) -> dict:
    out = {}
    for manifest in sorted((case / "nodes_1800").glob("device_*/manifest.json")):
        node_dir = manifest.parent
        report = read_json(node_dir / "report.json", {}) or {}
        manifest_data = read_json(manifest, {}) or {}
        node = node_short(report.get("node_id", node_dir.name.replace("device_", "device:")))
        rows = []
        for row in report.get("template_rows", []):
            fields = row.get("fields") or {}
            selection = row.get("model_selection") or {}
            rows.append({
                "id": row.get("id"), "subject": row.get("subject"), "pattern": row.get("pattern"),
                "protected": bool(row.get("protected")), "use": selection.get("use"),
                "action": row.get("catalogue_action"), "handling": row.get("handling"),
                "meaning": short(fields.get("meaning"), 160), "unit": fields.get("metric_unit"),
                "whole": fields.get("whole_prefix"), "first": fields.get("first_trigger_values"),
                "extremes": (fields.get("extremes") or [])[:3], "timeline": (fields.get("timeline") or [])[:3],
            })
        counts = Counter(r["use"] or r["action"] or "unprocessed" for r in rows)
        out[node] = {
            "node_id": report.get("node_id", manifest_data.get("node_id")),
            "overview": short(report.get("overview_text"), 520),
            "template_rows": rows, "counts": dict(counts),
            "reported": int(manifest_data.get("reported", 0) or 0),
            "executed_self_requests": manifest_data.get("executed_self_requests", 0),
            "stop_reason": manifest_data.get("stop_reason"),
            "node_mode": report.get("node_mode"),
            "questions": (report.get("questions") or [])[:3],
            "deferred": (report.get("defer") or [])[:3],
        }
    return out


def node_calls(case: Path):
    by_node = Counter()
    by_phase = Counter()
    path = case / "nodes_1800" / "calls.jsonl"
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            node = item.get("node_id") or item.get("node") or "?"
            by_node[node_short(node)] += 1
            by_phase[item.get("phase", "?")] += 1
    return dict(by_node), dict(by_phase)


def evidence_catalog(case: Path):
    data = read_json(case / "coordinator_1800" / "input_0.json", {}) or {}
    view = data.get("view") or {}
    evidence = view.get("evidence") or []
    by_node = Counter()
    items = []
    for item in evidence:
        node = node_short(item.get("node_id", "?"))
        by_node[node] += 1
        items.append({
            "id": item.get("id"), "node": node,
            "fact": short(item.get("tool_fact"), 300),
            "time_marks": item.get("time_marks", [])[:4],
        })
    return {
        "evidence": items, "by_node": dict(by_node),
        "node_status": view.get("node_status", []),
        "candidate_coverage": view.get("candidate_coverage"),
        "candidate_slate": view.get("candidate_slate"),
        "node_summaries": view.get("node_summaries"),
        "scope": view.get("scope"),
        "system_context": view.get("system_context"),
    }


def coordinator(case: Path):
    base = case / "coordinator_1800"
    decisions, hypotheses, requests = [], [], []
    for path in sorted(base.glob("decision_*.json")):
        data = read_json(path, {}) or {}
        reply = data.get("reply") or {}
        index = path.stem.split("_")[-1]
        for h in reply.get("hypotheses", []):
            hypotheses.append({**h, "round": index})
        for r in reply.get("requests", []):
            requests.append({**r, "round": index})
        decisions.append({
            "round": index, "reply": reply,
            "attempts": data.get("attempts", []), "file": path.name,
        })
    final = read_json(base / "final_answer.json", {}) or {}
    complete = read_json(base / "complete.json", {}) or {}
    calls = []
    call_path = base / "calls.jsonl"
    if call_path.exists():
        for line in call_path.read_text(errors="replace").splitlines():
            try:
                calls.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    followups = []
    for d in sorted(base.glob("node_followup_*")):
        manifest = read_json(d / "manifest.json", {}) or {}
        report = read_json(d / "report.json", {}) or {}
        followups.append({
            "dir": d.name, "node_id": report.get("node_id") or manifest.get("node_id"),
            "requests": manifest.get("requests"), "files": manifest.get("files"),
            "overview": short(report.get("overview_text"), 260),
        })
    return {
        "decisions": decisions, "hypotheses": hypotheses, "requests": requests,
        "final": final, "complete": complete,
        "calls": len(calls), "followups": followups,
    }


def build_case(case: Path) -> dict:
    question = read_json(case / "frames" / "public_question.json", {}) or {}
    raw = raw_overview(case)
    frames = node_frames(case)
    reports = node_reports(case)
    calls, phases = node_calls(case)
    central = evidence_catalog(case)
    coord = coordinator(case)
    global_filter = read_json(case / "coordinator_1800" / "global_filter.json", {}) or {}
    gf = global_filter.get("filter") or {}
    global_before = gf.get("available")
    global_after = (len(gf.get("ids", [])) if not gf.get("fallback") else global_before)
    global_filter_info = {"before": {"count": global_before},
                          "after": {"count": global_after},
                          "filter": gf,
                          "file": "coordinator_1800/global_filter.json"}
    evaluation = read_json(case / "evaluation" / "result.json", {}) or {}
    nodes = sorted(set(frames) | set(reports))
    final_faults = ((coord["final"].get("reply") or {}).get("faults") or [])
    final_components = {f.get("component") for f in final_faults}
    evidence_node = {e["id"]: e["node"] for e in central["evidence"]}
    evidence_by_node = Counter(evidence_node.values())
    hypothesis_components = Counter(h.get("component") for h in coord["hypotheses"] if h.get("component"))
    node_rows = []
    for node in nodes:
        rep = reports.get(node, {})
        rows = rep.get("template_rows", [])
        selected = [r for r in rows if r.get("use") in ("report", "uncertain")]
        node_rows.append({
            "node": node,
            "raw": raw.get("by_node", {}).get(f"device:{node}", 0),
            "facts": frames.get(node, {}).get("facts", 0),
            "fact_kinds": frames.get(node, {}).get("kinds", {}),
            "report_rows": len(rows),
            "selected": len(selected),
            "reported": rep.get("reported", 0),
            "evidence": evidence_by_node.get(node, 0),
            "hypotheses": hypothesis_components.get(node, 0),
            "final": node in final_components,
            "calls": calls.get(node, 0),
            "overview": rep.get("overview"),
            "stop_reason": rep.get("stop_reason"),
            "template_rows": rows,
            "facts_sample": frames.get(node, {}).get("samples", []),
            "raw_by_modality": raw.get("by_node_modality", {}).get(f"device:{node}", {}),
            "raw_samples": [s for s in raw.get("samples", []) if s["node"] == node],
        })
    data = {
        "case": {"path": str(case), "row": question.get("row"), "instruction": question.get("instruction")},
        "raw": {
            "total": raw.get("total", 0), "by_modality": raw.get("by_modality", {}),
            "by_node": raw.get("by_node", {}), "by_minute": raw.get("by_minute", []),
            "device_total": sum(v for k, v in raw.get("by_node", {}).items() if str(k).startswith("device:")),
            "business_total": sum(v for k, v in raw.get("by_node", {}).items() if not str(k).startswith("device:")),
            "samples": raw.get("samples", []), "time_min": raw.get("time_min"),
            "time_max": raw.get("time_max"),
        },
        "nodes": node_rows,
        "central": {
            "evidence": central["evidence"], "by_node": central["by_node"],
            "node_status": central["node_status"],
            "candidate_coverage": central["candidate_coverage"],
            "candidate_slate": central["candidate_slate"],
            "scope": central["scope"], "system_context": central["system_context"],
            "model_calls": coord["calls"], "followups": coord["followups"],
            "decisions": coord["decisions"], "hypotheses": coord["hypotheses"],
            "requests": coord["requests"], "final": coord["final"],
            "complete": coord["complete"],
            "global_filter": global_filter_info,
        },
        "evaluation": {
            "score": evaluation.get("score"), "fully_correct": evaluation.get("fully_correct"),
            "any_field_correct": evaluation.get("any_field_correct"),
            "final_reply_valid": evaluation.get("final_reply_valid"),
            "passed": evaluation.get("passed", []), "failed": evaluation.get("failed", []),
            "scoring_points": evaluation.get("scoring_points"),
            "prediction": evaluation.get("prediction"),
            "model_calls": evaluation.get("model_calls"),
            "delivered_rows": evaluation.get("delivered_rows"),
            "time_tolerance_seconds": evaluation.get("time_tolerance_seconds"),
        },
        "flow": {
            "raw": raw.get("total", 0),
            "facts": sum(n["facts"] for n in node_rows),
            "report_rows": sum(n["report_rows"] for n in node_rows),
            "selected": sum(n["selected"] for n in node_rows),
            "evidence": len(central["evidence"]),
            "hypotheses": len(coord["hypotheses"]),
            "requests": len(coord["requests"]),
            "final": len(final_faults),
            "batch_pages": sum(len(n.get("batch_pages", [])) for n in (reports.values())),
            "global_before": global_before,
            "global_after": global_after,
        },
        "generated_by": "tools/build_trace_view.py",
    }
    return enrich_pipeline(case, enrich_trace(case, data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, action="append", required=True,
                        help="completed case directory; repeat for multiple cases")
    parser.add_argument("--out", type=Path, default=Path("web/trace/data"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index = {"generated_by": "tools/build_trace_view.py", "cases": []}
    for case in args.case:
        data = build_case(case.resolve())
        row = data["case"].get("row")
        slug = f"row_{row}" if row is not None else case.name
        name = f"case_{slug}.json"
        (args.out / name).write_text(json.dumps(data, ensure_ascii=False, indent=1))
        index["cases"].append({
            "name": name, "row": row, "path": str(case),
            "label": f"row {row} · {((data['evaluation'].get('prediction') or [{}])[0].get('root cause component') or '?')}",
            "score": data["evaluation"].get("score"),
            "raw": data["flow"]["raw"], "nodes": len(data["nodes"]),
        })
    if index["cases"]:
        index["default"] = index["cases"][0]["name"]
    (args.out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))
    print(json.dumps(index, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
