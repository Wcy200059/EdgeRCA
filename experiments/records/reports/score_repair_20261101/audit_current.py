"""Read-only audit of the completed baseline cases and saved final replies."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

from edge_rca.case_runtime import register_displayed_briefs, validate_final_answers
from edge_rca.coordinator import occurrence_ms
from edge_rca.data import format_time
from edge_rca.node_probe import decode_object
from edge_rca.query import parse_failure_count
from edge_rca.scoring import evaluate


ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    source = ROOT.parents[2] / "OpenRCA/main/evaluate.py"
    spec = importlib.util.spec_from_file_location("official", source)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    rows, mismatches, errors = [], [], Counter()
    for row in manifest["completed_rows_at_start"]:
        case = ROOT.parent / f"bank_case_{154057 if row < 46 else 154058}_{row}"
        report = json.loads((case / "evaluation/result.json").read_text())
        old = json.loads((ROOT.parent / "rollback_20261030/cases" / str(row) / "evaluation/result.json").read_text())
        item = {"row": row, "task": report["task_index"], "valid": report["final_reply_valid"],
                "scores": {}, "prior_scores": {}}
        for tolerance in (60, 300):
            item["scores"][str(tolerance)] = evaluate(json.dumps(report["prediction"]), report["scoring_points"],
                                                     time_tolerance_seconds=tolerance)[2]
            item["prior_scores"][str(tolerance)] = evaluate(json.dumps(old["prediction"]), old["scoring_points"],
                                                           time_tolerance_seconds=tolerance)[2]
        if item["scores"]["60"] != official.evaluate(json.dumps(report["prediction"]), report["scoring_points"])[2]:
            mismatches.append(row)
        final = json.loads((case / "coordinator_1800/final_answer.json").read_text())
        errors.update(attempt["error"] for attempt in final["attempts"] if not attempt["valid"])
        if not item["valid"]:
            full = json.loads((case / "coordinator_1800/last_input.json").read_text())["view"]
            original = {fact["id"]: fact for fact in full["evidence"]}
            item["revalidated"] = []
            for line in (case / "coordinator_1800/calls.jsonl").open():
                call = json.loads(line)
                if not call["phase"].startswith("final_commit/"):
                    continue
                payload = json.JSONDecoder().raw_decode(call["messages"][1]["content"])[0]
                payload["evidence"] = [dict(zip(("id", "node_id", "tool_fact", "time_marks"), fact)) for fact in payload["evidence"]]
                system = call["messages"][0]["content"]
                candidates = json.JSONDecoder().raw_decode(system.split("再输出：", 1)[1])[0] if "再输出：" in system else []
                view = register_displayed_briefs(payload, candidates, original)
                try:
                    reply = validate_final_answers(decode_object(call["raw_reply"]), view, parse_failure_count(report["instruction"]))
                    predictions = []
                    for fault in reply["faults"]:
                        stamp = occurrence_ms(fault["time"], view["window_start_ms"], view["cutoff_ms"])
                        predictions.append({"root cause occurrence datetime": format_time(stamp)[:19].replace("T", " ") if stamp is not None else "",
                                            "root cause component": fault["component"] or "", "root cause reason": fault["reason"] or ""})
                    item["revalidated"].append({"phase": call["phase"], "accepted": True,
                        "scores": {str(t): evaluate(json.dumps(predictions), report["scoring_points"], time_tolerance_seconds=t)[2] for t in (60, 300)},
                        "raw_reply_sha256": hashlib.sha256(call["raw_reply"].encode()).hexdigest()})
                except (ValueError, KeyError, TypeError) as error:
                    item["revalidated"].append({"phase": call["phase"], "accepted": False, "error": str(error)})
        rows.append(item)
    totals = {}
    for key in ("scores", "prior_scores"):
        totals[key] = {str(t): {"full": sum(r[key][str(t)] == 1 for r in rows),
                               "any": sum(r[key][str(t)] > 0 for r in rows),
                               "mean": sum(r[key][str(t)] for r in rows) / len(rows)} for t in (60, 300)}
    result = {"completed_at_start": len(rows), "uncompleted_at_start": sorted(set(range(136)) - {r["row"] for r in rows}),
              "matched_totals": totals, "official_scoring_mismatches": mismatches,
              "validation_errors": dict(errors), "rows": rows,
              "revalidation_scope": "Unchanged saved model outputs; no rerun or modification of baseline scores."}
    (ROOT / "current_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("completed_at_start", "matched_totals", "validation_errors")}, indent=2))
    for row in rows:
        if "revalidated" in row:
            print(row["row"], row["revalidated"])


if __name__ == "__main__":
    main()
