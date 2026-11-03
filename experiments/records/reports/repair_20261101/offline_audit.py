"""Replay saved validators and independently check scores; never call a model."""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

from edge_rca.case_runtime import validate_final_answers
from edge_rca.coordinator import occurrence_ms
from edge_rca.node_probe import decode_object
from edge_rca.query import parse_failure_count
from edge_rca.scoring import evaluate
from edge_rca.data import format_time


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "rollback_20261030"
OFFICIAL = ROOT.parents[2] / "OpenRCA/main/evaluate.py"


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def predictions(reply, view):
    result = []
    for fault in reply["faults"]:
        when = ""
        if fault["time"]:
            ms = occurrence_ms(fault["time"], view["window_start_ms"], view["cutoff_ms"])
            when = format_time(ms)[:19].replace("T", " ")
        result.append({"root cause occurrence datetime": when,
                       "root cause component": fault["component"] or "",
                       "root cause reason": fault["reason"] or ""})
    return result


def main():
    spec = importlib.util.spec_from_file_location("official_bank_evaluate", OFFICIAL)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    mismatches, replays = [], []
    groups = defaultdict(lambda: {"count": 0, "fully_correct": 0,
                                  "any_field_correct": 0, "invalid_replies": 0})
    count = 0
    for path in sorted(BASELINE.glob("cases/*/evaluation/result_60s.json")):
        report = json.loads(path.read_text())
        count += 1
        predicted = json.dumps(report["prediction"])
        expected = official.evaluate(predicted, report["scoring_points"])
        current = evaluate(predicted, report["scoring_points"], time_tolerance_seconds=60)
        if (expected[2] != report["score"] or expected[2] != current[2]
                or set(expected[0]) != set(current[0]) or set(expected[1]) != set(current[1])):
            mismatches.append({"row": report["row"], "saved": report["score"],
                               "official": expected, "current": current})
        group = groups[report["task_index"]]
        group["count"] += 1
        group["fully_correct"] += report["fully_correct"]
        group["any_field_correct"] += report["any_field_correct"]
        group["invalid_replies"] += not report["final_reply_valid"]
        if report["final_reply_valid"]:
            continue
        call_path = path.parents[1] / "coordinator_1800/calls.jsonl"
        accepted, errors = [], []
        for line in call_path.open():
            call = json.loads(line)
            if not call["phase"].startswith("final_commit/"):
                continue
            view = json.JSONDecoder().raw_decode(call["messages"][1]["content"])[0]
            view["evidence"] = [dict(zip(("id", "node_id", "tool_fact", "time_marks"), row))
                                for row in view["evidence"]]
            try:
                value = validate_final_answers(decode_object(call["raw_reply"]), view,
                                               parse_failure_count(report["instruction"]))
                prediction = predictions(value, view)
                scores = {str(t): evaluate(json.dumps(prediction), report["scoring_points"],
                                          time_tolerance_seconds=t)[2] for t in (60, 300)}
                accepted.append({"phase": call["phase"], "unchanged_raw_reply": value,
                                 "raw_reply_sha256": hashlib.sha256(call["raw_reply"].encode()).hexdigest(),
                                 "offline_reply_scores": scores})
            except (ValueError, KeyError, TypeError) as error:
                errors.append({"phase": call["phase"], "error": str(error)})
        replays.append({"row": report["row"], "accepted_after_validator_repairs": bool(accepted),
                        "accepted": accepted, "errors_remaining": errors})
    save("scoring_parity.json", {"checked": count, "official_source": str(OFFICIAL),
                                "official_source_sha256": hashlib.sha256(OFFICIAL.read_bytes()).hexdigest(),
                                "mismatches": mismatches, "baseline_by_task": dict(sorted(groups.items()))})
    save("saved_reply_revalidation.json", {
        "scope": "Offline validation replay of unchanged saved replies; no model rerun or change to baseline scores.",
        "source_sha256": json.loads((ROOT / "source_manifest.json").read_text())["source_sha256"],
        "rows": replays})
    print(json.dumps({"scoring_checked": count, "scoring_mismatches": len(mismatches),
                      "accepted_saved_rows": [r["row"] for r in replays if r["accepted_after_validator_repairs"]]}))


if __name__ == "__main__":
    main()
