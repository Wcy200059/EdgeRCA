"""Deterministic recovery from frozen model replies, followed by separate scoring."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from edge_rca.case_runtime import retain_valid_fields_with_unknown_time
from edge_rca.coordinator import occurrence_ms
from edge_rca.data import format_time
from edge_rca.node_probe import decode_object


ROOT = Path(__file__).resolve().parent


def repair():
    if not (ROOT / "coverage_complete.json").exists():
        raise ValueError("Model runs must finish first")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    completed = []
    for row in manifest["rows"]:
        for parent in ("candidate", "coverage"):
            original = ROOT / "replays" / str(row) / parent
            final = json.loads((original / "final_answer.json").read_text())
            view = json.loads((original / "final_input.json").read_text())["view"]
            result, changes = final["reply"], []
            if result is None:
                calls = [json.loads(line) for line in (original / "calls.jsonl").read_text().splitlines()]
                for call in reversed(calls):
                    try:
                        result, changes = retain_valid_fields_with_unknown_time(
                            decode_object(call["raw_reply"]), view, final["expected_failures"])
                    except (ValueError, KeyError, TypeError):
                        continue
                    break
            prediction = json.loads((original / "prediction.json").read_text())
            if changes:
                prediction = []
                for fault in result["faults"]:
                    stamp = occurrence_ms(fault["time"], view["window_start_ms"], view["cutoff_ms"])
                    prediction.append({"root cause occurrence datetime": format_time(stamp)[:19].replace("T", " ") if stamp is not None else "",
                                       "root cause component": fault["component"] or "", "root cause reason": fault["reason"] or ""})
            output = ROOT / "partial_replays" / str(row) / parent
            output.mkdir(parents=True, exist_ok=False)
            (output / "prediction.json").write_text(json.dumps(prediction, ensure_ascii=False, indent=2) + "\n")
            record = {"row": row, "parent_variant": parent, "valid": result is not None,
                      "reply": result, "field_repairs": changes, "labels_used": False,
                      "original_calls_sha256": hashlib.sha256((original / "calls.jsonl").read_bytes()).hexdigest()}
            (output / "repair.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            completed.append({"row": row, "parent_variant": parent, "repaired": bool(changes)})
    (ROOT / "partial_complete.json").write_text(json.dumps({"completed": completed, "labels_used": False}, indent=2) + "\n")


def evaluate():
    if not (ROOT / "partial_complete.json").exists():
        raise ValueError("Freeze recovered outputs before scoring")
    from edge_rca.scoring import evaluate as score
    manifest = json.loads((ROOT / "manifest.json").read_text())
    records = []
    for row in manifest["rows"]:
        case = ROOT.parent / f"bank_case_{154057 if row < 46 else 154058}_{row}"
        truth = json.loads((case / "evaluation/result.json").read_text())["scoring_points"]
        for parent in ("candidate", "coverage"):
            output = ROOT / "partial_replays" / str(row) / parent
            prediction = (output / "prediction.json").read_bytes()
            repair = json.loads((output / "repair.json").read_text())
            records.append({"row": row, "variant": parent + "_partial", "valid": repair["valid"],
                "repaired": bool(repair["field_repairs"]), "prediction_sha256": hashlib.sha256(prediction).hexdigest(),
                "scores": {str(t): score(prediction.decode(), truth, time_tolerance_seconds=t)[2] for t in (60, 300)}})
    summary = {}
    for parent in ("candidate", "coverage"):
        selected = [r for r in records if r["variant"] == parent + "_partial"]
        summary[parent + "_partial"] = {"n": len(selected), "valid": sum(r["valid"] for r in selected),
            "scores": {str(t): {"full": sum(r["scores"][str(t)] == 1 for r in selected),
                                "mean": sum(r["scores"][str(t)] for r in selected) / len(selected)} for t in (60, 300)}}
    (ROOT / "partial_scores.json").write_text(json.dumps({"summary": summary, "rows": records,
        "scope": "Deterministic postprocessing of unchanged frozen model replies, not a new model run."}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("repair", "evaluate"))
    args = parser.parse_args()
    (repair if args.phase == "repair" else evaluate)()
