"""CPU analysis: does the final 9B candidate pool contain the truth?"""

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from edge_rca.query import parse_failure_count
from edge_rca.scoring import COMPONENT, REASON, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default="143858")
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    rows = []
    for directory in sorted(glob.glob(str(args.root / f"bank_case_{args.job}_*")),
                            key=lambda value: int(value.rsplit("_", 1)[-1])):
        row = int(directory.rsplit("_", 1)[-1]); path = Path(directory)
        try:
            question = json.loads((path / "frames/public_question.json").read_text())
            with Path(question["query_file"]).open(encoding="utf-8-sig", newline="") as handle:
                item = next(item for index, item in enumerate(csv.DictReader(handle)) if index == question["row"])
            prediction = json.loads((path / "coordinator_1800/prediction.json").read_text())
            passed, failed, score = evaluate(json.dumps(prediction), item["scoring_points"], time_tolerance_seconds=300)
        except (FileNotFoundError, StopIteration, ValueError, KeyError, json.JSONDecodeError):
            continue
        truth_components, truth_reasons = set(COMPONENT.findall(item["scoring_points"])), set(REASON.findall(item["scoring_points"]))
        hypotheses = []
        for decision in glob.glob(str(path / "coordinator_1800/decision_*.json")):
            reply = json.loads(Path(decision).read_text()).get("reply") or {}
            hypotheses.extend(reply.get("hypotheses", []))
        components = [h.get("component") for h in hypotheses if h.get("component")]
        reasons = [h.get("reason") for h in hypotheses if h.get("reason")]
        supported = [h for h in hypotheses if h.get("assessment") == "supported" and h.get("component")]
        supported.sort(key=lambda h: len(h.get("support_ids", [])), reverse=True)
        top3_components = []
        for h in supported + [h for h in hypotheses if h.get("component")]:
            if h["component"] not in top3_components:
                top3_components.append(h["component"])
            if len(top3_components) >= 3:
                break
        first = {p.get("root cause component") for p in prediction}
        rows.append({"row": row, "score": score,
                     "component_recall": bool(truth_components & set(components)),
                     "reason_recall": bool(truth_reasons & set(reasons)),
                     "supported_recall": bool(truth_components & {h["component"] for h in supported}),
                     "top3_component_recall": bool(truth_components & set(top3_components)),
                     "first_component": bool(truth_components & first),
                     "time_only": "only root cause occurrence time" in item["scoring_points"].lower()
                     and "component" not in item["scoring_points"].lower(),
                     "two_fault": parse_failure_count(item["instruction"]) == 2})
    summary = {"evaluated": len(rows),
               "component_recall": sum(r["component_recall"] for r in rows) / len(rows),
               "reason_recall": sum(r["reason_recall"] for r in rows) / len(rows),
               "supported_component_recall": sum(r["supported_recall"] for r in rows) / len(rows),
               "top3_component_recall": sum(r["top3_component_recall"] for r in rows) / len(rows),
               "first_component_correct": sum(r["first_component"] for r in rows) / len(rows),
               "component_recall_given_not_first": sum(r["component_recall"] and not r["first_component"] for r in rows)}
    by_type = defaultdict(list)
    for r in rows:
        by_type["time_only" if r["time_only"] else "two_fault" if r["two_fault"] else "single_other"].append(r)
    summary["by_type"] = {k: {"n": len(v),
                              "component_recall": sum(x["component_recall"] for x in v) / len(v),
                              "top3_recall": sum(x["top3_component_recall"] for x in v) / len(v),
                              "first_component": sum(x["first_component"] for x in v) / len(v)}
                          for k, v in by_type.items()}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
