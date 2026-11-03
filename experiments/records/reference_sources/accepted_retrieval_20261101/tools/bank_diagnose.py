"""Stage-oriented diagnosis of a Bank batch, not just a total score."""

import argparse
import csv
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from edge_rca.query import parse_failure_count
from edge_rca.scoring import COMPONENT, REASON, TIME, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--job", required=True)
    parser.add_argument("--time-tolerance-seconds", type=int, default=300)
    args = parser.parse_args()
    rows, missing = [], []
    for directory in sorted(glob.glob(str(args.root / f"bank_case_{args.job}_*")),
                            key=lambda value: int(value.rsplit("_", 1)[-1])):
        row = int(directory.rsplit("_", 1)[-1])
        path = Path(directory)
        try:
            question = json.loads((path / "frames/public_question.json").read_text())
            with Path(question["query_file"]).open(encoding="utf-8-sig", newline="") as handle:
                item = next(item for index, item in enumerate(csv.DictReader(handle)) if index == question["row"])
            prediction = json.loads((path / "coordinator_1800/prediction.json").read_text())
            passed, failed, score = evaluate(json.dumps(prediction), item["scoring_points"],
                                             time_tolerance_seconds=args.time_tolerance_seconds)
        except (FileNotFoundError, StopIteration, ValueError, KeyError, json.JSONDecodeError):
            missing.append(row)
            continue
        truth_components = COMPONENT.findall(item["scoring_points"])
        truth_reasons = REASON.findall(item["scoring_points"])
        truth_times = TIME.findall(item["scoring_points"])
        hypotheses, queried = [], set()
        for decision in glob.glob(str(path / "coordinator_1800/decision_*.json")):
            reply = json.loads(Path(decision).read_text()).get("reply") or {}
            hypotheses.extend(reply.get("hypotheses", []))
            queried.update(request.get("node_id") for request in reply.get("requests", []))
        for request in glob.glob(str(path / "coordinator_1800/queries/*/request.json")):
            queried.add(json.loads(Path(request).read_text()).get("request", {}).get("node_id"))
        hypothesis_components = {h.get("component") for h in hypotheses}
        hypothesis_reasons = {h.get("reason") for h in hypotheses}
        predicted_components = {p.get("root cause component") for p in prediction}
        truth_nodes = {f"device:{c}" for c in truth_components}
        if score == 1.0:
            stage = "correct"
        elif not any(any(p.get(key) for key in ("root cause component", "root cause reason", "root cause occurrence datetime"))
                     for p in prediction):
            stage = "blank_final"
        elif not set(truth_components) & hypothesis_components:
            stage = "candidate_not_hypothesized"
        elif truth_nodes and not truth_nodes & queried and not truth_nodes & hypothesis_components:
            stage = "candidate_not_investigated"
        elif not set(truth_components) & predicted_components:
            stage = "final_component_selection"
        elif not set(truth_reasons) & hypothesis_reasons:
            stage = "reason_not_hypothesized"
        elif any("datetime" in str(f) or re.match(r"\d{4}-", str(f)) for f in failed):
            stage = "time_selection"
        elif not set(truth_reasons) & {p.get("root cause reason") for p in prediction}:
            stage = "final_reason_selection"
        else:
            stage = "other"
        coverage = []
        try:
            view = json.loads((path / "coordinator_1800/input_0.json").read_text()).get("view", {})
            coverage = view.get("candidate_coverage", [])
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        truth_coverage = [item for item in coverage if item.get("component") in truth_components]
        rows.append({"row": row, "score": score, "failed": failed, "stage": stage,
                     "time_only": "only root cause occurrence time" in item["scoring_points"].lower()
                     and "component" not in item["scoring_points"].lower(),
                     "two_fault": parse_failure_count(item["instruction"]) == 2,
                     "truth": truth_components, "predicted": sorted(x for x in predicted_components if x),
                     "hypotheses": sorted(x for x in hypothesis_components if x),
                     "queried_nodes": sorted(x for x in queried if x),
                     "truth_coverage": truth_coverage,
                     "model_calls": json.loads((path / "evaluation/result.json").read_text()).get("model_calls")
                     if (path / "evaluation/result.json").exists() else None})
    stages = Counter(item["stage"] for item in rows)
    by_type = defaultdict(list)
    for item in rows:
        kind = "time_only" if item["time_only"] else "two_fault" if item["two_fault"] else "single_other"
        by_type[kind].append(item["score"])
    output = args.root / f"bank_diagnose_{args.job}"
    output.mkdir(parents=True, exist_ok=False)
    summary = {"job": args.job, "evaluated": len(rows), "missing": missing,
               "mean": sum(item["score"] for item in rows) / len(rows) if rows else None,
               "stages": dict(stages),
               "by_type": {k: {"n": len(v), "mean": sum(v) / len(v)} for k, v in by_type.items()},
               "rows": rows}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Bank 分阶段诊断", "", f"- 已评分：{len(rows)}；缺失：{len(missing)}",
             f"- 平均分：{summary['mean']:.3f}" if rows else "- 平均分：无", "", "## 阶段", "",
             "| 阶段 | 数量 |", "|---|---:|"]
    for stage, count in stages.most_common():
        lines.append(f"| {stage} | {count} |")
    lines += ["", "## 题型", "", "| 类型 | 题数 | 平均分 |", "|---|---:|---:|"]
    for kind, values in by_type.items():
        lines.append(f"| {kind} | {len(values)} | {sum(values) / len(values):.3f} |")
    lines += ["", "## 明细", "", "| row | 分数 | 阶段 | 真值组件 | 预测组件 | 查询节点 |", "|---:|---:|---|---|---|---|"]
    for item in rows:
        lines.append(f"| {item['row']} | {item['score']} | {item['stage']} | {','.join(item['truth']) or '-'} | "
                     f"{','.join(item['predicted']) or '-'} | {','.join(item['queried_nodes']) or '-'} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"evaluated": len(rows), "missing": len(missing), "mean": summary["mean"],
                      "stages": summary["stages"], "by_type": summary["by_type"]}, ensure_ascii=False, indent=2))
    print(output / "summary.md")


if __name__ == "__main__":
    main()
