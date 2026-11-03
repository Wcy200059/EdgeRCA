"""Re-score frozen Bank predictions with a configurable time window."""

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path

from edge_rca.scoring import evaluate


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
            prediction = (path / "coordinator_1800/prediction.json").read_text()
            with Path(question["query_file"]).open(encoding="utf-8-sig", newline="") as handle:
                item = next(item for index, item in enumerate(csv.DictReader(handle)) if index == question["row"])
            passed, failed, score = evaluate(prediction, item["scoring_points"],
                                             time_tolerance_seconds=args.time_tolerance_seconds)
        except (FileNotFoundError, StopIteration, ValueError, KeyError, json.JSONDecodeError):
            missing.append(row)
            continue
        old = path / "evaluation/result.json"
        old_result = json.loads(old.read_text()) if old.exists() else {}
        rows.append({"row": row, "score": score, "fully_correct": score == 1.0,
                     "any_field_correct": score > 0.0, "passed": passed, "failed": failed,
                     "prediction": json.loads(prediction),
                     "model_calls": old_result.get("model_calls"),
                     "time_only": "only root cause occurrence time" in item["scoring_points"].lower()
                     and "component" not in item["scoring_points"].lower(),
                     "failure_count": 2 if "two" in item["instruction"].lower() else 1})
    scores = [item["score"] for item in rows]
    output = args.root / f"bank_accuracy_{args.job}_{args.time_tolerance_seconds}s"
    output.mkdir(parents=True, exist_ok=False)
    summary = {"job": args.job, "time_tolerance_seconds": args.time_tolerance_seconds,
               "questions_with_evaluation": len(rows), "missing_or_failed": missing,
               "mean_field_score": statistics.mean(scores) if scores else None,
               "mean_official_field_score": statistics.mean(scores) if scores and args.time_tolerance_seconds == 60 else None,
               "exact_match_rate": sum(item["fully_correct"] for item in rows) / len(rows) if rows else None,
               "any_field_correct_rate": sum(item["any_field_correct"] for item in rows) / len(rows) if rows else None,
               "conservative_mean_over_136": sum(scores) / 136 if scores else None,
               "time_only": {"n": sum(item["time_only"] for item in rows),
                             "exact": sum(item["fully_correct"] for item in rows if item["time_only"])},
               "two_fault": {"n": sum(item["failure_count"] == 2 for item in rows),
                             "exact": sum(item["fully_correct"] for item in rows if item["failure_count"] == 2)},
               "rows": rows}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"# Bank 批量结果（±{args.time_tolerance_seconds}秒时间窗）", "",
             "±60秒为官方口径，其他容差单独报告。", "",
             f"- 评分题目：{len(rows)}；缺失/失败：{len(missing)} {missing}",
             f"- 平均字段得分：{summary['mean_field_score']:.3f}" if scores else "- 平均字段得分：无",
             f"- 完全正确率：{summary['exact_match_rate']:.1%}",
             f"- 至少一个字段正确：{summary['any_field_correct_rate']:.1%}",
             f"- 按136题分母计分：{summary['conservative_mean_over_136']:.3f}" if scores else "- 按136题分母计分：无",
             f"- 时间题：{summary['time_only']['n']} 道，完全正确 {summary['time_only']['exact']}",
             f"- 双故障题：{summary['two_fault']['n']} 道，完全正确 {summary['two_fault']['exact']}", "",
             "| row | 得分 | 完全正确 | 失败字段 |", "|---:|---:|---|---|"]
    for item in rows:
        lines.append(f"| {item['row']} | {item['score']} | {item['fully_correct']} | {', '.join(item['failed']) or '-'} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "questions_with_evaluation", "missing_or_failed", "mean_field_score", "mean_official_field_score",
        "exact_match_rate", "any_field_correct_rate", "conservative_mean_over_136",
        "time_only", "two_fault")}, ensure_ascii=False, indent=2))
    print(f"summary: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
