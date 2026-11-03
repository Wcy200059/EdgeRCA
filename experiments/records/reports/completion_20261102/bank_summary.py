"""Aggregate per-question Bank evaluations into one accuracy report."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--job", required=True, help="Slurm array job id")
    parser.add_argument("--design-row", type=int, default=1,
                        help="Row used during design; report inclusive and exclusive")
    args = parser.parse_args()
    rows, missing = [], []
    for directory in sorted(glob.glob(str(args.root / f"bank_case_{args.job}_*"))):
        row = int(directory.rsplit("_", 1)[-1])
        path = Path(directory) / "evaluation" / "result.json"
        if not path.exists():
            missing.append(row)
            continue
        result = json.loads(path.read_text())
        rows.append({
            "row": row,
            "score": result.get("score"),
            "fully_correct": result.get("fully_correct"),
            "any_field_correct": result.get("any_field_correct"),
            "final_reply_valid": result.get("final_reply_valid"),
            "prediction": result.get("prediction"),
            "failed": result.get("failed", []),
            "model_calls": result.get("model_calls"),
            "model_generation_seconds": result.get("model_generation_seconds"),
            "checkpoints": result.get("checkpoints", []),
            "top3_fully_correct": result.get("top3_fully_correct"),
            "candidate_count": result.get("ranked_evaluation", {}).get("candidate_count"),
        })
    scores = [item["score"] for item in rows if isinstance(item["score"], (int, float))]
    unseen = [item for item in rows if item["row"] != args.design_row]
    unseen_scores = [item["score"] for item in unseen if isinstance(item["score"], (int, float))]
    summary = {
        "job": args.job,
        "questions_with_evaluation": len(rows),
        "missing_or_failed": missing,
        "mean_official_field_score": statistics.mean(scores) if scores else None,
        "exact_match_rate": (sum(item["fully_correct"] is True for item in rows) / len(rows)) if rows else None,
        "any_field_correct_rate": (sum(item["any_field_correct"] is True for item in rows) / len(rows)) if rows else None,
        "unseen_questions": len(unseen),
        "unseen_mean_official_field_score": statistics.mean(unseen_scores) if unseen_scores else None,
        "unseen_exact_match_rate": (sum(item["fully_correct"] is True for item in unseen) / len(unseen)) if unseen else None,
        "design_row": args.design_row,
        "note": "单题官方评分均值；Bank每题的字段数和故障数可能不同，不是统一题数加权。",
        "rows": rows,
    }
    ranked_rows = [item for item in rows if item["top3_fully_correct"] is not None]
    if ranked_rows:
        summary["questions_with_ranked_candidates"] = len(ranked_rows)
        summary["top3_correct_count"] = sum(item["top3_fully_correct"] is True for item in ranked_rows)
        summary["top3_exact_match_rate"] = summary["top3_correct_count"] / len(ranked_rows)
    output = args.root / f"bank_accuracy_{args.job}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    lines = ["# Bank 批量结果", "",
             f"- 有评分题目：{len(rows)}；缺失/失败：{len(missing)} {missing}",
             f"- 平均官方字段得分：{summary['mean_official_field_score']:.3f}" if scores else "- 平均官方字段得分：无",
             f"- 完全正确率：{summary['exact_match_rate']:.1%}" if rows else "- 完全正确率：无",
             f"- 至少一个字段正确：{summary['any_field_correct_rate']:.1%}" if rows else "- 至少一个字段正确：无",
             f"- 未参与设计题目：{len(unseen)}；平均分 {summary['unseen_mean_official_field_score']:.3f}"
             if unseen_scores else "- 未参与设计题目：无",
             "", "| row | 得分 | 完全正确 | 失败字段 | 模型调用 |", "|---:|---:|---|---|---:|"]
    for item in rows:
        lines.append(f"| {item['row']} | {item['score']} | {item['fully_correct']} | "
                     f"{', '.join(item['failed']) or '-'} | {item['model_calls']} |")
    if ranked_rows:
        lines += ["", f"前三候选完整命中率：{summary['top3_exact_match_rate']:.1%}"
                  f"（{summary['top3_correct_count']}/{len(ranked_rows)}题）。"
                  "这里只统计实际生成候选文件的题目；首选正确率保持单独报告。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "questions_with_evaluation", "missing_or_failed", "mean_official_field_score",
        "exact_match_rate", "unseen_questions", "unseen_mean_official_field_score",
        "unseen_exact_match_rate")}, ensure_ascii=False, indent=2), flush=True)
    print(f"summary: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
