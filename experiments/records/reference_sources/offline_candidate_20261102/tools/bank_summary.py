"""Aggregate frozen Bank evaluations with an explicit, consistent time tolerance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from edge_rca.ranked_answers import score as score_ranked
from edge_rca.scoring import evaluate


def summarize(root: Path, job: str, *, time_tolerance_seconds: int | None = None,
              expected_rows: int | None = None, design_rows: tuple[int, ...] = ()) -> dict:
    if time_tolerance_seconds is not None and time_tolerance_seconds < 0:
        raise ValueError("time tolerance must be nonnegative")
    if expected_rows is not None and expected_rows < 1:
        raise ValueError("expected_rows must be positive")
    directories = {int(path.name.rsplit("_", 1)[-1]): path
                   for path in root.glob(f"bank_case_{job}_*")}
    expected = set(range(expected_rows)) if expected_rows is not None else set(directories)
    unexpected = set(directories) - expected
    if unexpected:
        raise ValueError(f"unexpected rows outside requested dataset: {sorted(unexpected)}")
    rows, missing, tolerances = [], [], set()
    for row in sorted(expected):
        directory = directories.get(row)
        path = directory / "evaluation/result.json" if directory else None
        if path is None or not path.exists():
            missing.append(row)
            continue
        result = json.loads(path.read_text())
        if result.get("row", row) != row:
            raise ValueError(f"evaluation row disagrees with directory: {path}")
        saved_tolerance = result.get("time_tolerance_seconds")
        tolerance = time_tolerance_seconds if time_tolerance_seconds is not None else saved_tolerance
        tolerances.add(tolerance)
        score, failed = result.get("score"), result.get("failed", [])
        if time_tolerance_seconds is not None:
            if "prediction" not in result or not result.get("scoring_points"):
                raise ValueError(f"cannot rescore without frozen prediction and scoring points: {path}")
            _, failed, score = evaluate(json.dumps(result["prediction"], ensure_ascii=False),
                                        result["scoring_points"], time_tolerance_seconds=tolerance)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError(f"invalid evaluation score: {path}")
        ranked = result.get("ranked_evaluation", {})
        top3 = result.get("top3_fully_correct")
        if time_tolerance_seconds is not None and saved_tolerance != tolerance:
            top3, ranked = None, {}
            ranked_path = directory / "coordinator_1800/ranked_predictions.json"
            if ranked_path.exists():
                ranked = score_ranked(json.loads(ranked_path.read_text()), result["prediction"],
                                      result["scoring_points"], time_tolerance_seconds=tolerance)
                top3 = ranked["fully_correct_within_top3"]
        rows.append({"row": row, "score": score, "fully_correct": score == 1,
                     "any_field_correct": score > 0, "failed": failed,
                     "prediction": result.get("prediction"),
                     "final_reply_valid": result.get("final_reply_valid"),
                     "model_calls": result.get("model_calls"),
                     "model_generation_seconds": result.get("model_generation_seconds"),
                     "checkpoints": result.get("checkpoints", []),
                     "top3_fully_correct": top3, "candidate_count": ranked.get("candidate_count")})
    if len(tolerances) > 1:
        raise ValueError("mixed or unknown saved time tolerances; request an explicit rescore")
    tolerance = next(iter(tolerances), time_tolerance_seconds)
    mean = statistics.mean(item["score"] for item in rows) if rows else None
    summary = {"job": job, "time_tolerance_seconds": tolerance,
               "questions_expected": len(expected), "questions_with_evaluation": len(rows),
               "missing_or_failed": missing, "mean_field_score": mean,
               "mean_official_field_score": mean if tolerance == 60 else None,
               "fully_correct_count": sum(item["fully_correct"] for item in rows),
               "any_field_correct_count": sum(item["any_field_correct"] for item in rows),
               "exact_match_rate": statistics.mean(item["fully_correct"] for item in rows) if rows else None,
               "any_field_correct_rate": statistics.mean(item["any_field_correct"] for item in rows) if rows else None,
               "invalid_final_rows": [item["row"] for item in rows if item["final_reply_valid"] is False],
               "note": "±60秒为官方口径；其他容差单列。已有Bank样例不自动视作未参与设计的测试集。",
               "rows": rows}
    if design_rows:
        subset = [item for item in rows if item["row"] not in design_rows]
        summary["excluding_design_rows"] = {
            "excluded_rows": sorted(set(design_rows)), "questions": len(subset),
            "mean_field_score": statistics.mean(item["score"] for item in subset) if subset else None,
            "note": "仅排除指定行，不代表其他题未用于开发。"}
    ranked_rows = [item for item in rows if item["top3_fully_correct"] is not None]
    if ranked_rows:
        summary["questions_with_ranked_candidates"] = len(ranked_rows)
        summary["top3_correct_count"] = sum(item["top3_fully_correct"] is True for item in ranked_rows)
        summary["top3_exact_match_rate"] = summary["top3_correct_count"] / len(ranked_rows)
    return summary


def write_summary(summary: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    tolerance = summary["time_tolerance_seconds"]
    if tolerance == 60:
        label = "官方±60秒"
    elif tolerance is None:
        label = "原记录容差未知"
    else:
        label = f"±{tolerance}秒（非官方口径）"
    lines = [f"# Bank 批量结果：{label}", "",
             f"- 有评分题目：{summary['questions_with_evaluation']}/{summary['questions_expected']}",
             f"- 缺失/失败：{summary['missing_or_failed']}",
             f"- 平均字段得分：{summary['mean_field_score']:.6f}" if summary["rows"] else "- 平均字段得分：无",
             f"- 整题完全正确：{summary['fully_correct_count']}",
             f"- 至少一项正确：{summary['any_field_correct_count']}",
             f"- 无效最终回复：{summary['invalid_final_rows']}",
             "", summary["note"], "",
             "| row | 得分 | 完全正确 | 失败字段 | 模型调用 |", "|---:|---:|---|---|---:|"]
    for item in summary["rows"]:
        lines.append(f"| {item['row']} | {item['score']} | {item['fully_correct']} | "
                     f"{', '.join(item['failed']) or '-'} | {item['model_calls']} |")
    if "questions_with_ranked_candidates" in summary:
        lines += ["", f"前三候选完整命中率：{summary['top3_exact_match_rate']:.1%}"
                  f"（{summary['top3_correct_count']}/{summary['questions_with_ranked_candidates']}题）。"
                  "只统计相同容差下有候选评分的题目。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--job", required=True)
    parser.add_argument("--design-row", type=int, action="append", default=[])
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--time-tolerance-seconds", type=int,
                        help="Rescore frozen answers; omit to use consistent saved scores")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.root, args.job, time_tolerance_seconds=args.time_tolerance_seconds,
                        expected_rows=args.expected_rows, design_rows=tuple(args.design_row))
    suffix = f"_{args.time_tolerance_seconds}s" if args.time_tolerance_seconds is not None else ""
    output = args.output or args.root / f"bank_accuracy_{args.job}{suffix}"
    write_summary(summary, output)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"summary: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
