"""Summarize the two original arrays at both tolerances without changing predictions."""
from datetime import datetime, timezone
import json
from pathlib import Path

from edge_rca.scoring import evaluate


def main():
    root = Path(__file__).resolve().parent
    rows = []
    for row in range(136):
        path = root.parent / f"bank_case_{154057 if row < 46 else 154058}_{row}/evaluation/result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        rows.append({"row": row, "valid": result["final_reply_valid"],
            "scores": {str(t): evaluate(json.dumps(result["prediction"]), result["scoring_points"],
                                        time_tolerance_seconds=t)[2] for t in (60, 300)}})
    n = len(rows)
    summary = {"checked_at": datetime.now(timezone.utc).isoformat(), "completed": n, "expected": 136,
        "missing": sorted(set(range(136)) - {row["row"] for row in rows}),
        "invalid": [row["row"] for row in rows if not row["valid"]],
        "scores": {str(t): {"full": sum(row["scores"][str(t)] == 1 for row in rows),
                            "any": sum(row["scores"][str(t)] > 0 for row in rows),
                            "mean": sum(row["scores"][str(t)] for row in rows) / n if n else None} for t in (60, 300)},
        "scope": "Original baseline arrays 154057 and 154058; these predictions do not use the new repairs."}
    (root / "live_status.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 原全量实验当前结果", "", f"已完成 {n}/136。本表为154057/154058原基线，使用修复前源码。", "",
             "| 容差 | 完全正确 | 至少一项正确 | 平均字段分 |", "|---|---:|---:|---:|"]
    for t in (60, 300):
        score = summary["scores"][str(t)]
        lines.append(f"| ±{t}秒 | {score['full']}/{n} | {score['any']}/{n} | {score['mean']} |")
    lines.extend(["", f"缺失：{summary['missing']}；无效最终回复：{summary['invalid']}。", "",
                  "±60秒是官方时间口径；±300秒单独报告，不能称为官方准确率。", ""])
    (root / "原全量实验结果.md").write_text("\n".join(lines))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
