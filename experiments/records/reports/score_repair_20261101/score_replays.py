"""Independent scoring, only after all inference outputs have been frozen."""
import hashlib
import json
from pathlib import Path

from edge_rca.scoring import evaluate


ROOT = Path(__file__).resolve().parent


def main():
    if not (ROOT / "inference_complete.json").exists():
        raise ValueError("Inference has not finished")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    rows = []
    for row in manifest["rows"]:
        case = ROOT.parent / f"bank_case_{154057 if row < 46 else 154058}_{row}"
        truth = json.loads((case / "evaluation/result.json").read_text())["scoring_points"]
        for name in manifest["variants"]:
            output = ROOT / "replays" / str(row) / name
            data = (output / "prediction.json").read_bytes()
            final = json.loads((output / "final_answer.json").read_text())
            rows.append({"row": row, "variant": name,
                "split": "development" if row in manifest["development_rows"] else "holdout",
                "valid": final["reply"] is not None, "prediction": json.loads(data),
                "prediction_sha256": hashlib.sha256(data).hexdigest(),
                "scores": {str(t): evaluate(data.decode(), truth, time_tolerance_seconds=t)[2] for t in (60, 300)}})
    summary = {}
    lines = ["# 相同证据、按题固定随机种子的最终裁决对照", "",
             "这是最终裁决阶段的开发验证，复用已生成的节点与补查证据，不代表完整Bank新准确率。",
             "全部预测冻结后，独立评分进程才读取标准答案。", "",
             "| 分组 | 版本 | 题数 | 合法回复 | ±60秒完全正确 | ±60秒平均分 | ±300秒完全正确 |", "|---|---|---:|---:|---:|---:|---:|"]
    for split in ("development", "holdout", "all"):
        summary[split] = {}
        for name in manifest["variants"]:
            selected = [r for r in rows if r["variant"] == name and (split == "all" or r["split"] == split)]
            item = {"n": len(selected), "valid": sum(r["valid"] for r in selected),
                    "scores": {str(t): {"full": sum(r["scores"][str(t)] == 1 for r in selected),
                                        "mean": sum(r["scores"][str(t)] for r in selected) / len(selected)} for t in (60, 300)}}
            summary[split][name] = item
            lines.append(f"| {split} | {name} | {item['n']} | {item['valid']} | {item['scores']['60']['full']} | "
                         f"{item['scores']['60']['mean']:.3f} | {item['scores']['300']['full']} |")
    result = {"summary": summary, "rows": rows, "scope": manifest["evaluation_scope"]}
    (ROOT / "paired_scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "paired_scores.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
