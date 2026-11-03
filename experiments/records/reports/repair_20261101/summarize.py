"""Paired development verification; never label two cases as Bank accuracy."""
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "source_manifest.json").read_text())
    baseline = root.parent / "rollback_20261030"
    rows = []
    for row in manifest["rows"]:
        item = {"row": row, "complete": False, "scores": {}}
        for tolerance in (60, 300):
            filename = f"evaluation/result_{tolerance}s.json"
            old = json.loads((baseline / "cases" / str(row) / filename).read_text())
            path = root / "cases" / str(row) / filename
            new = json.loads(path.read_text()) if path.exists() else None
            item["scores"][str(tolerance)] = {
                "before": old["score"], "after": new["score"] if new else None,
                "before_valid": old["final_reply_valid"], "after_valid": new["final_reply_valid"] if new else None,
                "before_prediction": old["prediction"], "after_prediction": new["prediction"] if new else None,
            }
        item["complete"] = all(value["after"] is not None for value in item["scores"].values())
        rows.append(item)
    result = {"scope": manifest["scope"], "source_sha256": manifest["source_sha256"],
              "completed": sum(row["complete"] for row in rows), "total": len(rows), "rows": rows}
    (root / "paired_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["修复后的两题开发复测；不能当作全量Bank准确率。", "",
             f"完成 {result['completed']}/{len(rows)}。", "",
             "| 题号 | 原回复合法 | 新回复合法 | ±60秒原分→新分 | ±300秒原分→新分 |", "|---|---|---|---|---|"]
    for row in rows:
        official, relaxed = row["scores"]["60"], row["scores"]["300"]
        lines.append(f"| {row['row']} | {official['before_valid']} | {official['after_valid']} | "
                     f"{official['before']} → {official['after']} | {relaxed['before']} → {relaxed['after']} |")
    (root / "paired_results.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"completed": result["completed"], "total": len(rows)}))


if __name__ == "__main__":
    main()
