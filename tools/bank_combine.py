"""Collect two completed Bank arrays into a new, non-destructive report bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from tools.bank_summary import summarize, write_summary


def inference_source_hash(case: Path) -> str | None:
    modules = sorted((case / "source/edge_rca").glob("*.py"))
    if not modules:
        return None
    digest = hashlib.sha256()
    for module in modules:
        digest.update(module.name.encode())
        digest.update(module.read_bytes())
    return digest.hexdigest()


def combine(root: Path, output: Path, a_job: str, b_job: str, *,
            split_row: int = 46, expected_rows: int = 136) -> dict:
    if not 0 <= split_row <= expected_rows or expected_rows < 1:
        raise ValueError("split_row must be within the expected dataset")
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", job) for job in (a_job, b_job)):
        raise ValueError("array identifiers must contain only letters, digits, underscores or hyphens")
    sources = []
    for row in range(expected_rows):
        case = (root / f"bank_case_{a_job if row < split_row else b_job}_{row}").resolve()
        result_path = case / "evaluation/result.json"
        result_bytes = result_path.read_bytes()
        result = json.loads(result_bytes)
        if result.get("row") != row:
            raise ValueError(f"wrong row in {result_path}")
        prediction_path = case / "coordinator_1800/prediction.json"
        prediction_bytes = prediction_path.read_bytes()
        prediction_hash = hashlib.sha256(prediction_bytes).hexdigest()
        if result.get("prediction_sha256") != prediction_hash or json.loads(prediction_bytes) != result["prediction"]:
            raise ValueError(f"frozen prediction differs from evaluation: {case}")
        sources.append({"row": row, "source": str(case),
                        "evaluation_sha256": hashlib.sha256(result_bytes).hexdigest(),
                        "prediction_sha256": prediction_hash,
                        "inference_source_sha256": inference_source_hash(case)})
    versions = sorted({item["inference_source_sha256"] for item in sources
                       if item["inference_source_sha256"] is not None})
    if len(versions) > 1:
        raise ValueError("mixed inference source versions; do not combine as one frozen experiment")
    source_missing = [item["row"] for item in sources if item["inference_source_sha256"] is None]
    if versions and source_missing:
        raise ValueError(f"some inference source snapshots are missing: {source_missing}")
    # Never reuse or delete a case directory. An interrupted bundle must get a new name.
    output.mkdir(parents=True, exist_ok=False)
    for item in sources:
        (output / f"bank_case_combined_{item['row']}").symlink_to(item["source"], target_is_directory=True)
    summaries = {str(t): summarize(output, "combined", time_tolerance_seconds=t,
                                  expected_rows=expected_rows) for t in (60, 300)}
    for item in sources:
        for name, relative in (("evaluation", "evaluation/result.json"),
                               ("prediction", "coordinator_1800/prediction.json")):
            actual = hashlib.sha256((Path(item["source"]) / relative).read_bytes()).hexdigest()
            if actual != item[f"{name}_sha256"]:
                raise ValueError("source changed while building report: " + item["source"])
    for tolerance, summary in summaries.items():
        write_summary(summary, output / f"scores_{tolerance}s")
    manifest = {"source_arrays": [a_job, b_job], "split_row": split_row,
                "expected_rows": expected_rows, "sources": sources,
                "inference_source_versions": versions,
                "source_snapshot_missing_rows": source_missing,
                "scope": "Frozen answers from the specified arrays; recovery acceptance is a separate historical comparison."}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Bank 全量评测结果", "", f"来源数组：{a_job}、{b_job}。",
             "原始预测与评价文件均未改写，逐题指纹见manifest.json。", "",
             "| 时间口径 | 完全正确 | 至少一项正确 | 平均字段分 |", "|---|---:|---:|---:|"]
    for tolerance, summary in summaries.items():
        label = "官方±60秒" if tolerance == "60" else "历史±300秒"
        lines.append(f"| {label} | {summary['fully_correct_count']}/{expected_rows} | "
                     f"{summary['any_field_correct_count']}/{expected_rows} | {summary['mean_field_score']:.6f} |")
    lines.extend(["", "源码版本：" + (versions[0] if versions else "缺少快照，无法核验"),
                  "两种容差分开报告；是否恢复历史效果须结合对应历史版本、同题结果及评分口径核对。", ""])
    (output / "结果.md").write_text("\n".join(lines))
    return {t: {key: value for key, value in summary.items() if key != "rows"}
            for t, summary in summaries.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--a-job", required=True)
    parser.add_argument("--b-job", required=True)
    parser.add_argument("--split-row", type=int, default=46)
    parser.add_argument("--expected-rows", type=int, default=136)
    args = parser.parse_args()
    print(json.dumps(combine(args.root, args.output, args.a_job, args.b_job,
                             split_row=args.split_row, expected_rows=args.expected_rows),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
