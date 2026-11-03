"""Review frozen recovery predictions; never mark the recovery accepted."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE))

from edge_rca.scoring import evaluate
from tools.bank_combine import inference_source_hash


def main():
    historical_root = WORKSPACE / "outputs/rollback_20261030"
    restore = json.loads((historical_root / "restore_manifest.json").read_text())
    historical = json.loads((historical_root / "historical_144524_300s_rows.json").read_text())
    historical_rows = {item["row"]: item for item in historical["rows"]}
    if set(historical_rows) != set(range(136)):
        raise ValueError("historical comparison rows are incomplete")
    expected = json.loads((WORKSPACE / "outputs/offline_baseline_20261102/manifest.json").read_text())["inference_source_sha256"]
    rows, versions, missing = [], Counter(), []
    for row in range(136):
        case = WORKSPACE / f"outputs/bank_case_{154464 if row < 46 else 154465}_{row}"
        result_file = case / "evaluation/result.json"
        if not result_file.exists():
            missing.append(row)
            continue
        result = json.loads(result_file.read_text())
        prediction_file = case / "coordinator_1800/prediction.json"
        raw = prediction_file.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if result["row"] != row or result["prediction_sha256"] != digest or json.loads(raw) != result["prediction"]:
            raise ValueError(f"frozen answer disagrees with evaluation: {row}")
        scores = {str(t): evaluate(raw.decode(), result["scoring_points"], time_tolerance_seconds=t)[2]
                  for t in (60, 300)}
        if prediction_file.read_bytes() != raw:
            raise ValueError(f"answer changed during independent evaluation: {row}")
        version = inference_source_hash(case)
        versions[version or "missing"] += 1
        rows.append({"row": row, "source_sha256": version,
                     "final_reply_valid": result["final_reply_valid"], "prediction_sha256": digest,
                     "scores": scores, "historical_300s_score": historical_rows[row]["score"]})
    n = len(rows)
    totals = {str(t): {"evaluated": n, "fully_correct": sum(r["scores"][str(t)] == 1 for r in rows),
                       "any_field_correct": sum(r["scores"][str(t)] > 0 for r in rows),
                       "mean_field_score": sum(r["scores"][str(t)] for r in rows) / n if n else None}
              for t in (60, 300)}
    regressions = [r["row"] for r in rows if r["historical_300s_score"] > r["scores"]["300"]]
    lost_exact = [r["row"] for r in rows if r["historical_300s_score"] == 1 and r["scores"]["300"] < 1]
    stamp = datetime.now(timezone.utc)
    version_ok = bool(rows) and dict(versions) == {expected: n}
    state = "full_test_pending" if missing else "historical_review_required" if version_ok else "source_mismatch"
    report = {"checked_at": stamp.isoformat(), "recovery_acceptance": "not_accepted",
              "evaluation_state": state, "arrays": [154464, 154465], "combine_job": 154466,
              "completed": n, "expected": 136, "missing": missing,
              "source_versions": dict(versions), "source_matches_selected_candidate": version_ok,
              "current_scores": totals, "historical_144524_reference": restore["historical_accuracy"],
              "historical_reference_is_unique_user_target": False,
              "historical_300s_matched_regression_rows": regressions,
              "historical_300s_lost_exact_rows": lost_exact,
              "invalid_final_rows": [r["row"] for r in rows if not r["final_reply_valid"]],
              "rows": rows,
              "scope": "Completed frozen answers only. Partial results are not full accuracy; historical effect reproduction remains unaccepted.",
              "limits": ["Historical official per-row scores are not recovered; official global counts are preserved.",
                         "The user's best pre-streaming version is still being traced; 144524 is a verified reference."]}
    snapshots = ROOT / "snapshots"
    snapshots.mkdir(exist_ok=True)
    (snapshots / (stamp.strftime("%Y%m%dT%H%M%S%fZ") + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "current_status.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (WORKSPACE / "outputs/offline_baseline_20261102/recovery_status.json").write_text(json.dumps(
        {key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 恢复效果：未验收", "", f"核对时间（UTC）：{stamp.isoformat()}", "",
             f"当前恢复候选全量测试：{n}/136；数组154464/154465；汇总154466。",
             f"已完成题的源码与选定候选一致：{version_ok}。", "",
             "流程恢复、备份和程序检查完成均不能替代历史效果复现。",
             "历史144524是已核实参照，仍继续追查用户所指的更好历史版本。", ""]
    if missing:
        lines.append("完整测试尚未完成，当前不宣布恢复成功。")
    else:
        lines.extend(["全部题目已保存，等待同口径历史比较和回归原因核查；本脚本不自动接受恢复。", "",
                      "| 时间口径 | 当前整题正确 | 历史144524整题正确 |", "|---|---:|---:|"])
        for tolerance, key in ((60, "time_60s"), (300, "time_300s")):
            lines.append(f"| ±{tolerance}秒 | {totals[str(tolerance)]['fully_correct']}/136 | "
                         f"{restore['historical_accuracy'][key]['fully_correct']}/136 |")
        lines.extend(["", f"历史300秒口径曾整题正确、本轮失去整题正确的题目：{lost_exact}。",
                      f"历史300秒口径字段分下降的题目：{regressions}。"])
    lines.extend(["", "完整明细、源码指纹和阶段限制见current_status.json；旧状态保存在snapshots/。", ""])
    (ROOT / "恢复状态.md").write_text("\n".join(lines))
    print(json.dumps({key: report[key] for key in ("checked_at", "recovery_acceptance", "evaluation_state", "completed", "expected", "source_matches_selected_candidate")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
