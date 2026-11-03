"""Evaluate a frozen one-case answer after inference has ended.

Only this process reads scoring_points. The runtime and node workers do not
import this module. Scoring uses the existing OpenRCA object-matching scorer.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

from .case_runtime import save
from .scoring import evaluate as score_prediction


def evaluate_case(root, scorer, time_tolerance_seconds=300):
    root = Path(root)
    question = json.loads((root / "frames" / "public_question.json").read_text())
    frames = json.loads((root / "frames" / "complete.json").read_text())
    available = [seconds for seconds in frames["checkpoints"]
                 if (root / f"nodes_{seconds}" / "reports.json").exists()
                 and (root / f"coordinator_{seconds}" / "complete.json").exists()]
    if not available:
        raise FileNotFoundError("no completed checkpoint output to evaluate")
    statuses = []
    for seconds in available:
        report = json.loads((root / f"nodes_{seconds}" / "reports.json").read_text())
        central = json.loads((root / f"coordinator_{seconds}" / "complete.json").read_text())
        if len(report["reports"]) != frames["device_nodes"] or central["device_nodes_received"] != frames["device_nodes"]:
            raise ValueError("cannot evaluate an incomplete node cluster")
        statuses.append({"seconds": seconds, "nodes": len(report["reports"]),
                         "central_queries": central["executed_central_queries"]})
    last = max(available)
    prediction_file = root / f"coordinator_{last}" / "prediction.json"
    frozen = prediction_file.read_bytes()
    prediction = frozen.decode()
    final = json.loads((prediction_file.parent / "final_answer.json").read_text())
    with Path(question["query_file"]).open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        item = next(row for index, row in enumerate(rows) if index == question["row"])
    if item["instruction"] != question["instruction"]:
        raise ValueError("evaluation question changed after inference")
    if not Path(scorer).exists():
        raise FileNotFoundError(scorer)
    passed, failed, score = score_prediction(prediction, item["scoring_points"],
                                              time_tolerance_seconds=time_tolerance_seconds)
    if prediction_file.read_bytes() != frozen:
        raise ValueError("prediction changed during evaluation")
    calls = []
    for path in root.glob("**/*calls*.jsonl"):
        if "source" in path.parts:
            continue
        calls.extend(json.loads(line) for line in path.read_text().splitlines())
    report = {"query_file": question["query_file"], "row": question["row"], "instruction": item["instruction"],
              "task_index": item.get("task_index"), "scoring_points": item["scoring_points"],
              "prediction": json.loads(prediction), "prediction_sha256": hashlib.sha256(frozen).hexdigest(),
              "final_reply_valid": final["reply"] is not None, "score": score,
              "fully_correct": score == 1.0, "any_field_correct": score > 0.0,
              "passed": passed, "failed": failed, "checkpoints": statuses,
              "delivered_rows": frames["delivered_rows"], "device_nodes": frames["device_nodes"],
              "model_calls": len(calls), "model_generation_seconds": sum(c["seconds"] for c in calls),
              "hit_output_limit_calls": sum(c["hit_output_limit"] for c in calls),
              "inference_labels_used": False, "scorer": str(Path(scorer).resolve()),
              "time_tolerance_seconds": time_tolerance_seconds,
              "scoring_rule": f"components/reasons exact; occurrence time <= {time_tolerance_seconds}s",
              "scope": "单道已有Bank样例的加速回放验收，不代表335题准确率或实时部署性能。"}
    ranked_path = prediction_file.parent / "ranked_predictions.json"
    if ranked_path.exists():
        from .ranked_answers import score as score_ranked
        ranked_frozen = ranked_path.read_bytes()
        ranked = score_ranked(json.loads(ranked_frozen), report["prediction"], item["scoring_points"],
                              time_tolerance_seconds=time_tolerance_seconds)
        if ranked_path.read_bytes() != ranked_frozen:
            raise ValueError("ranked predictions changed during evaluation")
        report["ranked_evaluation"] = ranked
        report["top3_fully_correct"] = ranked["fully_correct_within_top3"]
        report["ranked_prediction_sha256"] = hashlib.sha256(ranked_frozen).hexdigest()
    output = root / "evaluation"
    if output.exists():
        raise ValueError("evaluation already exists")
    output.mkdir()
    save(output / "result.json", report)
    pairs = [("组件", "root cause component", r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)"),
             ("原因", "root cause reason", r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)"),
             ("时间", "root cause occurrence datetime", r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes \(i.e., <=1min\) of ([^\n]+)")]
    lines = ["# EdgeRCA：一道Bank题的完整集群测试", "", question["instruction"], "",
             f"**整题完全正确：{'是' if score == 1.0 else '否'}；官方字段得分：{score:.0%}。**", "",
             "| 字段 | 模型回答 | 标准答案 |", "|---|---|---|"]
    for label, key, pattern in pairs:
        truth = re.findall(pattern, item["scoring_points"])
        answers = "；".join(f"故障{i + 1}：{fault.get(key) or '未知'}"
                          for i, fault in enumerate(report["prediction"]))
        lines.append(f"| {label} | {answers} | {'；'.join(truth) or '本题不评分'} |")
    lines += ["", "每个故障的字段按同一编号对应；模型答案与标准答案的排列顺序可以不同，评分器按完整故障对象匹配。"]
    if "ranked_evaluation" in report:
        ranked = report["ranked_evaluation"]
        lines += ["", f"另保存{ranked['candidate_count']}组候选（含首选，最多{ranked['candidate_limit']}组）："
                  f"前三候选中{'有' if ranked['fully_correct_within_top3'] else '没有'}一组完整正确；"
                  f"首次命中排名：{ranked['first_correct_rank'] or '未命中'}。",
                  "候选命中独立统计，不把多个竞争猜测计作同时发生的故障，也不替代首选成绩。"]
    checkpoints_text = "、".join(f"第{c['seconds'] // 60}分钟" for c in statuses)
    lines += ["", f"{frames['device_nodes']}个逻辑节点，4B处理本地信息、9B协调；{checkpoints_text}更新，共{len(calls)}次模型调用。",
              f"回放{frames['delivered_rows']:,}条原始记录；标准答案仅在最终答案保存后由独立评分进程读取。",
              "", "这是加速回放，单张显卡顺序模拟节点；模型处理有排队延迟，尚不能称为实时边缘部署。",
              "单道既有调试样例的结果只能用于流程验收，后续需扩大到未参与设计的题目。", ""]
    (output / "结果.md").write_text("\n".join(lines))
    print(json.dumps({"fully_correct": score == 1.0, "score": score, "passed": passed, "failed": failed,
                      "result": str(output / "结果.md")}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--time-tolerance-seconds", type=int, default=300)
    args = parser.parse_args()
    evaluate_case(args.root, args.scorer, args.time_tolerance_seconds)


if __name__ == "__main__":
    main()
