"""Replay only the final 9B decision against a saved pre-final view."""

import argparse
import csv
import json
from pathlib import Path

from edge_rca.case_runtime import final_answer
from edge_rca.node_probe import LocalModel
from edge_rca.scoring import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-tolerance-seconds", type=int, default=300)
    args = parser.parse_args()
    view = json.loads((args.root / "coordinator_1800/last_input.json").read_text())["view"]
    question = json.loads((args.root / "frames/public_question.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    model = LocalModel(args.model, args.output / "calls.jsonl", max_context=20480)
    final_answer(model, view, question["instruction"], args.output)
    prediction = (args.output / "prediction.json").read_text()
    with Path(question["query_file"]).open(encoding="utf-8-sig", newline="") as handle:
        item = next(row for index, row in enumerate(csv.DictReader(handle)) if index == question["row"])
    passed, failed, score = evaluate(prediction, item["scoring_points"],
                                     time_tolerance_seconds=args.time_tolerance_seconds)
    print(json.dumps({"score": score, "passed": passed, "failed": failed,
                      "prediction": json.loads(prediction)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
