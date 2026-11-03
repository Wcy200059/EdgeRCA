"""CPU stage recall: truth component at 4B page, 9B input and hypotheses."""

import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path

from edge_rca.scoring import COMPONENT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", default="143858")
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    counts = Counter(); total = 0; details = []
    for directory in sorted(glob.glob(str(args.root / f"bank_case_{args.job}_*")),
                            key=lambda value: int(value.rsplit("_", 1)[-1])):
        row = int(directory.rsplit("_", 1)[-1]); path = Path(directory)
        try:
            question = json.loads((path / "frames/public_question.json").read_text())
            with Path(question["query_file"]).open(encoding="utf-8-sig", newline="") as handle:
                item = next(item for index, item in enumerate(csv.DictReader(handle)) if index == question["row"])
            truth = set(COMPONENT.findall(item["scoring_points"]))
            reports = json.loads((path / "nodes_1800/reports.json").read_text())["reports"]
            by_node = {r["node_id"]: r for r in reports}
            decision = []
            for d in glob.glob(str(path / "coordinator_1800/decision_*.json")):
                reply = json.loads(Path(d).read_text()).get("reply") or {}
                decision.extend(reply.get("hypotheses", []))
            hyp = {h.get("component") for h in decision}
            prediction = json.loads((path / "coordinator_1800/prediction.json").read_text())
            first = {p.get("root cause component") for p in prediction}
        except (FileNotFoundError, StopIteration, ValueError, KeyError, json.JSONDecodeError):
            continue
        if not truth:
            continue
        total += 1
        states = []
        for component in truth:
            node = by_node.get(f"device:{component}")
            rows = node.get("template_rows", []) if node else []
            checked = [r for r in rows if r.get("model_selection")]
            flagged = [r for r in checked if (r.get("model_selection") or {}).get("use") in ("report", "uncertain")]
            if not checked:
                stage = "4b_not_on_page"
            elif not flagged:
                stage = "4b_seen_not_flagged"
            elif component in hyp:
                stage = "9b_hypothesized"
            else:
                stage = "4b_flagged_9b_not_hypothesized"
            states.append(stage)
            counts[stage] += 1
        details.append({"row": row, "truth": sorted(truth), "states": states,
                        "first": sorted(x for x in first if x)})
    print(json.dumps({"rows": total, "component_states": dict(counts),
                      "details": details}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
