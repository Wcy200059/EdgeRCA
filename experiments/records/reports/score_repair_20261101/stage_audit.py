"""Independent component-level coverage audit; labels never enter model calls."""
from collections import Counter
import json
from pathlib import Path

from edge_rca.scoring import COMPONENT, REASON


ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    totals, details = Counter(), []
    for row in manifest["completed_rows_at_start"]:
        case = ROOT.parent / f"bank_case_{154057 if row < 46 else 154058}_{row}"
        report = json.loads((case / "evaluation/result.json").read_text())
        truth = set(COMPONENT.findall(report["scoring_points"]))
        reasons = set(REASON.findall(report["scoring_points"]))
        if not truth:
            continue
        reports = json.loads((case / "nodes_1800/reports.json").read_text())["reports"]
        by_node = {node["node_id"]: node for node in reports}
        hypotheses = []
        for path in (case / "coordinator_1800").glob("decision_*.json"):
            hypotheses.extend((json.loads(path.read_text()).get("reply") or {}).get("hypotheses", []))
        loaded_nodes = set()
        for line in (case / "coordinator_1800/calls.jsonl").open():
            call = json.loads(line)
            try:
                payload = json.JSONDecoder().raw_decode(call["messages"][1]["content"])[0]
                loaded_nodes.update(fact[1] for fact in payload.get("evidence", []) if isinstance(fact, list))
            except (ValueError, KeyError, IndexError):
                pass
        for component in sorted(truth):
            node = "device:" + component
            facts = by_node.get(node, {}).get("template_rows", [])
            checked = [f for f in facts if f.get("model_selection")]
            flagged = [f for f in checked if f["model_selection"]["use"] in ("report", "uncertain")]
            item = {"row": row, "component": component, "node_report_exists": node in by_node,
                    "node_model_expanded_observation": bool(checked), "node_model_flagged_observation": bool(flagged),
                    "some_component_observation_visible_to_coordinator": node in loaded_nodes,
                    "ever_hypothesized": any(h.get("component") == component for h in hypotheses),
                    "selected_finally": any(p.get("root cause component") == component for p in report["prediction"]),
                    "expanded_count": len(checked), "available_count": len(facts)}
            details.append(item)
            totals["scored_components"] += 1
            for key, value in item.items():
                if isinstance(value, bool):
                    totals[key] += value
    result = {"scope": "Only questions with component labels; component observations present does not imply decisive evidence present.",
              "totals": dict(totals), "details": details}
    (ROOT / "stage_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
