"""Attach immutable tool facts to tiny-model notes before coordinator handoff.

This is a report envelope, not a semantic judge or an RCA decision. Missing
summaries remain outstanding; model wording never overwrites tool statistics.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from .compact_tools import TOOL_ORDER, response_packet
from .node_probe import write_json
from .staged_probe import validate_findings


def build_handoff(request, report):
    if request["request_id"] != report["request_id"] or request["cutoff_ms"] != report["cutoff_ms"]:
        raise ValueError("report belongs to a different request or cutoff")
    bundle = request["bundle"]
    required = set(response_packet(bundle)["scope"]["must_report_ids"])
    steps = {}
    for step in report["steps"]:
        name = step["tool"]
        if name not in TOOL_ORDER or name in steps:
            raise ValueError("unknown or duplicate tool report")
        steps[name] = step
    findings, outstanding, reported = [], {}, set()
    for name in TOOL_ORDER:
        facts = {f["id"]: f for f in bundle["tools"][name]}
        step = steps.get(name, {})
        if "reply" not in step:
            outstanding[name] = list(facts)
            continue
        if not any(a["accepted"] for a in step["tool_attempts"]):
            raise ValueError("cannot summarize a tool that was not executed")
        checked = validate_findings(step["reply"], list(facts.values()), required & facts.keys())
        outstanding[name] = checked["not_summarized_ids"]
        for finding in checked["reply"]["findings"]:
            ids = finding["ids"]
            reported.update(ids)
            findings.append({"tool": name, "model_text_unverified": finding["text"],
                "request_id": request["request_id"], "fact_ids": ids,
                "tool_facts": [deepcopy(facts[i]) for i in ids],
                "source_references": [deepcopy(bundle["catalog"][i]) for i in ids]})
    return {"request_id": request["request_id"], "cutoff_ms": request["cutoff_ms"],
        "node_id": bundle["context"]["node_id"],
        "scope": deepcopy(bundle["context"]["request_scope"]),
        "alert_transitions": deepcopy(bundle["context"]["alerts"]),
        "observed_links": deepcopy(bundle["links"]),
        "findings": findings, "not_summarized_ids": outstanding,
        "alert_evidence_not_summarized": sorted(required - reported),
        "cpu_only_model_unread_ids": deepcopy(bundle["cpu_only_observation_ids"]),
        "semantics_verified": False,
        "note": "模型文字是待核查的本地观测。工具数值及来源独立保留；未总结不代表正常。不是根因结论。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(args.run.glob("r*_report.json"))
    if not paths:
        raise ValueError("no completed request reports found")
    for path in paths:
        report = json.loads(path.read_text())
        rid = report["request_id"]
        # IDs come from local probe output, but still disallow path traversal.
        if not rid.startswith("r") or not rid[1:].isdigit():
            raise ValueError("invalid probe request ID")
        request = json.loads((args.inputs / f"{rid}.json").read_text())
        write_json(args.output / f"{rid}.json", build_handoff(request, report))
    write_json(args.output / "manifest.json", {"inputs": str(args.inputs.resolve()),
        "run": str(args.run.resolve()), "reports": len(paths), "model_calls": 0})


if __name__ == "__main__":
    main()
