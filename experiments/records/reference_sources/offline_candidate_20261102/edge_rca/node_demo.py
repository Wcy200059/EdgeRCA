"""Export a synthetic node protocol example, with no language-model inference."""

from __future__ import annotations

import json
from pathlib import Path

from .data import parse_time
from .node_state import NodeState
from .prompts import build_coordinator_prompt, build_node_prompt


def _write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_node_demo(output: Path) -> dict:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = parse_time("2021-03-04 18:00:00")
    state = NodeState("synthetic-protocol-example", "device:Tomcat01", start,
                      start+1_800_000, output/"node_journal.jsonl")

    def receive(identifier, seconds, kind, description):
        at = start+seconds*1000
        item = {"observation_id": identifier, "kind": kind,
                "observed_at_ms": at, "available_at_ms": at,
                "description": description, "evidence_ids": ["synthetic:"+identifier]}
        evidence = {"synthetic:"+identifier: {
            "node_id": state.node_id, "observed_at_ms": at,
            "available_at_ms": at, "modality": "metric" if kind.startswith("metric") else "log"}}
        state.receive(item, evidence, at+1000)

    receive("o1", 8, "metric.cpu_high.open", "18:00:08，本实例CPU使用率95%，超过90%的试运行阈值。")
    first = state.begin_update(start+10_000)
    _write(output/"node_request_1.json", first)
    _write(output/"node_prompt_1.json", build_node_prompt(first["packet"]))

    # Inference may be slow: new observations are stored without completing the
    # older request. This checks the state protocol, not real worker concurrency.
    receive("o2", 28, "log.error", "18:00:28，本实例出现java.net.ConnectException，文字为Connection refused。")
    _write(output/"while_busy.json", state.snapshot())
    reply = {"notes": [{"observation_id": "o1", "summary": "观测到CPU偏高，是否影响请求尚未确定。"}],
             "questions": ["同期是否有请求延迟变化？"]}
    _write(output/"manual_reply_fixture.json", {
        "source": "Handwritten protocol fixture; NOT a model response", "response": reply})
    accepted = state.complete_update(first["request_id"], reply)
    assert accepted["remaining"] == 1
    second = state.begin_update(start+30_000)
    _write(output/"node_request_2.json", second)
    _write(output/"node_prompt_2.json", build_node_prompt(second["packet"]))

    findings = []
    for report in state.reports():
        if report["type"] == "cpu_observation":
            findings.append({"finding_id": report["finding_id"],
                             "observation": report["observation"]})
        elif report["type"] == "model_annotation":
            for finding in findings:
                if finding["finding_id"] == report["finding_id"]:
                    finding["model_note"] = report["model_note"]
    coordinator = {
        "cutoff_ms": start+30_000,
        "node_reports": [{"node_id": state.node_id, "findings": findings,
                          "questions": state.snapshot()["questions"]}],
        "observed_links": [], "existing_incidents": [],
    }
    _write(output/"coordinator_packet.json", coordinator)
    _write(output/"coordinator_prompt.json", build_coordinator_prompt(coordinator))
    state.fail_update(second["request_id"], "Demo only exports the next input; no model is connected")
    _write(output/"node_state.json", state.snapshot())
    summary = {
        "status": "protocol_example_complete", "source": "synthetic, not OpenRCA telemetry",
        "language_model_calls": 0, "gpu_used": False, "api_calls": 0,
        "cpu_findings_available_to_coordinator": len(findings),
        "model_annotations_from_manual_fixture": 1,
        "pending_observations": state.snapshot()["pending_ids"],
        "checks": {"received_while_previous_request_in_flight": True,
                   "old_reply_did_not_ack_new_observation": accepted["remaining"] == 1,
                   "coordinator_has_observation_without_model_note": "model_note" not in findings[1]},
        "note": "State/protocol demonstration only; no real model, asynchronous worker or RCA evaluation",
    }
    _write(output/"summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
