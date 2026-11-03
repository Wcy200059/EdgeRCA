"""A bounded local information loop, usable with initial or follow-up snapshots.

The small model proposes a local query while summarizing. The program executes
fixed tools, returns new facts, preserves unanswered questions for the
coordinator, and stops on an explicit budget. This is a frozen-prefix worker,
not the continuous 18-node scheduler.
"""

import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path

from .followup import NodeRequestBroker, answer, node_request_view, request_key
from .node_probe import LocalModel, write_json


def run_node(model, broker, initial, output, *, max_self_requests=2, node_mode="prose", screen_limit=None,
             detailed=False):
    if node_mode not in ("prose", "template", "catalogue"):
        raise ValueError("unknown node_mode")
    if detailed and node_mode != "catalogue":
        raise ValueError("detailed node replies require catalogue mode")
    answer_fn = answer
    if node_mode in ("template", "catalogue"):
        from .node_templates import answer as answer_fn
    if isinstance(max_self_requests, bool) or not isinstance(max_self_requests, int) or max_self_requests < 0:
        raise ValueError("max_self_requests must be a nonnegative integer")
    if initial["case_id"] != broker.case_id or initial["cutoff_ms"] != broker.cutoff_ms:
        raise ValueError("node input and tool session must share case and cutoff")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pending, checked, completed, skipped = deque(), {}, [], []
    unanswered, errors = [], []
    state = {"case_id": initial["case_id"], "node_id": initial["node_id"],
             "cutoff_ms": initial["cutoff_ms"], "files": [], "executed_self_requests": 0,
             "max_self_requests": max_self_requests, "labels_used": False,
             "stop_reason": "in_progress", "pending_requests": [], "questions": [], "request_errors": [],
             "node_mode": node_mode}
    if detailed:
        state["reply_schema"] = "node_slots_v2"

    def checkpoint():
        state.update(pending_requests=[{"node_id": initial["node_id"], "parent_request_id": parent,
                                       "request": deepcopy(request)} for request, _, parent in pending],
                     questions=deepcopy(unanswered), request_errors=deepcopy(errors),
                     completed_requests=deepcopy(completed), skipped_queries=deepcopy(skipped))
        write_json(output / "manifest.json", state)

    def inspect(packet):
        if packet["node_id"] != initial["node_id"] or packet["case_id"] != initial["case_id"] or packet["cutoff_ms"] != initial["cutoff_ms"]:
            raise ValueError("local investigation cannot change node, case or cutoff")
        step = output / f"step{len(state['files']):03d}"
        step.mkdir()
        packet = deepcopy(packet)
        packet["already_checked"] = list(checked.values())
        write_json(step / "input.json", packet)
        state["active_step"] = {"request_id": packet["request_id"], "directory": str(step.relative_to(output))}
        checkpoint()
        if node_mode in ("template", "catalogue"):
            result = answer_fn(model, packet, step, allow_self_requests=True,
                               screen_limit=screen_limit, catalogue=node_mode == "catalogue", detailed=detailed)
        else:
            result = answer_fn(model, packet, step, allow_self_requests=True)
        state.pop("active_step")
        state["files"].append(str((step / "node_answer.json").relative_to(output)))
        scope = node_request_view(packet)
        for request in result["self_requests"]:
            pending.append((request, scope, packet["request_id"]))
        for question in result["questions"]:
            if question not in [q["question"] for q in unanswered]:
                unanswered.append({"node_id": initial["node_id"], "request_id": packet["request_id"], "question": question})
        errors.extend({"request_id": packet["request_id"], **e} for e in result["request_errors"])
        checkpoint()

    # Initial full-node snapshots do not assert that every available series was
    # already inspected. A broker packet, however, records its exact query.
    if "request" in initial:
        request = initial["request"]
        key = request_key(request, node_request_view(initial))
        checked[key] = {k: deepcopy(request[k]) for k in ("start", "end", "tools", "metrics")}
    checkpoint()
    try:
        inspect(initial)
        while pending:
            request, view, parent = pending.popleft()
            key = request_key(request, view)
            if key in checked:
                skipped.append({"request_id": key, "parent": parent, "reason": "same_data_query_already_checked",
                                "request": deepcopy(request)})
                checkpoint()
                continue
            if state["executed_self_requests"] >= max_self_requests:
                pending.appendleft((request, view, parent))
                state["stop_reason"] = "budget_reached"
                break
            try:
                packet = broker.execute(request, view, output / "queries" / key)
            except Exception:
                pending.appendleft((request, view, parent))
                raise
            checked[key] = {k: deepcopy(request[k]) for k in ("start", "end", "tools", "metrics")}
            completed.append({"request_id": key, "parent": parent, "request": deepcopy(request)})
            state["executed_self_requests"] += 1
            checkpoint()
            inspect(packet)
        if state["stop_reason"] == "in_progress":
            state["stop_reason"] = "no_more_queries"
    except Exception as error:
        state.update(stop_reason="failed", failure=f"{type(error).__name__}: {error}")
        checkpoint()
        raise
    checkpoint()
    return deepcopy(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True, help="Initial snapshot or broker query packet")
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-self-requests", type=int, default=2)
    args = parser.parse_args()
    packet = json.loads(args.seed.read_text())
    # Load only when executed explicitly inside an allocated GPU job.
    journal = args.output.parent / (args.output.name + "-calls.jsonl")
    if args.output.exists() or journal.exists():
        raise ValueError("use a fresh output directory and journal")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NodeRequestBroker(args.prepared, packet["case_id"], packet["cutoff_ms"]) as broker:
        model = LocalModel(args.model, journal, max_context=12288)
        result = run_node(model, broker, packet, args.output, max_self_requests=args.max_self_requests)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
