"""Local 9B coordinator -> fixed node tools -> 4B node -> coordinator update.

Separate processes let all phases share one GPU without keeping both models in
memory. Every phase saves its inputs/results and fails visibly if invalid.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .coordination import ask, make_view, validate
from .followup import NodeRequestBroker, answer, request_key
from .node_probe import LocalModel, write_json
from .node_reports import ReportInbox
from .node_inquiry import run_node


def load_inbox(path):
    saved = json.loads(Path(path).read_text())
    inbox = ReportInbox()
    for report in saved["report_history"]:
        inbox.receive(report)
    cases = {r["case_id"] for r in inbox.history}
    if len(cases) != 1:
        raise ValueError("closed-loop probe requires exactly one case")
    return inbox, cases.pop()


def decide(args, updated=False):
    args.output.mkdir(parents=True, exist_ok=False)
    inbox, case = load_inbox(args.inbox)
    supplements, previous, pending, node_pending = [], [], [], []
    cutoff = max(r["cutoff_ms"] for r in inbox.history)
    if updated:
        plan = json.loads((args.plan / "decision.json").read_text())
        if plan["case_id"] != case or plan["cutoff_ms"] != cutoff:
            raise ValueError("plan belongs to another case/cutoff")
        prior_view = json.loads((args.plan / "input.json").read_text())["view"]
        mode = "overview_only" if args.overview_only else "full_reports"
        if prior_view.get("report_detail_mode", "full_reports") != mode:
            raise ValueError("coordinator detail mode cannot change between plan and update; evidence IDs would change")
        previous = plan["reply"]["hypotheses"]
        manifest = json.loads((args.answers / "manifest.json").read_text())
        if manifest["case_id"] != case or manifest["cutoff_ms"] != cutoff:
            raise ValueError("node answers belong to another case/cutoff")
        pending = manifest["pending_requests"]
        node_pending = manifest.get("pending_node_requests", [])
        supplements = [json.loads((args.answers / f).read_text())["answer"] for f in manifest["files"]]
    view, binding = make_view(inbox, case, cutoff, args.nodes, supplements, previous, pending, node_pending,
                              include_report_details=not args.overview_only)
    write_json(args.output / "input.json", {"view": view, "binding": binding})
    print(json.dumps({"phase": "update" if updated else "plan", "model": str(args.model),
                      "evidence": len(view["evidence"]), "supplements": len(supplements)}, ensure_ascii=False), flush=True)
    model = LocalModel(args.model, args.output / "calls.jsonl", max_context=16384)
    reply, attempts = ask(model, view, "coordinator_update" if updated else "coordinator_plan",
                          allow_partial_requests=not updated)
    record = {"case_id": case, "cutoff_ms": cutoff, "reply": reply, "attempts": attempts,
              "model": str(args.model), "evidence_count": len(view["evidence"]),
              "supplement_count": len(supplements), "peak_gpu_gib": model.torch.cuda.max_memory_allocated() / 1024**3,
              "protocol_valid": reply is not None and attempts[-1]["valid"],
              "actionable": reply is not None, "semantics_verified": False, "labels_used": False,
              "rca_evaluated": False, "inbox_sha256": hashlib.sha256(args.inbox.read_bytes()).hexdigest()}
    write_json(args.output / "decision.json", record)
    if reply is None:
        raise RuntimeError("coordinator failed output validation; see decision.json, no automatic fallback answer")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def tools(args):
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.plan / "decision.json").read_text())
    view = json.loads((args.plan / "input.json").read_text())["view"]
    validate(plan["reply"], view)
    # Repeated model questions must not execute or consume the request budget twice.
    requests, seen = [], set()
    for request in plan["reply"]["requests"]:
        key = request_key(request, view)
        if key not in seen:
            requests.append(request)
            seen.add(key)
    execute, pending = requests[:args.max_requests], requests[args.max_requests:]
    files = []
    if execute:
        with NodeRequestBroker(args.prepared, view["case_id"], view["cutoff_ms"]) as broker:
            for request in execute:
                rid = request_key(request, view)
                packet = broker.execute(request, view, args.output / rid)
                filename = rid + ".json"
                write_json(args.output / filename, packet)
                files.append(filename)
                print(json.dumps({"request_id": rid, "node_id": request["node_id"],
                                  "facts": len(packet["facts"]), "tools": packet["tools_executed"]}, ensure_ascii=False), flush=True)
    write_json(args.output / "manifest.json", {"case_id": view["case_id"], "cutoff_ms": view["cutoff_ms"],
               "files": files, "pending_requests": pending, "labels_used": False,
               "duplicate_requests": len(plan["reply"]["requests"]) - len(requests),
               "note": "只执行本次设置的请求数，其余保存待办，不视为已完成。"})


def node(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    files, node_pending = [], []
    if manifest["files"]:
        if args.max_self_requests < 0:
            raise ValueError("--max-self-requests must be nonnegative")
        if args.max_self_requests and args.prepared is None:
            raise ValueError("--prepared is required to enable node-initiated tools")
        def process(broker=None):
            for filename in manifest["files"]:
                packet = json.loads((args.inputs / filename).read_text())
                folder = args.output / packet["request_id"]
                if broker is None:
                    folder.mkdir()
                    result = answer(model, packet, folder)
                    files.append(packet["request_id"] + "/node_answer.json")
                else:
                    result = run_node(model, broker, packet, folder, max_self_requests=args.max_self_requests)
                    files.extend(packet["request_id"] + "/" + f for f in result["files"])
                    node_pending.extend(result["pending_requests"])
                print(json.dumps({"request_id": packet["request_id"], "node_id": packet["node_id"],
                                  "local_queries_enabled": broker is not None}, ensure_ascii=False), flush=True)
        if args.max_self_requests:
            with NodeRequestBroker(args.prepared, manifest["case_id"], manifest["cutoff_ms"]) as broker:
                model = LocalModel(args.model, args.output / "calls.jsonl", max_context=12288)
                process(broker)
        else:
            model = LocalModel(args.model, args.output / "calls.jsonl", max_context=12288)
            process()
    write_json(args.output / "manifest.json", {**manifest, "files": files, "node_model": str(args.model),
                                               "pending_node_requests": node_pending})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="phase", required=True)
    for name in ("plan", "update"):
        p = subs.add_parser(name)
        p.add_argument("--inbox", type=Path, required=True)
        p.add_argument("--model", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--nodes", nargs="+", default=["device:Tomcat04", "device:dockerA2"])
        p.add_argument("--overview-only", action="store_true",
                       help="Keep initial node details in the archive; load facts via selected queries")
        if name == "update":
            p.add_argument("--plan", type=Path, required=True)
            p.add_argument("--answers", type=Path, required=True)
    p = subs.add_parser("tools")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-requests", type=int, default=1)
    p = subs.add_parser("node")
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--prepared", type=Path)
    p.add_argument("--max-self-requests", type=int, default=0,
                   help="Enable bounded local tool requests; requires --prepared")
    args = parser.parse_args()
    if args.phase in ("plan", "update"):
        decide(args, updated=args.phase == "update")
    elif args.phase == "tools":
        if args.max_requests < 1:
            parser.error("--max-requests must be positive")
        tools(args)
    else:
        node(args)


if __name__ == "__main__":
    main()
