"""Exercise one local tiny model on representative, frozen alert pages.

The SOP chooses each read-only task. A successful reply covers that page only;
unselected pages and unsuccessful facts remain pending, not healthy.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

from .compact_tools import tool_schema
from .fact_cards import basic_fact_checks, fact_card
from .node_probe import LocalModel, decode_object, write_json
from .staged_probe import call_messages, decode_summary, summary_messages, validate_findings


def execute_page_call(call, request, page):
    if (page["request_id"] != request["request_id"]
            or request["node_id"] != request["bundle"]["context"]["node_id"]):
        raise ValueError("page belongs to a different request or node")
    if (not isinstance(call, dict) or set(call) != {"name", "arguments"}
            or call["name"] != page["tool"] or call["arguments"] != {"request_id": page["page_id"]}):
        raise ValueError("only the bound tool and page request_id are allowed")
    facts = {f["id"]: f for values in request["bundle"]["tools"].values() for f in values}
    if (not 1 <= len(page["facts"]) <= 2 or len({f["id"] for f in page["facts"]}) != len(page["facts"])
            or set(page["mandatory_ids"]) != {f["id"] for f in page["facts"]}):
        raise ValueError("each page must declare one or two distinct required facts")
    if any(f["id"] not in facts or f != facts[f["id"]] for f in page["facts"]):
        raise ValueError("page facts disagree with the frozen source bundle")
    return {"request_id": page["page_id"], "node_id": request["node_id"],
            "scope": deepcopy(request["bundle"]["context"]["request_scope"]),
            "facts": deepcopy(page["facts"]), "observed_links": []}


def page_prompt(value, page):
    messages = summary_messages(value, [], [], page["mandatory_ids"], page["tool"])
    body = json.loads(messages[-1]["content"])
    body["facts"] = [{"id": f["id"],
                      "signal": f.get("signal", f.get("metric", f.get("kind"))),
                      "observation": fact_card(f, value["scope"], page["tool"])}
                     for f in value["facts"]]
    body["task"] = ("只整理本小任务给出的事实，每个编号分别输出一项。准确陈述其数值、方向、时间和限制；"
                    "长指标名由程序按编号附回，无需抄写。当前值、基线、阈值不能混用；"
                    "没有给出的对照不要补充。解除、加重等状态必须保留。必须完整输出："
                    + "、".join(page["mandatory_ids"]) + "。")
    messages[-1]["content"] = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    return messages


def run(inputs, output, model_path, device):
    inputs, output = Path(inputs), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        (snapshot / path.name).write_bytes(path.read_bytes())
    started = time.monotonic()
    manifest = json.loads((inputs / "manifest.json").read_text())
    # Refuse known CPU audit failures before loading weights.
    audit_path = inputs.parent / "summary.json"
    audit = json.loads(audit_path.read_text())
    if audit["coverage_error_tasks"] or not audit["full_window_replayed"]:
        raise ValueError("CPU evidence coverage did not pass")
    model = LocalModel(model_path, output / "calls.jsonl", device=device)
    records = []
    for entry in manifest["requests"]:
        request = json.loads((inputs / entry["path"]).read_text())
        for page in request["selected_pages"]:
            pid = page["page_id"]
            record = {"page_id": pid, "request_id": request["request_id"], "node_id": request["node_id"],
                "cutoff_ms": request["cutoff_ms"], "tool": page["tool"], "features": page["features"],
                "mandatory_ids": page["mandatory_ids"], "tool_attempts": [], "summary_attempts": [],
                "semantics_verified": False, "connections_routed_separately": True}
            messages = call_messages(page["tool"], pid)
            value = None
            for attempt in range(2):
                raw = model.invoke(messages, [tool_schema(page["tool"])], 120, f"{pid}/call/{attempt}")
                try:
                    value = execute_page_call(decode_object(raw, True), request, page)
                except (ValueError, TypeError, KeyError) as error:
                    record["tool_attempts"].append({"accepted": False, "error": str(error)})
                    messages = call_messages(page["tool"], pid)
                    messages[-1]["content"] += f"上次没有执行：{error}。"
                    continue
                record["tool_attempts"].append({"accepted": True})
                break
            if value is not None:
                messages = page_prompt(value, page)
                for attempt in range(2):
                    try:
                        raw = model.invoke(messages, None, 384, f"{pid}/summary/{attempt}")
                        checked = validate_findings(decode_summary(raw), value["facts"], page["mandatory_ids"])
                        issues = basic_fact_checks(checked["reply"], value["facts"], page["tool"])
                        if issues:
                            raise ValueError("；".join(issues))
                    except (ValueError, TypeError, KeyError) as error:
                        record["summary_attempts"].append({"valid": False, "error": str(error)})
                        messages = page_prompt(value, page)
                        messages[-1]["content"] += "\n上次未通过：" + str(error) + "。只输出每个编号的短观测数组。"
                        continue
                    record["summary_attempts"].append({"valid": True})
                    record.update(checked)
                    record["basic_fact_checks_passed"] = True
                    record["tool_facts"] = value["facts"]
                    record["source_references"] = {i: deepcopy(request["bundle"]["catalog"][i]) for i in page["mandatory_ids"]}
                    break
            record["pending_ids"] = [] if "reply" in record else list(page["mandatory_ids"])
            records.append(record)
            write_json(output / f"{pid}.json", record)
            print(json.dumps({"page": pid, "node": request["node_id"], "valid": "reply" in record,
                              "pending": record["pending_ids"]}), flush=True)
    expected = {f for r in records for f in r["features"]}
    passed = {f for r in records if "reply" in r for f in r["features"]}
    summary = {"inputs": str(inputs.resolve()), "model": str(model_path), "device": device,
        "pages": len(records), "nodes_tested": sorted({r["node_id"] for r in records}),
        "tool_first_try": sum(r["tool_attempts"][0]["accepted"] for r in records),
        "summary_first_try": sum(bool(r["summary_attempts"]) and r["summary_attempts"][0]["valid"] for r in records),
        "summary_valid": sum("reply" in r for r in records),
        "facts_expected": sum(len(r["mandatory_ids"]) for r in records),
        "facts_pending": sum(len(r["pending_ids"]) for r in records),
        "model_calls": len(model.calls), "calls": model.calls,
        "covered_features": sorted(expected), "valid_format_features": sorted(passed),
        "failed_feature_coverage": sorted(expected - passed), "semantics_verified": False,
        "acceptance_policy": "format_citations_and_narrow_fact_checks_v2",
        "labels_used": False, "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_allocated_gib": model.torch.cuda.max_memory_allocated() / 1024**3 if device != "cpu" else 0,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob("*.py")},
        "limits": ["代表性页的格式/引用通过不等于语义正确，需人工对照统计。",
                   "未运行全部计划页、后台并发、协调者或最终RCA评分。"]}
    write_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args()
    try:
        run(args.inputs, args.output, args.model, args.device)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
