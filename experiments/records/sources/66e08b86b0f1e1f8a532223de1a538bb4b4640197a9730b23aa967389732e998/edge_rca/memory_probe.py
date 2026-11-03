"""Tiny-model check of frozen, successive versions of one real signal record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .node_probe import LocalModel, decode_object, write_json
from .rolling_memory import memory_messages, validate_memory_reply


def run(inputs, output, model_path):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        (output / "source" / path.name).write_bytes(path.read_bytes())
    data = json.loads(Path(inputs).read_text())
    if data["labels_used"]:
        raise ValueError("online node inputs must not use answers")
    requests = data["requests"]
    if not requests:
        raise ValueError("no real memory requests available")
    write_json(output / "frozen_inputs.json", data)
    started = time.monotonic()
    model = LocalModel(model_path, output / "calls.jsonl")
    results = []
    for packet in requests:
        item = {"request_id": packet["request_id"], "cutoff_ms": packet["snapshot_cutoff_ms"],
                "observation_id": packet["fact"]["observation_id"],
                "version": packet["fact"]["version"], "attempts": [], "semantics_verified": False}
        messages = memory_messages(packet)
        for attempt in range(2):
            raw = model.invoke(messages, None, 300, f"{item['request_id']}/{attempt}")
            try:
                reply = decode_object(raw)
                validate_memory_reply(packet, reply, qualitative_only=True)
            except (ValueError, KeyError, TypeError) as error:
                item["attempts"].append({"valid": False, "error": str(error)})
                messages = memory_messages(packet)
                messages[-1]["content"] += "\n上次未通过：" + str(error)
                continue
            item["attempts"].append({"valid": True})
            item["reply"] = reply
            item["cpu_fact"] = packet["fact"]
            item["handoff_policy"] = "程序保留时间线、数值、状态及来源；模型只提供未验证的文字说明。"
            break
        results.append(item)
        write_json(output / (item["request_id"] + ".json"), item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    write_json(output / "summary.json", {
        "requests": len(results), "accepted": sum("reply" in x for x in results),
        "first_try": sum(x["attempts"][0]["valid"] for x in results),
        "model": str(model_path), "model_calls": len(model.calls),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_gib": model.torch.cuda.max_memory_allocated() / 1024**3,
        "labels_used": False, "previous_model_prose_used": False,
        "prompt_policy": "qualitative_note_with_unchanged_cpu_timeline_v3",
        "limits": ["真实连续时刻的冻结记录试验，未让模型处理全部节点。",
                   "引用与状态词检查不等于完整语义核验，也不是RCA评分。"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output, args.model)
