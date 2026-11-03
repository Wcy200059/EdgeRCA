"""Measure one 4B generation with and without the flash-linear-attention path.

This is a throughput probe only. It does not read Bank data or make any RCA
claim; it loads the configured local model, warms up, and reports generated
tokens per second.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


PROMPT = (
    "你是边缘节点信息整理助手。只根据工具事实整理本节点观测，不判断全局根因。"
    "请说明哪些指标发生变化、哪些规则触发或解除、哪些数值缺少对照，并保留未知。"
    "不要编造数值，不要把普通GC写成OOM，不要把调用耗时写成网络延迟。"
)
SAMPLE = (
    "工具事实：Redis02 在 18:10:00 触发内存占用告警，首次采样值为 98%，"
    "同一指标整窗中位数为 23%，范围 11%—98%；18:11:00 无缓存内存为 92.6%，"
    "此前中位数为 10.6%；CPU 使用率为 8.1%，此前中位数为 0.8%；"
    "本节点到 18:30 没有日志记录；规则在 18:16 解除，但解除只表示规则条件不再满足。"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.tokens < 1:
        raise ValueError("--tokens must be positive")

    try:
        import fla
        fla_version = getattr(fla, "__version__", "unknown")
    except Exception as error:  # Importing fla on a CPU node also fails; record it.
        fla_version = f"unavailable: {type(error).__name__}: {error}"

    torch.set_num_threads(2)
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.bfloat16,
        device_map={"": args.device}, attn_implementation="sdpa").eval()
    messages = [{"role": "system", "content": PROMPT},
                {"role": "user", "content": SAMPLE + "\n请用不超过160字整理。"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(args.device)
    input_tokens = int(encoded.input_ids.shape[1])
    with torch.inference_mode():
        model.generate(**encoded, max_new_tokens=16, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **encoded, max_new_tokens=args.tokens, do_sample=True, temperature=.7,
            top_p=.8, top_k=20, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    output_tokens = int(output.shape[1] - input_tokens)
    result = {
        "python": sys.executable,
        "fla": fla_version,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "seconds": seconds,
        "tokens_per_second": output_tokens / seconds if seconds else None,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "note": "单次短生成吞吐检查；不读取Bank数据，不评价RCA。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
