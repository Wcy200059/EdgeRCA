"""Run paired final-stage variants against frozen evidence without reading labels."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parent


def source_hash(path):
    digest = hashlib.sha256()
    for file in sorted((path / "edge_rca").glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def runtime(name):
    directory = ROOT / f"{name}_source" / "edge_rca"
    namespace = "replay_" + name
    spec = importlib.util.spec_from_file_location(namespace, directory / "__init__.py",
                                                submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = package
    spec.loader.exec_module(package)
    return importlib.import_module(namespace + ".case_runtime")


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest["source_hashes"].items():
        if source_hash(ROOT / f"{name}_source") != expected:
            raise ValueError("Frozen source changed: " + name)
    modules = {name: runtime(name) for name in manifest["variants"]}
    from transformers import set_seed
    model = modules["baseline"].LocalModel(
        Path("/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-9B"), ROOT / "unused_calls.jsonl",
        max_context=20480)
    started = time.monotonic()
    completed = []
    for row in manifest["rows"]:
        case = ROOT.parent / f"bank_case_{154057 if row < 46 else 154058}_{row}"
        input_file = case / "coordinator_1800/last_input.json"
        if hashlib.sha256(input_file.read_bytes()).hexdigest() != manifest["input_hashes"][str(row)]:
            raise ValueError("Saved evidence changed")
        instruction = json.loads((case / "frames/public_question.json").read_text())["instruction"]
        for name in manifest["variants"]:
            output = ROOT / "replays" / str(row) / name
            output.mkdir(parents=True, exist_ok=False)
            set_seed(42 + row)
            model.journal_path = output / "calls.jsonl"
            model.calls = []
            view = json.loads(input_file.read_text())["view"]
            modules[name].final_answer(model, view, instruction, output)
            completed.append({"row": row, "variant": name, "seed": 42 + row,
                              "model_calls": len(model.calls),
                              "seconds": sum(call["seconds"] for call in model.calls)})
            modules[name].save(ROOT / "inference_progress.json", {
                "completed": completed, "expected": len(manifest["rows"]) * len(manifest["variants"]),
                "labels_used": False, "elapsed_seconds": time.monotonic() - started})
            print(json.dumps(completed[-1]), flush=True)
    modules["baseline"].save(ROOT / "inference_complete.json", {
        "labels_used": False, "elapsed_seconds": time.monotonic() - started,
        "completed": completed, "peak_gpu_gib": model.torch.cuda.max_memory_allocated() / 1024**3})


if __name__ == "__main__":
    main()
