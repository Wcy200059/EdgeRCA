"""Second-seed paired confirmation; inference and scoring are separate processes."""
import argparse
import hashlib
import json
from pathlib import Path

from replay_variants import ROOT, runtime, source_hash


def infer():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    final = json.loads((ROOT / "final_manifest.json").read_text())
    if source_hash(ROOT / "final_source") != final["source_sha256"]:
        raise ValueError("Final source changed")
    if source_hash(ROOT / "baseline_source") != manifest["baseline_source_sha256"]:
        raise ValueError("Baseline source changed")
    modules = {name: runtime(name) for name in ("baseline", "final")}
    from transformers import set_seed
    model = modules["baseline"].LocalModel(Path("/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-9B"),
                                           ROOT / "unused_confirm_calls.jsonl", max_context=20480)
    completed = []
    for row in manifest["rows"]:
        case = ROOT.parent / f"bank_case_{154057 if row < 46 else 154058}_{row}"
        input_file = case / "coordinator_1800/last_input.json"
        if hashlib.sha256(input_file.read_bytes()).hexdigest() != manifest["input_hashes"][str(row)]:
            raise ValueError("Saved evidence changed")
        instruction = json.loads((case / "frames/public_question.json").read_text())["instruction"]
        for name, module in modules.items():
            output = ROOT / "confirmation" / str(row) / name
            output.mkdir(parents=True, exist_ok=False)
            set_seed(1042 + row)
            model.calls = []
            model.journal_path = output / "calls.jsonl"
            module.final_answer(model, json.loads(input_file.read_text())["view"], instruction, output)
            completed.append({"row": row, "variant": name, "seed": 1042 + row,
                "model_calls": len(model.calls), "seconds": sum(call["seconds"] for call in model.calls)})
            module.save(ROOT / "confirmation_progress.json", {"completed": completed, "expected": 2 * len(manifest["rows"]), "labels_used": False})
            print(json.dumps(completed[-1]), flush=True)
    modules["final"].save(ROOT / "confirmation_complete.json", {"completed": completed, "labels_used": False,
        "peak_gpu_gib": model.torch.cuda.max_memory_allocated() / 1024**3})


def evaluate():
    if not (ROOT / "confirmation_complete.json").exists():
        raise ValueError("Freeze all inference outputs before scoring")
    from edge_rca.scoring import evaluate as score
    manifest = json.loads((ROOT / "manifest.json").read_text())
    records = []
    for row in manifest["rows"]:
        case = ROOT.parent / f"bank_case_{154057 if row < 46 else 154058}_{row}"
        truth = json.loads((case / "evaluation/result.json").read_text())["scoring_points"]
        for name in ("baseline", "final"):
            output = ROOT / "confirmation" / str(row) / name
            data = (output / "prediction.json").read_bytes()
            record = json.loads((output / "final_answer.json").read_text())
            records.append({"row": row, "variant": name, "valid": record["reply"] is not None,
                "prediction_sha256": hashlib.sha256(data).hexdigest(),
                "scores": {str(t): score(data.decode(), truth, time_tolerance_seconds=t)[2] for t in (60, 300)}})
    summary = {}
    for name in ("baseline", "final"):
        selected = [r for r in records if r["variant"] == name]
        summary[name] = {"n": len(selected), "valid": sum(r["valid"] for r in selected),
            "scores": {str(t): {"full": sum(r["scores"][str(t)] == 1 for r in selected),
                                "mean": sum(r["scores"][str(t)] for r in selected) / len(selected)} for t in (60, 300)}}
    (ROOT / "confirmation_scores.json").write_text(json.dumps({"summary": summary, "rows": records,
        "scope": "Second seed on the same development cases and saved pre-final evidence; not a new dataset accuracy run."}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("infer", "evaluate"))
    args = parser.parse_args()
    (infer if args.phase == "infer" else evaluate)()
