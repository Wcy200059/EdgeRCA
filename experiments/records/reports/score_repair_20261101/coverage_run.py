"""Separate follow-up experiment for evidence retention; reuse the same frozen inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from replay_variants import ROOT, runtime, source_hash


def infer():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    coverage = json.loads((ROOT / "coverage_manifest.json").read_text())
    if source_hash(ROOT / "coverage_source") != coverage["source_sha256"]:
        raise ValueError("Frozen coverage source changed")
    module = runtime("coverage")
    from transformers import set_seed
    model = module.LocalModel(Path("/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-9B"),
                              ROOT / "unused_coverage_calls.jsonl", max_context=20480)
    completed = []
    for row in manifest["rows"]:
        case = ROOT.parent / f"bank_case_{154057 if row < 46 else 154058}_{row}"
        input_file = case / "coordinator_1800/last_input.json"
        if hashlib.sha256(input_file.read_bytes()).hexdigest() != manifest["input_hashes"][str(row)]:
            raise ValueError("Saved evidence changed")
        instruction = json.loads((case / "frames/public_question.json").read_text())["instruction"]
        output = ROOT / "replays" / str(row) / "coverage"
        output.mkdir(parents=True, exist_ok=False)
        set_seed(42 + row)
        model.calls = []
        model.journal_path = output / "calls.jsonl"
        module.final_answer(model, json.loads(input_file.read_text())["view"], instruction, output)
        completed.append({"row": row, "variant": "coverage", "model_calls": len(model.calls),
                          "seconds": sum(call["seconds"] for call in model.calls)})
        module.save(ROOT / "coverage_progress.json", {"completed": completed, "expected": len(manifest["rows"]), "labels_used": False})
        print(json.dumps(completed[-1]), flush=True)
    module.save(ROOT / "coverage_complete.json", {"completed": completed, "labels_used": False,
        "peak_gpu_gib": model.torch.cuda.max_memory_allocated() / 1024**3})


def evaluate():
    if not (ROOT / "coverage_complete.json").exists():
        raise ValueError("Inference must finish before independent evaluation")
    from edge_rca.scoring import evaluate as score
    manifest = json.loads((ROOT / "manifest.json").read_text())
    rows = json.loads((ROOT / "paired_scores.json").read_text())["rows"]
    for row in manifest["rows"]:
        case = ROOT.parent / f"bank_case_{154057 if row < 46 else 154058}_{row}"
        truth = json.loads((case / "evaluation/result.json").read_text())["scoring_points"]
        output = ROOT / "replays" / str(row) / "coverage"
        data = (output / "prediction.json").read_bytes()
        rows.append({"row": row, "variant": "coverage", "prediction": json.loads(data),
            "split": "development" if row in manifest["development_rows"] else "holdout",
            "valid": json.loads((output / "final_answer.json").read_text())["reply"] is not None,
            "prediction_sha256": hashlib.sha256(data).hexdigest(),
            "scores": {str(t): score(data.decode(), truth, time_tolerance_seconds=t)[2] for t in (60, 300)}})
    summary = {}
    for variant in [*manifest["variants"], "coverage"]:
        selected = [r for r in rows if r["variant"] == variant]
        summary[variant] = {"n": len(selected), "valid": sum(r["valid"] for r in selected),
            "scores": {str(t): {"full": sum(r["scores"][str(t)] == 1 for r in selected),
                                "mean": sum(r["scores"][str(t)] for r in selected) / len(selected)} for t in (60, 300)}}
    result = {"summary": summary, "rows": rows,
              "scope": "Adaptive development comparison. Coverage repair used aggregate error diagnostics; these are not untouched test cases."}
    (ROOT / "coverage_scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("infer", "evaluate"))
    args = parser.parse_args()
    (infer if args.phase == "infer" else evaluate)()
