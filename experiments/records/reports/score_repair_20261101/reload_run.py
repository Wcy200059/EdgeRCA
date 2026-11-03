"""Verify bounded retrieval on only the cases whose rejected references exist in the archive."""
import argparse
import hashlib
import json
from pathlib import Path

from replay_variants import ROOT, runtime, source_hash


def infer():
    manifest = json.loads((ROOT / "retrieval_manifest.json").read_text())
    inputs = json.loads((ROOT / "manifest.json").read_text())["input_hashes"]
    if source_hash(ROOT / "retrieval_source") != manifest["source_sha256"]:
        raise ValueError("Frozen retrieval source changed")
    module = runtime("retrieval")
    from transformers import set_seed
    model = module.LocalModel(Path("/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-9B"),
                              ROOT / "unused_reload_calls.jsonl", max_context=20480)
    complete = []
    for item in manifest["affected"]:
        row, seed = item["row"], item["seed_base"] + item["row"]
        case = ROOT.parent / f"bank_case_{154057 if row < 46 else 154058}_{row}"
        input_file = case / "coordinator_1800/last_input.json"
        if hashlib.sha256(input_file.read_bytes()).hexdigest() != inputs[str(row)]:
            raise ValueError("Saved evidence changed")
        instruction = json.loads((case / "frames/public_question.json").read_text())["instruction"]
        output = ROOT / "reload_replays" / str(seed) / str(row)
        output.mkdir(parents=True, exist_ok=False)
        set_seed(seed)
        model.calls = []
        model.journal_path = output / "calls.jsonl"
        module.final_answer(model, json.loads(input_file.read_text())["view"], instruction, output)
        folder, variant = ("replays", "coverage") if item["seed_base"] == 42 else ("confirmation", "final")
        original = ROOT / folder / str(row) / variant / "calls.jsonl"
        before = json.loads(original.read_text().splitlines()[0])
        after = json.loads(model.journal_path.read_text().splitlines()[0])
        same_input = before["messages"] == after["messages"]
        same_reply = before["raw_reply"] == after["raw_reply"]
        complete.append({**item, "seed": seed, "same_first_input": same_input,
                         "same_first_reply": same_reply, "calls": len(model.calls)})
        print(json.dumps(complete[-1]), flush=True)
    module.save(ROOT / "reload_complete.json", {"completed": complete, "labels_used": False,
        "peak_gpu_gib": model.torch.cuda.max_memory_allocated() / 1024**3})


def evaluate():
    if not (ROOT / "reload_complete.json").exists():
        raise ValueError("Freeze inference outputs first")
    from edge_rca.scoring import evaluate as score
    runs = json.loads((ROOT / "reload_complete.json").read_text())["completed"]
    results = []
    for run in runs:
        row = run["row"]
        case = ROOT.parent / f"bank_case_{154057 if row < 46 else 154058}_{row}"
        truth = json.loads((case / "evaluation/result.json").read_text())["scoring_points"]
        output = ROOT / "reload_replays" / str(run["seed"]) / str(row)
        data = (output / "prediction.json").read_bytes()
        answer = json.loads((output / "final_answer.json").read_text())
        results.append({**run, "valid": answer["reply"] is not None,
            "prediction": json.loads(data), "prediction_sha256": hashlib.sha256(data).hexdigest(),
            "attempts": answer["attempts"],
            "scores": {str(t): score(data.decode(), truth, time_tolerance_seconds=t)[2] for t in (60, 300)}})
    (ROOT / "reload_scores.json").write_text(json.dumps({"rows": results,
        "scope": "Targeted regression of the only archived-reference rejection observed in both seeded replay groups."}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("infer", "evaluate"))
    args = parser.parse_args()
    (infer if args.phase == "infer" else evaluate)()
