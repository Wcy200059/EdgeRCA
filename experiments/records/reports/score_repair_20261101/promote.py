"""Apply an explicitly selected, tested variant after the baseline array ends."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from replay_variants import ROOT, source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("candidate", "reasoning", "coverage", "final", "retrieval"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    selected = ROOT / f"{args.variant}_source"
    workspace = ROOT.parents[1]
    manifest = json.loads((ROOT / "manifest.json").read_text())
    expected = (json.loads((ROOT / f"{args.variant}_manifest.json").read_text())["source_sha256"]
                if args.variant in ("coverage", "final", "retrieval") else manifest["source_hashes"][args.variant])
    if source_hash(selected) != expected:
        raise ValueError("Selected source changed since verification")
    if source_hash(workspace) != manifest["baseline_source_sha256"]:
        raise ValueError("Main source changed after baseline; refusing to overwrite other work")
    if not args.check_only and not (ROOT / "coverage_scores.json").exists():
        raise ValueError("All requested verification must finish before promotion")
    if not args.check_only:
        decision = json.loads((ROOT / "promotion_selection.json").read_text())
        if decision["variant"] != args.variant or decision["source_sha256"] != expected:
            raise ValueError("The selected variant has not been approved by the recorded experiment decision")
    changes = []
    for folder in ("edge_rca", "tests"):
        for source in sorted((selected / folder).glob("*.py")):
            relative = source.relative_to(selected)
            baseline = ROOT / "baseline_source" / relative
            target = workspace / relative
            if baseline.exists() and baseline.read_bytes() == source.read_bytes():
                continue
            if baseline.exists():
                if not target.exists() or target.read_bytes() != baseline.read_bytes():
                    raise ValueError("Changed target; refusing to overwrite " + str(relative))
            elif target.exists():
                raise ValueError("New target already exists; refusing to overwrite " + str(relative))
            changes.append((source, target, relative))
    queue = subprocess.run(["squeue", "-h", "-u", str(os.getuid()), "-o", "%i"],
                           check=True, text=True, capture_output=True).stdout.splitlines()
    active = [job for job in queue if job.strip().split("_")[0] in manifest["baseline_arrays"]]
    if args.check_only:
        print(json.dumps({"variant": args.variant, "changed_files": [str(p) for _, _, p in changes],
                          "baseline_jobs_still_active": active, "mutation_performed": False}))
        return
    if active:
        raise ValueError("Baseline jobs are still active; refusing to mix source versions")
    backup = ROOT / "before_promotion"
    backup.mkdir(exist_ok=False)
    for source, target, relative in changes:
        if target.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
        temporary = target.with_suffix(".repair.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    with (ROOT / "main_tests_after_promotion.log").open("w") as handle:
        result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                                cwd=workspace, stdout=handle, stderr=subprocess.STDOUT)
    if source_hash(workspace) != expected:
        raise ValueError("Main source changed during validation; refusing to overwrite concurrent work")
    report = {"variant": args.variant, "source_sha256": expected,
              "changed_files": [str(relative) for _, _, relative in changes], "tests_exit_code": result.returncode}
    if result.returncode:
        # Restore only the exact bytes we just installed, never concurrent work.
        for source, target, relative in changes:
            if target.read_bytes() != source.read_bytes():
                raise ValueError("Concurrent edit during validation; manual inspection required")
        for _, target, relative in changes:
            saved = backup / relative
            if saved.exists():
                shutil.copy2(saved, target)
            else:
                target.unlink()
        report["rolled_back"] = True
    (ROOT / "promotion_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (ROOT / "修复与复测.md").open("a") as handle:
        handle.write("\n## 延后应用结果\n\n" +
                     ("主目录检查失败，已恢复应用前的源码。" if result.returncode else
                      f"原基线作业已结束，已将`{args.variant}_source`合入主目录，主目录全部程序检查通过。") +
                     "详见`promotion_result.json`与`main_tests_after_promotion.log`。\n")
    print(json.dumps(report, ensure_ascii=False))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
