"""Rank complete alternative diagnoses without treating them as extra faults."""

from copy import deepcopy
import json

from .coordinator import occurrence_ms
from .data import format_time


SCHEMA = "ranked_fault_sets_v1"

INSTRUCTION = """\n此外输出alternatives列表，按可信程度给最多两个备选，首选仍放faults。
格式：{"faults":[首选中的故障对象],"alternatives":[{"faults":[另一组完整解释]}]}。
每一组里的故障数都遵循公开题目；单故障题的备选各只有一个故障，不能把竞争猜测当成同时发生的多故障。
每个故障沿用component/reason/time/evidence_ids/explanation字段，依据只能来自当前可见证据。
尚未排除、仍有当前观测可引用的其他解释，也应作为备选比较；在explanation说明哪里不确定。
优先比较previous_hypotheses中的不同解释，不仅考虑标为supported的一个。不能只为凑三个而枚举答案。
没有其他有线索的解释就返回空列表；不要用重复首选、调换双故障顺序来凑备选。
"""


def prediction(faults, view):
    rows = []
    for fault in faults:
        when = ""
        if fault and fault.get("time"):
            ms = occurrence_ms(fault["time"], view["window_start_ms"], view["cutoff_ms"])
            when = format_time(ms)[:19].replace("T", " ")
        rows.append({"root cause occurrence datetime": when,
                     "root cause component": (fault or {}).get("component") or "",
                     "root cause reason": (fault or {}).get("reason") or ""})
    return rows


def identity(faults):
    return tuple(sorted(tuple(f.get(k) or "" for k in ("component", "reason", "time")) for f in faults))


def build(primary, alternatives, view, count, maximum, validator):
    """Bad alternatives do not invalidate a separately validated first choice."""
    candidates, rejected, seen = [], [], set()
    if primary is not None:
        candidates.append({"rank": 1, "faults": deepcopy(primary["faults"]),
                           "prediction": prediction(primary["faults"], view)})
        seen.add(identity(primary["faults"]))
    if not isinstance(alternatives, list):
        rejected.append({"error": "alternatives must be a list"})
        alternatives = []
    if len(alternatives) > maximum - 1:
        rejected.append({"error": "extra alternatives exceed the configured maximum"})
    for index, item in enumerate(alternatives[:maximum - 1]):
        try:
            value = validator(item, view, count)
            faults = value["faults"]
            if not any(f[k] for f in faults for k in ("component", "reason", "time")):
                raise ValueError("an all-unknown group is not an additional candidate")
            key = identity(faults)
            if key in seen:
                raise ValueError("duplicate candidate, including fault-order permutations")
            if primary is None:
                raise ValueError("cannot rank alternatives without a valid first choice")
            seen.add(key)
            candidates.append({"rank": len(candidates) + 1, "faults": deepcopy(faults),
                               "prediction": prediction(faults, view)})
        except (ValueError, TypeError, KeyError) as error:
            rejected.append({"alternative": index + 1, "error": str(error)})
    return {"schema": SCHEMA, "max_candidates": maximum, "candidates": candidates,
            "rejected": rejected, "labels_used": False,
            "note": "每个候选是一整组故障解释；不同候选不是同时发生的额外故障。"}


def score(record, first_prediction, scoring_points, *, time_tolerance_seconds):
    """Runs only in the separate label-reading evaluation process."""
    from .scoring import evaluate
    if record.get("schema") != SCHEMA or record.get("max_candidates") not in (2, 3):
        raise ValueError("unknown ranked-answer schema or candidate limit")
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > record["max_candidates"]:
        raise ValueError("ranked candidates must be a bounded list")
    results = []
    for rank, item in enumerate(candidates, 1):
        if item.get("rank") != rank or not isinstance(item.get("prediction"), list):
            raise ValueError("candidate ranks must be contiguous and have prediction lists")
        if rank == 1 and item["prediction"] != first_prediction:
            raise ValueError("ranked first choice differs from frozen prediction.json")
        passed, failed, value = evaluate(json.dumps(item["prediction"], ensure_ascii=False), scoring_points,
                                          time_tolerance_seconds=time_tolerance_seconds)
        results.append({"rank": rank, "score": value, "fully_correct": value == 1.0,
                        "passed": passed, "failed": failed})
    hits = [item["rank"] for item in results if item["fully_correct"]]
    return {"candidate_count": len(results), "candidate_limit": record["max_candidates"],
            "fully_correct_within_top3": bool(hits), "first_correct_rank": min(hits) if hits else None,
            "candidates": results, "time_tolerance_seconds": time_tolerance_seconds}
