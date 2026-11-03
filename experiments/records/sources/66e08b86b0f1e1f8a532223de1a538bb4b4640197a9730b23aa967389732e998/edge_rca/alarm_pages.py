"""Split required alert observations into small, loss-accounted model tasks."""

from __future__ import annotations

from copy import deepcopy
import json

from .compact_tools import TOOL_ORDER, response_packet
from .data import format_time
from .staged_probe import brief_facts


def enrich_alert_facts(result, bundle):
    """Keep the actual trigger value/time beside later aggregate statistics.

    A median over an episode cannot stand in for a triggering spike. These are
    independent CPU facts, not model-created judgments or fault answer labels.
    """
    out = deepcopy(bundle)
    for i, fact in enumerate(f for f in result["observations"] if f["modality"] == "alert"):
        modality = fact["rule"].split(".")[0]
        tool = {"metric": "get_metrics", "log": "get_logs", "trace": "get_calls"}.get(modality)
        if tool is None:
            raise ValueError("device alert modality is unsupported")
        eid = f"a{i + 1}"
        out["tools"][tool].append({"id": eid, "kind": "alert_evaluation", "rule": fact["rule"],
            "action": fact["action"], "signal": fact["signal"],
            "time": format_time(fact["at_ms"]), "details": deepcopy(fact["details"])})
        out["catalog"][eid] = {"observation_id": fact["observation_id"],
            "evidence_ids": deepcopy(fact["evidence_ids"]), "cpu_fact": deepcopy(fact)}
    return out


def plan_pages(bundle, request_id, max_facts=2, max_characters=4200, extra_required=()):
    if max_facts not in (1, 2) or max_characters < 500:
        raise ValueError("tiny tasks allow one or two facts and a positive text budget")
    required = set(response_packet(bundle)["scope"]["must_report_ids"])
    required.update(extra_required)
    required.update(f["id"] for facts in bundle["tools"].values() for f in facts
                    if f.get("kind") == "alert_evaluation" or
                    any(f.get("quality", {}).get(k) for k in ("invalid", "conflicts", "counter_resets")))
    calls = bundle["tools"]["get_calls"]
    if any(a["rule"].startswith("trace.") for a in bundle["context"]["alerts"]):
        complete = [f for f in calls if f.get("kind") == "minute_latency" and not f["partial"]]
        if complete:
            required.add(complete[-1]["id"])
    pages = []
    for tool in TOOL_ORDER:
        facts = [f for f in bundle["tools"][tool] if f["id"] in required]
        # Targets are explicit. Other facts remain accounted for in the bundle.
        group = []
        for fact in facts:
            value = {"scope": bundle["context"]["request_scope"], "facts": group + [fact], "observed_links": []}
            size = len(json.dumps(brief_facts(value), ensure_ascii=False))
            if group and (len(group) >= max_facts or size > max_characters):
                pages.append({"tool": tool, "facts": deepcopy(group)})
                group = []
            group.append(fact)
        if group:
            pages.append({"tool": tool, "facts": deepcopy(group)})
    planned = [f["id"] for page in pages for f in page["facts"]]
    if len(planned) != len(set(planned)) or set(planned) != required:
        raise ValueError("required observations lost or duplicated while paging")
    for i, page in enumerate(pages):
        page.update(page_id=f"{request_id}-p{i + 1}", request_id=request_id,
                    mandatory_ids=[f["id"] for f in page["facts"]])
        value = {"scope": bundle["context"]["request_scope"], "facts": page["facts"], "observed_links": []}
        page["characters"] = len(json.dumps(brief_facts(value), ensure_ascii=False))
        page["oversized_single_fact"] = page["characters"] > max_characters
        page["features"] = page_features(bundle, page)
    return {"pages": pages, "required_ids": sorted(required),
        "deferred_ids": sorted(set(bundle["catalog"]) - required),
        "cpu_only_ids": deepcopy(bundle["cpu_only_observation_ids"]),
        "note": "每页一至两项；页面总数不限。未计划事实保留待查，不当成正常。"}


def page_features(bundle, page):
    features = {"node:" + bundle["context"]["node_id"]}
    for f in page["facts"]:
        if f.get("kind") == "alert_evaluation":
            features.update(("rule:" + f["rule"], "action:" + f["action"]))
        if "metric" in f:
            if "raw_units/second" == f.get("unit"):
                features.add("counter_rate")
            if "idle" in f["metric"].lower() or "FreeMemory" in f["metric"]:
                features.add("inverse_meaning")
            if f.get("quality", {}).get("comparison") == "limited_history":
                features.add("limited_history")
    if len(page["facts"]) == 2:
        features.add("two_observations")
    return sorted(features)


def choose_representatives(candidates):
    """Deterministic set cover of observed features, not correctness-based selection."""
    uncovered = set().union(*(set(c["features"]) for c in candidates)) if candidates else set()
    chosen, remaining = [], list(candidates)
    while uncovered:
        best = max(remaining, key=lambda c: (len(set(c["features"]) & uncovered), -c["characters"]))
        if not (set(best["features"]) & uncovered):
            raise ValueError("representative selection lost a feature")
        chosen.append(best)
        uncovered.difference_update(best["features"])
        remaining.remove(best)
    return chosen
