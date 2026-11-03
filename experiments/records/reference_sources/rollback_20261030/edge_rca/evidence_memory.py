"""Fit a revisitable working set, not an ever-growing concatenated transcript."""

from copy import deepcopy
import math

from .evidence_digest import compact, detail_pages, dumps, fact_overview, source_value, subject


PAGE_SIZE = 24


def page_index(view):
    return {f["id"]: len(detail_pages(source_value(f))) for f in view["evidence"]}


def validate_page_requests(requests, view):
    if not isinstance(requests, list) or len(requests) > 3 or any(not isinstance(r, str) for r in requests):
        raise ValueError("read_pages must contain at most 3 page names")
    if len(requests) != len(set(requests)):
        raise ValueError("read_pages must not repeat a page")
    index = view.get("evidence_page_index", {})
    directory_pages = view.get("directory_pages", 0)
    for name in requests:
        parts = name.split(":")
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError("page name must be eID:page or directory:page, starting at 0")
        source, page = parts[0], int(parts[1])
        if page >= (directory_pages if source == "directory" else index.get(source, 0)):
            raise ValueError("page does not exist in the current evidence catalogue")


def summary_payload(view, summaries, loaded_ids=None):
    items = []
    for summary in summaries:
        if loaded_ids is not None and summary.get("origin") == "followup":
            # V2 follow-up interpretations are already preserved, without
            # truncation, in investigation_updates beside the original question.
            continue
        node = summary.get("node_id")
        item = {key: summary.get(key) for key in (
            "node_id", "available_fact_count", "loaded_fact_count", "unloaded_fact_count",
            "protected_count", "rule_retained_count", "model_unprocessed_count")}
        if summary.get("text_unverified"):
            item["text_unverified"] = summary["text_unverified"][:220]
        if loaded_ids is not None:
            own = [f for f in view["evidence"] if f["node_id"] == node]
            item["available_fact_count"] = len(own)
            item["loaded_fact_count"] = sum(f["id"] in loaded_ids for f in own)
            item["unloaded_fact_count"] = len(own) - item["loaded_fact_count"]
        items.append(item)
    return items


def fit(model, view, system, output_tokens, public_question=None):
    """Preserve full audit; only loaded pages can be cited or supply times."""
    original = {f["id"]: f for f in view["evidence"]}
    index = page_index(view)
    # A node-level index, not a 25-page fact list. Details remain addressable
    # by key eIDs; unloaded observations are reached through node follow-up.
    node_ids = list(dict.fromkeys(f["node_id"] for f in view["evidence"]))
    directory = []
    for node in node_ids:
        own = [f for f in view["evidence"] if f["node_id"] == node]
        key_ids = [f["id"] for f in own if f.get("initial_display") or f.get("protected")
                   or f.get("pattern") in ("error_observed", "rule_signal", "latency_increased")][:6]
        directory.append([node, len(own), sum(f.get("initial_display", True) for f in own), key_ids])
    directory_pages = 1
    visible = deepcopy(view)
    visible["evidence_page_index"] = index
    visible["directory_pages"] = directory_pages
    requested = list(view.get("requested_pages", []))
    validate_page_requests(requested, visible)
    cursor = next((int(r.split(":")[1]) for r in reversed(requested) if r.startswith("directory:")), 0)
    pages = {r.split(":")[0]: int(r.split(":")[1]) for r in requested if not r.startswith("directory:")}
    shown = [fact_overview(f, pages.get(f["id"])) for f in view["evidence"]
             if f.get("initial_display", True) or f["id"] in pages]
    cited = set()
    for h in view.get("previous_hypotheses", []):
        cited.update(h.get("support_ids", []) + h.get("counter_ids", []))

    # Exact copies share a current display. Different intervals or values must
    # remain separate. Keep the latest copy; expose aliases in the directory.
    duplicate_of, identities = {}, {}
    for f in reversed(shown):
        identity = dumps([f["node_id"], source_value(original[f["id"]]), original[f["id"]].get("observation_scope")])
        if identity in identities and f["id"] not in pages:
            duplicate_of[f["id"]] = identities[identity]
        else:
            identities[identity] = f["id"]
    shown = [f for f in shown if f["id"] not in duplicate_of]
    summaries = view.get("node_summaries", [])
    short_metadata = False

    def render():
        visible["evidence"] = deepcopy(shown)
        ids = {f["id"] for f in shown}
        counted_ids = ids if view.get("node_reply_schema") == "node_slots_v2" else None
        visible["evidence_directory"] = directory
        visible["directory_page"] = 0
        visible["context_omissions"] = {
            "unloaded_count": len(original) - len(ids), "duplicates": duplicate_of,
            "requested_but_not_loaded": [r for r in requested if not r.startswith("directory:")
                                         and r.split(":")[0] not in ids],
            "note": "未加载不是正常。目录按节点汇总；key_ids可加载，其余用节点补查；不重新查询原始数据。"}
        payload = {k: deepcopy(view[k]) for k in (
            "case_id", "window_start_ms", "cutoff_ms", "system_context", "public_task",
            "node_status", "tool_capabilities", "query_policy", "request_window") if k in view}
        payload["evidence"] = [[f["id"], f["node_id"], f["tool_fact"], f["time_marks"]] for f in shown]
        payload["evidence_columns"] = ["可引用ID", "节点", "观测/当前详情页", "本次可引用时间"]
        payload["evidence_directory"] = visible["evidence_directory"]
        payload["directory_columns"] = ["节点", "可用事实", "已加载", "关键ID"]
        payload["directory_page"] = cursor
        payload["directory_pages"] = directory_pages
        payload["context_omissions"] = visible["context_omissions"]
        payload["previous_hypotheses"] = deepcopy(view.get("previous_hypotheses", []))
        # Model interpretation is kept beside facts; repeated copied numerical
        # paragraphs from every earlier response are not appended again.
        interpretations = {}
        for summary in summaries:
            for item in summary.get("filled_interpretations_unverified", []):
                if item["basis_id"] in ids:
                    interpretations[item["basis_id"]] = item
        payload["node_summaries"] = summary_payload(view, summaries, counted_ids)
        payload["candidate_coverage"] = deepcopy(view.get("candidate_coverage", []))
        payload["candidate_slate"] = deepcopy(view.get("candidate_slate", []))
        payload["node_questions"] = compact(view.get("node_questions", []), list_limit=8, text_limit=180)
        payload["followup_status"] = compact(view.get("followup_status", []), list_limit=6, text_limit=180)
        payload["node_interpretations_unverified"] = list(interpretations.values())
        if "node_reply_schema" in view:
            payload["node_reply_schema"] = view["node_reply_schema"]
        payload["investigation_updates"] = deepcopy(view.get("investigation_updates", []))
        payload["pending_requests"] = deepcopy(view.get("pending_requests", []))
        payload["paging_instruction"] = ('read_pages可选，例如["e17:0"]读取该事实详情第0页；'
            '["directory:0"]查看节点索引。最多3项。目录只是索引，不能作为支持证据；'
            '时间和结论只能引用本次evidence。关键ID优先看，未读部分保持未知。')
        if short_metadata:
            payload["previous_hypotheses"] = [{k: h.get(k) for k in ("component", "reason", "assessment")}
                                              for h in view.get("previous_hypotheses", [])]
            payload["investigation_updates"] = {"count": len(view.get("investigation_updates", [])),
                                               "full_responses_remain_local": True}
            payload["evidence_directory"] = [[r[0], r[1], r[2]] for r in visible["evidence_directory"]]
            payload["directory_columns"] = ["节点", "可用事实", "已加载"]
            payload["candidate_coverage"] = deepcopy(view.get("candidate_coverage", []))
            payload["candidate_slate"] = deepcopy(view.get("candidate_slate", []))
            payload["node_summaries"] = [{k: item.get(k) for k in (
                "node_id", "loaded_fact_count", "unloaded_fact_count", "text_unverified")}
                for item in summary_payload(view, summaries, counted_ids)]
            payload["node_interpretations_unverified"] = []
        if public_question is not None:
            payload["public_question"] = public_question
        prompt = [{"role": "system", "content": system}, {"role": "user", "content": dumps(payload)}]
        return prompt

    maximum = getattr(model, "max_context", 20480)
    while True:
        prompt = render()
        tokens = (model.count_input_tokens(prompt) if hasattr(model, "count_input_tokens")
                  else math.ceil(sum(len(m["content"]) for m in prompt) / 1.5))
        if tokens + output_tokens + 700 <= maximum:
            visible["input_budget"] = {"input_tokens": tokens, "output_tokens": output_tokens,
                                       "repair_reserve": 700, "context_limit": maximum,
                                       "loaded_facts": len(shown), "available_facts": len(original)}
            return prompt, visible
        if shown:
            # Requested pages outrank prior citations, which outrank the normal
            # working set. Nothing is pinned forever; every eviction is visible.
            victim = min(shown, key=lambda f: (f["id"] in pages, f["id"] in cited,
                original[f["id"]].get("origin") == "followup", original[f["id"]].get("protected", False)))
            shown.remove(victim)
        elif not short_metadata:
            short_metadata = True
        else:
            # This signals an unusably small configuration, not accumulated
            # evidence. Ordinary 20K runs keep metadata far below this floor.
            raise ValueError("context limit cannot fit instructions and one directory page; increase configuration minimum")
