"""Fit a revisitable working set, not an ever-growing concatenated transcript."""

from collections import Counter
from copy import deepcopy
import math
import re

from .evidence_digest import compact, detail_pages, dumps, fact_overview, source_value, subject


PAGE_SIZE = 24


def followup_status_payload(statuses):
    """Expose exact name examples when a requested metric name was missing.

    The full name list remains in the saved tool response. A fragment match is
    only a directory hint, not an alias, a metric value, or an executed query.
    """
    result = deepcopy(statuses)
    for status in result:
        names = status.get("available_metric_names")
        if not isinstance(names, list) or len(names) <= 6:
            continue
        fragments = []
        for missing in status.get("missing_metrics", []):
            if not isinstance(missing, str):
                continue
            for token in re.findall(r"[A-Za-z]{3,}", missing):
                if token.casefold() not in {part.casefold() for part in fragments}:
                    fragments.append(token)
        matches = []
        for fragment in fragments:
            found = [name for name in names if isinstance(name, str) and fragment.casefold() in name.casefold()]
            if found:
                matches.append({"fragment": fragment, "count": len(found), "examples": found[:3]})
            if len(matches) == 4:
                break
        status["available_metric_names"] = {
            "count": len(names), "fragment_matches": matches,
            "note": "仅展示真实名称片段示例；不表示指标等价或已经查值。更多名称可读取目录页或用list_metrics按片段查询。"}
    return compact(result, list_limit=6, text_limit=180)


def page_index(view):
    return {f["id"]: len(detail_pages(source_value(f))) for f in view["evidence"]}


def validate_page_requests(requests, view, *, max_requests=3):
    if (not isinstance(requests, list) or (max_requests is not None and len(requests) > max_requests)
            or any(not isinstance(r, str) for r in requests)):
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


def summary_payload(view, summaries, loaded_ids=None, *, omit_followup=False):
    items = []
    for summary in summaries:
        if omit_followup and summary.get("origin") == "followup":
            # V2 follow-up interpretations are already preserved, without
            # truncation, in investigation_updates beside the original question.
            continue
        node = summary.get("node_id")
        item = {key: summary.get(key) for key in (
            "node_id", "available_fact_count", "loaded_fact_count", "unloaded_fact_count",
            "protected_count", "rule_retained_count", "model_unprocessed_count", "catalogue_unprocessed_count")}
        if summary.get("text_unverified"):
            item["text_unverified"] = summary["text_unverified"][:220]
        if loaded_ids is not None and summary.get("origin") != "followup":
            # A node report's local total can be much larger than the subset
            # admitted to the coordinator archive.  Follow-up evidence for
            # the same node is a separate record, not part of this report.
            basis = summary.get("basis_ids")
            if isinstance(basis, list):
                own_ids = set(basis)
            else:
                own_ids = {f["id"] for f in view["evidence"]
                           if f["node_id"] == node and f.get("origin") != "followup"}
            declared = item["available_fact_count"]
            available = declared if isinstance(declared, int) and not isinstance(declared, bool) else 0
            item["available_fact_count"] = max(available, len(own_ids))
            item["loaded_fact_count"] = len(own_ids & loaded_ids)
            item["unloaded_fact_count"] = item["available_fact_count"] - item["loaded_fact_count"]
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
    # Per-round requests remain limited to three; accepted pages accumulate.
    validate_page_requests(requested, visible, max_requests=None)
    cursor = next((int(r.split(":")[1]) for r in reversed(requested) if r.startswith("directory:")), 0)
    pages = {}
    for request in requested:
        identifier, number = request.split(":")
        if identifier != "directory":
            pages.setdefault(identifier, []).append(int(number))
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
        omit_followup = view.get("node_reply_schema") == "node_slots_v2"
        loaded_by_node = Counter(f["node_id"] for f in shown)
        visible["evidence_directory"] = [[node, available, loaded_by_node[node], keys]
                                         for node, available, _, keys in directory]
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
        payload["directory_columns"] = ["节点", "协调者可查事实", "本次已加载", "关键ID"]
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
        payload["node_summaries"] = summary_payload(view, summaries, ids, omit_followup=omit_followup)
        payload["candidate_coverage"] = deepcopy(view.get("candidate_coverage", []))
        payload["candidate_slate"] = deepcopy(view.get("candidate_slate", []))
        payload["node_questions"] = compact(view.get("node_questions", []), list_limit=8, text_limit=180)
        payload["followup_status"] = followup_status_payload(view.get("followup_status", []))
        payload["node_interpretations_unverified"] = list(interpretations.values())
        payload["node_relations_unverified"] = [
            {"node_id": summary.get("node_id"), "from_id": relation["from_id"],
             "related_ids": relation["related_ids"],
             "unloaded_related_ids": [key for key in relation["related_ids"] if key not in ids]}
            for summary in summaries for relation in summary.get("relations_unverified", [])
            if relation.get("from_id") in ids and relation.get("related_ids")]
        if "node_reply_schema" in view:
            payload["node_reply_schema"] = view["node_reply_schema"]
        payload["investigation_updates"] = deepcopy(view.get("investigation_updates", []))
        payload["pending_requests"] = deepcopy(view.get("pending_requests", []))
        payload["paging_instruction"] = ('read_pages可选，例如["e17:0"]读取该事实详情第0页；'
            '["directory:0"]查看节点索引。最多3项。目录只是索引，不能作为支持证据；'
            '读取页面不查询原始数据，但会占用一轮调查；时间和结论只能引用本次evidence中的观测。关键ID优先看，未读部分保持未知。')
        if short_metadata:
            payload["previous_hypotheses"] = [{k: h.get(k) for k in ("component", "reason", "assessment")}
                                              for h in view.get("previous_hypotheses", [])]
            payload["investigation_updates"] = {"count": len(view.get("investigation_updates", [])),
                                               "full_responses_remain_local": True}
            payload["evidence_directory"] = [[r[0], r[1], r[2]] for r in visible["evidence_directory"]]
            payload["directory_columns"] = ["节点", "协调者可查事实", "本次已加载"]
            payload["candidate_coverage"] = deepcopy(view.get("candidate_coverage", []))
            payload["candidate_slate"] = deepcopy(view.get("candidate_slate", []))
            payload["node_summaries"] = [{k: item.get(k) for k in (
                "node_id", "loaded_fact_count", "unloaded_fact_count", "text_unverified")}
                for item in summary_payload(view, summaries, ids, omit_followup=omit_followup)]
            payload["node_interpretations_unverified"] = []
            # Keep compact relation IDs even when verbose metadata is reduced.
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
            # Preserve cross-node coverage before giving one node many extra
            # citations. Ties previously erased early nodes wholesale. An
            # explicit page request still has highest priority; if even one
            # fact per node cannot fit, ordinary eviction remains possible.
            per_node = Counter(fact["node_id"] for fact in shown)
            victim = min(shown, key=lambda f: (f["id"] in pages,
                per_node[f["node_id"]] == 1, f["id"] in cited,
                original[f["id"]].get("origin") == "followup", original[f["id"]].get("protected", False)))
            shown.remove(victim)
        elif not short_metadata:
            short_metadata = True
        else:
            # This signals an unusably small configuration, not accumulated
            # evidence. Ordinary 20K runs keep metadata far below this floor.
            raise ValueError("context limit cannot fit instructions and one directory page; increase configuration minimum")
