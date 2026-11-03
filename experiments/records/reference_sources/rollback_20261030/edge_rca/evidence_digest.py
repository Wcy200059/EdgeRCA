"""Small CPU overviews with addressable, unabridged detail pages.

Overviews describe what is shown and what remains in the detail pages. They
are not lossless replacements for raw records, nor claims of normality.
"""

from copy import deepcopy
import json


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def timeline_overview(timeline):
    """Keep separate trigger/clear episodes; never bridge a quiet gap."""
    if not timeline:
        return {"events": 0, "episodes": 0}
    episodes, opened, other = [], None, 0
    for row in timeline:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            other += 1
            continue
        when, state = row[:2]
        if state == "triggered":
            if opened is None:
                opened = when
        elif state == "rules_cleared":
            if opened is not None:
                episodes.append([opened, when])
                opened = None
        else:
            other += 1
    if opened is not None:
        episodes.append([opened, None])
    shown = episodes if len(episodes) <= 6 else episodes[:3] + episodes[-3:]
    return {"events": len(timeline), "first_event": timeline[0], "last_event": timeline[-1],
            "episodes": len(episodes), "shown_intervals": shown,
            "omitted_intervals": len(episodes) - len(shown), "other_events": other,
            "meaning": "区间末尾只表示规则解除，null表示尚未观察到解除；中间区间仍在详情页，不表示持续触发或恢复。"}


def compact(value, *, key="", list_limit=6, text_limit=350):
    """Whole values are kept or explicitly deferred, never silently cut."""
    if key == "timeline" and isinstance(value, list):
        return timeline_overview(value)
    if isinstance(value, dict):
        return {k: compact(v, key=k, list_limit=list_limit, text_limit=text_limit) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) <= list_limit:
            return [compact(v, list_limit=list_limit, text_limit=text_limit) for v in value]
        return {"items": len(value), "first": compact(value[0]), "last": compact(value[-1]),
                "middle_items_in_detail_pages": len(value) - 2}
    if isinstance(value, str) and len(value) > text_limit:
        return {"text_characters": len(value), "text_in_detail_pages": True}
    return value


def detail_pages(value, max_chars=2600):
    """Page complete JSON leaves with paths; string parts are explicitly numbered."""
    leaves = []

    def visit(item, path):
        if isinstance(item, dict) and item:
            for k, v in item.items():
                visit(v, path + [k])
        elif isinstance(item, list) and item:
            for i, v in enumerate(item):
                visit(v, path + [i])
        elif isinstance(item, str) and len(dumps([path, item])) > max_chars - 200:
            # Splitting an example message is a visible page boundary, not a
            # change to a numerical/statistical value or a discarded suffix.
            width = max(64, (max_chars - len(dumps(path)) - 300) // 2)
            parts = [item[i:i + width] for i in range(0, len(item), width)]
            for index, text in enumerate(parts):
                leaves.append({"path": path, "string_part": index, "parts": len(parts), "value": text})
        else:
            leaves.append({"path": path, "value": item})

    visit(value, [])
    pages, page = [], []
    for leaf in leaves:
        if page and len(dumps(page + [leaf])) > max_chars:
            pages.append(page)
            page = []
        page.append(leaf)
    if page:
        pages.append(page)
    return pages


def factual_fields(row):
    """CPU fields for a fill-in question; original fields remain in audit files."""
    return compact(row["fields"])


def source_value(fact):
    if "detail_fields" in fact:
        return fact["detail_fields"]
    if fact.get("template_payload"):
        return fact["template_payload"]["fields"]
    return fact["tool_fact"]


def subject(fact):
    return fact.get("template_payload", {}).get("subject", fact["tool_fact"].split("；", 1)[0])


def fact_overview(fact, page=None):
    value = deepcopy(fact)
    raw = source_value(fact)
    pages = detail_pages(raw)
    if page is None:
        content = {"subject": subject(fact), "observations": compact(raw), "detail_pages": len(pages)}
    else:
        content = {"subject": subject(fact), "detail_page": page, "detail_pages": len(pages),
                   "leaves": pages[page], "note": "path是原字段路径，数组下标从0开始；本页只展示部分详情，其他页未加载。"}
    value["tool_fact"] = dumps(content)
    # A timestamp hidden on another page cannot be used as a visible citation.
    value["time_marks"] = [t for t in fact.get("time_marks", []) if t in value["tool_fact"]]
    value.pop("template_payload", None)
    value.pop("detail_fields", None)
    return value
