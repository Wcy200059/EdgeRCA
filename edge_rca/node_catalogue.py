"""Present a bounded catalogue and keep unshown observations unprocessed."""

from copy import deepcopy
import json

from .evidence_digest import compact, dumps
from .node_probe import decode_object


SYSTEM = """你是单个设备的信息整理员。现在只选哪些观测需要展开，不诊断全局根因。
输入目录是本节点观测的受限截面，包括稳定背景、异常变化和材料不足；未展示的条目尚未检查。数据由工具计算，不需要你做算术。
逐项判断：需要上报/进一步看的写入 report；当前无法理解、需要保留待查的写入 defer；其余未列出的条目按 background 处理。
不要只选变化倍率最大的指标，不把普通GC当OOM，不因节点存在就硬找故障。多个同类指标可能相关，但名称不同不保证内容重复。
需回答补查时优先围绕request中的问题，检查支持和反证；不能回答的明确保留待查。没有异常可全选background。
输入提到详情页未加载时，不能把缺失内容认定为正常。每个目录ID恰好出现在一个列表中，可以选任意数量，不限三条。
只输出JSON：{"report":[],"defer":[]}。只列真正需要上报/待查的ID；未列出的本页ID自动记为background。不要为了凑数把所有ID都列出来。"""


def entry(row):
    f = row["fields"]
    if "metric" in f or "whole_prefix" in f:
        s = f.get("whole_prefix", f.get("current", {}))
        observations = {"unit": f.get("unit", f.get("metric_unit")),
                        "n,min,max,median": [s.get(k) for k in ("count", "min", "max", "median")],
                        "early,late_median": [f.get(k, {}).get("median") for k in ("early", "late")],
                        "extremes": f.get("extremes", []), "reference": f.get("reference")}
    else:
        observations = compact(f, text_limit=160, list_limit=4)
    return {"id": row["id"], "type": row["template"], "subject": row["subject"],
            "pattern": row["pattern_description"], "pattern_key": row.get("pattern"),
            "protected": bool(row.get("protected")), "observations": observations}


FAMILIES = ("cpu", "mem", "dsk", "disk", "net", "network", "jvm", "gc",
            "http", "tomcat", "mysql", "redis", "call", "log", "business", "alert")


def family(entry):
    subject = entry["subject"].lower()
    return next((name for name in FAMILIES if name in subject), entry["type"])


def balanced_entries(entries, max_items=12):
    """One bounded cross-section: modalities, families, then patterns.

    This is coverage, not importance. The model still chooses expand/background/
    defer; omitted observations remain at the node for a directed follow-up.
    """
    selected, ids = [], set()

    def take(entry):
        if len(selected) < max_items and entry["id"] not in ids:
            selected.append(entry)
            ids.add(entry["id"])

    # Objective safety floor: a protected change or explicit error/rule/latency
    # is never hidden by the 12-slot balancing. These are not semantic root
    # causes; activation/background is still the model's choice.
    for entry in entries:
        if (entry.get("protected") or entry.get("pattern_key") in
                ("error_observed", "rule_signal", "latency_increased")):
            take(entry)
    for key in ("type", "family", "pattern"):
        seen = set()
        for entry in entries:
            value = {"type": entry["type"], "family": family(entry),
                     "pattern": entry["pattern"]}[key]
            if value not in seen:
                seen.add(value)
                take(entry)
    for entry in entries:
        take(entry)
    return selected


def choose(model, rows, packet, page_size=12):
    """Show every observation, in bounded pages, before deciding.

    The previous single-page policy left 339/532 observations unseen in row 1
    and then recorded them as background. This constructor walks all balanced
    pages; every entry receives a model decision. Invalid pages become
    unprocessed, never background.
    """
    entries, aliases, seen = [], {}, {}
    for row in rows:
        item = entry(row)
        identity = dumps({k: v for k, v in row.items()
                          if k in ("subject", "fields", "template", "pattern")})
        if identity in seen:
            aliases.setdefault(seen[identity], []).append(row["id"])
        else:
            seen[identity] = row["id"]
            entries.append(item)
    ordered = balanced_entries(entries, max_items=len(entries)) if entries else []
    pages = [ordered[i:i + page_size] for i in range(0, len(ordered), page_size)]
    decisions, audits = {}, []
    for page_index, chosen in enumerate(pages):
        omitted = [item["id"] for item in entries if item["id"] not in {e["id"] for e in chosen}]
        data = {"node": packet["node_id"], "scope": packet["scope"],
                "request": packet.get("request"), "page": page_index + 1,
                "pages": len(pages), "catalogue": chosen,
                "coverage": {"available": len(entries), "page_shown": len(chosen),
                             "note": "按模态/家族/模式均衡的完整分批目录；本页必须逐项决定，"
                                     "未展示项会在后续页出现。"}}
        base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": dumps(data)}]
        reply, attempts = None, []
        for attempt in range(2):
            prompt = deepcopy(base)
            if attempts:
                prompt.append({"role": "user", "content": "修正：" + attempts[-1]["error"]})
            try:
                value = decode_object(model.invoke(
                    prompt, None, 650,
                    f"node_catalogue/{packet['request_id']}/{page_index}/{attempt}"))
                if not isinstance(value, dict) or not set(value) <= {"report", "expand", "defer", "background"}:
                    raise ValueError("use report/defer (optional background) lists")
                ids = []
                for action, picked in value.items():
                    if not isinstance(picked, list) or any(not isinstance(v, str) for v in picked):
                        raise ValueError("each choice must be a list of string IDs")
                    ids.extend(picked)
                expected = {item["id"] for item in chosen}
                if len(ids) != len(set(ids)) or not set(ids) <= expected:
                    raise ValueError("IDs must be unique and from this page")
                reply = value
            except (ValueError, TypeError, KeyError) as error:
                attempts.append({"valid": False, "error": str(error)})
                continue
            attempts.append({"valid": True})
            break
        selections = reply if reply is not None else {"unprocessed": [item["id"] for item in chosen]}
        page_decisions = {}
        for action, picked in selections.items():
            for identifier in picked:
                mapped = "expand" if action in ("report", "expand") else action
                page_decisions[identifier] = mapped
                for alias in aliases.get(identifier, []):
                    page_decisions[alias] = mapped
        # Observed-but-unlisted entries are an explicit model background decision.
        if reply is not None:
            for item in chosen:
                page_decisions.setdefault(item["id"], "background")
                for alias in aliases.get(item["id"], []):
                    page_decisions.setdefault(alias, "background")
        decisions.update(page_decisions)
        audits.append({"page": page_index, "pages": len(pages),
                       "ids": [item["id"] for item in chosen], "attempts": attempts,
                       "reply": reply, "omitted_ids": omitted})
    # Defensive: any entry without a decision is unprocessed, never background.
    for identifier in seen.values():
        decisions.setdefault(identifier, "unprocessed")
    return decisions, {"total_observations": len(rows), "unique_entries": len(entries),
                       "aliases": aliases, "pages": audits, "decisions": decisions,
                       "omitted_count": 0,
                       "coverage": {"available": len(entries), "shown": len(entries), "pages": len(pages)},
                       "note": "全部观测已分批展示并逐项决定；未展示项只可能来自无效页，标记为 unprocessed。"}
