"""Let the node model inspect every concise observation before expanding any."""

from copy import deepcopy
import json

from .evidence_digest import compact, dumps
from .node_probe import decode_object


SYSTEM = """你是单个设备的信息整理员。现在只选哪些观测需要展开，不诊断全局根因。
输入目录覆盖本节点已有的各类观测，包括稳定背景、异常变化和材料不足；由工具计算，不需要你做算术。
逐项分配：expand=需要查看完整统计再填写（包括有价值的反证）；background=概要已看，暂留本地；defer=当前无法理解、需保留待查。
不要只选变化倍率最大的指标，不把普通GC当OOM，不因节点存在就硬找故障。多个同类指标可能相关，但名称不同不保证内容重复。
需回答补查时优先围绕request中的问题，检查支持和反证；不能回答的明确保留待查。没有异常可全选background。
输入提到详情页未加载时，不能把缺失内容认定为正常。每个目录ID恰好出现在一个列表中，可以选任意数量，不限三条。
只输出JSON：{"expand":[],"background":[],"defer":[]}，三个列表只含本页真实ID。"""


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
    defer; omitted observations remain in the node report and the 9B directory.
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


def choose(model, rows, packet):
    # Equal, fully scoped observations can share one catalogue entry; merely
    # similar CPU/disk series are not silently merged or removed.
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
    chosen = balanced_entries(entries, max_items=12)
    omitted = [item["id"] for item in entries if item["id"] not in {e["id"] for e in chosen}]
    data = {"node": packet["node_id"], "scope": packet["scope"],
            "request": packet.get("request"), "page": 0, "pages": 1,
            "catalogue": chosen,
            "coverage": {"available": len(entries), "shown": len(chosen), "omitted": len(omitted),
                         "note": "这是按模态/家族/模式均衡的目录，不是按变化大小排序；未展示的观测仍在9B目录，可补查。"}}
    base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": dumps(data)}]
    reply, attempts = None, []
    for attempt in range(2):
        prompt = deepcopy(base)
        if attempts:
            prompt.append({"role": "user", "content": "修正：" + attempts[-1]["error"]})
        try:
            value = decode_object(model.invoke(prompt, None, 650, f"node_catalogue/{packet['request_id']}/0/{attempt}"))
            if set(value) != {"expand", "background", "defer"}:
                raise ValueError("use expand/background/defer lists")
            ids = []
            for action, picked in value.items():
                if not isinstance(picked, list) or any(not isinstance(v, str) for v in picked):
                    raise ValueError("each choice must be a list of string IDs")
                ids.extend(picked)
            expected = {item["id"] for item in chosen}
            if len(ids) != len(set(ids)) or set(ids) != expected:
                raise ValueError("every catalogue ID must occur exactly once")
            reply = value
        except (ValueError, TypeError, KeyError) as error:
            attempts.append({"valid": False, "error": str(error)})
            continue
        attempts.append({"valid": True})
        break
    decisions = {}
    for action, picked in (reply or {"background": [item["id"] for item in chosen]}).items():
        for identifier in picked:
            decisions[identifier] = action
            for alias in aliases.get(identifier, []):
                decisions[alias] = action
    audits = [{"page": 0, "ids": [item["id"] for item in chosen], "attempts": attempts,
               "reply": reply, "omitted_ids": omitted}]
    return decisions, {"total_observations": len(rows), "unique_entries": len(entries),
                       "aliases": aliases, "pages": audits, "decisions": decisions,
                       "omitted_count": len(omitted),
                       "note": "均衡目录已看；background不是正常，未展示的观测仍可补查。"}
    return decisions, {"total_observations": len(rows), "unique_entries": len(entries),
                       "aliases": aliases, "pages": audits, "decisions": decisions,
                       "note": "目录已看不等于明细已核实；background不是正常，未处理和暂留分开。"}
