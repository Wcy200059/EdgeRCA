"""Compact serialization of CPU observations; never summarize with an LLM.

The full tool text remains in the coordinator audit. Only known, complete CPU
formats are packed; an unfamiliar format passes through unchanged.
"""

import json
import re

from .node_templates import PATTERNS


METRIC_FIELDS = ("unit", "n,min,max,median", "extremes", "before_n,median",
                 "first5_n,median", "last5_n,median", "first_alarm_time,value,baseline")
SCHEMAS = {
    "metric": ["完整指标名", "变化分类", *METRIC_FIELDS],
    "observation": ["观测对象", "变化分类", "原字段"],
    "business": ["业务名", "各指标[名称,样本数,中位数,最小,最大,极值[时间,值]]"],
    "call_link": ["父节点", "子节点", "子调用次数", "P95毫秒", "最大毫秒", "时间不包含的匹配对数"],
}
BUSINESS = re.compile(r"([^；]+)：(\d+)点，中位数([^，]+)，范围([^—]+)—([^；]+)；极值观测(\[.*?\])(?:；|$)")
LINK = re.compile(r"实际父子ID匹配：(\S+) → (\S+)；子调用(\d+)次，P95=([^m]+)ms，最大=([^m]+)ms；其中(\d+)对时间区间不满足包含关系。")


def pack_fact(fact):
    payload = fact.get("template_payload")
    if payload is not None:
        fields = payload["fields"]
        if "n,min,max,median" in fields:
            return ["metric", payload["subject"], payload["pattern"], *[fields.get(k) for k in METRIC_FIELDS]]
        return ["observation", payload["subject"], payload["pattern"], fields]
    if fact.get("node_id") != "cluster:observations":
        return fact["tool_fact"]
    text = fact["tool_fact"]
    link = LINK.fullmatch(text)
    if link:
        source, target, count, p95, maximum, mismatches = link.groups()
        return ["call_link", source, target, int(count), float(p95), float(maximum), int(mismatches)]
    if text.startswith("business:") and "；" in text:
        name, body = text.split("；", 1)
        matches = list(BUSINESS.finditer(body))
        # Full coverage is mandatory. Never silently drop a new field.
        if not matches or "".join(m.group(0) for m in matches) != body:
            return text
        result = []
        try:
            for m in matches:
                label, n, median, low, high, extremes = m.groups()
                values = json.loads(extremes)
                if any(set(v) != {"time", "value"} for v in values):
                    return text
                result.append([label, int(n), float(median), float(low), float(high),
                               [[v["time"], v["value"]] for v in values]])
        except (ValueError, KeyError, TypeError):
            return text
        return ["business", name, result]
    return text


def legend():
    return {"rows": SCHEMAS, "patterns": PATTERNS,
            "note": "每行先写类型，再依次按对应列读取；null是缺少对照，不是0。单位未核实不可换算。"}
