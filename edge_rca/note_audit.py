"""Narrow regression checks for observed model mistakes, not semantic scoring.

Warnings preserve the original reply. An empty list does not prove truth;
causal reasoning, omitted evidence and usefulness still need human review.
"""

import math
import re


NUMBER = r"\d+(?:\.\d+)?"


def review_note(text, facts):
    warnings = []
    percentile, totals = [], []
    for fact in facts:
        values = [fact.get(k, {}) for k in ("current", "reference", "child_latency_ms", "first_trigger_values")]
        values += fact.get("latest", [])
        for value in values:
            for key in ("p95", "p95_ms"):
                if isinstance(value.get(key), (int, float)):
                    percentile.append(value[key])
            if "p95" in value or "p95_ms" in value:
                count = value.get("count", value.get("calls"))
                if isinstance(count, int) and count > 0:
                    totals.append(count)
    for match in re.finditer(r"P95\s*(?:延迟)?\s*(?:为|是|=|达|曾达|约|最高)?\s*(" + NUMBER + r")\s*(?:ms|毫秒)", text, re.I):
        spelling = match.group(1)
        decimals = len(spelling.split(".")[1]) if "." in spelling else 0
        tolerance = .5 * 10 ** (-decimals) + 1e-9
        if not any(math.isclose(float(spelling), p, rel_tol=0, abs_tol=tolerance) for p in percentile):
            warnings.append("P95数值不匹配：不能把最大耗时当成P95")
    for match in re.finditer(r"(" + NUMBER + r")\s*次慢调用", text):
        if float(match.group(1)) in totals:
            warnings.append("把全部调用次数写成慢调用次数；输入未提供相应慢调用计数")
    if re.search(r"(?:平均延迟|平均耗时)\s*(?:为|是|=|约)?\s*" + NUMBER, text):
        warnings.append("输入未提供平均调用耗时；P50不能用作均值")
    if re.search(NUMBER + r"\s*(?:GiB|MiB|KiB|GB|MB|KB)\b", text):
        if not any(f.get("unit") in ("GiB", "MiB", "KiB", "GB", "MB", "KB") for f in facts):
            warnings.append("报告换算了容量单位，但输入没有确认该单位")
    gc = [f for f in facts if f.get("kind") == "log" and f.get("category") == "gc" and f.get("status") == "available"]
    if gc:
        count = sum(f["count"] for f in gc)
        patterns = (r"(?:发生|共有|累计|共|观察到)\s*(\d+)\s*(?:次|条)\s*GC",
                    r"GC\s*(?:共|出现|发生|累计)\s*(\d+)\s*(?:次|条)")
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                if int(match.group(1)) != count:
                    warnings.append(f"GC总条数不匹配：引用日志共{count}条，不能把类别数当条数")
    return list(dict.fromkeys(warnings))
