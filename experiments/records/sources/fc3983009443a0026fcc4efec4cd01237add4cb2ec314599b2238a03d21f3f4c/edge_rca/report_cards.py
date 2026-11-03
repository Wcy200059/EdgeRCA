"""Readable tool facts for node and coordinator inputs; no model paraphrasing.

Full precision, timelines and raw-record references remain in the snapshot.
These cards label statistics and scopes, without inferring a root cause.
"""

import json

from .compact_tools import _when
from .fact_cards import fmt


STATES = {"triggered": "规则触发", "rules_cleared": "规则条件已解除"}
DIRECTIONS = {"up": "上升", "down": "下降", "unchanged": "相等", "unknown": "对照不足，无法判断变化"}


def unit_note(unit):
    if unit == "%":
        return "单位为百分比。"
    if unit == "raw_units/second":
        return "数值为每秒变化率，不是累计计数；原始计数单位未确认。"
    return "原始单位未确认，不能自行换算为GB、百分比等单位。"


def latency_text(stats):
    count = stats.get("count", stats.get("calls"))
    parts = [f"全部调用{fmt(count)}次（不是慢调用次数）"]
    if "p50" in stats:
        parts.append(f"P50={fmt(stats['p50'])}毫秒（中位数，不是平均值）")
    if "p95" in stats:
        parts.append(f"P95={fmt(stats['p95'])}毫秒")
    if "max" in stats:
        parts.append(f"最大单次耗时={fmt(stats['max'])}毫秒")
    return "；".join(parts) + "。"


def evaluation_text(value):
    if "p95_ms" in value:
        text = latency_text({"calls": value.get("calls"), "p95": value["p95_ms"], "max": value.get("max_ms")})
    elif "value" in value:
        text = f"采样值={fmt(value['value'])}。"
        if value.get("kind") == "rate_per_second":
            text += "这是每秒变化率。"
        # A std rule compares a mean, not a median. Name both if present;
        # never reuse the stored direction against the wrong baseline.
        for key, label in (("baseline_mean", "此前均值"), ("baseline_median", "此前中位数")):
            if key in value:
                text += f"{label}={fmt(value[key])}。"
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if "threshold_ms" in value:
        text += f"规则阈值={fmt(value['threshold_ms'])}毫秒。"
    return text


def card(fact, snapshot):
    """Render one fact. Do not merge objects, drop timelines or invent units."""
    kind = fact.get("kind")
    if kind == "alarm_history":
        signal = {"completed_call_p95": "调用P95", "maximum_call_duration": "最大单次调用耗时"}.get(fact["signal"], fact["signal"])
        text = f"信号{signal}；截至报告时{STATES.get(fact['state'], fact['state'])}。"
        text += "规则状态变化：" + " → ".join(f"{t} {STATES.get(s, s)}" for t, s in fact["timeline"]) + "。"
        text += "首次触发时：" + evaluation_text(fact["first_trigger_values"]) + "首次触发值不是正常基线。"
        for v in fact["latest"]:
            text += f"规则{v['rule']}最近一次评估（{v['time']}）：" + evaluation_text(v)
        text += "最近评估不一定是当前采样；解除只指规则，不证明故障恢复。"
        if "whole_prefix" in fact:
            s = fact["whole_prefix"]
            text += (f"本报告范围共{s['count']}点，范围{fmt(s['min'])}—{fmt(s['max'])}，"
                     f"中位数{fmt(s['median'])}；范围不能证明涨跌。" + unit_note(fact["metric_unit"]))
            text += f"含义：{fact['meaning']}。"
        early, late = fact.get("early"), fact.get("late")
        if early and late and early.get("count") and late.get("count"):
            direction = DIRECTIONS[fact.get("early_late_change", {}).get("direction", "unknown")]
            text += (f"窗口前后5分钟：前段{early['count']}点中位数{fmt(early['median'])}，"
                     f"后段{late['count']}点中位数{fmt(late['median'])}；比较：{direction}。")
        for extreme in fact.get("extremes", []):
            text += f"极值观测：{_when(extreme['time_ms'])}={fmt(extreme['value'])}。"
    elif "metric" in fact:
        a, b = fact["reference"], fact["current"]
        text = (f"指标{fact['metric']}；{fact['meaning']}。本报告范围{b['count']}点，"
                f"范围{fmt(b['min'])}—{fmt(b['max'])}，中位数{fmt(b['median'])}。"
                f"参考段{a['count']}点，中位数{fmt(a['median'])}；"
                f"参考段比较：{DIRECTIONS[fact['change']['direction']]}。范围本身不能证明趋势。" + unit_note(fact["unit"]))
        early, late = fact.get("early"), fact.get("late")
        if early and late and early.get("count") and late.get("count"):
            direction = DIRECTIONS[fact.get("early_late_change", {}).get("direction", "unknown")]
            text += (f"窗口前后5分钟：前段{early['count']}点中位数{fmt(early['median'])}，"
                     f"后段{late['count']}点中位数{fmt(late['median'])}；比较：{direction}。")
        for extreme in fact.get("extremes", []):
            text += f"极值观测：{_when(extreme['time_ms'])}={fmt(extreme['value'])}。"
        q = fact["quality"]
        if any(q[k] for k in ("invalid", "conflicts", "counter_resets")):
            text += f"质量记录：{json.dumps(q, ensure_ascii=False)}。"
    elif kind == "latency_over_analysis_window":
        text = f"累计统计范围{snapshot['scope']['start']}—{snapshot['scope']['end']}：" + latency_text(fact["current"])
        text += "参考段：" + latency_text(fact["reference"])
        for sample in fact["slow_calls"]:
            text += f"长调用样本{sample['start']}—{sample['end']}，耗时{fmt(sample['duration_ms'])}毫秒。"
        text += f"无效记录{fact['invalid_count']}条；调用耗时不是网络RTT，也不表示调用失败。"
    elif kind == "observed_connection":
        text = f"父子ID观察到连接{fact['from']} → {fact['to']}；该连接的子调用统计：" + latency_text(fact["child_latency_ms"])
        text += (f"{'整个已到达前缀另有' if fact.get('mismatch_scope') == 'all_arrived_prefix' else '其中'}{fact['interval_mismatch_count']}对时间区间不满足包含关系，"
                 "不可据此计算网络耗时、相减父子时长或确定故障传播方向。")
    elif kind == "minute_latency":
        text = f"{fact['start']}—{fact['end']}（{'不足一分钟' if fact['partial'] else '一分钟'}）：" + latency_text(fact)
        text += "按调用完成时间归入本分钟；这里只描述这一分钟，不是整窗统计。"
    elif kind == "http_access":
        text = (f"HTTP状态码{fact['status']}，共{fact['count']}条，"
                f"出现范围{fact['first']}—{fact['last']}；合并{fact['template_count']}种模板。"
                "只描述本组状态码，不能据此认为全部请求成功。")
    elif kind == "log":
        if fact["status"] != "available":
            text = "本范围未收到日志；不是没有错误，也不是没有其他模态数据。"
        else:
            text = (f"日志类别{fact['category']}，共{fact['count']}条，"
                    f"出现范围{fact['first']}—{fact['last']}；"
                    f"内容{json.dumps(fact['details'], ensure_ascii=False)}；"
                    f"明确错误标记={fact['explicit_error']}。")
            if fact["category"] == "gc":
                text += f"最大GC暂停={fmt(fact['gc_pause_max_ms'])}毫秒；普通GC或Allocation Failure不等于OOM。"
    elif kind == "metric_catalog":
        text = f"实际指标名称目录第{fact['part']}/{fact['parts']}部分：" + json.dumps(fact["names"], ensure_ascii=False)
        if fact.get("name_filters"):
            text += "；本次按名称包含任一片段筛选：" + json.dumps(fact["name_filters"], ensure_ascii=False)
        text += "。这是名称索引，不是已检查结论；目录包括本次时段及参考段有记录的名称。"
    elif kind == "missing":
        source = snapshot.get("catalog", {}).get(fact["id"], {})
        modality = fact.get("modality") or source.get("cpu_fact", {}).get("modality")
        noun = {"trace": "已完成调用", "metric": "指标", "log": "日志"}.get(modality)
        if noun is None:
            raise ValueError("missing-data fact needs a modality; cannot call all data missing")
        missing_names = source.get("missing_requested_names")
        if modality == "metric" and missing_names:
            text = ("本范围未找到请求的指标名称：" + json.dumps(missing_names, ensure_ascii=False)
                    + "；不代表其他指标缺失，也不表示设备正常。")
        else:
            text = f"本范围没有已到达的{noun}记录；不代表其他模态缺失，也不表示设备正常。"
    else:
        raise ValueError(f"unsupported report fact kind: {kind}")
    return {"id": fact["id"], "tool_fact": text}


def fact_view(fact, snapshot):
    """Merged groups retain every original card, including latency and times."""
    if "cpu_facts" in fact:
        return {"id": fact["id"], "unverified_node_note": fact["summary"],
                "tool_facts": [card(f, snapshot)["tool_fact"] for f in fact["cpu_facts"]]}
    return card(fact, snapshot)
