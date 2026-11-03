"""Readable, bounded CPU fact cards and narrow factual checks for tiny models.

Original names, full precision and all source fields remain in the bundle. These
cards change presentation; they do not infer causes or add fault answer labels.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re

from .node_probe import write_json


def fmt(value):
    return "未知" if value is None else f"{value:.6g}" if isinstance(value,(int,float)) else str(value)


def when(value):
    return value.split("T",1)[-1].split("+",1)[0] if value else "未知"


def fact_card(f, scope, tool):
    direction = {"up":"上升","down":"下降","unchanged":"相等未变化","unknown":"无法确定"}
    kind = f.get("kind")
    if kind == "missing":
        noun = {"get_metrics":"指标采样","get_logs":"日志","get_calls":"已完成调用"}[tool]
        return f"本范围没有{noun}；不代表其他数据也没有，更不能判定设备正常。"
    if kind == "alert_evaluation":
        d = f["details"]
        status = {"open":"新触发","cleared":"条件已解除","escalated":"进一步加重","updated":"有新证据"}[f["action"]]
        labels = {"metric.std_up":"3σ上侧","metric.std_down":"3σ下侧","metric.relative_up":"相对上升",
                  "metric.relative_down":"相对下降","metric.memory_high":"内存占用偏高",
                  "metric.cpu_high":"CPU使用率偏高","trace.extreme":"特别慢调用","trace.latency":"P95偏高"}
        text = f"{when(f['time'])}，{labels.get(f['rule'],f['rule'])}告警：{status}。"
        if "value" in d:
            unit = d.get("unit", "")
            text += f"当时采样={fmt(d['value'])}{unit}。"
            if d.get("kind") == "rate_per_second":
                text += "这是每秒变化率，不是累计值。"
            base_key = "baseline_mean" if f["rule"].startswith("metric.std_") else "baseline_median"
            if base_key in d:
                text += f"此前{'均值' if base_key=='baseline_mean' else '中位数'}={fmt(d[base_key])}；相对该对照{direction[d['direction']]}。"
            else:
                text += "本条没有历史基线，不可编造均值或涨跌。"
            if "threshold" in d:
                text += f"绝对阈值={fmt(d['threshold'])}{unit}。"
            if f["rule"] == "metric.memory_high":
                text += "占用可能含缓存，不能单凭占用认定内存故障。"
        elif "p95_ms" in d:
            text += f"该次评估{d['calls']}次调用，P95={fmt(d['p95_ms'])}、最大={fmt(d['max_ms'])}毫秒。"
            if d.get("baseline_p95_median_ms") is not None:
                text += f"此前{d['baseline_windows']}个窗口的P95中位数={fmt(d['baseline_p95_median_ms'])}毫秒。"
            if "threshold_ms" in d:
                text += f"本规则阈值={fmt(d['threshold_ms'])}毫秒。"
        elif "pause_ms" in d:
            text += f"GC暂停={fmt(d['pause_ms'])}毫秒。"
        if f["action"] == "cleared":
            text += "只是规则解除，不是故障恢复结论。"
        return text
    if "metric" in f:
        a,b = f["reference"],f["current"]
        text = (f"参考{a['count']}点，中位={fmt(a['median'])}；当前{b['count']}点，中位={fmt(b['median'])}，"
                f"范围{fmt(b['min'])}—{fmt(b['max'])}；方向{direction[f['change']['direction']]}。")
        if f["unit"] == "raw_units/second":
            text += "数值是每秒变化率，不是累计量。"
        elif f["unit"] == "%":
            text += "单位%。"
        else:
            text += "原始单位未确认。"
        if "idle" in f["metric"].lower():
            text += "这是空闲值，空闲下降不能说使用率下降。"
        if f["quality"]["comparison"] == "limited_history":
            text += "对照样本不足。"
        for key,label in (("invalid","无效记录"),("conflicts","冲突记录"),("counter_resets","计数器重置")):
            if f["quality"][key]:text += f"{label}{f['quality'][key]}次。"
        return text
    if kind == "latency_over_analysis_window":
        a,b=f["reference"],f["current"]
        text=(f"累计当前范围{scope['current'][0]}—{scope['current'][1]}：{b['count']}次，P50={fmt(b['p50'])}，"
              f"P95={fmt(b['p95'])}，最大={fmt(b['max'])}毫秒。参考段{a['count']}次。")
        if f["slow_calls"]:
            e=f["slow_calls"][-1]
            text+=f"{'当前段' if b['count'] else '参考段'}最长实例{e['start']}—{e['end']}。"
        return text
    if kind == "minute_latency":
        return (f"{f['start']}—{f['end']}（{'未满一分钟' if f['partial'] else '完整一分钟'}）："
                f"{f['count']}次，P50={fmt(f['p50'])}、P95={fmt(f['p95'])}、最大={fmt(f['max'])}毫秒。不是网络RTT。")
    if kind == "gc":
        return (f"普通GC：参考{f['reference_count']}条，当前{f['current_count']}条；"
                f"首次{f['first_time']}，最近{f['last_time']}，最大暂停{fmt(f['gc_pause_max_ms'])}毫秒。普通GC不等于OOM。")
    # Other log types keep their exact grouping; never replace an error by GC.
    return (f"日志组{kind}：{json.dumps(f['details'],ensure_ascii=False)}；参考{f['reference_count']}条，"
            f"当前{f['current_count']}条，首次{f['first_time']}，最近{f['last_time']}；明确错误={f['explicit_error']}。")


def basic_fact_checks(reply, facts, tool):
    by_id={f["id"]:f for f in facts}
    errors=[]
    for note in reply["findings"]:
        if len(note["ids"]) != 1:
            errors.append("每条观测只引用一个编号，避免把两条的数值混用")
            continue
        f=by_id[note["ids"][0]];text=note["text"]
        if f.get("kind")=="alert_evaluation":
            if f["action"]=="cleared" and not any(w in text for w in ("解除","不再满足","cleared")):
                errors.append(f["id"]+"必须说明规则条件已解除，不可只抄规则名")
            if f["action"]=="escalated" and not any(w in text for w in ("加重","升级","escalated","进一步")):
                errors.append(f["id"]+"必须说明告警进一步加重")
        allowed_mean = f.get("details",{}).get("baseline_mean") if f.get("kind")=="alert_evaluation" else None
        for match in re.finditer(r"(?:均值|平均值?)\s*[=:：为是]?\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",text,re.I):
            if allowed_mean is None or not math.isclose(float(match.group(1)),allowed_mean,rel_tol=.001,abs_tol=1e-8):
                errors.append(f["id"]+"没有支持该均值的输入，不能把当前值或中位数写成均值")
        is_rate = (f.get("unit") == "raw_units/second"
                   or f.get("details", {}).get("kind") == "rate_per_second")
        if is_rate and not any(w in text for w in ("每秒","速率","/s","/second")):
            errors.append(f["id"]+"需注明这是每秒变化率")
        if f.get("kind")=="missing":
            noun={"get_metrics":"指标","get_logs":"日志","get_calls":"调用"}[tool]
            if noun not in text:
                errors.append(f["id"]+"缺失范围必须明确是"+noun+"，不能说所有数据都没有")
    return errors


def revise_inputs(inputs, output):
    inputs,output=Path(inputs),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    corrections=[]
    for path in inputs.glob("r*.json"):
        r=json.loads(path.read_text())
        for facts in r["bundle"]["tools"].values():
            for f in facts:
                if f.get("kind")!="alert_evaluation" or "direction" not in f["details"]:continue
                d=f["details"];key="baseline_mean" if f["rule"].startswith("metric.std_") else "baseline_median"
                if key in d and d.get("value")==d[key] and d["direction"]!="unchanged":
                    corrections.append({"request":r["request_id"],"fact":f["id"],"field":"direction","old":d["direction"],"new":"unchanged"})
                    d["direction"]="unchanged"
                    r["bundle"]["catalog"][f["id"]]["cpu_fact"]["details"]["direction"]="unchanged"
        by_id={f["id"]:f for values in r["bundle"]["tools"].values() for f in values}
        for page in r["selected_pages"]:
            page["facts"]=[deepcopy(by_id[f["id"]]) for f in page["facts"]]
        write_json(output/path.name,r)
    manifest=json.loads((inputs/"manifest.json").read_text())
    manifest.update(derived_from=str(inputs.resolve()),corrections=corrections,
                    note="只纠正与基线恰好相等时的direction，不修改原始观测、标签或检测结果；旧输入保留。")
    write_json(output/"manifest.json",manifest)
    print(json.dumps({"corrections":len(corrections),"output":str(output)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();revise_inputs(args.inputs,args.output)
