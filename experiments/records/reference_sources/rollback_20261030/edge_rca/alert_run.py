"""Run CPU-only alert monitoring over the causal device stream replay."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

from .data import format_time
from .replay import ReplaySession


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


RULE_NAMES = {
    "metric.cpu_high": "CPU使用率偏高", "metric.memory_high": "内存占用偏高（需排除缓存）",
    "metric.relative_up": "指标明显上升", "metric.relative_down": "指标明显下降",
    "metric.std_up": "指标高于此前均值3σ", "metric.std_down": "指标低于此前均值3σ",
    "log.oom": "日志报告内存分配失败", "log.error": "明确错误日志",
    "log.gc_pause": "GC暂停偏长", "trace.latency": "一批调用的耗时偏高",
    "trace.extreme": "出现特别慢的调用", "business.success_low": "业务成功率偏低",
}


def _observation(alert: dict[str, Any]) -> str:
    details = alert.get("latest", {})
    if "p95_ms" in details:
        return f"{details['calls']}次调用，95%分位耗时{details['p95_ms']:g}毫秒，最大{details['max_ms']:g}毫秒"
    if "value" in details:
        value = f"最近值{details['value']:g}{details.get('unit', '')}"
        if "baseline_median" in details:
            value += f"，此前中位值{details['baseline_median']:g}"
        return value
    if "pause_ms" in details:
        return f"暂停{details['pause_ms']:g}毫秒"
    return "详见证据记录编号"


def _write_notes(output: Path, summary: dict[str, Any], alerts: list[dict[str, Any]]) -> None:
    window = summary["source_window"]
    lines = [
        "# 在线数据流告警运行说明",
        "",
        f"本次回放：{format_time(window['start_ms'])} 至 "
        f"{format_time(window['end_ms'])}，结束时暂停模拟。",
        "",
        "## 数据怎样到达、告警怎样产生",
        "",
        f"每 {summary['step_seconds']:g} 秒模拟时间投递一批已到达记录。"
        "这是数据运输步长，不是每隔这么久调用一次模型；本次没有加载模型、调用API或使用GPU。",
        "",
        "监测器按配置的统计窗口检查变化，生成、更新或解除告警；"
        "同节点的告警变化可整理成待调查任务。任务只是后续模型的工作入口，本次没有执行模型调查。",
        "",
        "所有到达的原始记录先存入 inbox.sqlite，再交给监测器。"
        "触发条件不会筛掉原始数据；没有告警的记录也保留，可供后续调查。"
        "调用记录按完成时间到达，不提前暴露未来记录。",
        "",
        "## 实际运行数量",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        f"| 数据投递批次 | {summary['ticks']:,} |",
        f"| 清单中应到达的记录 | {summary['expected_deliverable']:,} |",
        f"| 实际到达且保留的原始记录 | {summary['delivered_records']:,} |",
        f"| 窗口结束时尚未到达的记录 | {summary['pending_records']:,} |",
        f"| 告警事件（同一告警多次更新只计一个） | {summary['alert_count']:,} |",
        f"| 暂停时仍开启的告警 | {summary['open_alert_count']:,} |",
        f"| 已解除的告警 | {summary['cleared_alert_count']:,} |",
        f"| 告警产生、更新、解除等变化记录 | {summary['transition_count']:,} |",
        f"| 待模型调查的任务 | {summary['task_count']:,} |",
        "",
        "实际到达数量与原始清单一致。上述数量是规则监测结果，"
        "不是故障标签数量，也不是根因分析的正确率。告警多不等于诊断好；"
        "是否真正重要，要由后续调查和结果核对判断。",
        "规则解除只表示新观测不再满足这条规则，不代表已证明故障恢复。"
        "任务数量也是待调查入口数量，并不等于已经执行了这些次模型调用。",
        "",
        "### 按规则统计告警事件",
        "",
        "| 规则 | 告警事件数 |",
        "|---|---:|",
    ]
    for rule, count in summary["alerts_by_rule"].items():
        lines.append(f"| {_cell(RULE_NAMES.get(rule, rule))} | {count:,} |")
    if not summary["alerts_by_rule"]:
        lines.append("| 本次没有触发告警 | 0 |")
    lines.extend([
        "",
        "### 候选示例",
        "",
        "下表只展示部分告警，用来检查触发了什么，不作为根因结论。"
        "完整内容及对应原始记录编号见 alerts.json。",
        "",
        "| 节点 | 触发现象 | 首次发现 | 规则状态 | 最近观测 |",
        "|---|---|---|---|---|",
    ])
    examples = sorted(alerts, key=lambda alert: alert.get("first_seen_ms", 0))[:8]
    for alert in examples:
        status = {"open": "仍开启", "cleared": "已解除"}.get(
            alert.get("status"), str(alert.get("status", "未知")))
        lines.append(
            f"| {_cell(alert['node_id'])} | "
            f"{_cell(RULE_NAMES.get(alert['rule'], alert['rule']))} | "
            f"{format_time(alert['first_seen_ms'])} | {_cell(status)} | {_cell(_observation(alert))} |"
        )
    if not examples:
        lines.append("| — | 本次无告警 | — | — | — |")
    lines.extend([
        "",
        "## 结果文件",
        "",
        "- [summary.json](summary.json)：投递核对、规则和节点计数、监测统计与完整配置。",
        "- [alerts.json](alerts.json)：每个告警的状态、观测和原始证据编号。",
        "- [transitions.jsonl](transitions.jsonl)：按监测顺序保存的告警变化。",
        "- [tasks.jsonl](tasks.jsonl)：后续设备智能体可领取的调查任务。",
        "- [inbox.sqlite](inbox.sqlite)：已经到达的全部原始记录，保留供后续查询。",
        "",
        "本次只验证数据流与告警入口，没有使用标准答案，也尚未验证模型能否找到根因。",
        "历史样本不足的序列尚不能做相对变化判断；内存占用高还可能包含文件缓存。"
        "因此本轮不声称已找到全部重要异常，后续需检查告警噪声、冷启动及稀疏序列的巡检入口。",
        "",
    ])
    (output / "运行说明.md").write_text("\n".join(lines), encoding="utf-8")


def run_alerts(args: Any) -> dict[str, Any]:
    """CLI callback: replay a prepared window into a new inbox and alert engine."""
    step_seconds = float(args.step_seconds)
    if not math.isfinite(step_seconds) or round(step_seconds * 1000) <= 0:
        raise ValueError("--step-seconds must be finite and at least one millisecond")
    step_ms = round(step_seconds * 1000)
    prepared = Path(args.prepared).resolve()
    output = Path(args.output).resolve()
    # Never overwrite an earlier run and its preserved inbox.
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    delivered = 0
    ticks = 0
    clock_ms = None
    try:
        from .alerts import AlertConfig, AlertEngine

        overrides: dict[str, Any] = {}
        config_path = getattr(args, "config", None)
        if config_path is not None:
            overrides = json.loads(Path(config_path).read_text(encoding="utf-8"))
            if not isinstance(overrides, dict):
                raise ValueError("--config must contain a JSON object of AlertConfig fields")
        config = AlertConfig(**overrides)
        used_config = asdict(config) if is_dataclass(config) else dict(vars(config))
        by_node: Counter[str] = Counter()
        by_modality: Counter[str] = Counter()
        with ReplaySession(prepared, inbox_path=output / "inbox.sqlite") as session:
            engine = AlertEngine(session.start_ms, config=config)
            manifest = session.manifest
            expected = manifest["totals"]["deliverable"]
            next_progress = session.start_ms + 300_000
            for batch in session.iter_ticks(step_ms):
                ticks += 1
                clock_ms = batch["clock_ms"]
                for event in batch["events"]:
                    if event["available_at_ms"] >= clock_ms:
                        raise RuntimeError("Future observation was delivered to the alert engine")
                    engine.feed(event)
                    by_node[event["node_id"]] += 1
                    by_modality[event["modality"]] += 1
                    delivered += 1
                engine.advance_to(clock_ms)
                if clock_ms >= next_progress or batch["paused"]:
                    print(f"{format_time(clock_ms)} received={delivered:,} "
                          f"alerts={len(engine.alerts)} tasks={len(engine.tasks)}", flush=True)
                    while next_progress <= clock_ms:
                        next_progress += 300_000
            engine.finish(session.end_ms)
            if delivered != expected or delivered != session.total_delivered:
                raise RuntimeError(
                    f"Delivery count mismatch: processed={delivered}, "
                    f"inbox={session.total_delivered}, manifest={expected}"
                )
            alerts = list(engine.alerts.values())
            alerts_by_rule = Counter(alert["rule"] for alert in alerts)
            alerts_by_node = Counter(alert["node_id"] for alert in alerts)
            statuses = Counter(alert["status"] for alert in alerts)
            summary = {
                "schema_version": 1, "status": "complete", "prepared": str(prepared),
                "source_window": manifest["window"], "clock_ms": session.clock_ms,
                "step_seconds": step_seconds, "effective_step_ms": step_ms, "ticks": ticks,
                "prepared_records": manifest["totals"]["prepared"],
                "expected_deliverable": expected, "delivered_records": delivered,
                "pending_records": manifest["totals"]["pending"],
                "delivery_count_matches": delivered == expected,
                "delivered_by_modality": dict(sorted(by_modality.items())),
                "delivered_by_node": dict(sorted(by_node.items())),
                "alert_count": len(alerts), "open_alert_count": statuses["open"],
                "cleared_alert_count": statuses["cleared"],
                "transition_count": len(engine.transitions), "task_count": len(engine.tasks),
                "alerts_by_rule": dict(sorted(alerts_by_rule.items())),
                "alerts_by_node": dict(sorted(alerts_by_node.items())),
                "transitions_by_action": dict(sorted(Counter(
                    item["action"] for item in engine.transitions).items())),
                "tasks_by_node": dict(sorted(Counter(
                    item["node_id"] for item in engine.tasks).items())),
                "engine_stats": dict(engine.stats), "used_config": used_config,
                "checks": {"no_future_records_delivered": True,
                           "raw_data_filtered_by_anomaly": False,
                           "models_loaded": False, "labels_used": False},
                "inbox": str(output / "inbox.sqlite"),
                "wall_seconds": round(time.monotonic() - started, 3),
            }
            _write_json(output / "alerts.json", alerts)
            _write_jsonl(output / "transitions.jsonl", engine.transitions)
            _write_jsonl(output / "tasks.jsonl", engine.tasks)
            _write_notes(output, summary, alerts)
            _write_json(output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    except BaseException as error:
        _write_json(output / "failure.json", {
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "prepared": str(prepared), "clock_ms": clock_ms, "ticks": ticks,
            "delivered_before_failure": delivered,
            "wall_seconds": round(time.monotonic() - started, 3),
        })
        raise
