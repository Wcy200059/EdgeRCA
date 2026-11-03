"""CPU-owned cumulative observation records with versioned model annotations.

One record is one node/modality/signal, never one inferred root cause. Independent
rules retain their own episodes. A cleared rule cannot clear a different rule.
Full transitions and references stay local; compact views are replaceable indexes.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import re

from .data import format_time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


class RollingNodeMemory:
    def __init__(self, case_id, node_id, start_ms, end_ms):
        if end_ms <= start_ms:
            raise ValueError("invalid case window")
        self.case_id, self.node_id = case_id, node_id
        self.start_ms, self.end_ms, self.clock_ms = start_ms, end_ms, start_ms
        self.records = {}
        self.transitions = {}
        self.rows = Counter()
        self.last_data_ms = None
        self.requests, self.notes = {}, {}

    def _cutoff(self, cutoff):
        if not self.clock_ms <= cutoff <= self.end_ms:
            raise ValueError("cutoff moves backwards or outside this case")

    def receive_counts(self, events, cutoff):
        self._cutoff(cutoff)
        for event in events:
            if (event["node_id"] != self.node_id
                    or not self.start_ms <= event["available_at_ms"] < cutoff):
                raise ValueError("nonlocal or future raw event")
        # ReplaySession owns deduplication and the complete raw inbox.
        self.rows.update(e["modality"] for e in events)
        if events:
            self.last_data_ms = max(e["available_at_ms"] for e in events)

    def group_id(self, rule, signal):
        return "o-" + digest([self.case_id, self.node_id, rule.split(".")[0], signal])[:16]

    def sync(self, transitions, alerts, cutoff):
        """Consume the new transition prefix and the detector's current snapshots.

        Repeated hits may update counts/latest values without a new transition.
        No output is marked model-processed by this CPU operation.
        """
        self._cutoff(cutoff)
        for item in transitions:
            if (item["node_id"] != self.node_id
                    or not self.start_ms <= item["at_ms"] <= cutoff
                    or item["action"] not in {"open", "cleared", "escalated", "updated"}):
                raise ValueError("invalid transition node, time or action")
        for alert in alerts:
            if (alert["node_id"] != self.node_id
                    or not self.start_ms <= alert["first_seen_ms"] <= alert["last_seen_ms"] <= cutoff
                    or alert.get("cleared_at_ms", cutoff) > cutoff):
                raise ValueError("invalid or future alert snapshot")
        dirty = set()
        for item in transitions:
            tid = digest(item)
            if tid in self.transitions:
                continue
            gid = self.group_id(item["rule"], item["signal"])
            record = self.records.setdefault(gid, {
                "observation_id": gid, "node_id": self.node_id,
                "modality": item["rule"].split(".")[0], "signal": item["signal"],
                "version": 0, "episodes": {}, "transition_ids": [], "cutoff_ms": cutoff})
            self.transitions[tid] = deepcopy(item)
            record["transition_ids"].append(tid)
            dirty.add(gid)
        for alert in alerts:
            gid = self.group_id(alert["rule"], alert["signal"])
            if gid not in self.records:
                raise ValueError("alert snapshot lacks its opening transition")
            record = self.records[gid]
            if record["episodes"].get(alert["alert_id"]) != alert:
                record["episodes"][alert["alert_id"]] = deepcopy(alert)
                dirty.add(gid)
        for gid in dirty:
            self.records[gid]["version"] += 1
            self.records[gid]["cutoff_ms"] = cutoff
        self.clock_ms = cutoff
        return sorted(dirty)

    def project(self, gid):
        record = self.records[gid]
        changes = [self.transitions[t] for t in record["transition_ids"]]
        episodes = list(record["episodes"].values())
        if not episodes:
            raise ValueError("observation has no detector snapshot")
        active, cycles, state_changes = set(), 0, []
        by_time = {}
        for change in changes:
            by_time.setdefault(change["at_ms"], []).append(change)
        for at, items in sorted(by_time.items()):
            before = bool(active)
            for item in items:
                if item["action"] == "open":
                    active.add(item["alert_id"])
                elif item["action"] == "cleared":
                    active.discard(item["alert_id"])
            if not before and active:
                cycles += 1
            if before != bool(active):
                state_changes.append({"time": format_time(at),
                                      "status": "triggered" if active else "rules_cleared"})
        expected_active = {a["alert_id"] for a in episodes if a["status"] == "open"}
        if active != expected_active:
            raise ValueError("transition history disagrees with detector state")
        # Keep each rule's latest evaluation; a clearing evaluation can differ
        # from the last hit stored in AlertEngine.latest.
        latest = {}
        for item in sorted(changes, key=lambda t: t["at_ms"]):
            latest[item["rule"]] = {"at_ms": item["at_ms"], "details": deepcopy(item["details"])}
        for a in episodes:
            if a["status"] == "open" and a["last_seen_ms"] > latest[a["rule"]]["at_ms"]:
                latest[a["rule"]] = {"at_ms": a["last_seen_ms"], "details": deepcopy(a["latest"])}
        selected_fields = ("value", "kind", "unit", "baseline_median", "baseline_mean",
                           "direction", "threshold", "p95_ms", "max_ms", "calls", "pause_ms")
        latest_values = []
        for rule, item in sorted(latest.items()):
            d = item["details"]
            latest_values.append({"rule": rule, "time": format_time(item["at_ms"]),
                                  **{k: d[k] for k in selected_fields if k in d}})
        evidence = {eid for t in changes for eid in t["evidence_ids"]}
        for a in episodes:
            evidence.update(a["evidence_ids"])
            evidence.update(a.get("latest_evidence_ids", []))
        actions = Counter(t["action"] for t in changes)
        note = self.notes.get((gid, record["version"]))
        return {"observation_id": gid, "version": record["version"],
                "node_id": self.node_id, "modality": record["modality"], "signal": record["signal"],
                "cutoff_ms": record["cutoff_ms"],
                "first_seen": format_time(min(a["first_seen_ms"] for a in episodes)),
                "last_trigger": format_time(max(a["last_seen_ms"] for a in episodes)),
                "last_change": format_time(max(t["at_ms"] for t in changes)),
                "status": "triggered" if active else "rules_cleared",
                "active_rules": sorted({a["rule"] for a in episodes if a["status"] == "open"}),
                "rule_episodes": len(episodes), "rule_hits": sum(a["hits"] for a in episodes),
                "signal_activation_cycles": cycles, "state_changes": state_changes,
                "actions": dict(actions), "latest_values": latest_values,
                "transition_count": len(changes), "source_reference_count": len(evidence),
                "error_variants": sorted({t["details"]["error_context"] for t in changes
                                           if "error_context" in t["details"]}),
                "model_status": "annotated_unverified" if note else "pending",
                "model_note": deepcopy(note),
                "detail_ref": gid,
                "note": "规则段与触发次数不是故障数；规则解除不等于故障恢复。"}

    def overview(self):
        items = [self.project(gid) for gid in self.records]
        return {"case_id": self.case_id, "node_id": self.node_id,
                "window_start_ms": self.start_ms, "cutoff_ms": self.clock_ms,
                "raw_rows_received": dict(self.rows), "last_data_ms": self.last_data_ms,
                "observation_groups": len(items),
                "last_known_triggered_groups": sum(i["status"] == "triggered" for i in items),
                "groups_with_multiple_activations": sum(i["signal_activation_cycles"] > 1 for i in items),
                "model_pending_groups": sum(i["model_status"] == "pending" for i in items),
                "groups_by_modality": dict(Counter(i["modality"] for i in items)),
                "details_ref": self.node_id,
                "note": "节点目录，不是诊断证据全文；无告警、模型未读均不代表正常。"}

    def changes_since(self, versions):
        """Upserts replace the same stable ID; a consumer needn't append history."""
        if any(g not in self.records or v > self.records[g]["version"] for g, v in versions.items()):
            raise ValueError("unknown/future version cursor")
        return [self.project(g) for g, r in self.records.items() if r["version"] > versions.get(g, 0)]

    def freeze(self, gid):
        fact = self.project(gid)
        # Previous generated prose is never fed back as an established fact.
        fact.pop("model_note")
        packet = {"case_id": self.case_id, "node_id": self.node_id,
                  "snapshot_cutoff_ms": self.clock_ms, "fact": fact}
        request_id = "mem-" + digest(packet)[:18]
        packet["request_id"] = request_id
        self.requests[request_id] = deepcopy(packet)
        return deepcopy(packet)

    def complete(self, request_id, response):
        if request_id not in self.requests:
            raise ValueError("unknown frozen request")
        packet = self.requests[request_id]
        text = validate_memory_reply(packet, response)
        fact = packet["fact"]
        key = (fact["observation_id"], fact["version"])
        self.notes[key] = {"text": text, "request_id": request_id,
                           "semantics_verified": False}
        return {"applies_to_current": self.records[key[0]]["version"] == key[1],
                "observation_id": key[0], "version": key[1]}

    def export(self):
        return {"overview": self.overview(), "records": deepcopy(self.records),
                "transitions": deepcopy(self.transitions),
                "current_views": [self.project(g) for g in self.records]}


def memory_messages(packet):
    f = packet["fact"]
    status = "仍有规则触发" if f["status"] == "triggered" else "本信号的规则条件已解除"
    # Temporal/numeric fields are delivered unchanged beside the note. The tiny
    # model previously confused a clearing timestamp with a recurrence time.
    phenomenon = {"completed_call_p95": "窗口调用延迟偏高",
                  "maximum_call_duration": "单次调用特别慢"}.get(f["signal"], f["signal"])
    history = ("曾经解除后再次触发，存在反复。" if f["signal_activation_cycles"] > 1
               else "到目前只有一轮触发，未记录解除后再次触发。")
    facts = f"观测现象：{phenomenon}。历史：{history}目前最后已知状态：{status}。"
    return [{"role": "system", "content":
        "你是组件的观测文字助手。用一句话概括观测现象、是否反复和目前规则状态。"
        "准确时间、数字和证据由程序附带，不要重写或猜测。规则解除不等于故障恢复，不判断根因。"
        "只输出JSON对象，只有observation_id和text两个键；text是短字符串，不是对象或列表，最多80字。"
        "有反复的历史要保留，过去再次触发不等于目前仍触发。不要补充原因。"},
        {"role": "user", "content": json.dumps({"observation_id": f["observation_id"],
             "已核对的程序观测": facts}, ensure_ascii=False)}]


def validate_memory_reply(packet, reply, *, qualitative_only=False):
    f = packet["fact"]
    if not isinstance(reply, dict) or set(reply) != {"observation_id", "text"}:
        raise ValueError("reply needs only observation_id and text")
    if reply["observation_id"] != f["observation_id"]:
        raise ValueError("wrong observation ID")
    text = reply["text"]
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 180:
        raise ValueError("text must be 1..180 characters")
    if qualitative_only and re.search(r"\d", text):
        raise ValueError("数值与时间由程序附带，文字说明不要重写")
    count_expression = re.search(r"(?:出现|触发)(?:了)?\s*(\d+)\s*[轮次]", text)
    count_is_correct = (count_expression is not None
                        and int(count_expression.group(1)) == f["signal_activation_cycles"])
    if (f["signal_activation_cycles"] > 1
            and not any(w in text for w in ("再次", "反复", "多次", "重新"))
            and not count_is_correct):
        raise ValueError("must retain recurrence, not only the latest event")
    if f["status"] == "rules_cleared" and not any(w in text for w in ("已解除", "不再满足")):
        raise ValueError("must describe cleared rule conditions")
    if f["status"] == "triggered" and not any(w in text for w in ("仍", "当前", "目前")):
        raise ValueError("must describe the current trigger state")
    return text
