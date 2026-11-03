"""Disk-backed, arrival-causal Bank parent/child links with span latency.

No topology is inferred from component names. Span duration is not network RTT
or exclusive local processing time. The source inbox remains the raw archive.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import sqlite3

from .tool_stats import distribution, identifier, number


class TraceIndex:
    def __init__(self, path=":memory:"):
        self.connection = sqlite3.connect(path)
        self.clock_ms = None
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS spans (event_id TEXT PRIMARY KEY, trace_id TEXT, "
            "span_id TEXT, parent_id TEXT, node_id TEXT, start_ms INTEGER, "
            "available_ms INTEGER, duration REAL)"
        )
        if self.connection.execute("SELECT 1 FROM spans LIMIT 1").fetchone():
            self.close()
            raise ValueError("trace index must be fresh")
        self.connection.execute("CREATE INDEX IF NOT EXISTS span_key ON spans(trace_id, span_id)")
        self.connection.execute("CREATE INDEX IF NOT EXISTS span_arrival ON spans(available_ms)")

    def ingest(self, events, cutoff_ms):
        if (isinstance(cutoff_ms, bool) or not isinstance(cutoff_ms, int)
                or (self.clock_ms is not None and cutoff_ms < self.clock_ms)):
            raise ValueError("trace cutoff must be a nondecreasing integer")
        with self.connection:
            for event in events:
                if event["available_at_ms"] >= cutoff_ms:
                    raise ValueError("trace index cannot receive future records")
                if event["modality"] != "trace":
                    continue
                raw = event["raw"]
                duration = number(raw.get("duration"))
                if duration is not None and duration < 0:
                    duration = None
                self.connection.execute("INSERT INTO spans VALUES (?,?,?,?,?,?,?,?)", (
                    event["event_id"], str(raw.get("trace_id") or ""),
                    str(raw.get("span_id") or ""), str(raw.get("parent_id") or ""),
                    event["node_id"], event["observed_at_ms"], event["available_at_ms"], duration))
        self.clock_ms = cutoff_ms

    def snapshot(self, cutoff_ms, reference_start_ms, anchor_ms):
        if (self.clock_ms is None or cutoff_ms > self.clock_ms
                or not reference_start_ms <= anchor_ms <= cutoff_ms):
            raise ValueError("invalid or future trace snapshot range")
        # Both keys are scoped to the visible prefix. A later duplicate cannot
        # contaminate an old snapshot; once visible, ambiguity retracts the link.
        sql = """WITH visible AS (SELECT * FROM spans WHERE available_ms < ?),
            keys AS (SELECT trace_id, span_id, count(*) AS n FROM visible
                     GROUP BY trace_id, span_id)
            SELECT c.event_id,c.trace_id,c.span_id,c.parent_id,c.node_id,c.start_ms,
                   c.available_ms,c.duration,k.n,coalesce(pk.n,0),
                   p.event_id,p.node_id,p.start_ms,p.available_ms,p.duration
            FROM visible c JOIN keys k ON c.trace_id=k.trace_id AND c.span_id=k.span_id
            LEFT JOIN keys pk ON c.trace_id=pk.trace_id AND c.parent_id=pk.span_id
            LEFT JOIN visible p ON pk.n=1 AND c.trace_id=p.trace_id AND c.parent_id=p.span_id
            ORDER BY c.available_ms,c.event_id"""
        counts, groups = Counter(), defaultdict(list)
        for row in self.connection.execute(sql, (cutoff_ms,)):
            (eid, trace, span, parent, node, start, arrived, duration, own_n, parent_n,
             peid, pnode, pstart, parrived, pduration) = row
            counts["arrived_spans"] += 1
            if not trace or not span:
                counts["missing_identity"] += 1
                continue
            if own_n != 1 or parent_n > 1:
                counts["ambiguous_identity"] += 1
                continue
            if span == parent:
                counts["self_reference"] += 1
                continue
            if not parent:
                counts["no_parent_id"] += 1
                continue
            if parent_n == 0:
                counts["unmatched_parent"] += 1
                continue
            counts["matched_parent"] += 1
            if node == pnode:
                counts["same_node_parent"] += 1
                continue
            counts["cross_node_pairs"] += 1
            # Arrival of the parent may lag the child. Connection becomes known
            # only after both arrive; latency buckets use child completion time.
            groups[(pnode, node)].append({
                "child_event_id": eid, "parent_event_id": peid, "trace_id": trace,
                "child_start_ms": start, "parent_start_ms": pstart,
                "child_available_ms": arrived, "parent_available_ms": parrived,
                "known_at_ms": max(arrived, parrived), "child_duration_ms": duration,
                "parent_duration_ms": pduration,
                "child_within_parent_interval": (
                    pstart <= start and start + duration <= pstart + pduration
                    if duration is not None and pduration is not None else None),
            })
        links = []
        for (source, target), pairs in sorted(groups.items()):
            latency = {}
            for name, low, high in (("reference", reference_start_ms, anchor_ms),
                                     ("current", anchor_ms, cutoff_ms)):
                selected = [p for p in pairs if low <= p["child_available_ms"] < high]
                latency[name] = distribution([p["child_duration_ms"] for p in selected
                                               if p["child_duration_ms"] is not None])
                latency[name]["calls"] = len(selected)
            valid = [p for p in pairs if p["child_duration_ms"] is not None]
            representatives = {p["child_event_id"]: p for p in pairs[:1] + pairs[-1:]}
            if valid:
                slowest = max(valid, key=lambda p: p["child_duration_ms"])
                representatives[slowest["child_event_id"]] = slowest
            links.append({"link_id": identifier(source, target), "source_node": source,
                          "target_node": target, "matched_calls": len(pairs),
                          "matched_calls_scope": "all_arrived_prefix",
                          "first_child_start_ms": min(p["child_start_ms"] for p in pairs),
                          "last_child_start_ms": max(p["child_start_ms"] for p in pairs),
                          "child_span_latency_ms": latency,
                          "interval_mismatch_count": sum(p["child_within_parent_interval"] is False for p in pairs),
                          "examples": list(representatives.values())})
        return {"cutoff_ms": cutoff_ms, "counts": dict(counts), "links": links,
                "latency_scope": {"reference_start_ms": reference_start_ms,
                                  "anchor_ms": anchor_ms, "basis": "child_completion"},
                "limitations": ["已到达记录中的唯一父子匹配；未匹配不等于断链。",
                                 "子调用耗时不是网络传输延迟；父子耗时可能重叠，不能相加。",
                                 "Bank调用表无成功/失败状态；没有边的候选组件不能排除。"]}

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
