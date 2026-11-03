"""Causal, shared-clock replay of prepared per-device observation streams.

The SQLite inbox contains only delivered events. Files on disk may additionally
contain calls which started during the window but have not completed by its end;
those records never become visible through this session's query interface.
"""

from __future__ import annotations

import heapq
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterator, TextIO


MODALITIES = frozenset({"metric", "log", "trace", "business_metric"})


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer number of milliseconds")
    return value


class ReplaySession:
    """Merge node files by availability time and expose only the arrived prefix.

    ``advance_to(t)`` delivers records in ``[previous_clock, t)``. In particular,
    stopping at the window end does not reveal records available at that end.
    ``query`` filters by observation time, not arrival time, and never reads the
    source files. It therefore also works for calls which started before replay.

    An optional ``inbox_path`` preserves the received records for inspection;
    it must not already contain records. Without it, the SQLite file is temporary.
    """

    def __init__(self, prepared_dir: str | Path, inbox_path: str | Path | None = None):
        self.prepared_dir = Path(prepared_dir).resolve()
        self.manifest = json.loads((self.prepared_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported prepared stream schema_version")
        window = self.manifest["window"]
        self.start_ms = _integer(window["start_ms"], "window.start_ms")
        self.end_ms = _integer(window["end_ms"], "window.end_ms")
        if self.end_ms <= self.start_ms:
            raise ValueError("window.end_ms must be greater than window.start_ms")
        self.clock_ms = self.start_ms
        self.total_delivered = 0
        self.delivered_counts = {modality: 0 for modality in sorted(MODALITIES)}
        self._files: dict[str, TextIO] = {}
        self._previous_keys: dict[str, tuple[int, str]] = {}
        self._line_numbers: dict[str, int] = {}
        self._heap: list[tuple[int, str, str, dict[str, Any]]] = []
        self._closed = False
        self._temporary_dir: tempfile.TemporaryDirectory | None = None
        self._connection: sqlite3.Connection | None = None
        self._nodes: dict[str, dict[str, Any]] = {}
        try:
            for node in self.manifest["nodes"]:
                node_id = node["node_id"]
                if not isinstance(node_id, str) or not node_id:
                    raise ValueError("manifest node_id must be a nonempty string")
                if node_id in self._nodes:
                    raise ValueError(f"duplicate manifest node_id: {node_id}")
                path = (self.prepared_dir / node["path"]).resolve()
                if not path.is_relative_to(self.prepared_dir):
                    raise ValueError(f"node path escapes prepared directory: {node_id}")
                self._nodes[node_id] = node
                self._files[node_id] = path.open(encoding="utf-8")
                self._line_numbers[node_id] = 0
            if inbox_path is None:
                self._temporary_dir = tempfile.TemporaryDirectory(prefix="opsagent-edge-inbox-")
                self.inbox_path = Path(self._temporary_dir.name) / "inbox.sqlite3"
            else:
                self.inbox_path = Path(inbox_path).resolve()
                self.inbox_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.inbox_path)
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "event_id TEXT PRIMARY KEY, node_id TEXT NOT NULL, modality TEXT NOT NULL, "
                "observed_at_ms INTEGER NOT NULL, available_at_ms INTEGER NOT NULL, "
                "payload TEXT NOT NULL)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS events_node_time "
                "ON events (node_id, available_at_ms, event_id)"
            )
            if self._connection.execute("SELECT 1 FROM events LIMIT 1").fetchone():
                raise ValueError("inbox already contains records; use a fresh inbox_path")
            for node_id in self._nodes:
                self._read_next(node_id)
        except Exception:
            self.close()
            raise

    @property
    def paused(self) -> bool:
        return self.clock_ms == self.end_ms

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("replay session is closed")

    def _read_next(self, node_id: str) -> None:
        stream = self._files[node_id]
        line = stream.readline()
        if not line:
            return
        self._line_numbers[node_id] += 1
        context = f"{node_id}, line {self._line_numbers[node_id]}"
        try:
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError("event must be a JSON object")
            event_id = event["event_id"]
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("event_id must be a nonempty string")
            available = _integer(event["available_at_ms"], "available_at_ms")
            observed = _integer(event["observed_at_ms"], "observed_at_ms")
            if event["node_id"] != node_id:
                raise ValueError("event belongs to a different node")
            if event["source_id"] != self._nodes[node_id]["source_id"]:
                raise ValueError("source_id disagrees with node manifest")
            if event["modality"] not in MODALITIES:
                raise ValueError("unknown modality")
            if not isinstance(event["raw"], dict):
                raise ValueError("raw must preserve the source row as an object")
            if available < self.start_ms:
                raise ValueError("event is available before replay window")
            if available < observed:
                raise ValueError("event cannot be available before it was observed")
            key = available, event_id
            previous = self._previous_keys.get(node_id)
            if previous is not None and key <= previous:
                raise ValueError("node stream must be strictly sorted by (available_at_ms, event_id)")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid prepared event at {context}: {error}") from error
        self._previous_keys[node_id] = key
        heapq.heappush(self._heap, (available, event_id, node_id, event))

    def advance_to(self, clock_ms: int) -> dict[str, Any]:
        """Advance the shared clock without exposing records at/after it."""
        self._ensure_open()
        target = _integer(clock_ms, "clock_ms")
        if target < self.clock_ms:
            raise ValueError("simulation clock cannot move backwards")
        if target > self.end_ms:
            raise ValueError("simulation clock cannot pass the replay window end")
        previous_clock = self.clock_ms
        events: list[dict[str, Any]] = []
        counts = {modality: 0 for modality in sorted(MODALITIES)}
        assert self._connection is not None
        try:
            with self._connection:
                while self._heap and self._heap[0][0] < target:
                    _, event_id, node_id, event = heapq.heappop(self._heap)
                    # Validate the next row before making this one visible. This
                    # catches an out-of-order stream as soon as its prefix is read.
                    self._read_next(node_id)
                    self._connection.execute(
                        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                        (event_id, node_id, event["modality"], event["observed_at_ms"],
                         event["available_at_ms"], json.dumps(event, ensure_ascii=False)),
                    )
                    events.append(event)
                    counts[event["modality"]] += 1
        except Exception:
            # A malformed stream cannot be resumed after cursor consumption.
            self.close()
            raise
        self.clock_ms = target
        self.total_delivered += len(events)
        for modality, count in counts.items():
            self.delivered_counts[modality] += count
        return {
            "previous_clock_ms": previous_clock,
            "clock_ms": self.clock_ms,
            "paused": self.paused,
            "events": events,
            "delivered": len(events),
            "counts": counts,
            "total_delivered": self.total_delivered,
        }

    def tick(self, step_ms: int) -> dict[str, Any]:
        self._ensure_open()
        step = _integer(step_ms, "step_ms")
        if step <= 0:
            raise ValueError("step_ms must be positive")
        return self.advance_to(min(self.end_ms, self.clock_ms + step))

    def iter_ticks(self, step_ms: int) -> Iterator[dict[str, Any]]:
        self._ensure_open()
        step = _integer(step_ms, "step_ms")
        if step <= 0:
            raise ValueError("step_ms must be positive")
        while not self.paused:
            yield self.tick(step)

    def query(
        self,
        node_id: str,
        modality: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int | None = 100,
        available_before_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read arrived events, optionally filtering observation time [start,end)."""
        self._ensure_open()
        if node_id not in self._nodes:
            raise KeyError(f"unknown node_id: {node_id}")
        clauses = ["node_id = ?"]
        arguments: list[Any] = [node_id]
        if available_before_ms is not None:
            cutoff = _integer(available_before_ms, "available_before_ms")
            if cutoff > self.clock_ms:
                raise ValueError("availability cutoff cannot exceed the current replay clock")
            clauses.append("available_at_ms < ?")
            arguments.append(cutoff)
        if modality is not None:
            if modality not in MODALITIES:
                raise ValueError(f"unknown modality: {modality}")
            clauses.append("modality = ?")
            arguments.append(modality)
        if start_ms is not None:
            start_ms = _integer(start_ms, "start_ms")
            clauses.append("observed_at_ms >= ?")
            arguments.append(start_ms)
        if end_ms is not None:
            end_ms = _integer(end_ms, "end_ms")
            clauses.append("observed_at_ms < ?")
            arguments.append(end_ms)
        if start_ms is not None and end_ms is not None and end_ms < start_ms:
            raise ValueError("end_ms must not be before start_ms")
        statement = "SELECT payload FROM events WHERE " + " AND ".join(clauses)
        statement += " ORDER BY available_at_ms, event_id"
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError("limit must be a nonnegative integer or None")
            statement += " LIMIT ?"
            arguments.append(limit)
        assert self._connection is not None
        return [json.loads(row[0]) for row in self._connection.execute(statement, arguments)]

    def close(self) -> None:
        if self._closed:
            return
        for stream in self._files.values():
            stream.close()
        if self._connection is not None:
            self._connection.close()
        if self._temporary_dir is not None:
            self._temporary_dir.cleanup()
        self._closed = True

    def __enter__(self) -> "ReplaySession":
        self._ensure_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
