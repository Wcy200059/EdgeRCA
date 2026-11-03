"""Prepare lossless Bank observation streams using bounded-memory disk sorting."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from zoneinfo import ZoneInfo


TIMEZONE = ZoneInfo("Asia/Shanghai")
SOURCES = (
    ("metric/metric_container.csv", "metric", "device", "cmdb_id", 1000),
    ("metric/metric_app.csv", "business_metric", "business", "tc", 1000),
    ("log/log_service.csv", "log", "device", "cmdb_id", 1000),
    ("trace/trace_span.csv", "trace", "device", "cmdb_id", 1),
)


def parse_time(value: str) -> int:
    """Interpret an explicit local datetime as integer epoch milliseconds."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TIMEZONE)
    return int(parsed.timestamp() * 1000)


def format_time(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, TIMEZONE).isoformat(
        timespec="milliseconds"
    )


def _milliseconds(value: str, factor: int) -> int:
    try:
        number = Decimal(value) * factor
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError("must be finite and representable in integer milliseconds")
        return int(number)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"Invalid time value {value!r}: {error}") from error


def _date_directories(root: Path, start_ms: int, end_ms: int) -> list[Path]:
    # Adjacent calendar directories also contain boundary-spilling records.
    first = datetime.fromtimestamp(start_ms / 1000, TIMEZONE).date() - timedelta(days=1)
    last = datetime.fromtimestamp((end_ms - 1) / 1000, TIMEZONE).date() + timedelta(days=1)
    return [root / (first + timedelta(days=index)).strftime("%Y_%m_%d")
            for index in range((last - first).days + 1)
            if (root / (first + timedelta(days=index)).strftime("%Y_%m_%d")).is_dir()]


def _node_path(node_id: str) -> str:
    readable = "".join(character if character.isascii() and character.isalnum()
                       else "_" for character in node_id)[:70]
    digest = hashlib.sha256(node_id.encode()).hexdigest()[:12]
    return f"nodes/{readable}_{digest}.jsonl"


def prepare_bank(
    telemetry_root: Path,
    start: str,
    end: str,
    output_dir: Path,
    query_source: dict | None = None,
) -> dict:
    """Keep all selected original rows; no label, anomaly filter or deduplication.

    Point observations are selected in [start, end). Spans overlapping the window
    are retained, including unfinished spans, but replay only releases completed
    spans with available_at_ms < end. Arrival time is simulated because the CSVs
    do not contain actual collection/ingestion timestamps.
    """
    telemetry_root = Path(telemetry_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    start_ms, end_ms = parse_time(start), parse_time(end)
    if end_ms <= start_ms:
        raise ValueError("End must be later than start; use explicit dates across midnight")
    source_files = [
        (directory / filename, modality, kind, identity, factor)
        for directory in _date_directories(telemetry_root, start_ms, end_ms)
        for filename, modality, kind, identity, factor in SOURCES
        if (directory / filename).is_file()
    ]
    if not source_files:
        raise FileNotFoundError(f"No Bank source CSVs for this window in {telemetry_root}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building-",
                                    dir=output_dir.parent))
    connection = None
    try:
        connection = sqlite3.connect(staging / "sort.sqlite")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute("CREATE TABLE events (node_id TEXT, available INTEGER, "
                           "event_id TEXT PRIMARY KEY, modality TEXT, payload TEXT)")
        nodes: dict[str, dict] = {}
        source_audits = []
        total_prepared, deliverable = 0, 0
        by_modality: Counter = Counter()
        insert_batch = []
        for path, modality, kind, identity, factor in source_files:
            relative_path = path.relative_to(telemetry_root).as_posix()
            audit = {"path": relative_path, "modality": modality,
                     "rows_scanned": 0, "rows_prepared": 0, "rows_excluded": 0,
                     "deliverable": 0, "pending": 0, "input_time_reversals": 0}
            previous_time = None
            print(f"Reading {relative_path}", flush=True)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"timestamp", identity}
                if modality == "trace":
                    required.update({"duration", "trace_id", "span_id", "parent_id"})
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    raise ValueError(f"{path}: missing required columns {required}")
                if len(set(reader.fieldnames)) != len(reader.fieldnames):
                    raise ValueError(f"{path}: duplicate CSV column names")
                for row_number, raw in enumerate(reader, start=1):
                    audit["rows_scanned"] += 1
                    try:
                        if None in raw or any(value is None for value in raw.values()):
                            raise ValueError("CSV row does not match header")
                        source_id = raw[identity]
                        if not source_id or not source_id.strip():
                            raise ValueError(f"Missing {identity}; record cannot be silently dropped")
                        observed = _milliseconds(raw["timestamp"], factor)
                        available = observed
                        if modality == "trace":
                            duration = _milliseconds(raw["duration"], 1)
                            if duration < 0:
                                raise ValueError("Negative span duration")
                            available += duration
                        node_id = f"{kind}:{source_id}"
                        node = nodes.setdefault(node_id, {
                            "node_id": node_id, "kind": kind, "source_id": source_id,
                            "path": _node_path(node_id), "counts": {},
                            "deliverable": 0, "pending": 0,
                        })
                        node["counts"].setdefault(modality, 0)
                        if previous_time is not None and available < previous_time:
                            audit["input_time_reversals"] += 1
                        previous_time = available
                        selected = (observed < end_ms and available >= start_ms
                                    if modality == "trace"
                                    else start_ms <= observed < end_ms)
                        if not selected:
                            audit["rows_excluded"] += 1
                            continue
                        event_id = f"Bank:{relative_path}:{row_number:012d}"
                        event = {
                            "event_id": event_id, "modality": modality,
                            "node_id": node_id, "source_id": source_id,
                            "observed_at_ms": observed, "available_at_ms": available,
                            "source_file": relative_path, "source_row": row_number,
                            "raw": raw,
                        }
                        insert_batch.append((node_id, available, event_id, modality,
                                             json.dumps(event, ensure_ascii=False,
                                                        separators=(",", ":"))))
                        node["counts"][modality] = node["counts"].get(modality, 0) + 1
                        is_deliverable = available < end_ms
                        key = "deliverable" if is_deliverable else "pending"
                        node[key] += 1
                        audit[key] += 1
                        audit["rows_prepared"] += 1
                        total_prepared += 1
                        deliverable += int(is_deliverable)
                        by_modality[modality] += 1
                        if len(insert_batch) >= 4096:
                            connection.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                                                   insert_batch)
                            connection.commit()
                            insert_batch.clear()
                    except (KeyError, ValueError) as error:
                        raise ValueError(f"{path}, data row {row_number}: {error}") from error
            source_audits.append(audit)
            print(f"  scanned={audit['rows_scanned']} kept={audit['rows_prepared']} "
                  f"pending={audit['pending']}", flush=True)
        if insert_batch:
            connection.executemany("INSERT INTO events VALUES (?, ?, ?, ?, ?)", insert_batch)
            connection.commit()
        connection.execute("CREATE INDEX delivery ON events(node_id, available, event_id)")
        (staging / "nodes").mkdir()
        for node in sorted(nodes.values(), key=lambda item: item["node_id"]):
            rows_written = 0
            digest = hashlib.sha256()
            with (staging / node["path"]).open("wb") as handle:
                for payload, in connection.execute(
                    "SELECT payload FROM events WHERE node_id=? ORDER BY available,event_id",
                    (node["node_id"],),
                ):
                    line = (payload + "\n").encode("utf-8")
                    handle.write(line)
                    digest.update(line)
                    rows_written += 1
            if rows_written != sum(node["counts"].values()):
                raise RuntimeError(f"Node count mismatch: {node['node_id']}")
            node["sha256"] = digest.hexdigest()
        if total_prepared != sum(sum(node["counts"].values()) for node in nodes.values()):
            raise RuntimeError("Source and partition counts do not match")
        manifest = {
            "schema_version": 1, "benchmark": "Bank", "timezone": "Asia/Shanghai",
            "telemetry_root": str(telemetry_root),
            "window": {"start_ms": start_ms, "end_ms": end_ms,
                       "start": format_time(start_ms), "end": format_time(end_ms),
                       "timezone": "Asia/Shanghai", "bounds": "[start,end)"},
            "arrival_policy": {"metric_log_business": "observation_timestamp",
                               "trace": "start_timestamp_plus_duration",
                               "note": "Simulated availability; actual ingestion times are absent"},
            "source_scope": "Window calendar dates plus existing adjacent calendar days; "
                            "source paths below define the exact audited scope",
            "inventory_policy": "All identities in scanned files, including zero-event nodes; "
                                "only routing metadata, not future observations",
            "source_files": source_audits,
            "nodes": sorted(nodes.values(), key=lambda item: item["node_id"]),
            "totals": {"prepared": total_prepared, "deliverable": deliverable,
                       "pending": total_prepared - deliverable,
                       "by_modality": dict(by_modality)},
        }
        if query_source is not None:
            # Explicitly whitelist metadata: never copy the query row or answers.
            manifest["window_source"] = {
                key: query_source[key] for key in ("query_file", "row_index")
                if key in query_source
            }
            manifest["window_source"]["use"] = "Question time interval only"
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        connection.close()
        connection = None
        (staging / "sort.sqlite").unlink()
        # A failed preparation never leaves an apparently usable output directory.
        os.rename(staging, output_dir)
        return manifest
    except BaseException:
        if connection is not None:
            connection.close()
        shutil.rmtree(staging)
        raise
