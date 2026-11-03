"""A node's CPU-owned observations and incremental model-summary protocol.

This is a state-machine prototype, not a model runner or a concurrent scheduler.
The ingestion owner calls these methods; inference uses frozen request copies.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .data import format_time


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _text(value, name, maximum=None):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    if maximum is not None and len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters; split the work, do not truncate")
    return value


class NodeState:
    def __init__(self, case_id: str, node_id: str, start_ms: int, end_ms: int,
                 journal_path: str | Path | None = None):
        self.case_id = _text(case_id, "case_id")
        self.node_id = _text(node_id, "node_id")
        self.start_ms = _integer(start_ms, "start_ms")
        self.end_ms = _integer(end_ms, "end_ms")
        if end_ms <= start_ms:
            raise ValueError("end_ms must follow start_ms")
        self.clock_ms = start_ms
        self._observations = {}
        self._notes = {}
        self._questions = []
        self._processed = set()
        self._pending_order = []
        self._in_flight = None
        self._request_number = 0
        self._outbox = []
        self.journal_path = Path(journal_path) if journal_path is not None else None
        if self.journal_path is not None:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("x", encoding="utf-8"):
                pass

    def _record(self, record):
        if self.journal_path is not None:
            with self.journal_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _check_clock(self, cutoff_ms):
        cutoff = _integer(cutoff_ms, "cutoff_ms")
        if not self.clock_ms <= cutoff <= self.end_ms:
            raise ValueError("cutoff must not move backwards or exceed the case window")
        return cutoff

    def _finding_id(self, observation_id):
        value = json.dumps([self.case_id, self.node_id, observation_id], ensure_ascii=False)
        return "f-" + hashlib.sha256(value.encode()).hexdigest()[:16]

    @property
    def needs_update(self):
        return any(identifier not in self._processed for identifier in self._observations)

    @property
    def busy(self):
        return self._in_flight is not None

    def receive(self, observation: dict, evidence_catalog: dict, cutoff_ms: int) -> bool:
        """Store an immutable CPU observation and make it reportable immediately.

        The caller's CPU tools must compute descriptions from actual records.
        This method checks provenance/time, not the truth of that computation.
        """
        cutoff = self._check_clock(cutoff_ms)
        if not isinstance(evidence_catalog, dict):
            raise ValueError("evidence_catalog must be a dictionary")
        required = {"observation_id", "kind", "observed_at_ms", "available_at_ms",
                    "description", "evidence_ids"}
        if not isinstance(observation, dict) or set(observation) != required:
            raise ValueError("observation must contain exactly the protocol fields")
        identifier = _text(observation["observation_id"], "observation_id")
        _text(observation["kind"], "kind")
        _text(observation["description"], "description")
        observed = _integer(observation["observed_at_ms"], "observed_at_ms")
        available = _integer(observation["available_at_ms"], "available_at_ms")
        if observed > available or not self.start_ms <= available < cutoff:
            raise ValueError("observation is not yet visible in this case window")
        identifiers = observation["evidence_ids"]
        if (not isinstance(identifiers, list) or not identifiers
                or any(not isinstance(item, str) or not item for item in identifiers)
                or len(set(identifiers)) != len(identifiers)):
            raise ValueError("evidence_ids must be unique nonempty IDs")
        evidence = {}
        for evidence_id in identifiers:
            item = evidence_catalog.get(evidence_id)
            if not isinstance(item, dict) or item.get("node_id") != self.node_id:
                raise ValueError("evidence must exist and belong to this node")
            at = _integer(item.get("available_at_ms"), "evidence.available_at_ms")
            event_time = _integer(item.get("observed_at_ms"), "evidence.observed_at_ms")
            if event_time > at or not self.start_ms <= at <= available or at >= cutoff:
                raise ValueError("evidence has an invalid or future availability time")
            evidence[evidence_id] = deepcopy(item)
        stored = {"observation": deepcopy(observation), "evidence": evidence}
        if identifier in self._observations:
            if self._observations[identifier] != stored:
                raise ValueError("an observation ID cannot overwrite earlier facts")
            self.clock_ms = cutoff
            return False
        self._record({"type": "observation", "case_id": self.case_id,
                      "node_id": self.node_id, "cutoff_ms": cutoff, **stored})
        self._observations[identifier] = stored
        self._pending_order.append(identifier)
        self.clock_ms = cutoff
        self._outbox.append({
            "type": "cpu_observation", "case_id": self.case_id, "node_id": self.node_id,
            "finding_id": self._finding_id(identifier), "cutoff_ms": cutoff,
            "observation": deepcopy(observation),
            "note": "Program observation; model wording is optional and cannot replace this record",
        })
        return True

    def begin_update(self, cutoff_ms: int, max_items: int = 6,
                     previous_limit: int = 4) -> dict | None:
        """Freeze one small page. New arrivals may continue while it is in flight."""
        cutoff = self._check_clock(cutoff_ms)
        if self.busy:
            raise RuntimeError("one model update is already in flight")
        if not 1 <= _integer(max_items, "max_items") <= 12:
            raise ValueError("max_items must be between 1 and 12")
        if not 0 <= _integer(previous_limit, "previous_limit") <= 12:
            raise ValueError("previous_limit must be between 0 and 12")
        pending = [identifier for identifier in self._pending_order if identifier not in self._processed]
        self.clock_ms = cutoff
        if not pending:
            return None
        selected = pending[:max_items]
        rows = []
        for identifier in selected:
            item = deepcopy(self._observations[identifier]["observation"])
            item["time"] = format_time(item["observed_at_ms"])
            rows.append(item)
        previous = []
        prior_ids = list(self._notes)[-previous_limit:] if previous_limit else []
        for identifier in prior_ids:
            item = self._observations[identifier]["observation"]
            previous.append({"observation_id": identifier,
                             "time": format_time(item["observed_at_ms"]),
                             "description": item["description"],
                             "model_note": self._notes[identifier]})
        packet = {
            "node_id": self.node_id, "window_start_ms": self.start_ms, "cutoff_ms": cutoff,
            "observations": rows, "previous_findings": previous,
            "scope": {"case_id": self.case_id, "cutoff_time": format_time(cutoff),
                      "received_observations": len(self._observations),
                      "selected_for_this_call": len(selected),
                      "pending_outside_this_page": len(pending)-len(selected),
                      "previous_findings_scope": f"latest {previous_limit} annotated observations only"},
            "limitations": [
                "Only one input page; remaining observations are stored for later pages",
                "CPU observations remain reportable even when the model omits or misdescribes them",
                "Model notes are unverified wording, not replacements for measured facts",
                "No global topology or root-cause label is supplied",
            ],
        }
        self._request_number += 1
        request = {"request_id": f"{self.node_id}:update:{self._request_number}", "packet": packet}
        self._record({"type": "request_started", **request})
        # Rotate attempted items so a valid but empty/partial response cannot
        # keep all later pages out of the model forever.
        self._pending_order = pending[len(selected):] + selected
        self._in_flight = deepcopy(request)
        return deepcopy(request)

    def complete_update(self, request_id: str, response: dict) -> dict:
        """Accept selected annotations only. Omitted/new observations stay pending."""
        if not self.busy or request_id != self._in_flight["request_id"]:
            raise ValueError("response does not match the in-flight request")
        if not isinstance(response, dict) or set(response) != {"notes", "questions"}:
            raise ValueError("response must contain only notes and questions")
        notes, questions = response["notes"], response["questions"]
        if not isinstance(notes, list) or not isinstance(questions, list) or len(questions) > 3:
            raise ValueError("notes/questions must be lists; at most three questions")
        for question in questions:
            _text(question, "question", 160)
        allowed = {row["observation_id"] for row in self._in_flight["packet"]["observations"]}
        annotations = {}
        for note in notes:
            if not isinstance(note, dict) or set(note) != {"observation_id", "summary"}:
                raise ValueError("each note needs only observation_id and summary")
            identifier = _text(note["observation_id"], "note.observation_id")
            if identifier not in allowed or identifier in annotations:
                raise ValueError("note must reference a unique observation from this request")
            annotations[identifier] = _text(note["summary"], "summary", 160)
        cutoff = self._in_flight["packet"]["cutoff_ms"]
        self._record({"type": "model_annotations", "request_id": request_id,
                      "cutoff_ms": cutoff, "response": deepcopy(response),
                      "validation": "format and provenance only; semantics not verified"})
        for identifier, summary in annotations.items():
            self._notes[identifier] = summary
            self._processed.add(identifier)
            self._outbox.append({
                "type": "model_annotation", "case_id": self.case_id, "node_id": self.node_id,
                "finding_id": self._finding_id(identifier), "cutoff_ms": cutoff,
                "observation_id": identifier, "model_note": summary,
                "validation": "format and provenance only; semantics not verified",
            })
        new_questions = list(dict.fromkeys(question for question in questions if question not in self._questions))
        self._questions = list(dict.fromkeys(self._questions + questions))
        if new_questions:
            self._outbox.append({
                "type": "node_questions", "case_id": self.case_id, "node_id": self.node_id,
                "cutoff_ms": cutoff, "request_id": request_id,
                "questions": deepcopy(new_questions), "note": "Open questions, not confirmed facts",
            })
        self._pending_order = [identifier for identifier in self._pending_order
                               if identifier not in self._processed]
        self._in_flight = None
        return {"annotated_ids": list(annotations),
                "remaining": len(self._observations)-len(self._processed),
                "questions": deepcopy(questions)}

    def fail_update(self, request_id: str, reason: str):
        if not self.busy or request_id != self._in_flight["request_id"]:
            raise ValueError("failure does not match the in-flight request")
        self._record({"type": "request_failed", "request_id": request_id, "reason": str(reason)})
        self._in_flight = None

    def reports(self, after: int = 0) -> list[dict]:
        """Return an incremental outbox; its cursor does not delete local history."""
        if not 0 <= _integer(after, "after") <= len(self._outbox):
            raise ValueError("after must be a valid report cursor")
        return deepcopy(self._outbox[after:])

    def snapshot(self) -> dict:
        return {"case_id": self.case_id, "node_id": self.node_id,
                "window_start_ms": self.start_ms, "window_end_ms": self.end_ms,
                "clock_ms": self.clock_ms, "busy": self.busy,
                "observations": deepcopy(self._observations), "model_notes": deepcopy(self._notes),
                "pending_ids": [key for key in self._pending_order if key not in self._processed],
                "questions": deepcopy(self._questions), "report_count": len(self._outbox)}
