"""Read public question constraints; never use scoring labels for inference."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
import re


MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)
MONTH_NUMBERS = {name.lower(): index for index, name in enumerate(MONTH_NAMES, 1)}
DATE = re.compile(r"\b(" + "|".join(MONTH_NAMES) + r")\s+(\d{1,2}),\s+(\d{4})\b", re.I)
TIME = re.compile(r"\b(?:[01]\d|2[0-3]):[0-5]\d\b")
FAILURE_COUNT = re.compile(
    r"\b(single|one|two|[1-9]\d*|a)\s+"
    r"(?:(?:system|reported|distinct|documented|identified|recorded)\s+)*failures?\b", re.I)


def parse_failure_count(instruction: str) -> int | None:
    """Use only explicit incident counts in the public question.

    An unspecified count is unknown, not an implicit single-fault task.
    The independent data replay still needs only parse_question_window.
    """
    words = {"single": 1, "one": 1, "a": 1, "two": 2}
    counts = {words[token.lower()] if token.lower() in words else int(token)
              for token in FAILURE_COUNT.findall(instruction)}
    if len(counts) > 1:
        raise ValueError("public question contains conflicting failure counts")
    return next(iter(counts)) if counts else None


def parse_question_window(instruction: str) -> tuple[str, str]:
    dates = DATE.findall(instruction)
    times = TIME.findall(instruction)
    if not dates or len(times) < 2:
        raise ValueError("Question needs an explicit date and start/end times")

    def dated_time(date, clock):
        month, day, year = date
        hour, minute = map(int, clock.split(":"))
        return datetime(int(year), MONTH_NUMBERS[month.lower()], int(day), hour, minute)

    start = dated_time(dates[0], times[0])
    end = dated_time(dates[1] if len(dates) > 1 else dates[0], times[1])
    if end < start and len(dates) == 1:
        end += timedelta(days=1)
    if end <= start:
        raise ValueError("Question has an empty or reversed time window")
    return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")


def read_query_window(path: Path, row_index: int) -> tuple[str, str, dict]:
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise ValueError("row_index must be a nonnegative integer")
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "instruction" not in (reader.fieldnames or []):
            raise ValueError("Query CSV needs an instruction column")
        for index, row in enumerate(reader):
            if index == row_index:
                start, end = parse_question_window(row["instruction"])
                return start, end, {"query_file": str(Path(path).resolve()), "row_index": index}
    raise ValueError(f"Query row {row_index} not found in {path}")
