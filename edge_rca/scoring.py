"""OpenRCA object-matching scorer with a configurable occurrence-time window.

The upstream scorer hardcodes a 60-second time tolerance.  This project keeps
the same component/reason/permutation logic but uses a 5-minute window by
default; callers can pass a smaller value for parity checks.
"""

from datetime import datetime
import itertools
import re


PREDICTION = re.compile(
    r'{\s*'
    r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
    r'(?:"root cause component":\s*"(.*?)")?,?\s*'
    r'(?:"root cause reason":\s*"(.*?)")?\s*}')
COMPONENT = re.compile(r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)")
REASON = re.compile(r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)")
TIME = re.compile(r"The (?:\d+-th|only) root cause occurrence time is within \d+ minutes \(i\.e\., <=\d+min\) of ([^\n]+)")


def evaluate(prediction: str, scoring_points: str, *, time_tolerance_seconds=300):
    matches = PREDICTION.findall(prediction)
    predictions = [{"root cause occurrence datetime": when, "root cause component": component,
                    "root cause reason": reason} for when, component, reason in matches]
    components = COMPONENT.findall(scoring_points)
    reasons = REASON.findall(scoring_points)
    times = TIME.findall(scoring_points)
    length = max(len(components), len(reasons), len(times))
    total = len(components) + len(reasons) + len(times)
    passing, score = [], 0
    if length == len(predictions):
        best = -1
        for permutation in itertools.permutations(predictions):
            current, accepted = 0, []
            for index in range(length):
                if len(components) == length and permutation[index]["root cause component"] == components[index]:
                    current += 1
                    accepted.append(components[index])
                if len(reasons) == length and permutation[index]["root cause reason"] == reasons[index]:
                    current += 1
                    accepted.append(reasons[index])
                if len(times) == length and _time_close(times[index],
                                                        permutation[index]["root cause occurrence datetime"],
                                                        time_tolerance_seconds):
                    current += 1
                    accepted.append(times[index])
            if current > best:
                best, passing = current, accepted
        score = best
    failed = list(set(components + reasons + times) - set(passing))
    return passing, failed, round(score / total, 2) if total else 0.0


def _time_close(left, right, tolerance):
    try:
        first = datetime.strptime(left, "%Y-%m-%d %H:%M:%S")
        second = datetime.strptime(right, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return False
    return abs((first - second).total_seconds()) <= tolerance
