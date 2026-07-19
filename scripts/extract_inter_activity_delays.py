"""Stream BPIC-17 and refresh active inter-activity waiting-time parameters.

The general extractor uses PM4Py because it mines every input table. For this
single additive lifecycle block, materializing the 552 MB XES as a DataFrame is
unnecessary and memory-heavy. This script keeps one trace in memory, reconstructs
the same visible occurrence boundaries as ``extract_log_info.segment_work_items``,
and updates only ``lifecycle.inter_activity_delays`` in the existing artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import iterparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from extract_log_info import (  # noqa: E402
    CASE_DURATION_ENVELOPE_RUNTIME_SCALE,
    _fit_delay_distribution,
)


TERMINALS = {"complete", "ate_abort", "withdraw"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attrs(element) -> dict[str, str]:
    result = {}
    for child in element:
        if _local(child.tag) in {"string", "date", "int", "float", "boolean"}:
            key = child.attrib.get("key")
            if key is not None:
                result[key] = child.attrib.get("value")
    return result


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _occurrences(trace) -> list[tuple[int, datetime, datetime, str]]:
    events = []
    for source_order, child in enumerate(trace):
        if _local(child.tag) != "event":
            continue
        attrs = _attrs(child)
        activity = attrs.get("concept:name")
        timestamp = attrs.get("time:timestamp")
        lifecycle = attrs.get("lifecycle:transition", "complete").lower()
        if activity and timestamp:
            events.append((
                _timestamp(timestamp), source_order, activity, lifecycle,
            ))
    events.sort(key=lambda item: (item[0], item[1]))
    ordered = [(*event, order) for order, event in enumerate(events)]

    tokens = []
    for timestamp, _source_order, activity, lifecycle, order in ordered:
        if not activity.startswith("W_") and lifecycle == "complete":
            tokens.append((order, timestamp, timestamp, activity))

    by_activity = defaultdict(list)
    for event in ordered:
        if event[2].startswith("W_"):
            by_activity[event[2]].append(event)
    for activity, activity_events in by_activity.items():
        current = None
        for timestamp, _source_order, _activity, lifecycle, order in activity_events:
            if lifecycle == "schedule":
                if current is None:
                    current = {"ready": timestamp}
            elif lifecycle == "start":
                if current is None:
                    current = {"ready": timestamp}
            elif lifecycle in TERMINALS:
                if current is None:
                    current = {"ready": timestamp}
                tokens.append((order, current["ready"], timestamp, activity))
                current = None

    tokens.sort(key=lambda item: item[0])
    return tokens


def extract(path: Path) -> dict:
    by_transition = defaultdict(list)
    all_delays = []
    case_durations = []
    traces = 0
    for _event, element in iterparse(path, events=("end",)):
        if _local(element.tag) != "trace":
            continue
        tokens = _occurrences(element)
        if tokens:
            case_durations.append(max(
                0.0, (max(token[2] for token in tokens)
                      - min(token[1] for token in tokens)).total_seconds()
            ))
        for current, nxt in zip(tokens, tokens[1:]):
            _order, _ready, current_end, current_activity = current
            _next_order, next_ready, _next_end, next_activity = nxt
            delay = max(0.0, (next_ready - current_end).total_seconds())
            by_transition[(current_activity, next_activity)].append(delay)
            all_delays.append(delay)
        traces += 1
        element.clear()

    transitions = {}
    for (current, nxt), values in sorted(by_transition.items()):
        transitions.setdefault(current, {})[nxt] = _fit_delay_distribution(values)
    return {
        "basis": "current terminal -> next schedule/start/milestone",
        "source_traces": traces,
        "fallback": _fit_delay_distribution(all_delays),
        "transitions": transitions,
    }, {
        **_fit_delay_distribution(case_durations),
        "runtime_scale": CASE_DURATION_ENVELOPE_RUNTIME_SCALE,
        "basis": (
            "first-to-last observed event; lower bound for omitted parallel "
            "business waits"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--artifact", type=Path,
        default=REPO_ROOT / "simulation_inputs_active.json",
    )
    args = parser.parse_args()

    fitted, case_envelope = extract(args.log)
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    lifecycle = artifact.setdefault("lifecycle", {})
    lifecycle["inter_activity_delays"] = fitted
    lifecycle["case_duration_envelope"] = case_envelope
    args.artifact.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"updated {args.artifact}: {fitted['source_traces']} traces, "
        f"{sum(len(v) for v in fitted['transitions'].values())} transitions"
    )


if __name__ == "__main__":
    main()
