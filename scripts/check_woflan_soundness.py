"""
check_woflan_soundness.py
==========================
Section 1.4 Advanced: verifies the manually modeled (Signavio) BPMN, once
converted to a Petri net, is a sound workflow net (live, bounded, no dead
transitions) via pm4py's Woflan implementation — the check the lecture
recommends to rule out structural design errors before trusting a net's
loop/branching behavior as real process behavior rather than a modeling
defect (see process_model.tex, and the loop analysis in
branching_decisions.tex, Section~\ref{sec:branching}).

Usage (from the repo root):
    python scripts/check_woflan_soundness.py
    python scripts/check_woflan_soundness.py --bpmn simulation/models/bpic17_process_im02.bpmn
"""

import argparse
import json
import sys
from pathlib import Path

import pm4py
from pm4py.algo.analysis.woflan import algorithm as woflan

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BPMN = REPO_ROOT / "simulation" / "models" / "bpic17_process.bpmn"
DEFAULT_OUT = REPO_ROOT / "output" / "validation" / "woflan_soundness.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpmn", type=Path, default=DEFAULT_BPMN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    model = pm4py.read_bpmn(str(args.bpmn))
    net, im, fm = pm4py.convert_to_petri_net(model)
    print(f"[woflan] Net: {len(net.places)} places, {len(net.transitions)} "
          f"transitions (from {args.bpmn.name})")

    is_sound, diagnostics = woflan.apply(
        net, im, fm,
        parameters={
            woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: False,
            woflan.Parameters.PRINT_DIAGNOSTICS: True,
            woflan.Parameters.RETURN_DIAGNOSTICS: True,
        },
    )

    out = {
        "bpmn": args.bpmn.name,
        "net_size": f"{len(net.places)}p/{len(net.transitions)}t",
        "is_sound": bool(is_sound),
        "dead_tasks": [str(t) for t in diagnostics.get("dead_tasks", [])],
        # Every other diagnostics key is a plain string (its enum .value),
        # but pm4py keys this one specifically by the Outputs enum member
        # itself rather than by 'diagnostic_messages' — verified directly
        # against the installed pm4py version, not assumed from docs.
        "diagnostic_messages": diagnostics.get(woflan.Outputs.DIAGNOSTIC_MESSAGES, []),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print()
    print(f"[woflan] is_sound: {is_sound}")
    print(f"[woflan] dead_tasks: {out['dead_tasks']}")
    print(f"[save] -> {args.output}")


if __name__ == "__main__":
    main()
