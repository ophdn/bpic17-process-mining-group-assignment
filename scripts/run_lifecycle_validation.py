"""Regenerate the report-facing active-lifecycle sampler comparison.

The three modes use common random numbers and an identical resource roster.
Artifacts include the experiment runner's resolved configuration and a complete
code/input fingerprint, so ``notebooks/03_process_times.ipynb`` can reject stale
or incomparable evidence before producing report values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_lifecycle import (
    REPORT_LIFECYCLE_CONFIGURATION,
    evaluate_dataframe,
    validate_lifecycle_validation_artifact,
)
from scripts.run_experiments import ACTIVE_INPUTS_PATH, DEFAULT_ROSTER_SEED, run_once


MODES = ("distribution", "ml_model", "ml_probabilistic")
OUTPUT_DIR = REPO_ROOT / "output/validation/lifecycle_active"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected = REPORT_LIFECYCLE_CONFIGURATION

    for mode in MODES:
        df, meta = run_once(
            expected["policy"],
            expected["seed"],
            expected["horizon_days"],
            expected["scenario"],
            expected["crn"],
            expected["process_model"],
            expected["branching_mode"],
            lifecycle_mode=expected["lifecycle_mode"],
            processing_time_mode=mode,
            permissions=expected["permissions"],
            roster_seed=DEFAULT_ROSTER_SEED,
            capacity=expected["capacity"],
            atomic_duration_scale=expected["atomic_duration_scale"],
            drain_days=expected["drain_days"],
        )
        result = evaluate_dataframe(
            df,
            set(meta["completed_case_ids"]),
            ACTIVE_INPUTS_PATH,
            meta["configuration"],
            case_duration_seconds={
                case_id: (
                    meta["completion_times"][case_id]
                    - meta["arrival_times"][case_id]
                ).total_seconds()
                for case_id in meta["completed_case_ids"]
            },
        )
        result["configuration"]["label"] = mode
        validate_lifecycle_validation_artifact(result, mode)

        output_path = OUTPUT_DIR / f"{mode}.json"
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            f"[lifecycle] {mode}: {result['configuration']['completed_cases']} "
            f"completed cases -> {output_path}"
        )


if __name__ == "__main__":
    main()
