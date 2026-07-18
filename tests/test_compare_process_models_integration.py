from __future__ import annotations

import pm4py

from scripts import compare_process_models as cpm
from scripts import metrics


def test_advanced_probs_replays_on_the_reference_petri_net_and_completes_cases():
    reference = metrics.load_reference(cpm.LEGACY_REFERENCE_PATH)
    bpmn_model = pm4py.read_bpmn(str(cpm.BPMN_PATH))
    net, im, fm = pm4py.convert_to_petri_net(bpmn_model)

    df, df_all, stats = cpm.run_sim(
        True,
        bpmn_path=cpm.BPMN_PATH,
        branching_mode="probs",
        lifecycle_mode="legacy",
    )
    result = metrics.evaluate(df, reference, net, im, fm, df_all=df_all)

    # enforce_terminal_outcomes (default True, docs/ROADMAP.md A1) force-ends
    # a case on A_Pending/A_Denied/A_Cancelled, which is a domain rule rather
    # than a Petri-legal move -- so not every trace replays cleanly to the
    # net's own final marking (percentage_of_fitting_traces < 100%), but
    # average/log fitness stays high. That's an accepted, documented
    # trade-off, not a regression.
    assert result["control_flow"]["fitness"]["average_trace_fitness"] >= 0.9
    assert stats["completion_rate"] > 0.01
    assert stats["petri_debug"]["allow_end_opportunities"] > 0
    end_reasons = stats["petri_debug"]["end_reasons"]
    # Every completed case is attributed to exactly one end reason.
    assert sum(end_reasons.values()) == stats["cases_completed"]