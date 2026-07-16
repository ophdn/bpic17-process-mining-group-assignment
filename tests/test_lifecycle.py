from __future__ import annotations

import unittest
from datetime import datetime

from simulation.components.lifecycle_params import LifecycleParameters
from simulation.components.permissions import StaticPermissions
from simulation.components.process import ProcessComponent
from simulation.components.resource import ResourceComponent
from simulation.core.engine import SimulationEngine


WORK = "W_Test"
NEXT = "A_Next"


class _FixedRandom:
    def __init__(self, value: float):
        self.value = value

    def random(self) -> float:
        return self.value


class _ScriptedProcess(ProcessComponent):
    """Small deterministic seam around the lifecycle orchestration.

    Production sampling remains covered by the extraction/training checks; these
    tests pin exact state-machine sequences and timing without depending on a
    particular scipy RNG implementation.
    """

    def __init__(self, *args, durations=(10.0,), gaps=None,
                 session_draws=None, suspend_draws=None,
                 withdraw_delay=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_durations = list(durations)
        self._test_gaps = dict(gaps or {})
        self._test_session_draws = dict(session_draws or {})
        self._test_suspend_draws = dict(suspend_draws or {})
        self._test_withdraw_delay = withdraw_delay

    def _active_session_len(self, engine, w, session):
        return self._test_durations[session]

    def _sample_gap(self, engine, case_id, activity, session):
        return self._test_gaps[session]

    def _sample_withdraw_delay(self, case_id, activity, visit):
        return self._test_withdraw_delay

    def _draw_rng(self, case_id, activity, kind, visit=1):
        if kind == "session_end":
            return _FixedRandom(self._test_session_draws.get(visit, 0.0))
        if kind == "susp_end":
            return _FixedRandom(self._test_suspend_draws.get(visit, 0.0))
        if kind.startswith("term_"):
            return _FixedRandom(0.0)
        return super()._draw_rng(case_id, activity, kind, visit)


class _CyclingPolicy:
    def __init__(self):
        self.calls = 0

    def select(self, activity, candidates, state):
        desired = "r1" if self.calls == 0 else "r2"
        self.calls += 1
        return desired if desired in candidates else candidates[0]


class _CountingResource(ResourceComponent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release_calls = []

    def release(self, engine, resource, activity=None):
        self.release_calls.append((resource, activity))
        super().release(engine, resource, activity)


def _params(*, continuation=None) -> LifecycleParameters:
    return LifecycleParameters(
        processing_times={},
        session_end_probs={WORK: 0.5},
        suspend_end_probs={WORK: 0.5},
        resume_gap_params={},
        withdraw_hazard={},
        terminal_continuation={
            WORK: continuation or {
                "complete": [("__CASE_END__", 1.0)],
                "ate_abort": [(NEXT, 1.0)],
                "withdraw": [(NEXT, 1.0)],
            }
        },
    )


def _run(*, duration=100.0, resources=("r1",), busy=False,
         policy=None, continuation=None, **process_script):
    engine = SimulationEngine(
        sim_duration=duration,
        start_datetime=datetime(2016, 1, 1),
        lifecycle_mode="active",
    )
    permissions = StaticPermissions({
        resource: {WORK, NEXT} for resource in resources
    })
    resource = _CountingResource(
        capacity_per_resource=1,
        seed=7,
        permissions=permissions,
        policy=policy,
    )
    if busy:
        resource._busy[resources[0]] = 1
    process = _ScriptedProcess(
        seed=7,
        lifecycle_mode="active",
        lifecycle_params=_params(continuation=continuation),
        resource_component=resource,
        **process_script,
    )
    process._repeat_counts["c1"] = {}
    process._ctx["c1"] = {
        "start_t": 0.0, "position": 0, "prev_act": None, "attrs": {}
    }
    engine.register(resource)
    engine.register(process)
    process._fire_start(engine, "c1", WORK)
    engine.run()
    return engine, resource, process


def _work_rows(engine):
    return [r for r in engine.logger._rows if r["concept:name"] == WORK]


class LifecycleStateMachineTests(unittest.TestCase):
    def test_complete_without_churn(self):
        engine, resource, _ = _run(
            durations=(10.0,), session_draws={0: 0.0})
        rows = _work_rows(engine)
        self.assertEqual(
            [r["lifecycle:transition"] for r in rows],
            ["schedule", "start", "complete"],
        )
        self.assertEqual(len({r["work_item_id"] for r in rows}), 1)
        self.assertEqual(engine.stats["cases_completed"], 1)
        self.assertEqual(resource.release_calls, [("r1", WORK)])
        self.assertEqual(resource._busy["r1"], 0)

    def test_multi_suspend_resume_exact_timing_and_active_sum(self):
        engine, resource, _ = _run(
            duration=100.0,
            durations=(10.0, 20.0, 30.0),
            gaps={1: 5.0, 2: 7.0},
            session_draws={0: 0.9, 1: 0.9, 2: 0.1},
            suspend_draws={1: 0.1, 2: 0.1},
        )
        rows = _work_rows(engine)
        self.assertEqual(
            [r["lifecycle:transition"] for r in rows],
            ["schedule", "start", "suspend", "resume", "suspend", "resume", "complete"],
        )
        seconds = [
            (datetime.fromisoformat(r["time:timestamp"]) - datetime(2016, 1, 1)).total_seconds()
            for r in rows
        ]
        self.assertEqual(seconds, [0.0, 0.0, 10.0, 15.0, 35.0, 42.0, 72.0])
        self.assertEqual((10.0 - 0.0) + (35.0 - 15.0) + (72.0 - 42.0), 60.0)
        self.assertEqual(resource._busy["r1"], 0)

    def test_abort_continues_case_without_double_release(self):
        engine, resource, _ = _run(
            duration=10.0,
            durations=(10.0,),
            gaps={},
            session_draws={0: 0.9},
            suspend_draws={1: 0.9},
        )
        rows = _work_rows(engine)
        self.assertEqual(
            [r["lifecycle:transition"] for r in rows],
            ["schedule", "start", "suspend", "ate_abort"],
        )
        self.assertTrue(any(
            row["concept:name"] == NEXT
            and row["lifecycle:transition"] == "start"
            for row in engine.logger._rows
        ))
        self.assertEqual(engine.stats["cases_completed"], 0)
        self.assertEqual(resource.release_calls, [("r1", WORK)])
        # The one busy slot now belongs to the routed A_Next work item.  Abort
        # itself caused no second release after suspend.
        self.assertEqual(resource._busy["r1"], 1)

    def test_withdraw_removes_queued_item_before_start(self):
        engine, resource, _ = _run(
            duration=5.0,
            busy=True,
            durations=(10.0,),
            withdraw_delay=3.0,
        )
        rows = _work_rows(engine)
        self.assertEqual(
            [r["lifecycle:transition"] for r in rows],
            ["schedule", "withdraw"],
        )
        self.assertFalse(any(w.activity == WORK for w in resource._waiting))
        self.assertEqual(engine.stats["cases_completed"], 0)

    def test_stale_withdrawal_timer_after_allocation_is_noop(self):
        engine, _, _ = _run(
            duration=4.0,
            durations=(10.0,),
            withdraw_delay=3.0,
        )
        self.assertEqual(
            [r["lifecycle:transition"] for r in _work_rows(engine)],
            ["schedule", "start"],
        )

    def test_suspend_releases_and_resume_reacquires_from_pool(self):
        policy = _CyclingPolicy()
        engine, resource, _ = _run(
            resources=("r1", "r2"),
            policy=policy,
            durations=(10.0, 10.0),
            gaps={1: 5.0},
            session_draws={0: 0.9, 1: 0.1},
            suspend_draws={1: 0.1},
        )
        rows = _work_rows(engine)
        start = next(r for r in rows if r["lifecycle:transition"] == "start")
        resume = next(r for r in rows if r["lifecycle:transition"] == "resume")
        self.assertEqual(start["org:resource"], "r1")
        self.assertEqual(resume["org:resource"], "r2")
        self.assertEqual(resource._busy, {"r1": 0, "r2": 0})


if __name__ == "__main__":
    unittest.main()
