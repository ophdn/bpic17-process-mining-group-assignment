from __future__ import annotations

import unittest
from datetime import datetime

from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from simulation.components.lifecycle_params import LifecycleParameters
from simulation.components.permissions import StaticPermissions
from simulation.components.petri_process import PetriNetProcessComponent
from simulation.components.process import ProcessComponent
from simulation.components.resource import ResourceComponent
from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType, SimEvent


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


class _FirstPolicy:
    def select(self, activity, candidates, state):
        return candidates[0]


class _LastPolicy:
    def select(self, activity, candidates, state):
        return candidates[-1]


class _ScriptedPetri(PetriNetProcessComponent, _ScriptedProcess):
    """Petri path backed by the smallest legal W_Test → A_Next net."""

    def __init__(self, *args, **kwargs):
        # Avoid BPMN I/O while retaining PetriNetProcessComponent's terminal
        # handlers. The sampling seam comes from _ScriptedProcess.
        _ScriptedProcess.__init__(self, *args, **kwargs)
        self.net = PetriNet("lifecycle-parity")
        p0, p1, p2 = (PetriNet.Place(name) for name in ("p0", "p1", "p2"))
        tw = PetriNet.Transition("tw", WORK)
        tn = PetriNet.Transition("tn", NEXT)
        self.net.places.update({p0, p1, p2})
        self.net.transitions.update({tw, tn})
        petri_utils.add_arc_from_to(p0, tw, self.net)
        petri_utils.add_arc_from_to(tw, p1, self.net)
        petri_utils.add_arc_from_to(p1, tn, self.net)
        petri_utils.add_arc_from_to(tn, p2, self.net)
        self.im = Marking({p0: 1})
        self.fm = Marking({p2: 1})
        self._markings = {}
        self._fm_reach_cache = {}
        self.branching_mode = "probs"
        self.enforce_terminal_outcomes = False
        self._branching_by_visit = {}
        self._dp_probs = {}
        self._dp_visit_counts = {}
        self._decision_rules_path = None
        self._decision_models = None
        self._decision_encoders = None
        self._decision_feature_names = None
        self._decision_unknown = "__UNKNOWN__"
        self._case_attrs = {}
        self._advance_reasons = {}
        self._debug = {
            "allow_end_opportunities": 0,
            "allow_end_without_dp": 0,
            "end_label_choices": 0,
            "end_reasons": {
                "final_marking": 0,
                "end_label": 0,
                "terminal_outcome": 0,
                "loop_guard": 0,
                "dead_marking": 0,
                "terminal_continuation_end": 0,
                "terminal_allow_end_fallback": 0,
            },
        }


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


def _run_crn(policy):
    engine = SimulationEngine(
        sim_duration=1_000.0,
        start_datetime=datetime(2016, 1, 1),
        lifecycle_mode="active",
    )
    permissions = StaticPermissions({"r1": {WORK}, "r2": {WORK}})
    resource = ResourceComponent(
        capacity_per_resource=1, seed=99, permissions=permissions, policy=policy)
    params = LifecycleParameters(
        processing_times={WORK: ("expon", (0.0, 10.0))},
        # Seed 123 yields four suspend/resume cycles before session 5
        # completes, so this exercises every CRN draw kind rather than only a
        # single no-churn duration.
        session_end_probs={WORK: 0.17},
        suspend_end_probs={WORK: 1.0},
        resume_gap_params={WORK: ("expon", (0.0, 5.0))},
        terminal_continuation={
            WORK: {"complete": [("__CASE_END__", 1.0)]}
        },
    )
    process = ProcessComponent(
        seed=123,
        crn=True,
        lifecycle_mode="active",
        lifecycle_params=params,
        resource_component=resource,
    )
    process._repeat_counts["crn"] = {}
    process._ctx["crn"] = {
        "start_t": 0.0, "position": 0, "prev_act": None, "attrs": {}}
    engine.register(resource)
    engine.register(process)
    process._fire_start(engine, "crn", WORK)
    engine.run()
    return _work_rows(engine)


class LifecycleStateMachineTests(unittest.TestCase):
    def test_active_crn_session_timing_is_policy_independent(self):
        first = _run_crn(_FirstPolicy())
        repeated = _run_crn(_FirstPolicy())
        last = _run_crn(_LastPolicy())
        self.assertEqual(first, repeated)
        self.assertGreaterEqual(
            sum(row["lifecycle:transition"] == "suspend" for row in first), 2)
        # Allocation may change org:resource, but it must not perturb any
        # case-keyed lifecycle draw (session length/end or resume gap).
        structure = lambda rows: [
            (row["time:timestamp"], row["lifecycle:transition"], row["work_item_id"])
            for row in rows
        ]
        self.assertEqual(structure(first), structure(last))

    def test_basic_and_petri_paths_share_lifecycle_structure(self):
        common = dict(
            duration=100.0,
            durations=(10.0, 20.0),
            gaps={1: 5.0},
            session_draws={0: 0.9, 1: 0.1},
            suspend_draws={1: 0.1},
        )
        basic_engine, _, _ = _run(**common)

        petri_engine = SimulationEngine(
            sim_duration=100.0,
            start_datetime=datetime(2016, 1, 1),
            lifecycle_mode="active",
        )
        resource = _CountingResource(
            capacity_per_resource=1,
            seed=7,
            permissions=StaticPermissions({"r1": {WORK, NEXT}}),
        )
        process = _ScriptedPetri(
            seed=7,
            lifecycle_mode="active",
            lifecycle_params=_params(),
            resource_component=resource,
            durations=common["durations"],
            gaps=common["gaps"],
            session_draws=common["session_draws"],
            suspend_draws=common["suspend_draws"],
        )
        process._repeat_counts["c1"] = {}
        process._ctx["c1"] = {
            "start_t": 0.0, "position": 0, "prev_act": None, "attrs": {}}
        process._markings["c1"] = Marking(process.im)
        petri_engine.register(resource)
        petri_engine.register(process)
        process._fire_start(petri_engine, "c1", WORK)
        petri_engine.run()

        structure = lambda engine: [
            (row["time:timestamp"], row["lifecycle:transition"])
            for row in _work_rows(engine)
        ]
        self.assertEqual(structure(basic_engine), structure(petri_engine))
        process._assert_marking_legal("c1")

    def test_piled_preference_skips_resume_ready_request(self):
        engine = SimulationEngine(
            sim_duration=1.0,
            start_datetime=datetime(2016, 1, 1),
            lifecycle_mode="active",
        )
        resource = ResourceComponent(
            capacity_per_resource=1,
            seed=7,
            permissions=StaticPermissions({"r1": {WORK, NEXT}}),
            piled=True,
        )
        engine.register(resource)
        resource._busy["r1"] = 1
        fresh = SimEvent(
            timestamp=0.0,
            event_type=EventType.ACTIVITY_REQUEST,
            case_id="fresh",
            activity=NEXT,
            payload={"work_item_id": "fresh:1"},
        )
        resume = SimEvent(
            timestamp=0.0,
            event_type=EventType.ACTIVITY_REQUEST,
            case_id="resume",
            activity=WORK,
            payload={"work_item_id": "resume:1", "resuming": True},
        )
        # FIFO says NEXT. Piled execution would jump to the later W_Test item
        # solely because it matches the completed activity unless resumes are
        # explicitly excluded from that preference.
        resource._waiting.extend([fresh, resume])
        resource.on_resource_available(engine, SimEvent(
            timestamp=0.0,
            event_type=EventType.RESOURCE_AVAILABLE,
            resource="r1",
            payload=WORK,
        ))
        engine.run()

        starts = [
            row for row in engine.logger._rows
            if row["lifecycle:transition"] in {"start", "resume"}
        ]
        self.assertEqual(
            [(row["case:concept:name"], row["lifecycle:transition"])
             for row in starts],
            [("fresh", "start")],
        )
        self.assertEqual(resource._waiting, [resume])

    def test_complete_without_churn(self):
        engine, resource, process = _run(
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
        self.assertNotIn("c1", process._witem_seq)
        self.assertFalse(process._witem)

    def test_atomic_duration_scale_zero_makes_active_ao_transition_instantaneous(self):
        engine = SimulationEngine(
            sim_duration=10.0,
            start_datetime=datetime(2016, 1, 1),
            lifecycle_mode="active",
        )
        process = ProcessComponent(
            seed=7,
            lifecycle_mode="active",
            lifecycle_params=_params(),
            atomic_duration_scale=0.0,
        )
        process._repeat_counts["c1"] = {}
        process._ctx["c1"] = {
            "start_t": 0.0, "position": 0, "prev_act": None, "attrs": {}
        }
        process._sample_duration = lambda activity, rng: 120.0

        process.on_activity_start(engine, SimEvent(
            timestamp=0.0,
            event_type=EventType.ACTIVITY_START,
            case_id="c1",
            activity=NEXT,
            resource="r1",
            payload={"work_item_id": "c1:1"},
        ))

        self.assertEqual(len(engine._queue), 1)
        self.assertEqual(engine._queue[0].timestamp, 0.0)

    def test_atomic_duration_scale_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, "atomic_duration_scale"):
            ProcessComponent(
                lifecycle_mode="active",
                lifecycle_params=_params(),
                atomic_duration_scale=-0.1,
            )

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

    def test_max_session_guard_records_forced_completion(self):
        session_count = 60
        engine, _, _ = _run(
            duration=200.0,
            durations=(1.0,) * session_count,
            gaps={session: 1.0 for session in range(1, session_count)},
            session_draws={session: 0.9 for session in range(session_count)},
            suspend_draws={session: 0.1 for session in range(1, session_count)},
        )

        rows = _work_rows(engine)
        opened_sessions = sum(
            row["lifecycle:transition"] in {"start", "resume"} for row in rows
        )
        self.assertEqual(opened_sessions, session_count)
        self.assertEqual(engine.stats["max_session_guard_reached"], 1)
        self.assertEqual(engine.stats["max_session_guard_forced_completions"], 1)
        self.assertEqual(engine.stats["max_session_guard_by_activity"], {WORK: 1})
        self.assertEqual(engine.stats["cases_completed"], 1)

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
