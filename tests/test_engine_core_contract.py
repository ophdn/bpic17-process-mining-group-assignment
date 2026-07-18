from __future__ import annotations

from pathlib import Path

from simulation.core.engine import SimulationEngine
from simulation.core.events import EventType, SimEvent


class _Recorder:
    HANDLES = {
        EventType.CASE_ARRIVAL: None,
        EventType.ACTIVITY_COMPLETE: None,
    }

    def __init__(self):
        self.calls = []

    def on_case_arrival(self, engine, event):
        self.calls.append(("arrival", engine.now, event.case_id))

    def on_activity_complete(self, engine, event):
        self.calls.append(("complete", engine.now, event.activity))


_Recorder.HANDLES = {
    EventType.CASE_ARRIVAL: _Recorder.on_case_arrival,
    EventType.ACTIVITY_COMPLETE: _Recorder.on_activity_complete,
}


def test_engine_routes_events_from_one_global_priority_queue():
    engine = SimulationEngine(sim_duration=20)
    recorder = _Recorder()
    engine.register(recorder)

    engine.schedule(SimEvent(timestamp=8, priority=10,
                             event_type=EventType.CASE_ARRIVAL, case_id="c-late"))
    engine.schedule(SimEvent(timestamp=5, priority=10,
                             event_type=EventType.CASE_ARRIVAL, case_id="c-mid"))
    engine.schedule(SimEvent(timestamp=5, priority=1,
                             event_type=EventType.ACTIVITY_COMPLETE,
                             case_id="c-mid", activity="A_Complete"))

    engine.run()

    assert recorder.calls == [
        ("complete", 5, "A_Complete"),
        ("arrival", 5, "c-mid"),
        ("arrival", 8, "c-late"),
    ]
    assert engine.stats["cases_started"] == 2
    assert engine.stats["events_processed"] == 3


def test_logger_saves_csv_for_activity_events(tmp_path):
    engine = SimulationEngine(sim_duration=5)
    engine.schedule(SimEvent(timestamp=1, event_type=EventType.ACTIVITY_START,
                             case_id="c1", activity="A_Create Application",
                             resource="r1"))
    engine.schedule(SimEvent(timestamp=2, event_type=EventType.ACTIVITY_COMPLETE,
                             case_id="c1", activity="A_Create Application",
                             resource="r1"))

    engine.run()
    out_path = engine.logger.save(tmp_path / "event_log.csv")

    saved = Path(out_path).read_text(encoding="utf-8").splitlines()
    assert saved[0] == "case:concept:name,concept:name,time:timestamp,lifecycle:transition,org:resource"
    assert saved[1].startswith("c1,A_Create Application,")
    assert saved[1].endswith(",start,r1")
    assert saved[2].endswith(",complete,r1")
