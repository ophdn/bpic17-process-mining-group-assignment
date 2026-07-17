"""Every fitted engine component runs by default, and main.py's wiring and the
Part II runner's wiring agree.

Team decision 2026-07-17: a fitted, validated component that the default engine
does not use is not a safe default, it is a silently wrong one. Two bugs of
exactly that shape have already been found:

  p_work            fitted, serialised, validated in notebook 01, and never
                    read at runtime -- the deployed calendar fielded ~123 people
                    on a Monday where the validated model expects ~37.
  USE_MDN_ARRIVALS  the parametric arrival model is statistically REJECTED
                    against the real log (inter-arrival KS p = 3.18e-24) where
                    the MDN is not (p = 0.389), yet the parametric one was the
                    default -- and run_experiments.py never even read the
                    switch, so the Part II grid could not have picked the MDN up
                    however that flag was set.

Neither was a broken function, so no unit test of a function could fail on
them. These tests assert the *wiring*: that what we fitted is what we run.
"""

from __future__ import annotations

import inspect
import unittest

from simulation.components.arrival import ArrivalComponent
from simulation.components.arrival_mdn import MDNArrivalComponent
from simulation.components.resource import DEFAULT_ROSTER_SEED


class ArrivalModelDefaultTests(unittest.TestCase):
    def test_mdn_is_the_default(self):
        from simulation.main import USE_MDN_ARRIVALS

        self.assertTrue(
            USE_MDN_ARRIVALS,
            "the parametric arrival model is rejected at KS p=3.18e-24; see "
            "output/arrival_model_eval.md",
        )

    def test_the_grid_uses_the_same_arrival_model_as_main(self):
        """The bug: run_experiments hardcoded the parametric component.

        A switch in main.py that the experiment runner ignores is not a switch.
        """
        from scripts.run_experiments import build_arrival_component
        from simulation.main import USE_MDN_ARRIVALS

        expected = MDNArrivalComponent if USE_MDN_ARRIVALS else ArrivalComponent
        self.assertIsInstance(build_arrival_component(1, "normal"), expected)

    def test_peak_scenario_still_scales_under_the_mdn(self):
        """--scenario peak must not silently lose its +30% when MDN is on."""
        from scripts.run_experiments import build_arrival_component

        peak = build_arrival_component(1, "peak")
        self.assertEqual(getattr(peak, "_scale_factor", None), 1.3)


class RosterDefaultTests(unittest.TestCase):
    def test_rostering_is_on_by_default_in_main(self):
        from simulation.main import main as sim_main

        self.assertEqual(
            inspect.signature(sim_main).parameters["roster_seed"].default,
            DEFAULT_ROSTER_SEED,
        )

    def test_rostering_is_on_by_default_in_the_grid(self):
        from scripts.run_experiments import run_once

        self.assertEqual(
            inspect.signature(run_once).parameters["roster_seed"].default,
            DEFAULT_ROSTER_SEED,
        )

    def test_rostering_can_still_be_switched_off(self):
        """Old evidence must stay reproducible via an explicit opt-out."""
        from analysis.availability import YearlyAvailability, WeeklyAvailability
        import datetime as dt

        cal = YearlyAvailability(
            weekly=WeeklyAvailability(
                windows={"r1": {0: (9.0, 17.0)}}, p_work={"r1": {0: 0.0}},
            ),
            holidays=set(), vacations={}, roster_seed=None,
        )
        self.assertTrue(cal.is_available("r1", dt.datetime(2016, 6, 6, 10, 0)),
                        "roster_seed=None must restore pre-rostering behaviour")


if __name__ == "__main__":
    unittest.main()
