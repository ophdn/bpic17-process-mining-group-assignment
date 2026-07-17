"""Section 1.6: the p_work roster draw.

Covers the verification plan in docs/PLAN_pwork_roster.md.

The bug these tests exist for was not a broken function -- `is_available` did
exactly what it said, checking holiday, vacation and window, and any honest
unit test of it passed. The bug was that a *fitted* parameter (`p_work`) was
never wired into the model that *runs*, so the calendar we validated in
notebook 01 and the calendar Part II simulated under were different models,
and nothing failed. A unit test cannot catch a missing connection; only a
consistency check between the two models can, which is what
`RosterMatchesValidatedModelTests` below is. It is the regression test for the
whole class of "the thing we measured is not the thing we ran".
"""

from __future__ import annotations

import datetime as dt
import subprocess
import sys
import unittest
from pathlib import Path

from analysis.availability import WeeklyAvailability, YearlyAvailability

MODEL_PATH = Path("models/availability_model.json")
MONDAY = dt.datetime(2016, 6, 6, 10, 0)      # the plan's reference instant


def _calendar(p: float, roster_seed=42, resources=("r1",)) -> YearlyAvailability:
    """A calendar of `resources`, each open 09-17 every weekday with P(work)=p."""
    return YearlyAvailability(
        weekly=WeeklyAvailability(
            windows={r: {d: (9.0, 17.0) for d in range(5)} for r in resources},
            p_work={r: {d: p for d in range(5)} for r in resources},
        ),
        holidays=set(),
        vacations={},
        roster_seed=roster_seed,
    )


class WorksTodayTests(unittest.TestCase):
    """Verification 1: the draw's contract."""

    def test_stable_within_a_day(self):
        """The whole reason this is a hash and not an RNG stream.

        is_available() is called once per allocation attempt, i.e. many times
        per simulated day. A fresh draw per call would flicker a resource on
        and off mid-day -- worse than not modelling p_work at all.
        """
        cal = _calendar(0.5)
        d = dt.date(2016, 6, 6)
        answers = {cal._works_today("r1", d) for _ in range(100)}
        self.assertEqual(len(answers), 1)

    def test_varies_across_days(self):
        cal = _calendar(0.5)
        days = [dt.date(2016, 6, 6) + dt.timedelta(days=i) for i in range(60)]
        drawn = {cal._works_today("r1", d) for d in days}
        self.assertEqual(drawn, {True, False}, "draw is constant across dates")

    def test_independent_across_resources(self):
        cal = _calendar(0.5, resources=tuple(f"r{i}" for i in range(40)))
        d = dt.date(2016, 6, 6)
        drawn = {cal._works_today(r, d) for r in cal.weekly.resources()}
        self.assertEqual(drawn, {True, False}, "all resources drew alike")

    def test_certain_probabilities_short_circuit(self):
        d = dt.date(2016, 6, 6)
        self.assertFalse(_calendar(0.0)._works_today("r1", d))
        self.assertTrue(_calendar(1.0)._works_today("r1", d))

    def test_unknown_resource_is_never_rostered(self):
        self.assertFalse(_calendar(0.5)._works_today("nobody", dt.date(2016, 6, 6)))

    def test_seed_changes_the_draw(self):
        d = dt.date(2016, 6, 6)
        rs = [f"r{i}" for i in range(60)]
        a = {r: _calendar(0.5, roster_seed=1, resources=tuple(rs))._works_today(r, d)
             for r in rs}
        b = {r: _calendar(0.5, roster_seed=2, resources=tuple(rs))._works_today(r, d)
             for r in rs}
        self.assertNotEqual(a, b)

    def test_reproducible_across_processes(self):
        """A hash, unlike an RNG stream, survives process boundaries."""
        code = (
            "import sys; sys.path.insert(0, '.');"
            "import datetime as dt;"
            "from tests.test_roster import _calendar;"
            "c = _calendar(0.5, resources=tuple(f'r{i}' for i in range(20)));"
            "print(''.join('1' if c._works_today(r, dt.date(2016, 6, 6)) else '0'"
            " for r in c.weekly.resources()))"
        )
        runs = {
            subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True).stdout.strip()
            for _ in range(2)
        }
        self.assertEqual(len(runs), 1, f"draw differed across processes: {runs}")
        self.assertIn("1", runs.pop(), "subprocess produced no draws at all")

    def test_rostering_off_by_default(self):
        """Backward compatibility: old calendars keep the pre-1.6 behaviour."""
        cal = _calendar(0.0, roster_seed=None)
        self.assertTrue(cal._works_today("r1", dt.date(2016, 6, 6)))
        self.assertTrue(cal.is_available("r1", MONDAY),
                        "roster_seed=None must not gate availability")

    def test_converges_to_p_work(self):
        """Verification 2: the draw actually samples the fitted probability.

        Weekdays only: p_work is per-weekday, and the fixture leaves Sat/Sun
        undefined, which correctly draws 0.0 and would drag the share to 5/7 of
        p if counted.
        """
        for p in (0.25, 0.5, 0.75):
            cal = _calendar(p)
            days = [d for d in (dt.date(2016, 1, 4) + dt.timedelta(days=i)
                                for i in range(2800))
                    if d.weekday() < 5]
            share = sum(cal._works_today("r1", d) for d in days) / len(days)
            self.assertAlmostEqual(share, p, delta=0.03, msg=f"p_work={p}")

    def test_weekend_is_never_rostered(self):
        """The 5/7 effect above, pinned deliberately."""
        cal = _calendar(1.0)
        saturday = dt.date(2016, 6, 11)
        self.assertEqual(saturday.weekday(), 5)
        self.assertFalse(cal._works_today("r1", saturday),
                         "a weekday-only p_work must not roster anyone on Saturday")


class RosterGatesAvailabilityTests(unittest.TestCase):
    def test_roster_off_day_is_unavailable_even_inside_the_window(self):
        cal = _calendar(0.0)
        self.assertTrue(cal.weekly.is_open("r1", MONDAY.weekday(), 10.0))
        self.assertFalse(cal.is_available("r1", MONDAY),
                         "p_work=0 resource available inside its window")

    def test_roster_on_day_still_respects_the_window(self):
        cal = _calendar(1.0)
        self.assertTrue(cal.is_available("r1", MONDAY))
        self.assertFalse(cal.is_available("r1", MONDAY.replace(hour=3)),
                         "rostered on must not mean available around the clock")


@unittest.skipUnless(MODEL_PATH.exists(), f"{MODEL_PATH} not fitted")
class RosterMatchesValidatedModelTests(unittest.TestCase):
    """Verification 3 and 4: the runtime and the validated model agree.

    This is the test that would have caught the original bug, and the reason it
    is worth more than any unit test of `is_available`: it compares the two
    models rather than checking one of them against itself.
    """

    def test_headline_headcount(self):
        """Verification 3: Mon 10:00 puts ~37 on shift, not 123."""
        cal = YearlyAvailability.from_json(MODEL_PATH, roster_seed=42)
        n = sum(cal.is_available(r, MONDAY) for r in cal.weekly.resources())
        self.assertTrue(30 <= n <= 45, f"expected ~37 on shift, got {n}")

    def test_without_rostering_the_old_overstaffing_is_reproduced(self):
        """Pins the bug itself, so the fix cannot silently regress."""
        cal = YearlyAvailability.from_json(MODEL_PATH)
        n = sum(cal.is_available(r, MONDAY) for r in cal.weekly.resources())
        self.assertGreater(n, 100, f"expected the ~123 pre-roster pool, got {n}")

    def test_weekday_headcount_matches_notebook_01(self):
        """Verification 4: the point of the whole exercise.

        notebook 01 cell 30 rolls p_work directly and reports a modelled
        weekday headcount of mean ~37.3 against a real 40.2. The runtime must
        reproduce that, because it is now the same model.
        """
        cal = YearlyAvailability.from_json(MODEL_PATH, roster_seed=42)
        resources = cal.weekly.resources()
        counts = []
        d = dt.date(2016, 1, 4)
        while d < dt.date(2016, 12, 31):
            if d.weekday() < 5 and d not in cal.holidays:
                at_10 = dt.datetime.combine(d, dt.time(10, 0))
                counts.append(sum(cal.is_available(r, at_10) for r in resources))
            d += dt.timedelta(days=1)

        mean = sum(counts) / len(counts)
        self.assertTrue(33.0 <= mean <= 42.0,
                        f"modelled weekday headcount {mean:.1f} outside the "
                        f"notebook-01 range (~37.3 modelled vs 40.2 real)")


if __name__ == "__main__":
    unittest.main()
