"""
case_attributes.py — sample case-level attributes when a case is created.

Section 1.7 Advanced needs this: an OrdinoR execution context is a triple
(case type, activity type, time type), so enforcing the case-type dimension at
runtime requires each simulated case to *have* a case type. Real BPIC-17 cases
carry a loan goal; simulated ones must too, or the model's case dimension can
never bind and we would be discovering structure we then throw away.

It also serves Section 1.5 Advanced I ("at spawn rate sample case attributes"),
which is why this is a standalone component rather than something buried in the
permission code.

Design decision — sample at case initialisation, not in the arrival component
----------------------------------------------------------------------------
The attribute is logically a property of the arriving case, so the arrival
component looks like its natural home. We put it in `ProcessComponent` instead,
where per-case context (`self._ctx`) already lives, for one reason: there are two
arrival components (parametric and MDN), and both would otherwise need the same
edit. Sampling where the case context is created means neither is touched, and
the attribute is available to every component downstream via the event payload.

The sampler is injected and defaults to None, so a simulation that does not need
case attributes behaves exactly as before.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional


class CaseAttributeSampler:
    """Draws case attributes from distributions fitted on the real log.

    Attributes are drawn independently. That is an assumption, and a defensible
    one here: the only attribute the permission model consumes is the loan goal,
    so cross-attribute correlation cannot affect the results we report. It would
    matter if a later section conditions processing times on several attributes
    at once.
    """

    def __init__(self, distributions: Dict[str, Dict[str, float]],
                 seed: Optional[int] = 42):
        self._rng = random.Random(seed)
        self._dists: Dict[str, tuple[List[str], List[float]]] = {}
        for attr, probs in distributions.items():
            values = list(probs)
            weights = [probs[v] for v in values]
            self._dists[attr] = (values, weights)

    def sample(self) -> Dict[str, str]:
        """One draw per attribute, for a newly created case."""
        return {
            attr: self._rng.choices(values, weights=weights, k=1)[0]
            for attr, (values, weights) in self._dists.items()
        }

    @property
    def attributes(self) -> List[str]:
        return list(self._dists)

    @classmethod
    def from_json(cls, path: str | Path, seed: Optional[int] = 42
                  ) -> "CaseAttributeSampler":
        return cls(json.loads(Path(path).read_text())["distributions"], seed=seed)


def save_distributions(distributions: Dict[str, Dict[str, float]],
                       path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"distributions": distributions}, indent=1))
    return path
