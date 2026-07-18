"""Deep-RL support for online resource allocation.

The implementation follows Middelhuis et al. (2025): the discrete-event
simulation is the environment, a decision is one feasible resource/activity
assignment (or POSTPONE), and infeasible actions are masked.  The normal
simulation has no dependency on the RL stack; Gymnasium, PyTorch and
sb3-contrib are imported only when training or loading a DRL model.

The reward used by :class:`ResourceAllocationEnv` is the negative area under
the work-in-progress curve.  For a fixed arrival stream this is proportional
to accumulated case sojourn time (Little's-law accounting), gives PPO a dense
signal, and makes postponing all work expensive even when few cases complete
within a finite horizon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np


class DRLDependencyError(RuntimeError):
    """Raised when the optional training/inference stack is unavailable."""


def _gymnasium():
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise DRLDependencyError(
            "DRL support requires the optional packages in requirements-drl.txt. "
            "Install them with `.venv/bin/python -m pip install -r "
            "requirements-drl.txt`."
        ) from exc
    return gym


def load_maskable_ppo(path: str | Path):
    """Load an sb3-contrib MaskablePPO model with an actionable error."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise DRLDependencyError(
            "Loading a DRL allocation policy requires requirements-drl.txt."
        ) from exc

    model_path = Path(path)
    # Stable-Baselines appends .zip when saving; accept either spelling.
    if not model_path.exists() and model_path.suffix != ".zip":
        zipped = model_path.with_suffix(".zip")
        if zipped.exists():
            model_path = zipped
    if not model_path.exists():
        raise FileNotFoundError(
            f"DRL model not found: {model_path}. Train one with "
            "`scripts/train_drl.py` or pass --drl-model."
        )
    return MaskablePPO.load(str(model_path))


try:  # Keep importing simulation.drl cheap for non-RL users and tests.
    import gymnasium as _gym
    _GymBase = _gym.Env
except ImportError:  # pragma: no cover - exercised when optional extra absent
    _gym = None
    _GymBase = object


class ResourceAllocationEnv(_GymBase):
    """Gymnasium wrapper that pauses a DES at allocation decision epochs.

    Parameters
    ----------
    simulation_factory:
        ``factory(seed) -> (engine, resource_component)``.  The resource
        component must have ``drl=True, drl_external_control=True``.
    base_seed:
        First episode seed; subsequent resets increment it unless Gymnasium
        supplies an explicit seed.
    reward_scale:
        Divide WIP-seconds by this value.  The default is 100 case-days, which
        keeps typical rewards near unit scale without changing the optimum.
    completion_reward:
        Small positive reward for each completed case.  This adds a direct
        throughput signal to the dense cycle-time objective.
    terminal_wip_penalty:
        Additional penalty per unfinished case at the fixed episode horizon.
        This prevents a policy from looking attractive by deferring work until
        after the training window.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        simulation_factory: Callable[[int], Tuple[object, object]],
        *,
        base_seed: int = 1,
        reward_scale: float = 100.0 * 86400.0,
        postpone_penalty: float = 0.0,
        completion_reward: float = 0.01,
        terminal_wip_penalty: float = 0.002,
        increment_seeds: bool = True,
    ):
        gym = _gymnasium()
        super().__init__()
        self._factory = simulation_factory
        self._base_seed = int(base_seed)
        self._next_seed = int(base_seed)
        self._reward_scale = float(reward_scale)
        self._postpone_penalty = float(postpone_penalty)
        self._completion_reward = float(completion_reward)
        self._terminal_wip_penalty = float(terminal_wip_penalty)
        self._increment_seeds = bool(increment_seeds)
        if self._reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        if min(
            self._postpone_penalty,
            self._completion_reward,
            self._terminal_wip_penalty,
        ) < 0:
            raise ValueError("DRL reward weights must be non-negative")

        self.engine = None
        self.resources = None
        self._terminated = False
        self._reuse_initial_episode = True
        self._build(self._next_seed)
        self._next_seed += 1

        self.action_space = gym.spaces.Discrete(self.resources.drl_action_count)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.resources.drl_observation_size,),
            dtype=np.float32,
        )

    def _build(self, seed: int) -> None:
        self.engine, self.resources = self._factory(int(seed))
        if not getattr(self.resources, "drl_external_control", False):
            raise ValueError(
                "simulation_factory must build ResourceComponent with "
                "drl=True and drl_external_control=True"
            )
        self._terminated = False
        self._advance_to_decision()

    def reset(self, *, seed: Optional[int] = None, options=None):
        if _gym is not None:
            super().reset(seed=seed)
        reuse_initial = self._reuse_initial_episode and seed is None
        self._reuse_initial_episode = False
        if reuse_initial:
            # Reuse the episode built in __init__; SB3 calls reset immediately.
            pass
        else:
            episode_seed = int(seed) if seed is not None else (
                self._next_seed if self._increment_seeds else self._base_seed
            )
            self._build(episode_seed)
            if seed is None and self._increment_seeds:
                self._next_seed += 1
        return self._observation(), self._info()

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("step() called after episode end; call reset()")
        if not self.resources.drl_decision_pending:
            raise RuntimeError("no DRL allocation decision is pending")

        action = int(action)
        postponed = action == self.resources.drl_postpone_action
        completed_before = int(self.engine.stats["cases_completed"])
        self.resources.apply_drl_action(self.engine, action)
        wip_area = self._advance_to_decision()
        completed_after = int(self.engine.stats["cases_completed"])
        completed_delta = max(0, completed_after - completed_before)
        reward = -float(wip_area) / self._reward_scale
        reward += self._completion_reward * completed_delta
        if postponed:
            reward -= self._postpone_penalty
        terminal_penalty = 0.0
        if self._terminated:
            wip = max(
                0,
                int(self.engine.stats["cases_started"])
                - int(self.engine.stats["cases_completed"]),
            )
            terminal_penalty = self._terminal_wip_penalty * wip
            reward -= terminal_penalty
        obs = self._observation()
        # A fixed simulation horizon is a Gymnasium time-limit truncation,
        # not an absorbing terminal state of the business process.
        return obs, reward, False, self._terminated, self._info(
            wip_area=wip_area,
            cases_completed_delta=completed_delta,
            terminal_penalty=terminal_penalty,
        )

    def action_masks(self) -> np.ndarray:
        """MaskablePPO convention: True means the action is feasible."""
        if self._terminated:
            mask = np.zeros(self.action_space.n, dtype=bool)
            mask[-1] = True
            return mask
        return self.resources.drl_action_mask(self.engine)

    def _advance_to_decision(self) -> float:
        """Run events until a DRL decision or horizon; return WIP-seconds."""
        area = 0.0
        while not self.resources.drl_decision_pending:
            old_t = float(self.engine.now)
            old_wip = max(
                0,
                int(self.engine.stats["cases_started"])
                - int(self.engine.stats["cases_completed"]),
            )
            if not self.engine.step():
                # Charge unfinished cases through the configured horizon.
                tail = max(0.0, float(self.engine.sim_duration) - old_t)
                area += old_wip * tail
                self._terminated = True
                break
            area += old_wip * max(0.0, float(self.engine.now) - old_t)
        return area

    def _observation(self) -> np.ndarray:
        if self._terminated:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        return self.resources.drl_observation(self.engine)

    def _info(
        self,
        *,
        wip_area: float = 0.0,
        cases_completed_delta: int = 0,
        terminal_penalty: float = 0.0,
    ) -> dict:
        return {
            "sim_time": float(self.engine.now),
            "wip": max(
                0,
                int(self.engine.stats["cases_started"])
                - int(self.engine.stats["cases_completed"]),
            ),
            "wip_area_seconds": float(wip_area),
            "cases_completed_delta": int(cases_completed_delta),
            "terminal_wip_penalty": float(terminal_penalty),
            "cases_completed": int(self.engine.stats["cases_completed"]),
            "resource_stats": self.resources.stats(),
        }
