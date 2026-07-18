#!/usr/bin/env python3
"""Train D3: a masked-PPO resource-allocation policy.

The environment pauses the existing discrete-event simulator whenever at
least one permitted resource/activity assignment is possible.  The policy
chooses an assignment or POSTPONE; impossible pairs are masked.  Observations
include queue age/load, resource capacity and shifts, plus cyclical time.

Example (from repository root)::

    PYTHONPATH=. .venv/bin/python scripts/train_drl.py \
      --timesteps 100000 --days 3 --out models/drl_resource_policy

Install the optional stack first with::

    .venv/bin/python -m pip install -r requirements-drl.txt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np

from analysis.availability import YearlyAvailability
from simulation.components.lifecycle_params import LifecycleParameters
from simulation.components.petri_process import PetriNetProcessComponent
from simulation.components.process import ProcessComponent
from simulation.components.resource import (
    DEFAULT_ROSTER_SEED,
    RESOURCE_PERMISSIONS,
    ResourceComponent,
    capacity_for_mode,
)
from simulation.core.engine import SimulationEngine
from simulation.drl import DRLDependencyError, ResourceAllocationEnv
from simulation.main import CaseCompletionTracker

from scripts.run_experiments import (
    ACTIVE_INPUTS_PATH,
    AVAILABILITY_MODEL_PATH,
    BPMN_PATH,
    CASE_ATTRIBUTES_PATH,
    START_DATETIME,
    build_arrival_component,
    load_permission_model,
    scenario_excluded_resources,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--days", type=int, default=3,
                        help="Simulated days per training episode.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path,
                        default=Path("models/drl_resource_policy"))
    parser.add_argument("--scenario", default="normal",
                        choices=["normal", "peak", "outage"])
    parser.add_argument("--process-model", default="advanced",
                        choices=["basic", "advanced"])
    parser.add_argument("--branching-mode", default="visit",
                        choices=["probs", "visit", "rules"])
    parser.add_argument("--permissions", default="orgmodel",
                        choices=["orgmodel", "observed", "hardcoded"])
    parser.add_argument("--lifecycle-mode", default="active",
                        choices=["legacy", "active"])
    parser.add_argument("--processing-time-mode", default="distribution",
                        choices=["distribution", "ml_model", "ml_probabilistic"])
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--roster-seed", type=int, default=DEFAULT_ROSTER_SEED)
    parser.add_argument("--no-roster", action="store_true")
    parser.add_argument(
        "--n-steps", type=int, default=None,
        help="Rollout steps per worker (default keeps 25,600 total across workers).",
    )
    parser.add_argument(
        "--n-envs", type=int, default=1,
        help="Parallel simulator workers; 4 is a useful GPU-training starting point.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Width of each of the two policy/value layers.")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument(
        "--ent-coef", type=float, default=0.005,
        help="Entropy bonus that preserves exploration after the SPT warm start.",
    )
    parser.add_argument(
        "--postpone-penalty", type=float, default=0.001,
        help="Scale-adjusted penalty for avoidable strategic idling.",
    )
    parser.add_argument(
        "--completion-reward", type=float, default=0.01,
        help="Reward per completed case (adds a direct throughput signal).",
    )
    parser.add_argument(
        "--terminal-wip-penalty", type=float, default=0.002,
        help="End-of-episode penalty per unfinished case.",
    )
    parser.add_argument("--action-version", type=int, default=2, choices=[1, 2])
    parser.add_argument(
        "--observation-version", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
        help="Training device. auto prefers Apple MPS, then CUDA, then CPU.",
    )
    parser.add_argument(
        "--spt-pretrain-decisions", type=int, default=10_000,
        help="Pull-SPT expert decisions used to warm-start the policy (0 disables).",
    )
    parser.add_argument("--spt-pretrain-epochs", type=int, default=3)
    parser.add_argument("--spt-pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--constant-learning-rate", action="store_true",
        help="Disable the paper's linear learning-rate decay.",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=0,
        help="Validate every N steps and retain the best checkpoint (0 disables).",
    )
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--eval-seed", type=int, default=1_000_000)
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2])
    return parser.parse_args()


def linear_schedule(initial_value: float):
    """Paper-compatible linear decay from initial_value to zero."""
    return lambda progress_remaining: float(progress_remaining) * initial_value


def resolve_device(requested: str) -> str:
    """Resolve a training device without silently ignoring the user's GPU."""
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for DRL training") from exc

    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        built = torch.backends.mps.is_built()
        raise SystemExit(
            "--device mps was requested, but PyTorch reports MPS unavailable "
            f"(MPS-enabled build: {built}). Check the Python/PyTorch build and "
            "macOS Metal support; training was not silently moved to CPU."
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda was requested, but PyTorch reports CUDA unavailable. "
            "On a supported AMD ROCm installation, PyTorch also uses the "
            "'cuda' device name."
        )
    return requested


def collect_spt_demonstrations(
    factory,
    *,
    decisions: int,
    base_seed: int,
    postpone_penalty: float,
    completion_reward: float,
    terminal_wip_penalty: float,
):
    """Generate deterministic state/action examples from the Pull-SPT rule."""
    env = ResourceAllocationEnv(
        factory,
        base_seed=base_seed,
        postpone_penalty=postpone_penalty,
        completion_reward=completion_reward,
        terminal_wip_penalty=terminal_wip_penalty,
    )
    observation, _ = env.reset()
    observations = []
    actions = []
    masks = []
    for _ in range(decisions):
        mask = env.action_masks()
        action = env.resources.drl_shortest_processing_action(env.engine)
        observations.append(observation.copy())
        actions.append(action)
        masks.append(mask.copy())
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()
    env.close()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        np.asarray(masks, dtype=bool),
    )


def pretrain_policy_from_demonstrations(
    model,
    demonstrations,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> float:
    """Behavior-clone the SPT expert into MaskablePPO's policy head."""
    import torch

    observations, actions, masks = demonstrations
    if not len(actions):
        return 0.0
    optimizer = model.policy.optimizer
    original_rates = [group["lr"] for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    rng = np.random.default_rng(seed)
    losses = []
    model.policy.set_training_mode(True)
    try:
        for _ in range(epochs):
            order = rng.permutation(len(actions))
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                obs_batch = torch.as_tensor(
                    observations[indices], device=model.device)
                action_batch = torch.as_tensor(
                    actions[indices], device=model.device)
                mask_batch = torch.as_tensor(
                    masks[indices], device=model.device)
                _, log_probability, _ = model.policy.evaluate_actions(
                    obs_batch, action_batch, action_masks=mask_batch)
                loss = -log_probability.mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
    finally:
        for group, rate in zip(optimizer.param_groups, original_rates):
            group["lr"] = rate
    return float(np.mean(losses)) if losses else 0.0


def make_simulation_factory(args):
    capacity = (args.capacity if args.capacity is not None
                else capacity_for_mode(args.lifecycle_mode))
    roster_base = None if args.no_roster else args.roster_seed

    def factory(seed: int):
        effective_roster_seed = None if roster_base is None else roster_base + seed
        calendar = YearlyAvailability.from_json(
            AVAILABILITY_MODEL_PATH, roster_seed=effective_roster_seed)
        permissions, case_attributes = load_permission_model(args.permissions, seed)
        resource_pool = (permissions.resources() if permissions is not None
                         else sorted(RESOURCE_PERMISSIONS))
        excluded = scenario_excluded_resources(
            args.scenario, seed, resource_pool)
        lifecycle_params = (
            LifecycleParameters.from_file(ACTIVE_INPUTS_PATH)
            if args.lifecycle_mode == "active" else None
        )

        engine = SimulationEngine(
            sim_duration=args.days * 86400,
            start_datetime=START_DATETIME,
            verbose=False,
            lifecycle_mode=args.lifecycle_mode,
        )
        resources = ResourceComponent(
            capacity_per_resource=capacity,
            seed=seed,
            calendar=calendar,
            start_datetime=START_DATETIME,
            permissions=permissions,
            excluded_resources=excluded,
            lifecycle_mode=args.lifecycle_mode,
            lifecycle_params=lifecycle_params,
            drl=True,
            drl_external_control=True,
            drl_action_version=args.action_version,
            drl_observation_version=args.observation_version,
        )
        arrivals = build_arrival_component(seed, args.scenario)
        process_kwargs = dict(
            seed=seed,
            mode=args.processing_time_mode,
            start_datetime=START_DATETIME,
            resource_component=resources,
            crn=True,
            case_attributes=case_attributes,
            lifecycle_mode=args.lifecycle_mode,
            lifecycle_params=lifecycle_params,
        )
        if args.process_model == "advanced":
            process = PetriNetProcessComponent(
                bpmn_path=str(BPMN_PATH),
                branching_mode=args.branching_mode,
                **process_kwargs,
            )
        else:
            process = ProcessComponent(**process_kwargs)

        engine.register(arrivals)
        engine.register(resources)
        engine.register(process)
        engine.register(CaseCompletionTracker())
        arrivals.bootstrap(engine)
        return engine, resources

    return factory


def main():
    args = parse_args()
    if args.timesteps <= 0 or args.days <= 0:
        raise SystemExit("--timesteps and --days must be positive")
    if args.n_envs <= 0:
        raise SystemExit("--n-envs must be positive")
    if args.n_steps is None:
        args.n_steps = math.ceil(25_600 / args.n_envs)
    if min(args.batch_size, args.n_steps, args.hidden_size) <= 0:
        raise SystemExit(
            "--batch-size, --n-steps, --n-envs and --hidden-size must be positive")
    if min(
        args.postpone_penalty,
        args.completion_reward,
        args.terminal_wip_penalty,
        args.eval_freq,
        args.spt_pretrain_decisions,
        args.ent_coef,
    ) < 0 or args.eval_episodes <= 0:
        raise SystemExit(
            "reward weights/--eval-freq must be non-negative and "
            "--eval-episodes positive"
        )
    if (args.spt_pretrain_epochs <= 0
            or args.spt_pretrain_learning_rate <= 0):
        raise SystemExit("SPT pretraining epochs/learning rate must be positive")
    device = resolve_device(args.device)

    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    except ImportError as exc:
        raise SystemExit(
            "Missing optional DRL dependencies. Run: "
            ".venv/bin/python -m pip install -r requirements-drl.txt"
        ) from exc

    try:
        factory = make_simulation_factory(args)
        env_kwargs = dict(
            postpone_penalty=args.postpone_penalty,
            completion_reward=args.completion_reward,
            terminal_wip_penalty=args.terminal_wip_penalty,
        )
        if args.n_envs == 1:
            env = ResourceAllocationEnv(
                factory, base_seed=args.seed, **env_kwargs)
        else:
            from stable_baselines3.common.vec_env import SubprocVecEnv

            def make_worker(rank: int):
                return lambda: ResourceAllocationEnv(
                    factory,
                    base_seed=args.seed + rank * 1_000_000,
                    **env_kwargs,
                )

            # spawn is safe with Apple's Metal runtime; forking a process after
            # PyTorch initializes MPS can leave the child in an invalid state.
            env = SubprocVecEnv(
                [make_worker(rank) for rank in range(args.n_envs)],
                start_method="spawn",
            )
    except DRLDependencyError as exc:
        raise SystemExit(str(exc)) from exc

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [args.hidden_size, args.hidden_size]},
        learning_rate=(
            args.learning_rate if args.constant_learning_rate
            else linear_schedule(args.learning_rate)
        ),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        seed=args.seed,
        device=device,
        verbose=args.verbose,
    )
    pretrain_loss = None
    if args.spt_pretrain_decisions:
        print(
            f"Collecting {args.spt_pretrain_decisions:,} Pull-SPT "
            "demonstration decisions..."
        )
        demonstrations = collect_spt_demonstrations(
            factory,
            decisions=args.spt_pretrain_decisions,
            base_seed=args.seed,
            postpone_penalty=args.postpone_penalty,
            completion_reward=args.completion_reward,
            terminal_wip_penalty=args.terminal_wip_penalty,
        )
        pretrain_loss = pretrain_policy_from_demonstrations(
            model,
            demonstrations,
            epochs=args.spt_pretrain_epochs,
            batch_size=args.batch_size,
            learning_rate=args.spt_pretrain_learning_rate,
            seed=args.seed,
        )
        print(f"SPT warm-start mean imitation loss: {pretrain_loss:.6f}")
    callback = None
    best_dir = args.out.parent / f"{args.out.name}_validation"
    if args.eval_freq:
        from stable_baselines3.common.monitor import Monitor

        eval_env = Monitor(
            ResourceAllocationEnv(
                factory,
                base_seed=args.eval_seed,
                postpone_penalty=args.postpone_penalty,
                completion_reward=args.completion_reward,
                terminal_wip_penalty=args.terminal_wip_penalty,
                # Every validation round sees the same distinct held-out seeds.
                # Since its length equals n_eval_episodes, the cycle realigns at
                # the beginning of every MaskableEvalCallback evaluation.
                episode_seeds=range(
                    args.eval_seed, args.eval_seed + args.eval_episodes),
            )
        )
        callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(best_dir),
            eval_freq=max(1, args.eval_freq // args.n_envs),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            use_masking=True,
            verbose=1,
        )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=False,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    best_model_path = best_dir / "best_model.zip"
    reward_best_model_path = args.out.with_name(
        f"{args.out.name}_reward_best"
    ).with_suffix(".zip")
    if best_model_path.exists():
        shutil.copy2(best_model_path, reward_best_model_path)
    metadata = {
        "method": "MaskablePPO",
        "paper": "Middelhuis et al., Information Systems 128 (2025) 102492",
        # PPO finishes complete n_steps rollout blocks, so the number actually
        # learned can be slightly larger than the requested CLI value.
        "timesteps": int(model.num_timesteps),
        "requested_timesteps": args.timesteps,
        "episode_days": args.days,
        "seed": args.seed,
        "scenario": args.scenario,
        "process_model": args.process_model,
        "branching_mode": args.branching_mode,
        "permissions": args.permissions,
        "lifecycle_mode": args.lifecycle_mode,
        "processing_time_mode": args.processing_time_mode,
        "capacity": args.capacity,
        "roster_seed": None if args.no_roster else args.roster_seed,
        "observation_version": args.observation_version,
        "observation_size": int(env.observation_space.shape[0]),
        "action_version": args.action_version,
        "action_count": int(env.action_space.n),
        "postpone_penalty": args.postpone_penalty,
        "completion_reward": args.completion_reward,
        "terminal_wip_penalty": args.terminal_wip_penalty,
        "device": device,
        "parallel_environments": args.n_envs,
        "spt_pretraining": {
            "decisions": args.spt_pretrain_decisions,
            "epochs": args.spt_pretrain_epochs,
            "learning_rate": args.spt_pretrain_learning_rate,
            "mean_imitation_loss": pretrain_loss,
        },
        "validation": {
            "frequency": args.eval_freq,
            "episodes": args.eval_episodes,
            "seed": args.eval_seed if args.eval_freq else None,
            "seeds": (
                list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
                if args.eval_freq else []
            ),
            "reward_best_model": (
                str(reward_best_model_path) if best_model_path.exists() else None
            ),
        },
        "hyperparameters": {
            "network": [args.hidden_size, args.hidden_size],
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "learning_rate_schedule": (
                "constant" if args.constant_learning_rate else "linear"
            ),
            "gamma": args.gamma,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
        },
    }
    metadata_path = args.out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved DRL model -> {args.out.with_suffix('.zip')}")
    print(f"Saved metadata  -> {metadata_path}")


if __name__ == "__main__":
    main()
