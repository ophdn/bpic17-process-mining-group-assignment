# Deep Reinforcement Learning Resource Allocation (D3)

This optional policy implements the Deep Reinforcement Learning alternative in
Final Task 2, following Middelhuis et al. (2025), *Learning policies for
resource allocation in business processes*, Information Systems 128, 102492.

## Method

The existing discrete-event simulation is wrapped as a Gymnasium environment.
It pauses whenever at least one queued task can be assigned to an available,
on-shift and permitted resource. MaskablePPO chooses one action:

- `(resource, activity)` assigns the oldest compatible queued instance;
- `POSTPONE` advances the simulation to a changed allocation state.

Infeasible assignments are masked. The mask enforces the live capacity,
calendar/roster, excluded-resource scenario, and the contextual OrdinoR
permission model. Consequently, the neural policy cannot bypass a simulation
constraint.

The version-3 normalized observation contains, per resource, free-capacity
fraction, on-shift status and the activity currently being executed; per
activity, queue length, oldest wait, oldest case age and expected duration; and
global queue, utilization and cyclical hour/weekday features. Case age helps the
policy avoid starving long-running cases, while expected duration supplies the
information needed to learn SPT-like behavior. Inference automatically
recognizes the smaller version-1 and version-2 observations used by preliminary
models.

Action-space version 2 removes resource/activity pairs that the permission
model can never allow. With the organizational permission model this reduces
the policy output from 3,457 actions to 1,942 without removing any feasible
decision. Old Cartesian-action models are detected and loaded as version 1.

The training reward combines negative WIP-seconds between decisions, a small
completion bonus, a penalty for choosing `POSTPONE`, and an end-of-episode
penalty for unfinished cases. The area under the WIP curve is accumulated case
sojourn time for a fixed arrival stream. The additional signals address both
the indefinite-postponement failure mode reported by Middelhuis et al. and the
finite-horizon loophole where work is merely pushed past the training window.
Final evaluation still uses the assignment's cycle-time, occupation, fairness,
throughput and backlog metrics, not the shaped training reward.

## Install

The RL stack is optional because PyTorch is large and the normal simulator does
not need it:

```bash
uv pip install --python .venv/bin/python -r requirements-drl.txt
```

## Train

The defaults use the final advanced configuration and the hyperparameters
reported by Middelhuis et al. as a starting point. Before PPO exploration, the
policy behavior-clones 10,000 decisions from Pull-SPT. This gives it a competent
starting rule while still allowing reinforcement learning to improve beyond
that heuristic:

```bash
PYTHONPATH=. .venv/bin/python scripts/train_drl.py \
  --timesteps 1000000 --days 3 --device mps --n-envs 4 \
  --eval-freq 100000 --eval-episodes 2 \
  --out models/drl_resource_policy_v3_1m
```

`--device auto` prefers MPS on Apple Silicon, then CUDA/ROCm, then CPU. An
explicit unavailable device fails immediately rather than silently training on
CPU. On an AMD machine with a supported ROCm PyTorch build, use `--device cuda`:
PyTorch deliberately exposes ROCm through its CUDA-compatible API.

`--n-envs 4` runs four independent simulations in parallel. This matters more
than the GPU for data collection because the discrete-event simulation is the
main bottleneck; it also gives the GPU larger batches of decisions to process.

Set `--spt-pretrain-decisions 0` for a from-scratch ablation. The demonstration
count, imitation loss, reward weights, action/state versions and training device
are saved in the model's JSON metadata.

`100000` steps only establish that the pipeline learns. In the BPIC model it
learned a postponement-heavy policy that completed 92.3 cases per evaluation
run. The older version-2 one-million-step policy completed 245.5, while Random
completed 268.4. A report-quality claim still requires a convergence study and
substantially more decisions; the paper allowed up to 20 million. Training
uses the paper's linear learning-rate decay. Validation checkpoints are ranked
by shaped reward, but the final model must be selected using held-out business
metrics because shaped reward and throughput can disagree.

## Evaluate

Use unseen seeds and the same model vocabulary/configuration used for training:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_experiments.py \
  --policies random,drl \
  --drl-model models/drl_resource_policy.zip \
  --seeds 10 --days 10 --warmup-days 2 \
  --process-model advanced --branching-mode visit \
  --permissions orgmodel --lifecycle-mode active \
  --out output/experiments_drl/
```

Never evaluate on training seeds. Compare the frozen policy with the same CRN
replication seeds as every baseline. A smoke-trained model is evidence that the
software path works, not evidence that DRL improves resource allocation.

## Files

- `simulation/drl.py`: Gym environment, dense reward, optional model loader.
- `simulation/components/resource.py`: observation, action mask and action
  application at the allocation seam.
- `scripts/train_drl.py`: reproducible MaskablePPO training CLI.
- `scripts/run_experiments.py`: frozen `drl` policy evaluation.
- `tests/test_drl.py`: masks, observations, postpone, assignment and reset.
