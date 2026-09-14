# MimicKit steering task (T1)

Deploys a steering policy trained in [MimicKit](https://github.com/AKCIT-RL/MimicKit)
(Isaac Gym, MCWAMP) through `booster_deploy`. Reproduction work for
arXiv:2511.03996.

The policy walks and turns on a velocity command. It is memoryless — one
observation frame in, one set of joint targets out.

## How it differs from `tasks/locomotion`

| | `locomotion` | `mimickit_steering` |
|---|---|---|
| action | residual: `default_joint_pos + action * action_scale` | **the joint position target itself** |
| observation | 53 dims × 10 frames of history | 200 dims, single frame |
| obs contents | ang vel, projected gravity, cmd, joint pos/vel, last action | root height, tan-norm rotations, key body positions, steering cmd |

The action difference matters: MimicKit's action normalizer **and** the training
environment's action clip are baked into the exported TorchScript module,
because both are derived from the asset's joint limits and are not the
consumer's to re-declare. **Do not add a default pose or an action scale to this
policy's output.**

## Requirements

Beyond `booster_deploy`'s own dependencies:

- a **MimicKit** checkout, by default a sibling of `booster_deploy/`; override
  with `MIMICKIT_PATH`
- `gymnasium` and `pyyaml` — `envs.char_env` imports them

`observation.py` calls MimicKit's own `compute_char_obs` instead of
reimplementing the 200-dim layout, which is why the checkout is needed. A copied
layout goes stale silently; an imported one cannot.

None of this is needed to `--list` tasks: MimicKit is imported on first use, so a
robot carrying only `booster_deploy` is unaffected.

## Running

```bash
cd booster_deploy
PYTHONPATH=../booster_assets/src DISPLAY=:1 \
    python3 scripts/deploy.py --task t1_mimickit_steering --mujoco
```

Type `x y yaw` + Enter in the terminal to command it.

For keyboard control in the viewer, and to choose the damping scheme:

```bash
PYTHONPATH=../booster_assets/src DISPLAY=:1 \
    python3 tasks/mimickit_steering/teleop.py --pd explicit --vx 1.0
```

```
W / X   forward speed   A / D   turn   Q / E   strafe   Z zero   P print
```

The viewer's own keys keep working (Space pause, ←/→ step, Backspace reset,
0-9 slot with S save / L load, R record, G ghost) — `teleop.py` chains into
`MujocoController`'s callback rather than replacing it. Speed is `W`/`X` and not
`W`/`S` because `S` is the viewer's save-state key.

`--log_states <path>` writes `<path>.npz` of the rollout for offline analysis.

## Damping schemes — read this before comparing runs

`--pd` selects one of three, defined in [`controllers.py`](controllers.py). They
share one `ctrl_step` and differ in exactly one term, so a comparison between
them isolates it.

| `--pd` | PD loop | MuJoCo joint damping |
|---|---|---|
| `explicit` (default) | `kp*(q* − q) − kd*q̇` | zero |
| `implicit` | `kp*(q* − q)` | `kd` written into `dof_damping` |
| `none` | `kp*(q* − q)` | zero |

`none` is not a design — it is what this checkout currently does. The upstream
`ctrl_step` sets `kd = np.zeros_like(kp)` and expects the MJCF to supply the
damping as passive joint damping, but `booster_assets/robots/T1/T1_23dof.xml`
declares none: `mj_model.dof_damping[6:]` is all zeros. So `RobotCfg.joint_damping`
is applied nowhere and the T1 runs completely undamped.

Measured, 60 s per run, forward command only:

| `--pd` | cmd 0.5 m/s | cmd 1.0 m/s | cmd 2.0 m/s |
|---|---|---|---|
| `none` | fell at 1.7 s | fell at 1.0 s | fell at 1.2 s |
| `explicit` | 0.68 (136%) | 1.03 (103%) | 1.53 (76%) |
| `implicit` | 0.68 (136%) | 1.03 (103%) | 1.51 (76%) |

Two conclusions. The upstream *scheme* is fine — `implicit` and `explicit` agree
to within noise for this policy, so the argument for implicit damping (no
horizontal contact force at a stance foot, hence no sliding) costs nothing here.
The upstream *state* is not: without damping the robot cannot stand.

`implicit` is the honest long-term answer, and the cleaner fix is to declare the
damping in the T1 MJCF so it applies to every task rather than only to this one.
`explicit` is what this policy was trained and validated against, so it is the
default.

The gains themselves are Booster's own `kp` with `kd = 0.0637·kp` — the ratio the
G1 asset uses uniformly across all 29 of its joints, validated in the training
engine. See `scripts/prepare_t1_asset.py --kd_ratio` in the htwk-gym repo.

## Rates are part of the policy

`policy_dt = 1/30` and `decimation = 4` (120 Hz physics), matching the training
run's `engine_config.yaml` (`control_freq: 30`, `sim_freq: 120`), not the 50 Hz /
500 Hz defaults. A position-target policy is a closed loop with its PD
controller: change `kp`, `kd` or the control period and the same targets produce
different motion.

Note that `MujocoControllerCfg.__post_init__` derives
`physics_dt = policy_dt / decimation`, so a cfg that sets `decimation` *after*
calling `super().__post_init__()` leaves `physics_dt` stale and silently runs at
the wrong rate.

## Regenerating the policy

`models/t1_steering.pt` is exported from a MimicKit checkpoint by the htwk-gym
repo:

```bash
python3 scripts/export_policy.py \
    --checkpoint MimicKit/output/t1_fixed_seed1/model.pt \
    --actor_net fc_2layers_1024units
```

The export self-checks: it scripts the module, saves it, reloads it from disk and
compares against the policy assembled from MimicKit's own classes. Correctness of
the observation adapter is covered by `tests/unit/test_deploy_observation.py`
(the 200 numbers must match the training environment's) and
`tests/unit/test_deploy_key_bodies.py` (MimicKit's forward kinematics against
MuJoCo's own), both in htwk-gym.

## Known gaps

**Hardware.** Runs under `--mujoco` only. 196 of the 200 observation dimensions
come straight off the hardware, but `booster_robot_controller.update_state`
fills `root_pos_w` and `root_lin_vel_w` with `np.zeros(3)` — only the IMU and the
encoders are real — so root height and root linear velocity would arrive as
lies. Both are recoverable with stance-foot kinematic odometry, which belongs in
this task; the exported policy needs no change for it.

**Torque.** Trained against the URDF limits (knee 130.5 Nm), which is what
`T1MimicKitSteeringControllerCfg` restates. `T1_23DOF_CFG.effort_limit` is
Booster's derated operating limit (knee 60 Nm). Before any hardware run, retrain
with `prepare_t1_asset.py --effort_source deploy` rather than raising the limit.

**Stability.** The policy tracks speed and yaw rate to within a few percent but
is marginally stable in MuJoCo: some command values fall and the boundary is not
smooth (yaw +0.29 and +0.31 rad/s walk; +0.30 falls at 1.6 s, reproducibly).
Applying a turn during the spawn transient is reliably bad — the robot spawns at
`default_joint_pos`, while training used reference-state initialisation from
motion frames, so the first moments are off-distribution. Whether this is the
policy or the Isaac Gym → MuJoCo crossing is not yet attributed.
