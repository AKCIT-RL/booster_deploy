"""MimicKit steering task for booster_deploy.

The MimicKit checkout has to be importable before observation.py loads, and
that is earlier than it looks: scripts/deploy.py walks and imports EVERY task
submodule to build its registry, so putting the path setup in the Policy
constructor is too late - `--list` would already have crashed. Hence it happens
here, in the package.

Set MIMICKIT_PATH when the checkout is not a sibling of booster_deploy.

Requires, on top of booster_deploy's own dependencies: a MimicKit checkout and
`gymnasium` (envs.char_env imports it for its action space, though the
observation function itself is plain torch). That is the price of importing the
observation definition instead of copying it into this repo, and it is worth
paying - a copied 200-dim layout goes stale silently, and tests/unit/
test_deploy_observation.py could not then compare against anything.

Neither is needed to register the task or to run `deploy.py --list`:
observation.py imports MimicKit on first use, not at module scope. They are
needed to construct the policy, and their absence reports itself there.
"""

import os
import sys

_DEFAULT_MIMICKIT = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "MimicKit", "mimickit"))

MIMICKIT_PATH = os.path.abspath(os.environ.get("MIMICKIT_PATH", _DEFAULT_MIMICKIT))
if MIMICKIT_PATH not in sys.path:
    sys.path.insert(0, MIMICKIT_PATH)

try:
    from .mimickit_steering import T1MimicKitSteeringControllerCfg
except ImportError as e:
    print("[mimickit_steering] not registered: {}. This task needs a MimicKit "
          "checkout (looked in {}; override with MIMICKIT_PATH) and gymnasium."
          .format(e, MIMICKIT_PATH))
else:
    from booster_deploy.utils.isaaclab.configclass import configclass
    from booster_deploy.utils.registry import register_task

    @configclass
    class T1MimicKitSteeringMeasControllerCfg(T1MimicKitSteeringControllerCfg):
        """Steering under a MEASURABLE-observation checkpoint.

        A separate task rather than a new default on the existing one, because
        what changes between the two checkpoints is not only the file. The
        observation layout is read off the module (`_frame_dim`), so that part
        needs no config - but the frame history length does have to agree with
        the control period, and that lives here: this checkpoint carries 30
        frames, which is 1 s at the 30 Hz this cfg runs. A checkpoint with 50
        frames is 1 s at 50 Hz and would need its own task with policy_dt 1/50
        and mujoco.decimation 10; driven at 1/30 it loads, runs, and is simply
        wrong, with nothing reporting it (see the RATES note in
        mimickit_steering.py). One task per contract is what keeps that from
        being a flag someone forgets.

        Everything else is inherited deliberately: the gains, the URDF effort
        ceiling, the prepare-state pose/gain asymmetry and the command envelope
        are properties of the robot and the handover, not of which observation
        the actor reads.

        WHY THIS ONE IS WORTH RUNNING. Measured with scripts/sim_viability.py,
        60 s at vx=1.0, `--pd explicit`:

          --degrade none        60.00 s   tracking 118.8%   root_h_min 0.6440
          --degrade zero_root   60.00 s   tracking 118.8%   root_h_min 0.6440

        Identical in every measured quantity - the two JSON summaries differ
        only in the field that records which flag was passed. `zero_root` is an
        exact no-op here, the same way it is on Booster's own t1_walk, because
        this observation never asks for root height or root linear velocity.
        The observability gap that makes the legacy checkpoint sim-only
        (docs/08, docs/12) does not apply to this one.

        WHAT IS STILL OPEN, and neither is closed by this task:

          - Actuator envelope. It crosses `--effort catalog --tn` (the physical
            reading) at 60 s for vx 0.5 and 1.0, but falls under `--effort
            derated`: 3.60 s at vx=1.0, 10.43 s at 0.5 with `--tn`. The fall is
            saturation at the flat firmware ceiling, not the T-N curve -
            n_joints_over_derated reads 0/23 there precisely because the demand
            is clipped to that ceiling rather than exceeding it. Gate 2 of
            docs/10 (60 s under derated --tn) therefore FAILS.
          - Overspeed. It tracks 118-138% across the range, against 86-103% for
            the legacy checkpoint. On the gantry that does not show; on the
            floor it is speed nobody commanded. vel_command is inherited at
            vx_max 2.5, which is worth a deliberate decision before any floor
            run rather than an inherited default.
        """

        def __post_init__(self):
            super().__post_init__()
            self.policy.checkpoint_path = \
                "../../models/t1_meas_dr_wrturn_seed1.pt"

    register_task("t1_mimickit_steering", T1MimicKitSteeringControllerCfg())
    register_task("t1_mimickit_steering_meas",
                  T1MimicKitSteeringMeasControllerCfg())
