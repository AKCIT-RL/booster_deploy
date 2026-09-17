"""Deploy a MimicKit steering policy through booster_deploy.

Produced by scripts/export_policy.py in the htwk-gym repo; see
MimicKit/mimickit/learning/policy_export.py for the export contract.

HOW THIS DIFFERS FROM tasks/locomotion

  action meaning   LocomotionPolicy receives a residual and builds the command
                   as `default_joint_pos + action * action_scale`. This policy's
                   output IS the joint position target: the action normalizer
                   and the environment's clip are baked into the exported
                   module, because both are derived from the asset's joint
                   limits and are not the consumer's to re-declare. So there is
                   no default pose and no action scale here, and adding one
                   would silently rescale every command.

  memory           depends on the checkpoint, see below. The older steering
                   observation is a single frame; the measurable one stacks a
                   frame history the way LocomotionPolicy does.

  observation      53 privileged-free numbers there. Here it is one of two
                   layouts, both built in observation.py.

TWO OBSERVATION LAYOUTS, CHOSEN BY THE CHECKPOINT

  legacy (200)     root height, tan-norm rotations, root linear velocity, key
                   body positions + steering. Memoryless. Four of its numbers
                   do not exist on hardware (see below).

  measurable (80)  projected gravity, base angular velocity, joint offsets,
                   joint velocities, previous action + steering, stacked as
                   [current frame | H past frames]. Everything in it comes from
                   the IMU, the encoders and the policy's own last output, so
                   nothing has to be faked on hardware.

Nothing configures this: the exported module carries a `_frame_dim` attribute
in the measurable case and not in the legacy one, so __init__ reads the layout
off the checkpoint. Pointing at the wrong one is therefore not possible.

RATES AND GAINS ARE PART OF THE POLICY

The checkpoint was trained at control 30 Hz / physics 120 Hz (the run's
engine_config.yaml) with the PD gains below. A position-target policy is a
closed loop with its PD controller: change kp, kd or the control period and the
same targets produce different motion. Hence policy_dt = 1/30 and decimation 4
rather than the 50 Hz / 500 Hz defaults, and hence the gains are restated here
instead of inherited from T1WalkControllerCfg, whose 200/5 belong to Booster's
own policy.

The rate is fixed at 30 Hz here, so a checkpoint trained at 50 Hz will load and
run but be driven at the wrong period. Making policy_dt follow the checkpoint is
its own change; until then, only use 30 Hz checkpoints.

HANDOVER: THE POSE IS OURS, THE PREPARE GAINS ARE NOT

The pose the ramp ends in is part of this policy - its output is an absolute
position target, so it commands its pose rather than correcting toward one, and
Booster's prepare pose is half this crouch. That pose is overridden below.

The prepare GAINS are not, and an earlier version of this file that overrode
them put the T1 on the floor during the ramp. Holding a pose statically and
running a policy are different control problems: a PD on a fixed target carries
the whole inverted pendulum on joint stiffness, while the policy re-plans 30
times a second and balances actively. Stiff to stand, soft to run. See the
PREPARE STATE note in the controller cfg for the measurements.

WHAT THIS STILL DOES NOT DO

Runs under `--mujoco`. On hardware the LEGACY layout is short 4 numbers:
booster_robot_controller.update_state fills root_pos_w and root_lin_vel_w with
zeros (only the IMU and the encoders are real), so root height and root linear
velocity arrive as lies - a constant 0 where training saw ~0.69 m and a constant
0 where it saw the speed it was asked for. Both are recoverable with stance-foot
kinematic odometry, which belongs in this file - the exported policy needs no
change for it. The MEASURABLE layout does not have this problem at all: it never
asks for either, which is the reason it exists.

The fall guard, the rate limiter and the low-pass below do NOT close that gap,
and were not built to. They bound what a fall costs and how abruptly the
actuators are driven into it; the policy fed a wrong root state still walks into
one. Nothing in this file makes the LEGACY task deployable on the floor - see
docs/08-caminho-para-hardware-mimickit.md for what would.

No measurable checkpoint exists yet, so that path is written and unexercised -
see the note on MimicKitSteeringPolicy._compute_measurable_observation.
"""

from __future__ import annotations

import math
import os
import sys

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg, PolicyCfg, PrepareStateCfg, VelocityCommandCfg
)
from booster_deploy.robots.booster import T1_23DOF_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.isaaclab import math as lab_math


from . import observation as obs_util


# The gains the policy was trained with: Booster's own kp (T1_23DOF_CFG), and
# kd = 0.0637 * kp - the ratio the G1 asset uses uniformly across all 29 of its
# joints and which is validated in the training engine. Booster's own kd is
# ~2.5x lighter on hip/knee/waist, tuned for their control loop rather than
# this one. See scripts/prepare_t1_asset.py --kd_ratio.
T1_KP = [4.0, 4.0,
         4.0, 4.0, 4.0, 4.0,
         4.0, 4.0, 4.0, 4.0,
         80.0,
         80.0, 80.0, 80.0, 80.0, 30.0, 30.0,
         80.0, 80.0, 80.0, 80.0, 30.0, 30.0]
T1_KD = [round(0.0637 * kp, 4) for kp in T1_KP]

# T1_23dof.urdf <limit effort>, i.e. the hardware maximum. T1_23DOF_CFG's own
# effort_limit is Booster's derated operating limit (knee 60 Nm vs 130.5), and
# mujoco_controller clips torque at whatever this says. The policy was trained
# against the URDF limits, so clipping at the derated ones would silently starve
# it of the torque it learned to use. Matching training is the right call for
# sim-to-sim; before any hardware run, retrain with
# `prepare_t1_asset.py --effort_source deploy` instead of raising the limits.
T1_EFFORT_URDF = [7.0, 7.0,
                  38.3, 38.3, 38.3, 38.3,
                  38.3, 38.3, 38.3, 38.3,
                  68.0,
                  98.8, 68.0, 68.0, 130.5, 73.1, 17.2,
                  98.8, 68.0, 68.0, 130.5, 73.1, 17.2]

# The pose the policy was trained to stand in, and therefore the ONLY pose the
# handover may start from. Named rather than repeated so that the controller's
# default_joint_pos and its prepare_state cannot drift apart - which is exactly
# the bug this constant was introduced to close. See PREPARE POSE below.
T1_TRAIN_POSE = [0, 0,
                 0.2, -1.3, 0, -0.5,
                 0.2, 1.3, 0, 0.5,
                 0.,
                 -0.2, 0, 0, 0.4, -0.2, 0.,
                 -0.2, 0, 0, 0.4, -0.2, 0.]


class MimicKitSteeringPolicy(Policy):
    """Steering policy: observation in, joint position targets out.

    Carries frame history and the previous action only for measurable
    checkpoints; for the legacy 200-dim ones it stays memoryless, as before.
    """

    def __init__(self, cfg: MimicKitSteeringPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        policy_path = cfg.checkpoint_path
        if not os.path.isabs(policy_path):
            policy_path = os.path.normpath(
                os.path.join(self.task_path, policy_path))
        # Printed because on hardware this is the only place the choice is
        # visible: deploy.py takes no --checkpoint, so the operator has no other
        # confirmation of WHICH policy is about to drive the robot.
        print("[mimickit_steering] policy: {}".format(policy_path))
        self._model: torch.jit.ScriptModule = torch.jit.load(
            policy_path, map_location="cpu")
        self._model.eval()

        asset_path = cfg.asset_path
        if not os.path.isabs(asset_path):
            from . import MIMICKIT_PATH
            asset_path = os.path.join(os.path.dirname(MIMICKIT_PATH), asset_path)
        self._char_model, self._key_body_ids = obs_util.build_char_model(
            asset_path, cfg.key_bodies)
        obs_util.assert_joint_order_matches(self._char_model,
                                            self.robot.cfg.joint_names)

        # resolved once, from the controller that is actually running
        self._lin_vel_is_world = obs_util.root_lin_vel_is_world(controller)
        self._face_heading = 0.0

        # Which observation a checkpoint wants is read off the module itself:
        # the measurable exports carry _frame_dim, the older 200-dim ones do
        # not. No flag to get wrong, and the 16 older policies keep working.
        self._frame_dim = getattr(self._model, "_frame_dim", None)
        if (self._frame_dim is not None):
            self._frame_dim = int(self._frame_dim)
            obs_size = int(self._model._obs_mean.shape[0])
            if (obs_size % self._frame_dim != 0):
                raise ValueError(
                    "checkpoint obs {} is not a whole number of {}-dim frames"
                    .format(obs_size, self._frame_dim))
            self._hist_steps = obs_size // self._frame_dim - 1

            num_dofs = len(self.robot.cfg.joint_names)
            # the training env measured joint offsets against its init_pose,
            # whose 23 DOF entries are all zero - see observation.py
            self._init_dof_pos = torch.zeros(num_dofs)
            self._prev_action = torch.zeros(num_dofs)
            self._frame_hist = None     # filled on the first observation

    def reset(self) -> None:
        self._face_heading = 0.0
        if (self._frame_dim is not None):
            # order matters, exactly as in the training env: zero the previous
            # action FIRST, so the frame that refills the history is the one a
            # fresh episode would see
            self._prev_action.zero_()
            self._frame_hist = None
        # The rate limiter's anchor. BaseController.run calls update_state()
        # before start(), so joint_pos here is the real measured pose at the end
        # of the prepare ramp - not an assumed one. Anchoring on the measurement
        # means that if the ramp did NOT reach the pose it was asked for, the
        # limiter walks from where the robot actually is.
        self._prev_targets = self.robot.data.joint_pos.clone()
        # seeded on the first inference rather than here: the filter's state is
        # a TARGET, and joint_pos is where the robot IS, which after the prepare
        # ramp is the same thing only if the ramp arrived
        self._filtered = None

    def _projected_gravity(self) -> torch.Tensor:
        """Gravity in the base frame; [2] ~ -1 upright, -> 0 on its side."""
        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        return lab_math.quat_apply_inverse(self.robot.data.root_quat_w, gravity_w)

    def _check_falling(self) -> None:
        """Stop the controller when the trunk tips past the guard angle.

        Ported from locomotion.py:63-70. `enable_safety_fallback` has always
        existed on PolicyCfg and defaulted to True; this task simply never read
        it, so the flag was on and did nothing - the failure mode where a safety
        switch reads as armed and is not wired to anything.

        This reads the IMU only, so it works identically on hardware and in
        MuJoCo. That matters here because the observation does NOT: the two root
        quantities this policy is missing on hardware are unavailable to any
        guard as well, so trunk attitude is the one honest fall signal the robot
        has. It is a detector, not a preventer - it fires once the fall is
        underway, and buys a controlled stop rather than a controlled recovery.
        """
        if (not self.cfg.enable_safety_fallback):
            return
        if (self._projected_gravity()[2] > self.cfg.fall_gravity_z):
            print("\nFalling detected, stopping policy for safety. "
                  "You can disable safety fallback by setting "
                  f"{self.cfg.__class__.__name__}.enable_safety_fallback "
                  "to False.")
            self.controller.stop()

    def _steering_command(self):
        command = self.controller.vel_command
        self._face_heading += command.ang_vel_yaw * self.cfg.policy_dt
        self._face_heading = math.atan2(math.sin(self._face_heading),
                                        math.cos(self._face_heading))
        return obs_util.steering_command(
            command.lin_vel_x, command.lin_vel_y, self._face_heading,
            self.cfg.tar_speed_min, self.cfg.tar_speed_max)

    def _compute_measurable_observation(self) -> torch.Tensor:
        """[current frame | H past frames, oldest first], the contract written
        on the exported module itself.

        WRITTEN AHEAD OF ITS PRODUCER. No exported checkpoint carries
        `_frame_dim` yet, and the training-side change this mirrors (MimicKit's
        measurable steering env and its `compute_proprio_frame`) is not on any
        published ref - only the soccer env has that function today. So this
        path is unexercised: it type-checks and it matches the contract as
        described, and nothing more than that has been verified. Validate it
        against the training env's own `_compute_measurable_frame` on identical
        state before trusting it, the way tests/unit/test_deploy_key_bodies.py
        validates the legacy path's forward kinematics.
        """
        data = self.robot.data
        tar_dir, tar_speed, face_dir = self._steering_command()

        frame = obs_util.compute_measurable_frame(
            root_rot_wxyz=data.root_quat_w,
            root_ang_vel_b=data.root_ang_vel_b,
            dof_pos=data.joint_pos,
            dof_vel=data.joint_vel,
            prev_action=self._prev_action,
            init_dof_pos=self._init_dof_pos,
            tar_dir=tar_dir, tar_speed=tar_speed, face_dir=face_dir)

        if (self._frame_hist is None):
            # after a reset, repeat the first measured frame instead of zeros -
            # the training env refills its history the same way, so a policy
            # never sees a transient it was not trained through
            self._frame_hist = frame.unsqueeze(0).repeat(self._hist_steps, 1)
        else:
            # roll with the CURRENT frame before assembling, so hist[-1] equals
            # the current frame - that is what the training env produces, since
            # it rolls in _update_task before recomputing the observation
            self._frame_hist = torch.cat(
                [self._frame_hist[1:], frame.unsqueeze(0)], dim=0)

        return torch.cat([frame, self._frame_hist.reshape(-1)], dim=-1)

    def compute_observation(self) -> torch.Tensor:
        # BEFORE the layout dispatch, not inside either branch: the guard reads
        # the IMU only, so it is the one part of this method that does not care
        # which observation the checkpoint wants. Putting it after the dispatch
        # is how a merge silently disarms it on one path.
        self._check_falling()

        if (self._frame_dim is not None):
            return self._compute_measurable_observation()

        data = self.robot.data

        root_quat = obs_util.quat_wxyz_to_xyzw(data.root_quat_w)
        root_lin_vel_w = obs_util.root_lin_vel_to_world(
            root_quat, data.root_lin_vel_b, self._lin_vel_is_world)

        tar_dir, tar_speed, face_dir = self._steering_command()

        return obs_util.compute_observation(
            self._char_model, self._key_body_ids,
            root_pos_w=data.root_pos_w,
            root_quat_w_xyzw=root_quat,
            root_lin_vel_w=root_lin_vel_w,
            root_ang_vel_b=data.root_ang_vel_b,
            dof_pos=data.joint_pos,
            dof_vel=data.joint_vel,
            tar_dir=tar_dir, tar_speed=tar_speed, face_dir=face_dir)

    def inference(self) -> torch.Tensor:
        obs = self.compute_observation()
        with torch.no_grad():
            # already the joint position target: unnormalized and clipped inside
            # the exported module. Do not add a default pose or an action scale.
            dof_targets = self._model(obs)
        dof_targets = self._rate_limit(self._lowpass(dof_targets))
        if (self._frame_dim is not None):
            # prev_action is fed back AFTER the filters, i.e. what was actually
            # commanded - not the raw module output. The training env stores as
            # prev_action the target it applied, so the faithful mirror of it is
            # the target this file applies. With the shipped defaults the two
            # are the same object anyway (low-pass off, rate limit unclipped
            # returns its input), so this is bit-exact in the configuration the
            # checkpoint was validated in; it only diverges when a filter is
            # deliberately switched on, and there the commanded value is the
            # honest one to report. Not a rescaling either way.
            self._prev_action = dof_targets.detach().clone()
        return dof_targets

    def _lowpass(self, dof_targets: torch.Tensor) -> torch.Tensor:
        """First-order low-pass on the position target. Off by default.

        Aimed at a specific measured problem: this checkpoint's target does not
        move smoothly, it DITHERS. Over 30 s at vx=1.0 the commanded target
        reverses direction 16.7 times per second at the hip and 10-13 at the
        ankle, with a median step of 0.20 rad at the hip and 0.10 rad at the
        ankle - at a 30 Hz update rate, where 15 reversals/s is the most a
        non-aliased signal can show. The reference is oscillating at its own
        control rate.

        On hardware that is a buzz, and the ankle is where it is loudest: its
        PD loop's natural frequency under this task's gains is ~3.3 Hz in pitch,
        so a ~12 Hz dither is a command the loop cannot follow and can only
        fight, and the T1's ankle pitch and roll share a parallel linkage driven
        by one motor pair, so the two dithers beat against each other in it.

        A rate limit is the wrong instrument for this - it bounds amplitude,
        and the dither's problem is its frequency. A low-pass attacks the
        frequency directly: the gait's own fundamental is 1-2 Hz, so a cutoff
        of 5-8 Hz passes the motion and removes the dither.

        Kept OFF by default because it changes the closed loop the checkpoint
        was validated in, and because the right fix is upstream: MimicKit's
        export has no action-smoothness term, so the chatter is baked into the
        checkpoint. This is a diagnostic instrument, not a solution - if a run
        only works with it on, that is a finding about the checkpoint to carry
        into retraining, not a setting to leave enabled and forget.
        """
        fc = self.cfg.target_lowpass_hz
        if (fc is None):
            return dof_targets
        # alpha for a first-order filter at cutoff fc, sampled at policy_dt
        tau = 1.0 / (2.0 * math.pi * fc)
        alpha = self.cfg.policy_dt / (tau + self.cfg.policy_dt)
        if (self._filtered is None):
            self._filtered = dof_targets.clone()
        else:
            self._filtered = self._filtered + alpha * (dof_targets - self._filtered)
        return self._filtered

    def _rate_limit(self, dof_targets: torch.Tensor) -> torch.Tensor:
        """Bound how far a single step may move the position target.

        The export's internal clip bounds the target's VALUE to the asset's
        joint limits; nothing bounds its DERIVATIVE. A position-target policy
        commands through a stiff PD loop, so a one-step jump of the full joint
        range is a torque spike limited only by effort_limit - which this task
        deliberately raises to the URDF maxima (T1_EFFORT_URDF, knee 130.5 Nm
        against the firmware's derated 60). In MuJoCo that costs a stumble. On
        hardware it is commanded into real actuators.

        The limit is a slew rate in rad/s, converted to a per-step delta with
        the policy period, so it means the same thing if policy_dt changes.

        Deliberately NOT a fix for a bad observation: a policy fed the wrong
        root state still walks into a fall, just less abruptly. And read
        max_target_rate's note before trusting it to do more than that - for
        THIS policy it is a sanity ceiling, not a gait guard.

        The unclipped path returns the input tensor itself rather than
        `prev + (target - prev)`. Those are equal in exact arithmetic and not in
        floating point, and the ~1e-7 residual is not harmless here: walking is
        chaotic, and a 30 s rollout amplified it into a 2-point difference in
        tracking (0.883 vs 0.905) with zero steps actually clipped. A guard that
        is off must be bit-exact off, or every measurement it touches drifts for
        a reason nobody will find.
        """
        if (self.cfg.max_target_rate is None):
            return dof_targets
        max_delta = self.cfg.max_target_rate * self.cfg.policy_dt
        delta = dof_targets - self._prev_targets
        if (bool(delta.abs().max() <= max_delta)):
            self._prev_targets = dof_targets
            return dof_targets
        limited = self._prev_targets + torch.clamp(delta, -max_delta, max_delta)
        self._prev_targets = limited
        return limited


@configclass
class MimicKitSteeringPolicyCfg(PolicyCfg):
    constructor = MimicKitSteeringPolicy

    # THE checkpoint for this task, and the one validated in sim. Relative paths
    # resolve against the TASK directory (see MimicKitSteeringPolicy.__init__),
    # hence the ../../ to reach the repo-root models/ that README.md prescribes.
    #
    # This default is load-bearing on hardware in a way it is not in sim:
    # scripts/deploy.py has no --checkpoint flag, so on the robot this field is
    # the ONLY thing selecting the policy. teleop.py and sim_viability.py can
    # both override it from the command line, which is exactly how sim and robot
    # drifted apart before - sim_viability defaulted to t1_wrturn_s1 while this
    # said t1_steering.pt, so "validated in sim" and "running on the robot"
    # silently meant two different files. Keep this pointing at whatever the
    # robot is meant to run, and let the sim tools be the ones that deviate.
    checkpoint_path: str = "../../models/t1_wrturn_s1/policy.pt"
    # relative to the MimicKit checkout resolved in this package's __init__
    asset_path: str = "data/assets/t1/t1.xml"
    # data/envs/wamp_t1_steering_env.yaml
    key_bodies: list[str] = [
        "left_ankle_roll_link", "right_ankle_roll_link", "aahead_pitch_link",
        "left_elbow_yaw_link", "right_elbow_yaw_link",
    ]
    tar_speed_min: float = 0.0
    tar_speed_max: float = 2.5
    policy_dt: float = 1.0 / 30.0

    # Trunk attitude at which the fall guard stops the controller, as
    # projected_gravity[2]: -1 is upright, 0 is on its side. -0.5 is ~60 deg off
    # vertical, the same threshold locomotion.py uses and the same default
    # scripts/sim_viability.py already scores falls with, so the sim number and
    # the robot's behaviour refer to one definition.
    fall_gravity_z: float = -0.5

    # Slew rate ceiling for the position target [rad/s]; None disables it.
    #
    # 120 is a SANITY CEILING, not a gait guard, and the distinction is the
    # whole point of this note. Measured demand of this policy, 30 s rollouts,
    # limiter off, worst joint per step:
    #
    #   vx      p50     p95     p99     max        (rad/s)
    #   0.5    14.3    30.8    45.5    61.1
    #   1.0    14.9    29.0    37.7    45.7
    #   2.0    30.7    58.5    71.1    93.3
    #   2.5    28.7    63.4    80.4    91.7
    #
    # The nominal gait routinely slews the target at tens of rad/s: the MEDIAN
    # step moves the worst joint ~0.5 rad (29 deg). A position target that
    # active leaves no gap between nominal and pathological for a rate limit to
    # sit in - a full-range jump, the thing worth catching, is ~75 rad/s and the
    # gait already peaks at 93. So 120 clears the measured envelope (0% of steps
    # clipped at every commanded speed) and catches only output that is out of
    # family entirely, e.g. a corrupted checkpoint.
    #
    # Do not tighten it hoping for more protection. Measured, at vx=1.0 over
    # 30 s: 60 -> 0% clipped, tracking 90.5%; 30 -> 4% clipped, 88.5%; 10 -> 83%
    # of steps clipped and 85.7%. The tight settings do not fail loudly, they
    # quietly reshape the closed loop the checkpoint was validated in - which is
    # worse, because the run still looks like it worked.
    #
    # The real lesson is about the policy, not the limit: MimicKit's export has
    # no action-smoothness term, so target chatter is baked into the checkpoint.
    # Bounding the actuator transient properly means retraining with one, not
    # filtering it here. See docs/11 section 5.
    max_target_rate: float | None = 120.0

    # Cutoff [Hz] of a first-order low-pass on the position target; None = off,
    # which is the default. See MimicKitSteeringPolicy._lowpass for what this is
    # for and why it does not ship enabled. Reachable on the robot with
    # `deploy.py --target-lowpass 6`, which is where it is meant to be used.
    target_lowpass_hz: float | None = None


@configclass
class T1MimicKitSteeringControllerCfg(ControllerCfg):
    robot = T1_23DOF_CFG.replace(  # type: ignore
        default_joint_pos=T1_TRAIN_POSE,
        joint_stiffness=T1_KP,
        joint_damping=T1_KD,
        effort_limit=T1_EFFORT_URDF,
        # PREPARE STATE - the POSE is overridden here, the GAINS deliberately
        # are not. The asymmetry is the whole point, and it was learned the
        # expensive way: an earlier version of this file set both to the
        # policy's values, and on the T1 the robot sank to the floor during the
        # ramp, joints tracking their targets to within a degree the whole way
        # down. Measured afterwards, holding this pose statically for 3 s with
        # the derated torque ceiling:
        #
        #   leg kp 350 (Booster's prepare)   base stays at 0.678 m
        #   leg kp 200 (locomotion's)        base stays at 0.676 m
        #   leg kp  80 (this policy's)       base ends at 0.025 m - on the floor
        #
        # Isolated: the ankles are not the cause (stiffening them alone changes
        # nothing); the hip and knee gain is. Below ~200 the leg cannot hold the
        # robot up with a fixed target.
        #
        # WHY THAT IS NOT A VERDICT ON THE POLICY. Statically holding a pose and
        # running a policy are different control problems. A PD on a fixed
        # target has to carry the whole inverted pendulum on joint stiffness
        # alone; the policy re-plans its target 30 times a second and balances
        # actively, which is why kp=80 walks perfectly well in MuJoCo. The two
        # regimes need different gains, and Booster's split - stiff to stand,
        # soft to run - is correct rather than an oversight to be smoothed away.
        # tasks/locomotion does exactly the same thing: its own gains are
        # 200/50 and it still ramps under the inherited 350.
        #
        # So the gains are inherited from T1_23DOF_CFG and only the pose is
        # replaced, because only the pose has to match what the first inference
        # step assumes: this policy's output is an ABSOLUTE position target, so
        # it commands its pose rather than correcting toward it, and Booster's
        # prepare pose (legs at [-0.1, 0, 0, 0.2, -0.1, 0]) is half this crouch.
        #
        # The gain step at handover (350 -> 80) is therefore real and stays.
        # It is the same step locomotion lives with, one factor larger.
        prepare_state=PrepareStateCfg(
            stiffness=list(T1_23DOF_CFG.prepare_state.stiffness),
            damping=list(T1_23DOF_CFG.prepare_state.damping),
            joint_pos=list(T1_TRAIN_POSE),
        ),
    )
    # training rates: engine_config.yaml control_freq 30, sim_freq 120
    policy_dt: float = 1.0 / 30.0
    vel_command: VelocityCommandCfg = VelocityCommandCfg(
        vx_max=2.5, vy_max=1.0, vyaw_max=1.0,
    )
    policy: MimicKitSteeringPolicyCfg = MimicKitSteeringPolicyCfg()

    def __post_init__(self):
        # decimation FIRST: ControllerCfg.__post_init__ derives
        # physics_dt = policy_dt / decimation, so changing decimation after it
        # leaves physics_dt stale and the effective control rate becomes
        # decimation * (policy_dt / 10) - 75 Hz instead of 30. Silent, and it
        # changes the closed loop the policy was trained in.
        self.mujoco.decimation = 4          # 30 Hz * 4 = 120 Hz physics
        super().__post_init__()
        # MujocoControllerCfg defaults to 0.60 m, below this robot's standing
        # waist height; 0.69 is the training env's init_pose z
        # (data/envs/wamp_t1_steering_env.yaml).
        self.mujoco.init_pos = [0.0, 0.0, 0.69]
        self.policy.policy_dt = self.policy_dt
