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

  memory           LocomotionPolicy stacks 10 frames. This one is memoryless -
                   MimicKit's steering observation is a single frame - so there
                   is no obs_history.

  observation      53 privileged-free numbers there, 200 here, of a completely
                   different kind (root height, tan-norm rotations, key body
                   positions). observation.py builds them by calling MimicKit's
                   own functions.

RATES AND GAINS ARE PART OF THE POLICY

The checkpoint was trained at control 30 Hz / physics 120 Hz (the run's
engine_config.yaml) with the PD gains below. A position-target policy is a
closed loop with its PD controller: change kp, kd or the control period and the
same targets produce different motion. Hence policy_dt = 1/30 and decimation 4
rather than the 50 Hz / 500 Hz defaults, and hence the gains are restated here
instead of inherited from T1WalkControllerCfg, whose 200/5 belong to Booster's
own policy.

WHAT THIS DOES NOT DO YET

Runs under `--mujoco`. On hardware, 4 of the 200 dimensions are unavailable as
booster_deploy ships: booster_robot_controller.update_state fills root_pos_w and
root_lin_vel_w with zeros (only the IMU and the encoders are real), so root
height and root linear velocity would arrive as lies. Both are recoverable with
stance-foot kinematic odometry, which belongs in this file - the exported policy
needs no change for it.
"""

from __future__ import annotations

import math
import os
import sys

import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import (
    ControllerCfg, PolicyCfg, VelocityCommandCfg
)
from booster_deploy.robots.booster import T1_23DOF_CFG
from booster_deploy.utils.isaaclab.configclass import configclass


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


class MimicKitSteeringPolicy(Policy):
    """Memoryless steering policy: 200-dim observation in, joint targets out."""

    def __init__(self, cfg: MimicKitSteeringPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        policy_path = cfg.checkpoint_path
        if not os.path.isabs(policy_path):
            policy_path = os.path.join(self.task_path, policy_path)
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

    def reset(self) -> None:
        self._face_heading = 0.0

    def compute_observation(self) -> torch.Tensor:
        data = self.robot.data

        root_quat = obs_util.quat_wxyz_to_xyzw(data.root_quat_w)
        root_lin_vel_w = obs_util.root_lin_vel_to_world(
            root_quat, data.root_lin_vel_b, self._lin_vel_is_world)

        command = self.controller.vel_command
        self._face_heading += command.ang_vel_yaw * self.cfg.policy_dt
        self._face_heading = math.atan2(math.sin(self._face_heading),
                                        math.cos(self._face_heading))

        tar_dir, tar_speed, face_dir = obs_util.steering_command(
            command.lin_vel_x, command.lin_vel_y, self._face_heading,
            self.cfg.tar_speed_min, self.cfg.tar_speed_max)

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
        return dof_targets


@configclass
class MimicKitSteeringPolicyCfg(PolicyCfg):
    constructor = MimicKitSteeringPolicy
    checkpoint_path: str = "models/t1_steering.pt"
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


@configclass
class T1MimicKitSteeringControllerCfg(ControllerCfg):
    robot = T1_23DOF_CFG.replace(  # type: ignore
        default_joint_pos=[
            0, 0,
            0.2, -1.3, 0, -0.5,
            0.2, 1.3, 0, 0.5,
            0.,
            -0.2, 0, 0, 0.4, -0.2, 0.,
            -0.2, 0, 0, 0.4, -0.2, 0.
        ],
        joint_stiffness=T1_KP,
        joint_damping=T1_KD,
        effort_limit=T1_EFFORT_URDF,
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
