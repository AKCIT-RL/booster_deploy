"""T2 full-body RL locomotion deployment task."""

from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
    VelocityCommandCfg,
)
from booster_deploy.robots.t2 import T2_31DOF_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from ...locomotion import T2LocomotionPolicyCfg


@configclass
class T2WalkControllerCfg(ControllerCfg):
    robot = T2_31DOF_CFG.replace(
        default_joint_pos=[
            0.0, 0.0,
            0.2, -1.3, 0.0, -0.5, 0.0, 0.0, 0.0,
            0.2, 1.3, 0.0, 0.5, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        joint_stiffness=[
            20.0, 30.0,
            67.1133, 54.0854, 54.0854, 54.0854, 2.0, 2.0, 2.0,
            67.1133, 54.0854, 54.0854, 54.0854, 2.0, 2.0, 2.0,
            155.9961, 206.9129, 110.9343,
            110.9343, 110.9343, 110.9343, 110.9343, 141.7042, 52.6468,
            110.9343, 110.9343, 110.9343, 110.9343, 141.7042, 52.6468,
        ],
        joint_damping=[
            0.5, 1.0,
            2.1362, 1.7215, 1.7215, 1.7215, 0.1, 0.1, 0.1,
            2.1362, 1.7215, 1.7215, 1.7215, 0.1, 0.1, 0.1,
            1.7215, 1.7215, 3.5311,
            3.5311, 3.5311, 3.5311, 3.5311, 1.7215, 1.7215,
            3.5311, 3.5311, 3.5311, 3.5311, 1.7215, 1.7215,
        ],
    )
    vel_command: VelocityCommandCfg = VelocityCommandCfg(
        # Direction-specific factors: normalized vx in [-1, 1] maps to
        # Normalized vx maps to [-0.7, 3.0] m/s.
        vx_forward_max=3.0,
        vx_backward_max=0.7,
        vy_max=0.8,
        vyaw_max=2.0,
    )
    policy: T2LocomotionPolicyCfg = T2LocomotionPolicyCfg(
        obs_dof_vel_scale=1.0,
        policy_joint_names=[
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "waist_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "waist_roll_joint",
            "left_elbow_pitch_joint",
            "right_elbow_pitch_joint",
            "waist_yaw_joint",
            "left_elbow_yaw_joint",
            "right_elbow_yaw_joint",
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "left_knee_pitch_joint",
            "right_knee_pitch_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
        ],
    )
    mujoco: MujocoControllerCfg = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 1.05],
    )


@configclass
class T2WalkTaskCfg(T2WalkControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "robots/t2/models/t2_walk.pt"


register_task("t2_walk", T2WalkTaskCfg())
