from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    VelocityCommandCfg,
)
from booster_deploy.robots.t1 import T1_23DOF_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from ...locomotion import T1LocomotionPolicyCfg


@configclass
class T1WalkControllerCfg(ControllerCfg):
    robot = T1_23DOF_CFG.replace(  # type: ignore
        default_joint_pos=[
            0, 0,
            0.2, -1.3, 0, -0.5,
            0.2, 1.3, 0, 0.5,
            0.,
            -0.2, 0, 0, 0.4, -0.2, 0.,
            -0.2, 0, 0, 0.4, -0.2, 0.,
        ],
        joint_stiffness=[
            4.0, 4.0,
            50.0, 50.0, 50.0, 50.0,
            50.0, 50.0, 50.0, 50.0,
            200.,
            200.0, 200.0, 200.0, 200.0, 50.0, 50.0,
            200.0, 200.0, 200.0, 200.0, 50.0, 50.0,
        ],
        joint_damping=[
            1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            5.0,
            5.0, 5.0, 5.0, 5.0, 2.0, 2.0,
            5.0, 5.0, 5.0, 5.0, 2.0, 2.0,
        ],
    )
    vel_command: VelocityCommandCfg = VelocityCommandCfg(
        vx_max=1.0,
        vy_max=1.0,
        vyaw_max=1.0,
    )
    policy: T1LocomotionPolicyCfg = T1LocomotionPolicyCfg(
        obs_dof_vel_scale=1.0,
        policy_joint_names=[
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "waist_yaw_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "left_elbow_pitch_joint",
            "right_elbow_pitch_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "left_elbow_yaw_joint",
            "right_elbow_yaw_joint",
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


@configclass
class T1WalkControllerCfg1(T1WalkControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "robots/t1/models/t1_walk.pt"


register_task("t1_walk", T1WalkControllerCfg1())
