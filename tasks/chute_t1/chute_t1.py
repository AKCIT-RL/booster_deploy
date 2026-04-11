from __future__ import annotations
from dataclasses import MISSING

from booster_deploy.controllers.controller_cfg import ControllerCfg, MujocoControllerCfg
from booster_deploy.robots.booster import T1_23DOF_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from ..beyond_mimic.beyond_mimic import (
    BeyondMimicPolicy,
    BeyondMimicPolicyCfg,
)


class T1BeyondMimicPolicy(BeyondMimicPolicy):
    """T1 subclass so that task_path resolves to tasks/t1_mj_dance_002/."""
    pass


@configclass
class T1BeyondMimicPolicyCfg(BeyondMimicPolicyCfg):
    constructor = T1BeyondMimicPolicy
    checkpoint_path: str = MISSING
    motion_path: str = MISSING
    anchor_body_name: str = "Trunk"


@configclass
class T1MjDance002ControllerCfg(ControllerCfg):
    # Joint order (23 DOF, URDF order):
    # AAHead_yaw, Head_pitch,
    # Left_Shoulder_Pitch, Left_Shoulder_Roll, Left_Elbow_Pitch, Left_Elbow_Yaw,
    # Right_Shoulder_Pitch, Right_Shoulder_Roll, Right_Elbow_Pitch, Right_Elbow_Yaw,
    # Waist,
    # Left_Hip_Pitch, Left_Hip_Roll, Left_Hip_Yaw, Left_Knee_Pitch, Left_Ankle_Pitch, Left_Ankle_Roll,
    # Right_Hip_Pitch, Right_Hip_Roll, Right_Hip_Yaw, Right_Knee_Pitch, Right_Ankle_Pitch, Right_Ankle_Roll
    robot = T1_23DOF_CFG.replace(  # type: ignore
        default_joint_pos=[
            0.0,  0.0,                            # head
            0.2, -1.3,  0.0, -0.5,               # left arm
            0.2,  1.3,  0.0,  0.5,               # right arm
            0.0,                                  # waist
           -0.2,  0.0,  0.0,  0.4, -0.2,  0.0,  # left leg
           -0.2,  0.0,  0.0,  0.4, -0.2,  0.0,  # right leg
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
        effort_limit=[
            7.0,  7.0,                                     # head
            18.0, 18.0, 18.0, 18.0,                       # left arm
            18.0, 18.0, 18.0, 18.0,                       # right arm
            25.0,                                          # waist
            45.0, 25.0, 25.0, 60.0, 24.0, 15.0,          # left leg (Pitch, Roll, Yaw, Knee, Ank_P, Ank_R)
            45.0, 25.0, 25.0, 60.0, 24.0, 15.0,          # right leg
        ],
    )
    enable_velocity_commands = False
    policy: T1BeyondMimicPolicyCfg = T1BeyondMimicPolicyCfg()
    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.70],
        visualize_reference_ghost=True,
    )
