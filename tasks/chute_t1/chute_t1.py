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
        # Gains derived from booster_train commit 651b7a5 — BoosterJointCfg formula:
        #   kp = armature * (2π * natural_freq)²   (natural_freq=10 Hz)
        #   kd = 2 * damping_ratio * armature * (2π * natural_freq)   (damping_ratio=2.0)
        # Motors per joint group:
        #   Head        → DM4310  (arm=0.0018,     kp=7.11,   kd=0.45)
        #   Arms        → E4310   (arm=0.0282528,  kp=111.54, kd=7.10)
        #   Waist/HipR/Y→ E6408   (arm=0.0478125,  kp=188.76, kd=12.02)
        #   Hip_Pitch   → E8112   (arm=0.0523908,  kp=206.83, kd=13.17)
        #   Knee_Pitch  → E8116   (arm=0.0636012,  kp=251.09, kd=15.98)
        #   Ankle       → E4315×2 (arm=0.0679104,  kp=268.10, kd=17.07)
        joint_stiffness=[
            7.11, 7.11,                                          # head   (DM4310)
            111.54, 111.54, 111.54, 111.54,                     # left arm  (E4310)
            111.54, 111.54, 111.54, 111.54,                     # right arm (E4310)
            188.76,                                              # waist  (E6408)
            206.83, 188.76, 188.76, 251.09, 268.10, 268.10,    # left leg
            206.83, 188.76, 188.76, 251.09, 268.10, 268.10,    # right leg
        ],
        # kd applied as passive XML joint damping (implicit, via MuJoCo solver).
        # These values are also sent to the real robot hardware controller.
        # In MuJoCo sim, ctrl_step zeros kd in the explicit PD to avoid double-counting.
        joint_damping=[
            0.45, 0.45,                                          # head   (DM4310)
            7.10, 7.10, 7.10, 7.10,                             # left arm  (E4310)
            7.10, 7.10, 7.10, 7.10,                             # right arm (E4310)
            12.02,                                               # waist  (E6408)
            13.17, 12.02, 12.02, 15.98, 17.07, 17.07,          # left leg
            13.17, 12.02, 12.02, 15.98, 17.07, 17.07,          # right leg
        ],
        effort_limit=[
            7.0,  7.0,                                           # head   (DM4310)
            38.3, 38.3, 38.3, 38.3,                             # left arm  (E4310)
            38.3, 38.3, 38.3, 38.3,                             # right arm (E4310)
            68.0,                                                # waist  (E6408)
            96.0, 68.0, 68.0, 130.0, 76.0, 76.0,               # left leg
            96.0, 68.0, 68.0, 130.0, 76.0, 76.0,               # right leg
        ],
    )
    enable_velocity_commands = False
    policy: T1BeyondMimicPolicyCfg = T1BeyondMimicPolicyCfg()
    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.70],
        visualize_reference_ghost=True,
    )
