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
        # Gains — per-group natural frequency, kp = arm*(2π*wn)², kd = 4*arm*(2π*wn),
        # damping_ratio=2.0. wn chosen so action_scale α=0.25*τ/Kp matches G1:
        #   Head        → DM4310  (arm=0.0018,     wn=6Hz, kp= 2.56, kd= 0.27)
        #   Arms        → E4310   (arm=0.0282528,  wn=6Hz, kp=40.15, kd= 4.26)
        #   Waist       → E6408   (arm=0.0478125,  wn=6Hz, kp=67.95, kd= 7.21)
        #   Hip_Pitch   → E8112   (arm=0.0523908,  wn=5Hz, kp=51.71, kd= 6.59)
        #   Hip_Roll/Yaw→ E6408   (arm=0.0478125,  wn=5Hz, kp=47.19, kd= 6.01)
        #   Knee_Pitch  → E8116   (arm=0.0636012,  wn=6Hz, kp=90.39, kd= 9.59)
        #   Ankle       → E4315×2 (arm=0.0679104,  wn=4Hz, kp=42.90, kd= 6.83)
        joint_stiffness=[
             2.56,  2.56,                                        # head   (DM4310,  wn=6Hz)
            40.15, 40.15, 40.15, 40.15,                         # left arm  (E4310, wn=6Hz)
            40.15, 40.15, 40.15, 40.15,                         # right arm (E4310, wn=6Hz)
            67.95,                                               # waist  (E6408,   wn=6Hz)
            51.71, 47.19, 47.19, 90.39, 42.90, 42.90,          # left leg
            51.71, 47.19, 47.19, 90.39, 42.90, 42.90,          # right leg
        ],
        # kd applied as passive XML joint damping (implicit, via MuJoCo solver).
        # These values are also sent to the real robot hardware controller.
        # In MuJoCo sim, ctrl_step zeros kd in the explicit PD to avoid double-counting.
        # MUST match damping= attributes in T1_23dof.xml.
        joint_damping=[
            0.27, 0.27,                                          # head   (DM4310,  wn=6Hz)
            4.26, 4.26, 4.26, 4.26,                             # left arm  (E4310, wn=6Hz)
            4.26, 4.26, 4.26, 4.26,                             # right arm (E4310, wn=6Hz)
            7.21,                                                # waist  (E6408,   wn=6Hz)
            6.59, 6.01, 6.01, 9.59, 6.83, 6.83,                # left leg  (per-group wn)
            6.59, 6.01, 6.01, 9.59, 6.83, 6.83,                # right leg (per-group wn)
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
