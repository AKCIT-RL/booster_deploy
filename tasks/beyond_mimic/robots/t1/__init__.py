"""T1 BeyondMimic task package."""
from __future__ import annotations

from booster_deploy.controllers.controller_cfg import (
    ControllerCfg,
    MujocoControllerCfg,
)
from booster_deploy.robots.t1 import T1_23DOF_CFG as T1_CFG
from booster_deploy.utils.registry import register_task
from booster_deploy.utils.isaaclab.configclass import configclass

from ...beyond_mimic import BeyondMimicPolicyCfg


@configclass
class T1BeyondMimicControllerCfg(ControllerCfg):
    robot = T1_CFG
    anchor_body_name = "trunk"
    enable_velocity_commands = False
    policy: BeyondMimicPolicyCfg = BeyondMimicPolicyCfg(
        # dance
        action_scale_factors=[
            0.25,
            0.25,
            0.5,
            0.5,
            0.5,
            0.7,
            0.5,
            0.5,
            0.5,
            0.7,
            0.16,
            0.35,
            0.35,
            0.28,
            0.3,
            0.17,
            0.28,
            0.35,
            0.35,
            0.28,
            0.3,
            0.17,
            0.28,
        ]
    )
    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 0.6442],
        visualize_reference_ghost=True,
    )


@configclass
class T1MotionTrackingControllerCfg(T1BeyondMimicControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = "robots/t1/motions/t1_dance.npz"
        self.policy.checkpoint_path = "robots/t1/models/t1_dance.pt"


register_task("t1_motion_tracking", T1MotionTrackingControllerCfg())
