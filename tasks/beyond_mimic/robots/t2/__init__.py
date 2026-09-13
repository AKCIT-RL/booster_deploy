"""T2 BeyondMimic task configurations and motion assets."""

from booster_deploy.controllers.controller_cfg import (
    BoosterRobotControllerCfg,
    ControllerCfg,
    MujocoControllerCfg,
)
from booster_deploy.robots.t2 import T2_31DOF_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task

from ...beyond_mimic import BeyondMimicPolicyCfg


@configclass
class T2BeyondMimicControllerCfg(ControllerCfg):
    """Base controller configuration for 31-DOF T2 motion policies.

    ``booster.exit_mode`` controls the robot mode entered after the custom
    controller exits. It can be configured as ``"walking"`` (default) or
    ``"damping"``.
    """

    booster: BoosterRobotControllerCfg = BoosterRobotControllerCfg(
        exit_mode="walking"
    )

    robot = T2_31DOF_CFG.replace(
        joint_stiffness=[
            18.214394, 18.214394,
            42.952518, 34.614676, 34.614676, 34.614676, 18.214394, 18.214394, 18.214394,
            42.952518, 34.614676, 34.614676, 34.614676, 18.214394, 18.214394, 18.214394,
            99.837519, 132.424295, 70.997986,
            70.997986, 70.997986, 70.997986, 70.997986, 90.690730, 33.693992,
            70.997986, 70.997986, 70.997986, 70.997986, 90.690730, 33.693992,
        ],
        joint_damping=[
            1.449455, 1.449455,
            3.418053, 2.754549, 2.754549, 2.754549, 1.449455, 1.449455, 1.449455,
            3.418053, 2.754549, 2.754549, 2.754549, 1.449455, 1.449455, 1.449455,
            2.754549, 2.754549, 5.649840,
            5.649840, 5.649840, 5.649840, 5.649840, 2.754549, 2.754549,
            5.649840, 5.649840, 5.649840, 5.649840, 2.754549, 2.754549,
        ],
        effort_limit=[
            22., 22.,
            74., 60., 60., 60., 22., 22., 22.,
            74., 60., 60., 60., 22., 22., 22.,
            138., 144., 135.,
            135., 135., 135., 135., 132., 72.,
            135., 135., 135., 135., 132., 72.,
        ],
    )
    enable_velocity_commands = False
    policy: BeyondMimicPolicyCfg = BeyondMimicPolicyCfg(
        anchor_body_name="trunk",
    )
    mujoco = MujocoControllerCfg(
        init_pos=[0.0, 0.0, 1.0],
        visualize_reference_ghost=True,
        decimation=4,
    )


@configclass
class T2DanceControllerCfg(T2BeyondMimicControllerCfg):
    """T2 Dance controller configuration."""

    def __post_init__(self):
        super().__post_init__()
        self.policy.motion_path = (
            "robots/t2/motions/t2_dance.npz"
        )
        self.policy.checkpoint_path = (
            "robots/t2/models/"
            "t2_dance.onnx"
        )
        self.policy.stop_at_motion_end = True
        self.booster.exit_mode = "walking"
        self.mujoco.init_pos = [0.0, 0.0, 1.05]
        self.robot.effort_limit = [
            22., 22.,
            74., 60., 60., 60., 22., 22., 22.,
            74., 60., 60., 60., 22., 22., 22.,
            138., 144., 135.,
            135., 135., 135., 135., 132., 72.,
            135., 135., 135., 135., 132., 72.,
        ]
        # PD gains are selected from the parallel-actuated joint armature ``J_eq``:
        #   omega_n = 2 * pi * natural_freq_hz
        #   joint_stiffness (Kp) = J_eq * omega_n**2
        #   joint_damping (Kd) = 2 * damping_ratio * J_eq * omega_n
        # The values below use natural_freq_hz=8 and damping_ratio=2.
        self.robot.joint_stiffness = [
            18.214394, 18.214394,  # head
            42.952518, 34.614676, 34.614676, 34.614676,18.214394, 18.214394, 18.214394,  # left arm
            42.952518, 34.614676, 34.614676, 34.614676,18.214394, 18.214394, 18.214394,  # right arm
            99.837519, 132.424295, 70.997986,  # waist (pitch, roll, yaw)
            70.997986, 70.997986, 70.997986, 70.997986,90.690730, 33.693992,  # left leg
            70.997986, 70.997986, 70.997986, 70.997986,90.690730, 33.693992,  # right leg
        ]
        self.robot.joint_damping = [
            1.449455, 1.449455,  # head
            3.418053, 2.754549, 2.754549, 2.754549, 1.449455, 1.449455, 1.449455,  # left arm
            3.418053, 2.754549, 2.754549, 2.754549, 1.449455, 1.449455, 1.449455,  # right arm
            # 7.944818, 10.537990, 5.649840,  # waist (pitch, roll, yaw)
            2.754549, 2.754549, 5.649840,  #waist motor-side damping
            # 5.649840, 5.649840, 5.649840, 5.649840, 7.216939, 2.681282,  # left leg
            5.649840, 5.649840, 5.649840, 5.649840, 2.754549, 2.754549,  # leg - motor-side damping
            # 5.649840, 5.649840, 5.649840, 5.649840, 7.216939, 2.681282,  # right leg
            5.649840, 5.649840, 5.649840, 5.649840, 2.754549, 2.754549,  # right - motor-side damping
        ]
        self.policy.fixed_action_scale = [
            0.25 * effort / stiffness
            for effort, stiffness in zip(
                self.robot.effort_limit,
                self.robot.joint_stiffness,
            )
        ]


register_task("t2_dance", T2DanceControllerCfg())
