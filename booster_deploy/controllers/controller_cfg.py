from typing import Callable, List, Optional
from dataclasses import MISSING
import torch

from ..utils.isaaclab.configclass import configclass


@configclass
class PrepareStateCfg:
    stiffness: List[float] = MISSING
    damping: List[float] = MISSING
    joint_pos: List[float] = MISSING


@configclass
class MujocoControllerCfg:
    init_pos: List[float] = [0.0, 0.0, 0.6]
    init_quat: List[float] = [1.0, 0.0, 0.0, 0.0]
    decimation: int = 10
    # physics_dt will automatically be set by ControllerCfg
    physics_dt: float = None  # type: ignore
    log_states: Optional[str] = None
    visualize_reference_ghost: bool = False
    ghost_rgba: List[float] = [0.2, 0.8, 0.2, 0.25]
    # Video recording (requires opencv-python).
    # If set, recording starts automatically; otherwise toggle with R key.
    video_path: Optional[str] = None
    video_width: int = 1280
    video_height: int = 720
    # Steps of actuation delay: dof_targets are buffered and the oldest applied.
    # mujoco_controller.run() reads this unconditionally; without it every task
    # raises AttributeError on start. 0 reproduces the undelayed behaviour.
    action_delay_steps: int = 0


@configclass
class BoosterRobotControllerCfg:
    low_state_dt: float = 0.002
    metrics_max_events: int = 2000


@configclass
class RobotCfg:
    name: str = MISSING

    joint_names: list[str] = MISSING
    body_names: list[str] = MISSING

    sim_joint_names: list[str] = MISSING
    sim_body_names: list[str] = MISSING

    joint_stiffness: List[float] = MISSING
    joint_damping: List[float] = MISSING

    default_joint_pos: List[float] = MISSING
    effort_limit: List[float] = MISSING

    # T-N curve parameters (BoosterDelayedPDActuator).
    # velocity_limit: no-load speed per joint [rad/s]. None disables T-N curve.
    # knee_point_velocity: speed below which full torque is available [rad/s].
    velocity_limit: Optional[List[float]] = None
    knee_point_velocity: Optional[List[float]] = None

    mjcf_path: str = MISSING

    prepare_state: PrepareStateCfg = MISSING

    def __post_init__(self):
        assert (
            len(self.joint_names)
            == len(self.joint_stiffness)
            == len(self.joint_damping)
            == len(self.default_joint_pos)
            == len(self.effort_limit)
        )
        if self.velocity_limit is not None:
            assert len(self.velocity_limit) == len(self.joint_names)
        if self.knee_point_velocity is not None:
            assert len(self.knee_point_velocity) == len(self.joint_names)


@configclass
class VelocityCommandCfg:
    vx_max: float = 1.0
    vy_max: float = 1.0
    vyaw_max: float = 1.0

    # Standing command applied with the operator's input at neutral, in the
    # physical units of the command (m/s, rad/s) - NOT normalized like the
    # joystick axes. VelocityCommand seeds itself from these, so they are also
    # the command the very first policy step sees. Set by deploy.py's
    # --vx/--vy/--vyaw; 0.0 leaves every path bit-identical to not having them.
    #
    # Why a baseline and not only an initial value: on hardware the command is
    # not state this process owns. _low_state_handler rewrites it from the
    # remote on every /low_state callback (~500 Hz) and update_vel_command
    # re-reads it every policy step, so a value merely seeded at construction
    # is gone within ~2 ms and the flag would be a silent no-op there. As a
    # baseline it survives, the remote keeps full authority to add to it or
    # cancel it, and the sum stays clipped to the *_max envelope.
    #
    # In MuJoCo the same fields read as a true initial value, because
    # MujocoController.update_vel_command only writes when stdin has a line:
    # the seed persists until the operator types a triple, which then replaces
    # it outright rather than adding to it.
    #
    # OPERATIONAL NOTE: a non-zero vx means the robot starts walking the moment
    # the policy takes over, with no operator action. That is the point on a
    # gantry and a hazard on the floor.
    lin_vel_x_init: float = 0.0
    lin_vel_y_init: float = 0.0
    ang_vel_yaw_init: float = 0.0


@configclass
class PolicyCfg:
    constructor: Callable = MISSING
    checkpoint_path: str = MISSING
    enable_safety_fallback: bool = True
    device: str | torch.device = "cpu"


@configclass
class EvaluatorCfg:
    constructor: Callable = MISSING
    # Rendering
    render: bool = True


@configclass
class ControllerCfg:
    """Controller configuration class.
    """

    policy_dt: float = 0.02
    robot: RobotCfg = MISSING
    vel_command: Optional[VelocityCommandCfg] = None
    policy: PolicyCfg = MISSING

    mujoco: MujocoControllerCfg = MujocoControllerCfg()
    booster: BoosterRobotControllerCfg = BoosterRobotControllerCfg()
    evaluator: Optional[EvaluatorCfg] = None

    def __post_init__(self):
        self.mujoco.physics_dt = self.policy_dt / self.mujoco.decimation
