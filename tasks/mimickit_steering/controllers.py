"""Three PD schemes for the MimicKit steering task, isolated for comparison.

A position-target policy is half of a closed loop; the PD controller is the
other half. Change where the damping term is applied and the same joint targets
produce different motion. This module makes that choice explicit and switchable
instead of implicit in whichever booster_deploy revision happens to be checked
out.

    ZeroDampingController      no damping anywhere
    ExplicitPDController       tau = kp*(q* - q) - kd*qd, every substep
    ImplicitDampingController  tau = kp*(q* - q); kd as MuJoCo joint damping

All three share one ctrl_step and differ only in that one term, so a comparison
between them isolates it. The T-N torque limit is shared and driven by the same
RobotCfg fields the upstream controller reads, so it is not a hidden variable.

WHY THIS EXISTS

booster_deploy/controllers/mujoco_controller.py:205 ctrl_step contains, on the
AKCIT fork:

    # kd is applied as passive joint damping in the XML (implicit, via MuJoCo
    # solver), not as an explicit torque in the PD loop. This matches IsaacLab's
    # ImplicitActuator behaviour and avoids generating horizontal contact forces
    # that cause sliding.
    kd = np.zeros_like(kp)

The reasoning is sound and the two schemes are genuinely different: an explicit
-kd*qd torque is computed from the velocity at the START of a substep and
applied as an external force, so at a foot in contact it shows up as a
horizontal contact force and the foot slides. MuJoCo's joint damping is folded
into the implicit integration of the same substep, which is unconditionally
stable and produces no such force.

But the premise is that the MJCF supplies the damping, and for the T1 it does
not. `mj_model.dof_damping[6:]` is all zeros with
booster_assets/robots/T1/T1_23dof.xml, so RobotCfg.joint_damping - the 0.0637*kp
we chose deliberately, matching the ratio the G1 asset uses uniformly and which
was validated in the training engine - is applied NOWHERE. The T1 runs
completely undamped, which is the condition that produced the hop-at-reset and
jittery gait this project already diagnosed once.

ImplicitDampingController is the fork's intent made to actually happen: it
writes RobotCfg.joint_damping into mj_model.dof_damping at construction, so the
damping exists and is integrated implicitly. ExplicitPDController is what the
policy was trained and validated against. ZeroDampingController is the current
effective behaviour, kept so it can be measured rather than assumed.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from booster_deploy.controllers.mujoco_controller import MujocoController
from booster_deploy.utils.isaaclab import math as lab_math

EFFORT_URDF = "urdf"
EFFORT_DERATED = "derated"
EFFORT_CATALOG = "catalog"
EFFORT_SOURCES = (EFFORT_URDF, EFFORT_DERATED, EFFORT_CATALOG)


def apply_effort_source(cfg, source):
    """Choose which torque ceiling the MuJoCo PD loop clips at.

    ctrl_step clips at RobotCfg.effort_limit, and the task restates the URDF
    limits (knee 130.5 Nm) because that is what the policy was trained against.
    The firmware clips at Booster's derated operating limits instead
    (T1_23DOF_CFG.effort_limit, knee 60 Nm), and those are never sent over the
    wire - booster_robot_controller.ctrl_step publishes only q, kp and kd, so the
    ceiling on hardware is whatever the motor board enforces.

    "derated" therefore isolates one hardware constraint the way --degrade
    zero_root isolates the other: same policy, same state, only the torque
    ceiling moved to what the robot will actually deliver.

    Caveat: a flat clip is the optimistic model. Real actuators fall off with
    speed, which is what RobotCfg.velocity_limit / knee_point_velocity describe
    (both None here, so the T-N curve is off). A policy that survives this is not
    yet proven to survive the real torque-speed envelope.
    """
    from booster_deploy.robots import booster as bst

    if (source == EFFORT_URDF):
        return cfg
    if (source == EFFORT_DERATED):
        cfg.robot.effort_limit = list(bst.T1_23DOF_CFG.effort_limit)
        return cfg
    if (source == EFFORT_CATALOG):
        cfg.robot.effort_limit = list(bst.T1_CATALOG_PEAK_TORQUE)
        return cfg
    raise ValueError("unknown effort source: {}".format(source))


def apply_tn_curve(cfg, enabled):
    """Turn on the torque-speed falloff the actuators actually have.

    RobotCfg.velocity_limit and knee_point_velocity are None for the T1, so
    ctrl_step's piecewise-linear T-N limit never runs and every experiment so far
    clipped torque at a FLAT ceiling. That is the optimistic model: a real
    actuator delivers full torque only up to its rated speed and decays to zero at
    its no-load speed.

    The numbers come from the manufacturer's own table
    (docs/info_sources/boosteractuatordata.pdf) via booster.T1_CATALOG_*: rated
    speed becomes knee_point_velocity, peak speed becomes velocity_limit. They are
    applied here rather than on T1_23DOF_CFG so that turning this on stays an
    explicit experimental choice and no existing task changes behaviour silently.

    Worth knowing before reading a result: hip-roll, hip-yaw and waist have a peak
    speed of only 70 rpm = 7.33 rad/s, the lowest on the robot, so they are where
    the curve bites first.
    """
    if (not enabled):
        return cfg

    from booster_deploy.robots import booster as bst
    cfg.robot.velocity_limit = list(bst.T1_CATALOG_VELOCITY_LIMIT)
    cfg.robot.knee_point_velocity = list(bst.T1_CATALOG_KNEE_POINT_VELOCITY)
    return cfg


DEGRADE_NONE = "none"
DEGRADE_ZERO_ROOT = "zero_root"
DEGRADE_MODES = (DEGRADE_NONE, DEGRADE_ZERO_ROOT)


def make_degraded(base_cls, degrade):
    """Subclass whose update_state reports what the HARDWARE reports.

    booster_robot_controller.py:244-249 fills root_pos_w and root_lin_vel_w with
    np.zeros(3) because the T1 has no sensor for either. MuJoCo supplies both, so
    a --mujoco run validates the policy against its own training assumptions
    rather than against the deployment target. This puts the hardware's gap back
    in, where falling is free.

    For t1_mimickit_steering that is 4 of 200 observation dimensions: root_h and
    the three of root_vel. The other 196 are invariant to the root's position and
    linear velocity, so those 4 carry all of the translational information - which
    is exactly why they are also the ones no onboard sensor can provide.
    """
    if (degrade == DEGRADE_NONE):
        return base_cls
    if (degrade != DEGRADE_ZERO_ROOT):
        raise ValueError("unknown degrade mode: {}".format(degrade))

    class Degraded(base_cls):
        def update_state(self):
            super().update_state()
            self.robot.data.root_pos_w.zero_()
            self.robot.data.root_lin_vel_b.zero_()

    Degraded.__name__ = "ZeroRoot{}".format(base_cls.__name__)
    return Degraded


DAMPING_EXPLICIT = "explicit"
DAMPING_IMPLICIT = "implicit"
DAMPING_NONE = "none"


class MimicKitPDController(MujocoController):
    """MujocoController with the damping scheme as an explicit choice.

    Subclasses set `damping_mode`. ctrl_step is reimplemented rather than
    inherited so that all three variants stay identical apart from that term,
    whichever booster_deploy revision this runs against.
    """

    damping_mode = DAMPING_EXPLICIT

    def __init__(self, cfg):
        super().__init__(cfg)

        kd = self.robot.joint_damping.numpy().astype(np.float64)
        # dof_damping is indexed by DOF: 0-5 are the floating base
        model_damping = self.mj_model.dof_damping[6:]
        if (model_damping.shape[0] != kd.shape[0]):
            raise ValueError(
                "asset has {} joint DOFs but the robot config lists {} damping "
                "values".format(model_damping.shape[0], kd.shape[0]))

        if (self.damping_mode == DAMPING_IMPLICIT):
            # what the fork's comment assumes the XML already provides
            model_damping[:] = kd
        else:
            # explicit and none both apply nothing through the solver; whatever
            # the MJCF declared is cleared so the scheme is the only difference
            model_damping[:] = 0.0

        self._pd_kd = kd if self.damping_mode == DAMPING_EXPLICIT else np.zeros_like(kd)

    def describe_pd(self):
        return ("{}: PD kd {} | MuJoCo joint damping {} | effort limit "
                "{:.1f}..{:.1f} Nm | T-N curve {}".format(
                    type(self).__name__,
                    "active" if self.damping_mode == DAMPING_EXPLICIT else "zero",
                    "active" if self.damping_mode == DAMPING_IMPLICIT else "zero",
                    float(self.robot.effort_limit.min()),
                    float(self.robot.effort_limit.max()),
                    "on" if getattr(self.robot, "velocity_limit", None) is not None
                    else "off"))

    def ctrl_step(self, dof_targets: torch.Tensor):
        dof_targets = dof_targets.cpu().numpy()  # type: ignore
        self.log_states(dof_targets)
        if self.vel_command is not None:
            self.update_vel_command()

        dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
        kp = self.robot.joint_stiffness.numpy()
        kd = self._pd_kd
        effort_limit = self.robot.effort_limit.numpy()

        velocity_limit = getattr(self.robot, "velocity_limit", None)
        knee = getattr(self.robot, "knee_point_velocity", None)
        if (velocity_limit is not None):
            velocity_limit = velocity_limit.numpy()
            knee = knee.numpy()
            denom = np.maximum(velocity_limit - knee, 1e-6)

        for _ in range(self.decimation):
            torque = kp * (dof_targets - dof_pos) - kd * dof_vel
            if (velocity_limit is not None):
                # piecewise-linear torque-speed limit, same form the upstream
                # controller uses so it is not a difference between variants
                tau_linear = effort_limit * (velocity_limit - np.abs(dof_vel)) / denom
                max_torque = np.clip(tau_linear, 0.0, effort_limit)
            else:
                max_torque = effort_limit
            self.mj_data.ctrl = np.clip(torque, -max_torque, max_torque)
            mujoco.mj_step(self.mj_model, self.mj_data)
            dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
            dof_vel = self.mj_data.qvel.astype(np.float32)[6:]



class ExplicitPDController(MimicKitPDController):
    """kd inside the PD loop. What the policy was trained and validated against.

    tau = kp*(q* - q) - kd*qd, recomputed every physics substep. The damping
    torque is an external force applied from the velocity at the start of the
    substep, which is where the sliding-foot objection comes from.
    """
    damping_mode = DAMPING_EXPLICIT


class ImplicitDampingController(MimicKitPDController):
    """kd as MuJoCo joint damping. The AKCIT fork's stated intent.

    tau = kp*(q* - q) only; RobotCfg.joint_damping is written into
    mj_model.dof_damping so the solver integrates it implicitly. Same numbers as
    ExplicitPDController, different integration - that is the whole comparison.
    """
    damping_mode = DAMPING_IMPLICIT


class ZeroDampingController(MimicKitPDController):
    """No damping at all. The fork's CURRENT effective behaviour on this asset.

    Not a design, a measurement: the fork zeroes kd in the PD loop and expects
    the MJCF to supply it, and T1_23dof.xml supplies none. Kept as a named
    variant so the cost of that gap is a number rather than an argument.
    """
    damping_mode = DAMPING_NONE


CONTROLLERS = {
    "explicit": ExplicitPDController,
    "implicit": ImplicitDampingController,
    "none": ZeroDampingController,
}


# ---------------------------------------------------------------------------
# IMU noise
# ---------------------------------------------------------------------------

IMU_NOISE_NONE = "none"

# WHERE THESE NUMBERS COME FROM - read this before quoting a result.
#
# They are ENGINEERING ESTIMATES for a MEMS AHRS on a walking humanoid, not
# measurements of this T1. Nothing in this repo has ever measured the T1's IMU,
# so a preset is a hypothesis generator: it says "if the sensor were this bad,
# would the gait change like that?". It does not say the sensor IS this bad.
#
# Replacing them with measured values is cheap and worth doing before any
# conclusion leaves this file: with the robot powered, still and visually
# vertical, read /low_state for ~30 s. The mean of imu_state.rpy[0:2] is
# roll/pitch bias, its std is tilt_noise, the mean of imu_state.gyro is
# gyro_bias, its std is gyro_noise, and the slope of rpy[2] is yaw_drift. Five
# numbers, one standing robot, no policy running.
#
# What each term models, and why it is in a different place in the math:
#
#   tilt bias      a BODY-FIXED constant: the IMU is bolted on at a small angle,
#                  or its zero is off. Composed on the right (q_true * q_err) so
#                  it rotates with the robot, which is what a mounting error
#                  does. This is the one that maps to a steady trunk lean:
#                  proj_gravity is yaw-invariant, so only roll and pitch reach
#                  it, and the policy drives the real trunk by -bias to make the
#                  sensor read upright.
#   tilt noise     white noise on roll/pitch. Bounded, because an AHRS has
#                  gravity as an absolute reference for these two axes.
#   yaw bias       the initial heading offset discussed in docs: _face_heading
#                  starts at 0 = "world +x", and on hardware world +x is
#                  wherever the IMU's yaw zero landed. Composed on the LEFT
#                  (q_err * q_true) because it is a world-frame rotation.
#   yaw drift      the same, growing: yaw has no absolute reference, so its
#                  error is unbounded. Also world-frame.
#   gyro bias/noise  added to the gyro directly; it is already a body-frame
#                  measurement and needs no rotation.
#
# NOT modelled, deliberately, and each would need its own evidence: IMU latency
# (the frame history makes it matter and DR already covers action delay),
# vibration coupling at specific frequencies, and dynamic tilt error during
# acceleration - an AHRS leans its gravity estimate into a turn, which a
# constant bias does not reproduce.
IMU_NOISE_PRESETS = {
    "low": dict(tilt_bias_deg=0.5, tilt_noise_deg=0.10, yaw_bias_deg=0.0,
                yaw_drift_dps=0.2, gyro_bias_dps=0.1, gyro_noise_dps=0.5),
    "typical": dict(tilt_bias_deg=1.5, tilt_noise_deg=0.30, yaw_bias_deg=0.0,
                    yaw_drift_dps=0.5, gyro_bias_dps=0.3, gyro_noise_dps=1.0),
    "high": dict(tilt_bias_deg=4.0, tilt_noise_deg=0.80, yaw_bias_deg=0.0,
                 yaw_drift_dps=2.0, gyro_bias_dps=1.0, gyro_noise_dps=3.0),
}

# Exact overrides. A preset DRAWS roll and pitch bias from +-tilt_bias_deg,
# which is right for asking "how bad can it get"; these pin one axis to one
# number, which is what a hypothesis test needs ("is the lean 3 deg of pitch
# bias?"). Setting either one makes that axis exact and ignores tilt_bias_deg
# for it.
_IMU_EXACT = ("roll_bias_deg", "pitch_bias_deg")

_IMU_KEYS = tuple(IMU_NOISE_PRESETS["typical"].keys()) + _IMU_EXACT


def parse_imu_noise(spec):
    """'none' | preset | 'preset,key=value,...' | 'key=value,...' -> dict|None.

    Refuses an unknown key rather than ignoring it, for the same reason
    steering_dr.load_ranges does: a typo that silently trains (or here, tests)
    against the default is a measurement nobody can reproduce and everybody
    believes.
    """
    if (spec is None or spec == IMU_NOISE_NONE):
        return None

    params = dict.fromkeys(_IMU_EXACT, None)
    items = [s.strip() for s in spec.split(",") if s.strip()]
    if (len(items) > 0 and "=" not in items[0]):
        name = items.pop(0).lower()
        if (name not in IMU_NOISE_PRESETS):
            raise ValueError("unknown IMU noise preset '{}'. Known: {}".format(
                name, ", ".join([IMU_NOISE_NONE] + sorted(IMU_NOISE_PRESETS))))
        params.update(IMU_NOISE_PRESETS[name])
        params["preset"] = name
    else:
        # a bare key=value list starts from silence, so `--imu-noise
        # pitch_bias_deg=3` is exactly one effect and not one effect plus a
        # preset nobody asked for
        params.update({k: 0.0 for k in IMU_NOISE_PRESETS["typical"]})
        params["preset"] = "custom"

    for item in items:
        if ("=" not in item):
            raise ValueError(
                "--imu-noise '{}' is not KEY=VALUE".format(item))
        key, raw = item.split("=", 1)
        key, raw = key.strip().lower(), raw.strip()
        if (key not in _IMU_KEYS):
            raise ValueError("unknown IMU noise key '{}'. Known: {}".format(
                key, ", ".join(sorted(_IMU_KEYS))))
        params[key] = float(raw)
    return params


def make_noisy_imu(base_cls, params, seed=None):
    """Subclass whose update_state reports a NOISY attitude and gyro.

    Only the two fields an IMU actually produces are touched: root_quat_w and
    root_ang_vel_b. joint_pos, joint_vel and the PD loop keep reading the
    simulator's truth, which is correct - the firmware's PD runs on encoders,
    not on the IMU, so corrupting the control loop as well would measure two
    things at once.

    Composes with make_degraded: the two model different hardware gaps (a sensor
    that lies vs a sensor that does not exist), and a run can want both.
    """
    if (params is None):
        return base_cls

    class NoisyIMU(base_cls):
        def __init__(self, cfg):
            super().__init__(cfg)
            self._imu = dict(params)
            self._imu_rng = np.random.default_rng(seed)

            # Drawn ONCE, at construction: an installation constant is not
            # noise. Redrawing it every step would model a sensor being
            # re-bolted 30 times a second, which averages to zero and hides
            # exactly the effect this exists to expose.
            def draw(mag):
                return float(self._imu_rng.uniform(-mag, mag))

            p = self._imu
            self._imu_roll_bias = np.radians(
                p["roll_bias_deg"] if p["roll_bias_deg"] is not None
                else draw(p["tilt_bias_deg"]))
            self._imu_pitch_bias = np.radians(
                p["pitch_bias_deg"] if p["pitch_bias_deg"] is not None
                else draw(p["tilt_bias_deg"]))
            self._imu_yaw_bias = np.radians(p["yaw_bias_deg"])
            self._imu_yaw_drift = np.radians(draw(p["yaw_drift_dps"]))
            self._imu_gyro_bias = np.radians(
                self._imu_rng.uniform(-p["gyro_bias_dps"], p["gyro_bias_dps"],
                                      size=3))
            self._imu_tilt_sigma = np.radians(p["tilt_noise_deg"])
            self._imu_gyro_sigma = np.radians(p["gyro_noise_dps"])
            self._imu_yaw_err = 0.0
            print(self.describe_imu())

        def describe_imu(self):
            """What was DRAWN, not what was asked for.

            The preset is a range; the run used one sample of it. Printing the
            range would describe a run that did not happen, and a bias of +1.4
            and one of -1.4 deg lean the robot opposite ways.
            """
            return ("[imu] preset={} seed={} | tilt bias roll {:+.2f} deg "
                    "pitch {:+.2f} deg (sigma {:.2f}) | yaw bias {:+.1f} deg "
                    "drift {:+.2f} deg/s | gyro bias [{:+.2f} {:+.2f} {:+.2f}] "
                    "deg/s (sigma {:.2f})".format(
                        self._imu.get("preset", "custom"), seed,
                        np.degrees(self._imu_roll_bias),
                        np.degrees(self._imu_pitch_bias),
                        np.degrees(self._imu_tilt_sigma),
                        np.degrees(self._imu_yaw_bias),
                        np.degrees(self._imu_yaw_drift),
                        *np.degrees(self._imu_gyro_bias),
                        np.degrees(self._imu_gyro_sigma)))

        def update_state(self):
            super().update_state()
            data = self.robot.data
            rng = self._imu_rng

            def angle(value):
                return torch.tensor(float(value), dtype=torch.float32)

            # body-fixed tilt error: composed on the RIGHT, so it rotates with
            # the robot the way a mounting angle does
            q_tilt = lab_math.quat_from_euler_xyz(
                angle(self._imu_roll_bias + rng.normal(0.0, self._imu_tilt_sigma)),
                angle(self._imu_pitch_bias + rng.normal(0.0, self._imu_tilt_sigma)),
                angle(0.0))

            # world-fixed heading error: composed on the LEFT, and unbounded,
            # because yaw has no absolute reference to be corrected against
            self._imu_yaw_err += self._imu_yaw_drift * self.cfg.policy_dt
            q_yaw = lab_math.quat_from_euler_xyz(
                angle(0.0), angle(0.0),
                angle(self._imu_yaw_bias + self._imu_yaw_err))

            data.root_quat_w = lab_math.normalize(
                lab_math.quat_mul(q_yaw,
                                  lab_math.quat_mul(data.root_quat_w, q_tilt)))

            gyro_err = self._imu_gyro_bias + rng.normal(
                0.0, self._imu_gyro_sigma, size=3)
            data.root_ang_vel_b = data.root_ang_vel_b + torch.from_numpy(
                gyro_err.astype(np.float32))

    NoisyIMU.__name__ = "NoisyIMU{}".format(base_cls.__name__)
    return NoisyIMU


def parse_hip_yaw_limit(spec):
    """'BASE[,GAIN]' in rad and rad per rad/s -> (base, gain), or None."""
    if (spec is None):
        return None
    parts = [float(v) for v in spec.split(",")]
    if (len(parts) not in (1, 2) or any(v < 0.0 for v in parts)):
        raise ValueError("--hip-yaw-limit expects BASE[,GAIN] >= 0, got "
                         "{!r}".format(spec))
    return (parts[0], parts[1] if len(parts) == 2 else 0.0)
