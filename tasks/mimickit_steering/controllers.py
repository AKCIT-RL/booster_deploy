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
        kd = self.robot.joint_damping.numpy()
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
