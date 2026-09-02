"""Build MimicKit's character observation from a booster_deploy robot state.

This module does NOT reimplement the observation. It converts conventions and
then calls MimicKit's own `compute_char_obs` and `compute_steering_observations`
- the exact functions the training environment calls. Reimplementing a 200-dim
observation and keeping it in step with a training repo is a losing game; the
only maintainable version is to import the definition.

What is left, then, is an adapter, and adapters are where the bugs live. Four
conventions differ between the two sides, none of which announces itself:

  quaternion order    MimicKit is xyzw (Isaac Gym). MuJoCo qpos[3:7] and
                      booster_deploy's lab_math are wxyz (IsaacLab). Feeding one
                      to the other yields a valid unit quaternion describing the
                      wrong rotation.

  angular velocity    compute_char_obs rotates root_ang_vel by the inverse
                      heading, so it wants WORLD frame. Both booster_deploy
                      controllers report body frame (MuJoCo qvel[3:6] is
                      body-local; the robot's is a gyro).

  linear velocity     `robot.data.root_lin_vel_b` is NOT the same frame in the
                      two controllers, despite the name. See
                      root_lin_vel_is_world() below - this one is a trap.

  joint order         MimicKit's DOF order is the asset's; booster_deploy's is
                      RobotCfg.joint_names. For the T1 these are identical, and
                      assert_joint_order_matches() checks rather than assumes.

Key body positions come from forward kinematics on the same kinematic model,
not from the simulator. For a rigid articulated body the two agree exactly, and
FK works on any host - including a real robot, which has no body-position
sensor. tests/unit/test_deploy_key_bodies.py checks the FK against MuJoCo's own.

Observation layout (200 dims for the T1, 23 DOF, 5 key bodies):

    root_h(1) root_rot(6) root_vel(3) root_ang_vel(3)
    joint_rot(24 x 6) dof_vel(23) key_pos(5 x 3) | steering(5)
"""

import math

import torch

_MIMICKIT = {}


def _mimickit():
    """MimicKit's own definitions - the point of this module is to call them.

    Imported on first use rather than at module scope: scripts/deploy.py builds
    its task registry with pkgutil.walk_packages, which imports this submodule
    DIRECTLY - the package __init__'s guard never sees it - and re-raises. A
    module-level import here would take every other task's registration down
    with it on a machine that has no MimicKit checkout.
    """
    if (len(_MIMICKIT) == 0):
        from anim.mjcf_char_model import MJCFCharModel
        from envs.char_env import compute_char_obs
        from envs.task_steering_env import compute_steering_observations
        import util.torch_util as torch_util
        _MIMICKIT.update(MJCFCharModel=MJCFCharModel,
                         compute_char_obs=compute_char_obs,
                         compute_steering_observations=compute_steering_observations,
                         torch_util=torch_util)
    return _MIMICKIT


def quat_wxyz_to_xyzw(q):
    """MuJoCo / IsaacLab quaternion order -> MimicKit / Isaac Gym order."""
    return torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)


def root_lin_vel_is_world(controller):
    """Whether this controller's `root_lin_vel_b` is really in the world frame.

    It is, for the MuJoCo controller, and the name is simply wrong there:
    mujoco_controller.update_state assigns `qvel[:3]` to `root_lin_vel_b`, and
    for a free joint MuJoCo defines qvel[0:3] as the linear velocity in the
    GLOBAL frame (qvel[3:6] is the angular velocity in the body frame - only
    the second half is body-local). Verified empirically, not from the name.

    booster_robot_controller.update_state does convert, with
    quat_apply_inverse(root_quat_w, root_lin_vel_w), so on hardware the field
    means what it says.

    Their own locomotion policy reads neither field, which is why the
    disagreement has never mattered upstream. It matters here: this quantity is
    3 of our 200 observation dimensions, and getting it wrong is NOT silent -
    the robot falls within seconds - but it is silent as to the cause.

    The MRO is walked rather than isinstance() so that this module needs no
    mujoco import (the hardware controller runs on a robot that has none), and
    rather than comparing type(controller).__name__ so that a SUBCLASS is still
    recognised. That last point is not hypothetical: teleop.KeyboardMujocoController
    is one, and the leaf-name version of this check silently sent it down the
    hardware branch. test_deploy_observation covers both.
    """
    return any(cls.__name__ == "MujocoController" for cls in type(controller).__mro__)


def root_lin_vel_to_world(root_quat_xyzw, root_lin_vel, is_world):
    """Normalise a controller's linear velocity to the world frame.

    Kept here rather than in the Policy so that the branch is covered by
    tests/unit/test_deploy_observation.py; see root_lin_vel_is_world() for why
    there is a branch at all.
    """
    if (is_world):
        return root_lin_vel.reshape(3)
    return _mimickit()["torch_util"].quat_rotate(
        root_quat_xyzw.reshape(1, 4), root_lin_vel.reshape(1, 3)).reshape(3)


def build_char_model(asset_path, key_body_names, device="cpu"):
    """Load the same asset the policy was trained on."""
    model = _mimickit()["MJCFCharModel"](device=device)
    model.load(asset_path)
    key_body_ids = torch.tensor([model.get_body_id(n) for n in key_body_names],
                                dtype=torch.long, device=device)
    return model, key_body_ids


def assert_joint_order_matches(char_model, robot_joint_names):
    """The exported policy's action is in the asset's DOF order.

    booster_deploy indexes joints by RobotCfg.joint_names. For the T1 the two
    coincide (scripts/prepare_t1_asset.py asserts the same thing from the other
    direction), so no remapping is needed - but a silent divergence would send
    every joint command to the wrong actuator, so it is checked on startup.
    """
    dof_names = []
    for j in range(1, char_model.get_num_joints()):
        joint = char_model.get_joint(j)
        if (joint.get_dof_dim() > 0):
            dof_names.append(joint.name)

    if (len(dof_names) != len(robot_joint_names)):
        raise ValueError("asset has {} DOFs but the robot config lists {} joints".format(
            len(dof_names), len(robot_joint_names)))

    # The two naming schemes differ in ways that have nothing to do with order:
    # the asset is lower_snake_case with a _joint suffix, the config is the
    # SDK's CamelCase; Booster's "AA" prefix appears on aahead_yaw_joint and
    # aahead_pitch_joint but only on AAHead_yaw in the config; and the asset
    # qualifies some joints the config leaves bare (waist_yaw_joint / Waist).
    # So names are matched by prefix after normalisation.
    #
    # That is loose enough to survive those differences and still strict enough
    # for the failure this guards against: no left name is a prefix of a right
    # one, and no pitch name is a prefix of a roll or yaw one, so any transposed
    # pair or shifted block is still caught. What it would NOT catch is a config
    # that renamed a joint to a strict prefix of its neighbour - which no
    # reordering can produce.
    def norm(name):
        name = name.lower().replace("_joint", "").replace("_", "")
        if (name.startswith("aa")):
            name = name[2:]
        return name

    def compatible(asset_name, robot_name):
        a, b = norm(asset_name), norm(robot_name)
        return a.startswith(b) or b.startswith(a)

    mismatched = [(a, b) for a, b in zip(dof_names, robot_joint_names)
                  if not compatible(a, b)]
    if (len(mismatched) > 0):
        raise ValueError(
            "joint order differs between the asset and the robot config; the "
            "policy's actions would go to the wrong actuators. First mismatches: "
            "{}".format(mismatched[:5]))
    return


def steering_command(vel_x, vel_y, face_heading, tar_speed_min, tar_speed_max):
    """Map a joystick velocity command to the steering task's variables.

    The task was trained on (tar_dir, tar_speed, face_dir) with tar_dir and
    face_dir as WORLD directions - not on a body-frame twist. This is a mapping
    choice, not a derivation:

      tar_speed  |(vx, vy)|, clamped to the range the task sampled during
                 training. Below tar_speed_min the policy is extrapolating: the
                 env samples speeds in [tar_speed_min, tar_speed_max] and never
                 showed it a stop command, so clamping is safer than passing 0.
      tar_dir    the joystick direction rotated into the world by the operator's
                 desired heading.
      face_dir   that desired heading. The training env ran with
                 rand_face_dir: False, i.e. face_dir was always world +x, so
                 what the policy actually learned is "face the commanded
                 direction"; integrating the yaw stick into face_heading is what
                 exposes that as a control.

    Returns (tar_dir[2], tar_speed[1], face_dir[2]) as tensors.
    """
    speed = math.sqrt(vel_x * vel_x + vel_y * vel_y)
    if (speed > 1e-6):
        stick = math.atan2(vel_y, vel_x)
    else:
        stick = 0.0
    heading = face_heading + stick

    tar_speed = min(max(speed, tar_speed_min), tar_speed_max)
    tar_dir = torch.tensor([math.cos(heading), math.sin(heading)], dtype=torch.float32)
    face_dir = torch.tensor([math.cos(face_heading), math.sin(face_heading)],
                            dtype=torch.float32)
    return tar_dir, torch.tensor([tar_speed], dtype=torch.float32), face_dir


def compute_observation(char_model, key_body_ids,
                        root_pos_w, root_quat_w_xyzw,
                        root_lin_vel_w, root_ang_vel_b,
                        dof_pos, dof_vel,
                        tar_dir, tar_speed, face_dir,
                        root_height_obs=True, global_obs=False):
    """The full observation, exactly as the training environment builds it.

    Every input is a single (unbatched) frame. Frames are named in the argument
    list because getting one wrong is silent; the caller is responsible for
    reaching this contract, and Policy._robot_state() is where that happens.
    """
    root_pos = root_pos_w.reshape(1, 3)
    root_rot = root_quat_w_xyzw.reshape(1, 4)
    root_vel = root_lin_vel_w.reshape(1, 3)
    dof_pos = dof_pos.reshape(1, -1)
    dof_vel = dof_vel.reshape(1, -1)

    # compute_char_obs rotates root_ang_vel by the inverse heading, so it must
    # arrive in the world frame; the controllers report the body frame
    mk = _mimickit()
    root_ang_vel = mk["torch_util"].quat_rotate(root_rot, root_ang_vel_b.reshape(1, 3))

    joint_rot = char_model.dof_to_rot(dof_pos)
    body_pos, _ = char_model.forward_kinematics(root_pos, root_rot, joint_rot)
    key_pos = body_pos[:, key_body_ids, :]

    char_obs = mk["compute_char_obs"](root_pos=root_pos, root_rot=root_rot,
                                root_vel=root_vel, root_ang_vel=root_ang_vel,
                                joint_rot=joint_rot, dof_vel=dof_vel,
                                key_pos=key_pos, global_obs=global_obs,
                                root_height_obs=root_height_obs)

    task_obs = mk["compute_steering_observations"](root_rot=root_rot,
                                             tar_dir=tar_dir.reshape(1, 2),
                                             tar_speed=tar_speed.reshape(1),
                                             face_dir=face_dir.reshape(1, 2))

    return torch.cat([char_obs, task_obs], dim=-1).reshape(-1)
