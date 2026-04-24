from __future__ import annotations

import sys
from time import sleep, strftime, monotonic
import select
import numpy as np
import torch
import mujoco
import mujoco.viewer
from booster_assets import BOOSTER_ASSETS_DIR
from .base_controller import BaseController, ControllerCfg, VelocityCommand


class MujocoController(BaseController):
    def __init__(self, cfg: ControllerCfg):
        super().__init__(cfg)

        mjcf_path = self._expand_assets_placeholder(self.robot.cfg.mjcf_path)
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_model.opt.timestep = self.cfg.mujoco.physics_dt
        self.decimation = self.cfg.mujoco.decimation
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        self.mj_data.qpos = np.concatenate(
            [
                np.array(self.cfg.mujoco.init_pos, dtype=np.float32),
                np.array(self.cfg.mujoco.init_quat, dtype=np.float32),
                self.robot.default_joint_pos.numpy(),
            ]
        )
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # render a second "ghost" robot (kinematic only) without
        # modifying the MuJoCo XML. This uses a second MjData to compute FK from
        # generalized coordinates and draws a duplicated set of geoms via
        # viewer.user_scn.
        self._ghost_mj_data = mujoco.MjData(self.mj_model)
        # Keep ghost initialized to the current simulated pose so it is valid
        # even before any policy calls set_reference_qpos().
        self._ghost_mj_data.qpos[:] = self.mj_data.qpos
        self._ghost_mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self._ghost_mj_data)
        self._ghost_rgba = np.array(
            self.cfg.mujoco.ghost_rgba, dtype=np.float32)
        self._ghost_scene_option = mujoco.MjvOption()

        # Reference qpos can be set explicitly by the policy.
        self._reference_qpos: np.ndarray | None = None

        # Video recording state
        self._recording: bool = False
        self._renderer: mujoco.Renderer | None = None
        self._ffmpeg_proc = None
        self._video_path: str = ""

    def start(self):
        # Clear reference; policy.reset() may set a fresh one.
        self._reference_qpos = None
        return super().start()

    def render_reference_robot(
        self,
        viewer,
        # mj_data: mujoco.MjData,
        *,
        rgba: np.ndarray | None = None,
    ) -> None:
        """Render a kinematic robot pose into viewer.user_scn using mj_data."""
        mujoco.mjv_updateScene(
            self.mj_model,
            self._ghost_mj_data,
            self._ghost_scene_option,
            None,
            viewer.cam,
            int(mujoco.mjtCatBit.mjCAT_DYNAMIC),
            viewer.user_scn,
        )
        if rgba is None:
            rgba = self._ghost_rgba

        for i in range(viewer.user_scn.ngeom):
            viewer.user_scn.geoms[i].rgba[:] = rgba

    def set_reference_qpos(
        self,
        qpos: np.ndarray | torch.Tensor | None,
    ) -> None:
        """Set the reference generalized coordinates (qpos) for ghost rendering.

        Policies should call this each step (or whenever updated). Pass None to
        clear the reference.
        """
        if qpos is None:
            self._reference_qpos = None
            return

        if isinstance(qpos, torch.Tensor):
            qpos_np = qpos.detach().cpu().numpy()
        else:
            qpos_np = np.asarray(qpos)

        qpos_np = qpos_np.astype(np.float32, copy=False).reshape(-1)
        if qpos_np.shape[0] != int(self.mj_model.nq):
            raise ValueError(
                f"reference qpos must have shape (nq,), got {qpos_np.shape} (nq={int(self.mj_model.nq)})"
            )
        self._reference_qpos = qpos_np.copy()
        # FK + offset
        self._ghost_mj_data.qpos[:] = self._reference_qpos
        self._ghost_mj_data.qvel[:] = 0.0
        mujoco.mj_forward(self.mj_model, self._ghost_mj_data)

    def _expand_assets_placeholder(self, path: str) -> str:
        """Replace {BOOSTER_ASSETS_DIR} placeholder in a path string.
        """
        try:
            return path.replace("{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR))
        except Exception:
            return path

    def update_vel_command(self):
        cmd: VelocityCommand = self.vel_command
        if select.select([sys.stdin], [], [], 0)[0]:
            try:
                parts = sys.stdin.readline().strip().split()
                if len(parts) == 3:
                    (cmd.lin_vel_x, cmd.lin_vel_y, cmd.ang_vel_yaw) = map(float, parts)
                    print(
                        f"Updated command to: x={cmd.lin_vel_x},"
                        f"y={cmd.lin_vel_y}, yaw={cmd.ang_vel_yaw}\n"
                        "Set command (x, y, yaw): ",
                        end="",
                    )
                else:
                    raise ValueError
            except ValueError:
                print(
                    "Invalid input. Enter three numeric values. "
                    "Set command (x, y, yaw): ",
                    end="",
                )

    def update_state(self) -> None:
        dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
        dof_torque = self.mj_data.qfrc_actuator[6:].astype(np.float32)

        base_pos_w = self.mj_data.qpos.astype(np.float32)[:3]
        base_quat = self.mj_data.qpos.astype(np.float32)[3:7]
        base_lin_vel_b = self.mj_data.qvel.astype(np.float32)[:3]
        base_ang_vel_b = self.mj_data.qvel.astype(np.float32)[3:6]

        self.robot.data.joint_pos = torch.from_numpy(
            dof_pos).to(self.robot.data.device)
        self.robot.data.joint_vel = torch.from_numpy(
            dof_vel).to(self.robot.data.device)
        self.robot.data.feedback_torque = torch.from_numpy(
            dof_torque).to(self.robot.data.device)
        self.robot.data.root_pos_w = torch.from_numpy(
            base_pos_w).to(self.robot.data.device)
        self.robot.data.root_quat_w = torch.from_numpy(
            base_quat).to(self.robot.data.device)
        self.robot.data.root_lin_vel_b = torch.from_numpy(
            base_lin_vel_b).to(self.robot.data.device)
        self.robot.data.root_ang_vel_b = torch.from_numpy(
            base_ang_vel_b).to(self.robot.data.device)

    def log_states(self, dof_targets: np.ndarray) -> None:
        if self.cfg.mujoco.log_states is not None:
            if not hasattr(self, '_states'):
                self._states = {
                    'root_pos_w': [],
                    'root_quat_w': [],
                    'root_lin_vel_b': [],
                    'root_ang_vel_b': [],
                    'joint_pos': [],
                    'joint_vel': [],
                    'joint_torque': [],
                    'dof_targets': [],
                }
            base_pos_w = self.mj_data.qpos.astype(np.float32)[:3]
            base_quat = self.mj_data.qpos.astype(np.float32)[3:7]
            base_lin_vel_b = self.mj_data.qvel.astype(np.float32)[:3]
            base_ang_vel_b = self.mj_data.qvel.astype(np.float32)[3:6]
            dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
            dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
            dof_torque = self.mj_data.qfrc_actuator[6:].astype(np.float32)

            self._states['root_pos_w'].append(base_pos_w)
            self._states['root_quat_w'].append(base_quat)
            self._states['root_lin_vel_b'].append(base_lin_vel_b)
            self._states['root_ang_vel_b'].append(base_ang_vel_b)
            self._states['joint_pos'].append(dof_pos)
            self._states['joint_vel'].append(dof_vel)
            self._states['joint_torque'].append(dof_torque)
            self._states['dof_targets'].append(dof_targets)
            if len(self._states['root_pos_w']) % 100 == 0:
                _states = {k: np.stack(v) for k, v in self._states.items()}
                np.savez(f'{self.cfg.mujoco.log_states}.npz', **_states)
                print(f'saved {self.cfg.mujoco.log_states}.npz '
                      f'at {self._step_count} steps')

    def ctrl_step(self, dof_targets: torch.Tensor):
        dof_targets = dof_targets.cpu().numpy()  # type: ignore
        self.log_states(dof_targets)
        if self.vel_command is not None:
            self.update_vel_command()

        dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
        dof_vel = self.mj_data.qvel.astype(np.float32)[6:]
        kp = self.robot.joint_stiffness.numpy()
        # kd is applied as passive joint damping in the XML (implicit, via MuJoCo solver),
        # not as an explicit torque in the PD loop. This matches IsaacLab's ImplicitActuator
        # behaviour and avoids generating horizontal contact forces that cause sliding.
        kd = np.zeros_like(kp)
        # ctrl_limit = [
        #     np.minimum(self.mj_model.actuator_forcerange[:, 0],
        #                self.mj_model.actuator_ctrlrange[:, 0]),
        #     np.maximum(self.mj_model.actuator_forcerange[:, 1],
        #                self.mj_model.actuator_ctrlrange[:, 1]),
        # ]
        effort_limit = self.robot.effort_limit.numpy()
        velocity_limit = (
            self.robot.velocity_limit.numpy()
            if self.robot.velocity_limit is not None else None
        )
        knee_point_velocity = (
            self.robot.knee_point_velocity.numpy()
            if self.robot.knee_point_velocity is not None else None
        )
        if velocity_limit is not None:
            denom = np.maximum(velocity_limit - knee_point_velocity, 1e-6)

        for i in range(self.decimation):
            torque = kp * (dof_targets - dof_pos) - kd * dof_vel
            if velocity_limit is not None:
                # Piecewise-linear T-N curve (BoosterDelayedPDActuator):
                #   |vel| <= knee_point_velocity  → max_torque = effort_limit
                #   |vel| in (knee, v_max)        → max_torque decreases linearly to 0
                #   |vel| >= velocity_limit        → max_torque = 0
                vel_abs = np.abs(dof_vel)
                tau_linear = effort_limit * (velocity_limit - vel_abs) / denom
                max_torque = np.clip(tau_linear, 0.0, effort_limit)
            else:
                max_torque = effort_limit
            self.mj_data.ctrl = np.clip(torque, -max_torque, max_torque)
            mujoco.mj_step(self.mj_model, self.mj_data)
            dof_pos = self.mj_data.qpos.astype(np.float32)[7:]
            dof_vel = self.mj_data.qvel.astype(np.float32)[6:]

    def _start_recording(self, viewer) -> None:
        import subprocess
        cfg = self.cfg.mujoco
        path = cfg.video_path or f"recording_{strftime('%Y%m%d_%H%M%S')}.mp4"
        w, h = cfg.video_width, cfg.video_height
        fps = 50
        if self._renderer is None:
            # Ensure the offscreen framebuffer is large enough before creating renderer
            self.mj_model.vis.global_.offwidth = max(self.mj_model.vis.global_.offwidth, w)
            self.mj_model.vis.global_.offheight = max(self.mj_model.vis.global_.offheight, h)
            self._renderer = mujoco.Renderer(self.mj_model, height=h, width=w)
        self._ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", f"{w}x{h}", "-framerate", str(fps),
                "-i", "pipe:",
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                path,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._recording = True
        self._video_path = path
        print(f"[sim] recording started → {path}  ({w}x{h} @ {fps} fps)")

    def _stop_recording(self) -> None:
        self._recording = False
        if self._ffmpeg_proc is not None:
            self._ffmpeg_proc.stdin.close()  # type: ignore[union-attr]
            self._ffmpeg_proc.wait()
            self._ffmpeg_proc = None
        print(f"[sim] recording saved → {self._video_path}")

    def _write_video_frame(self, viewer) -> None:
        self._renderer.update_scene(self.mj_data, camera=viewer.cam)
        frame_rgb = self._renderer.render()
        self._ffmpeg_proc.stdin.write(frame_rgb.tobytes())  # type: ignore[union-attr]

    def run(self):
        # Save initial qpos for reset
        self._init_qpos = self.mj_data.qpos.copy()

        self._last_video_frame_time: float = 0.0

        # --- Simulation control flags (written by key_callback, read by main loop) ---
        self._paused = False
        self._step_once = False   # advance exactly one policy step while paused
        self._do_reset = False
        self._do_save = False
        self._do_load = False
        self._key_slot = 0        # active slot for save/load (keys 0–9)
        # slot -> (qpos, qvel, step_count)
        self._saved_states: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}

        # GLFW key codes
        _KEY_SPACE     = 32
        _KEY_BACKSPACE = 259
        _KEY_RIGHT     = 262   # step one frame while paused
        _KEY_R         = 82    # toggle video recording
        _KEY_G         = 71    # toggle ghost visibility

        self._do_toggle_rec = False
        self._show_ghost = self.cfg.mujoco.visualize_reference_ghost

        def key_callback(keycode):
            if keycode == _KEY_SPACE:
                self._paused = not self._paused
                print("[sim] " + ("paused" if self._paused else "running"))
            elif keycode == _KEY_BACKSPACE:
                self._do_reset = True
            elif keycode == _KEY_RIGHT:
                self._step_once = True
            elif 48 <= keycode <= 57:          # 0–9: choose slot
                self._key_slot = keycode - 48
                print(f"[sim] key slot → {self._key_slot}")
            elif keycode == 83:                # S: save state
                self._do_save = True
            elif keycode == 76:                # L: load state
                self._do_load = True
            elif keycode == _KEY_R:            # R: toggle recording
                self._do_toggle_rec = True
            elif keycode == _KEY_G:            # G: toggle ghost
                self._show_ghost = not self._show_ghost
                print("[sim] ghost " + ("on" if self._show_ghost else "off"))

        with mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data,
                key_callback=key_callback) as viewer:

            self.viewer = viewer
            viewer.cam.elevation = -20
            if self.vel_command is not None:
                print("\nSet command (x, y, yaw): ", end="")

            print(
                "\n[sim] Controls:\n"
                "  Space      → pause / resume\n"
                "  → (right)  → step one frame (while paused)\n"
                "  Backspace  → reset to initial state\n"
                "  0-9        → select save/load slot\n"
                "  S          → save state to current slot\n"
                "  L          → load state from current slot\n"
                "  R          → toggle video recording\n"
                "  G          → toggle ghost robot\n"
            )

            with viewer.lock():
                self.update_state()
            self.start()

            # Auto-start recording if video_path is configured
            if self.cfg.mujoco.video_path:
                self._start_recording(viewer)

            while viewer.is_running() and self.is_running:

                # --- Toggle recording ---
                if self._do_toggle_rec:
                    self._do_toggle_rec = False
                    if self._recording:
                        self._stop_recording()
                    else:
                        self._start_recording(viewer)

                # --- Save state ---
                if self._do_save:
                    self._do_save = False
                    with viewer.lock():
                        self._saved_states[self._key_slot] = (
                            self.mj_data.qpos.copy(),
                            self.mj_data.qvel.copy(),
                            self._step_count,
                        )
                    print(f"[sim] saved state to slot {self._key_slot} "
                          f"(step {self._step_count})")

                # --- Load state ---
                if self._do_load:
                    self._do_load = False
                    if self._key_slot in self._saved_states:
                        qpos, qvel, step_count = self._saved_states[self._key_slot]
                        with viewer.lock():
                            self.mj_data.qpos[:] = qpos
                            self.mj_data.qvel[:] = qvel
                            mujoco.mj_forward(self.mj_model, self.mj_data)
                        self._step_count = step_count
                        self._elapsed_s = step_count * self.cfg.policy_dt
                        with viewer.lock():
                            self.update_state()
                        print(f"[sim] loaded state from slot {self._key_slot} "
                              f"(step {step_count})")
                    else:
                        print(f"[sim] slot {self._key_slot} is empty")

                # --- Reset ---
                if self._do_reset:
                    self._do_reset = False
                    if hasattr(self.policy, 'motion'):
                        sim2real = self.robot.data.sim2real_joint_indexes
                        init_joints = self.policy.motion.joint_pos[0][sim2real].cpu().numpy()
                    else:
                        init_joints = self._init_qpos[7:]
                    with viewer.lock():
                        mujoco.mj_resetData(self.mj_model, self.mj_data)
                        self.mj_data.qpos[:7] = self._init_qpos[:7]
                        self.mj_data.qpos[7:] = init_joints
                        self.mj_data.qvel[:] = 0.0
                        mujoco.mj_forward(self.mj_model, self.mj_data)
                        self.update_state()
                    self._step_count = 0
                    self._elapsed_s = 0.0
                    self.policy.reset()
                    print("[sim] reset to motion start")

                # --- Pause: hold until space or right-arrow ---
                if self._paused and not self._step_once:
                    sleep(0.01)
                    viewer.sync()
                    continue
                self._step_once = False

                # --- Normal simulation step ---
                _step_start = monotonic()
                with viewer.lock():
                    self.update_state()
                dof_targets = self.policy_step()
                with viewer.lock():
                    self.ctrl_step(dof_targets)

                if self._show_ghost:
                    self.render_reference_robot(viewer, rgba=self._ghost_rgba)

                self.viewer.cam.lookat[:] = self.mj_data.qpos.astype(np.float32)[0:3]
                self.viewer.sync()

                if self._recording:
                    now = monotonic()
                    if now - self._last_video_frame_time >= 1.0 / 30:
                        self._write_video_frame(viewer)
                        self._last_video_frame_time = now

                _remaining = self.cfg.policy_dt - (monotonic() - _step_start)
                if _remaining > 0:
                    sleep(_remaining)

            if self._recording:
                self._stop_recording()
