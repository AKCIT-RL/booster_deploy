"""Per-take CSV logs: the velocity command and the session's events.

What scripts/log_deploy.py cannot see. It records /low_state and /joint_ctrl,
but the velocity command never goes on a topic - it lives in a SyncedArray
between the portal and the inference process - and neither do the mode
changes. These two files fill that gap, one writer per process:

    cmd.csv      written by the controller, once per policy step:
                 t, step, vx, vy, vyaw, face_heading, event
                 vx/vy/vyaw are what the policy saw on that step (after the
                 remote/keyboard, the --vx baseline and the *_max clip);
                 face_heading is the mimickit task's integrated heading
                 [rad], empty for tasks that have none; event is "fall" on
                 the step the fall guard fired.

    events.csv   written by the robot portal (main process):
                 t, event, detail     custom_mode, prepare_done, rl_gait,
                 damping, exit, ...

`t` is time.monotonic(), i.e. CLOCK_MONOTONIC, which is one clock for every
process on the machine. log_deploy.py stores its own start on the same clock
(t0_monotonic in the .npz), so the three files line up by subtraction - see
docs/18-takes.md.

Files are line-buffered: a crash, a kill or a pulled cable loses at most the
line being written. Enabled by `deploy.py --log-dir DIR` (scripts/take.sh
passes it); with no log dir nothing here runs.
"""

from __future__ import annotations

import csv
import math
import os
import threading
import time


class CsvLog:
    """Append rows to a CSV with a header, line-buffered and thread-safe."""

    def __init__(self, path: str, header: list[str]):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._f = open(path, "a", buffering=1, newline="")
        self._w = csv.writer(self._f)
        self._lock = threading.Lock()
        if new:
            self._w.writerow(header)

    def row(self, *values) -> None:
        with self._lock:
            if self._f is None:
                return
            self._w.writerow(values)

    def close(self) -> None:
        with self._lock:
            if self._f is not None:
                self._f.close()
                self._f = None


class CommandLog(CsvLog):
    HEADER = ["t", "step", "vx", "vy", "vyaw", "face_heading", "event"]

    def __init__(self, log_dir: str):
        super().__init__(os.path.join(log_dir, "cmd.csv"), self.HEADER)
        self._pending = ""

    def mark(self, event: str) -> None:
        """Attach an event to the next command row (e.g. 'fall')."""
        self._pending = event if not self._pending else self._pending + "|" + event

    def record(self, step: int, vel_command, face_heading=None) -> None:
        def f(v):
            return "" if v is None else "{:.5f}".format(v)
        heading = None
        if face_heading is not None and math.isfinite(face_heading):
            heading = face_heading
        if vel_command is None:
            vx = vy = vyaw = None
        else:
            vx, vy, vyaw = (vel_command.lin_vel_x, vel_command.lin_vel_y,
                            vel_command.ang_vel_yaw)
        self.row("{:.6f}".format(time.monotonic()), step, f(vx), f(vy), f(vyaw),
                 f(heading), self._pending)
        self._pending = ""


class PolicyLog:
    """policy.csv: what the network produced, every step, next to its timing.

        t, step, infer_ms, late_ms, raw_<j>..., target_<j>..., obs_<i>...

    raw_*     the network output BEFORE the deploy-side filters (low-pass,
              rate limit, hip-yaw clamp). For mimickit it is already an
              absolute joint target; for locomotion it is the residual action
              in the policy's own joint order, before `default + a * scale`.
    target_*  what ctrl_step sends, i.e. the q of /joint_ctrl, robot order.
              raw vs target isolates the filters; target vs the measured q of
              log.npz is what the motors reached.
    obs_*     the CURRENT observation frame only (80 numbers for the
              measurable mimickit layout); the stacked history is the previous
              rows, so it is not stored 30 times over.
    infer_ms  wall time of policy.inference() on this step.
    late_ms   robot only: how late this step started against its 1/policy_hz
              schedule. The loop keeps its schedule, so after a stall the
              missed steps run back to back - late_ms is how that shows.

    Header is built on the first row, when the sizes are known.
    """

    def __init__(self, log_dir: str):
        self.path = os.path.join(log_dir, "policy.csv")
        os.makedirs(log_dir, exist_ok=True)
        self._f = None
        self._w = None
        self._lock = threading.Lock()

    @staticmethod
    def _flat(x):
        if x is None:
            return []
        try:
            return [float(v) for v in x.detach().reshape(-1).cpu().tolist()]
        except AttributeError:
            return [float(v) for v in x]

    def record(self, t: float, step: int, infer_s: float, late_s,
               raw, raw_names, target, target_names, obs) -> None:
        raw, target, obs = self._flat(raw), self._flat(target), self._flat(obs)
        with self._lock:
            if self._f is None:
                new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
                self._f = open(self.path, "a", buffering=1, newline="")
                self._w = csv.writer(self._f)
                if new:
                    rn = list(raw_names) if raw_names is not None and \
                        len(raw_names) == len(raw) else \
                        ["{:02d}".format(i) for i in range(len(raw))]
                    self._w.writerow(
                        ["t", "step", "infer_ms", "late_ms"]
                        + ["raw_" + n for n in rn]
                        + ["target_" + n for n in target_names]
                        + ["obs_{:03d}".format(i) for i in range(len(obs))])
            late = "" if late_s is None else "{:.3f}".format(late_s * 1e3)
            self._w.writerow(
                ["{:.6f}".format(t), step, "{:.3f}".format(infer_s * 1e3), late]
                + ["{:.6g}".format(v) for v in raw + target + obs])

    def close(self) -> None:
        with self._lock:
            if self._f is not None:
                self._f.close()
                self._f = None


class EventLog(CsvLog):
    HEADER = ["t", "event", "detail"]

    def __init__(self, log_dir: str):
        super().__init__(os.path.join(log_dir, "events.csv"), self.HEADER)

    def event(self, name: str, detail: str = "") -> None:
        self.row("{:.6f}".format(time.monotonic()), name, detail)
