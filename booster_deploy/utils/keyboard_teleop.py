"""Terminal keyboard teleop, teleop_twist_keyboard-style, for sim and robot.

The same keys drive `deploy.py --mujoco --teleop keyboard` and
`deploy.py --teleop keyboard` on the T1, so a layout rehearsed in MuJoCo is the
layout used on the robot. See docs/17-teleop-teclado.md for the operator guide.

    Moving around:          Holonomic (Shift):
       u    i    o             U    I    O
       j    k    l             J    K    L
       m    ,    .             M    <    >

    q/z : all speeds +/-10%    w/x : linear only    e/c : angular only
    k / space / any other key : stop (ramped)
    Enter : DAMPING (immediate, the robot goes limp)
    Ctrl-C : leave the policy (kWalking on the robot)
    x / r : custom mode / start policy - ONLY before the policy runs

THREE LAYERS, IN THIS ORDER

  1. key -> direction (latched, as in teleop_twist_keyboard: the last key holds
     until another one replaces it) times the current speed/turn scale;
  2. the target is clamped to a SAFE ENVELOPE (safe_vx/vy/vyaw), which is a
     separate and smaller set of numbers than the task's vx_max: the mimickit
     task declares vx_max 2.5 and overshoots its command by 18-38%;
  3. the command walks to the target at a bounded acceleration. Stopping is
     ramped as well - only damping is instantaneous.

Nothing here imports ROS, the SDK or MuJoCo: KeyboardTeleop is pure logic and
is unit-tested without a terminal (tests/unit/test_keyboard_teleop.py).
"""

from __future__ import annotations

import atexit
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from typing import Callable, Optional

# teleop_twist_keyboard's tables, minus its z axis (t/b): (x, y, yaw)
MOVE_BINDINGS = {
    "i": (1, 0, 0), "o": (1, 0, -1), "j": (0, 0, 1), "l": (0, 0, -1),
    "u": (1, 0, 1), ",": (-1, 0, 0), ".": (-1, 0, 1), "m": (-1, 0, -1),
    # holonomic
    "I": (1, 0, 0), "O": (1, -1, 0), "J": (0, 1, 0), "L": (0, -1, 0),
    "U": (1, 1, 0), "<": (-1, 0, 0), ">": (-1, -1, 0), "M": (-1, 1, 0),
}
# (linear factor, angular factor)
SPEED_BINDINGS = {
    "q": (1.1, 1.1), "z": (0.9, 0.9),
    "w": (1.1, 1.0), "x": (0.9, 1.0),
    "e": (1.0, 1.1), "c": (1.0, 0.9),
}
DAMPING_KEYS = ("\n", "\r")

# phases; the robot portal walks through the first three in order
PHASE_WAIT_CUSTOM = "wait_custom"   # 'x' enters custom mode + prepare ramp
PHASE_WAIT_GAIT = "wait_gait"       # 'r' starts the policy
PHASE_RUNNING = "running"
PHASE_DAMPING = "damping"

EVENT_CUSTOM = "custom"
EVENT_RL_GAIT = "rl_gait"
EVENT_DAMPING = "damping"

HELP = """
[teleop] teclado (estilo teleop_twist_keyboard)
     u    i    o        U    I    O     (Shift = holonomico/strafe)
     j    k    l        J    K    L
     m    ,    .        M    <    >
  q/z vel +-10%   w/x so linear   e/c so angular
  k / espaco / outra tecla : parar (com rampa)
  ENTER : DAMPING (imediato)    Ctrl-C : sair (kWalking no robo)
"""


@dataclass
class TeleopLimits:
    """Safe envelope and ramp, in physical units. Defaults are conservative."""
    safe_vx: float = 0.8          # m/s
    safe_vy: float = 0.3          # m/s
    safe_vyaw: float = 0.6        # rad/s
    speed: float = 0.3            # initial linear scale, m/s
    turn: float = 0.3             # initial angular scale, rad/s
    accel: float = 0.5            # m/s^2, vx and vy
    yaw_accel: float = 1.0        # rad/s^2
    timeout: Optional[float] = None   # s without a key -> target 0; None = latch


def _approach(current, target, max_step):
    if (target > current):
        return min(target, current + max_step)
    return max(target, current - max_step)


class KeyboardTeleop:
    """Key handling, safe envelope and ramp. No I/O, no clock of its own."""

    def __init__(self, limits: TeleopLimits | None = None,
                 phase: str = PHASE_RUNNING):
        self.limits = limits or TeleopLimits()
        self.phase = phase
        lim = self.limits
        self.speed = min(lim.speed, lim.safe_vx)
        self.turn = min(lim.turn, lim.safe_vyaw)
        self.direction = (0, 0, 0)
        self.cmd = [0.0, 0.0, 0.0]
        self._last_key_t = None

    # ---- keys --------------------------------------------------------------
    def handle_key(self, ch: str, now: float | None = None) -> str | None:
        """Apply one key press. Returns an EVENT_* for the caller to act on."""
        self._last_key_t = time.monotonic() if now is None else now
        if (ch in DAMPING_KEYS):
            self.enter_damping()
            return EVENT_DAMPING
        if (self.phase == PHASE_DAMPING):
            return None
        # phase keys only mean something before the policy runs; afterwards
        # 'x' is "linear -10%" again, as in teleop_twist_keyboard
        if (self.phase == PHASE_WAIT_CUSTOM and ch == "x"):
            self.phase = PHASE_WAIT_GAIT
            return EVENT_CUSTOM
        if (self.phase == PHASE_WAIT_GAIT and ch == "r"):
            self.phase = PHASE_RUNNING
            self.direction = (0, 0, 0)
            return EVENT_RL_GAIT
        if (self.phase != PHASE_RUNNING):
            # a movement key typed while waiting must not be latched and
            # released the moment the policy takes over
            return None
        if (ch in MOVE_BINDINGS):
            self.direction = MOVE_BINDINGS[ch]
        elif (ch in SPEED_BINDINGS):
            lin, ang = SPEED_BINDINGS[ch]
            self.speed = min(self.speed * lin, self.limits.safe_vx)
            self.turn = min(self.turn * ang, self.limits.safe_vyaw)
        else:
            # k, space, and - as in teleop_twist_keyboard - any unknown key
            self.direction = (0, 0, 0)
        return None

    def enter_damping(self) -> None:
        self.phase = PHASE_DAMPING
        self.direction = (0, 0, 0)
        self.cmd = [0.0, 0.0, 0.0]

    def resume(self) -> None:
        """Leave damping (MuJoCo reset). On the robot there is no way back."""
        self.phase = PHASE_RUNNING
        self.direction = (0, 0, 0)
        self.cmd = [0.0, 0.0, 0.0]

    # ---- command -----------------------------------------------------------
    def target(self, now: float | None = None) -> tuple[float, float, float]:
        lim = self.limits
        if (self.phase != PHASE_RUNNING):
            return (0.0, 0.0, 0.0)
        if (lim.timeout is not None and self._last_key_t is not None):
            now = time.monotonic() if now is None else now
            if (now - self._last_key_t > lim.timeout):
                return (0.0, 0.0, 0.0)
        dx, dy, dyaw = self.direction
        clip = lambda v, m: max(-m, min(m, v))  # noqa: E731
        return (clip(dx * self.speed, lim.safe_vx),
                clip(dy * self.speed, lim.safe_vy),
                clip(dyaw * self.turn, lim.safe_vyaw))

    def step(self, dt: float, now: float | None = None) -> tuple[float, float, float]:
        """Advance the ramp by dt and return (vx, vy, vyaw) in physical units."""
        if (self.phase == PHASE_DAMPING):
            self.cmd = [0.0, 0.0, 0.0]
            return (0.0, 0.0, 0.0)
        tx, ty, tyaw = self.target(now)
        lim = self.limits
        self.cmd[0] = _approach(self.cmd[0], tx, lim.accel * dt)
        self.cmd[1] = _approach(self.cmd[1], ty, lim.accel * dt)
        self.cmd[2] = _approach(self.cmd[2], tyaw, lim.yaw_accel * dt)
        return tuple(self.cmd)  # type: ignore[return-value]

    def status_line(self) -> str:
        tx, ty, tyaw = self.target()
        return ("[teleop] cmd vx={:+.2f} vy={:+.2f} yaw={:+.2f} | alvo {:+.2f} "
                "{:+.2f} {:+.2f} | speed {:.2f} turn {:.2f} | {}".format(
                    self.cmd[0], self.cmd[1], self.cmd[2], tx, ty, tyaw,
                    self.speed, self.turn, self.phase))

    def describe(self) -> str:
        lim = self.limits
        return ("[teleop] envelope vx<={:.2f} vy<={:.2f} vyaw<={:.2f} | "
                "speed {:.2f} turn {:.2f} | accel {:.2f} m/s2 {:.2f} rad/s2 | "
                "timeout {}".format(lim.safe_vx, lim.safe_vy, lim.safe_vyaw,
                                    self.speed, self.turn, lim.accel,
                                    lim.yaw_accel, lim.timeout))


class TerminalKeyReader:
    """Read single key presses from stdin in cbreak mode on a daemon thread.

    cbreak keeps ISIG, so Ctrl-C still raises SIGINT in the main thread and the
    portal's existing handler keeps working. The terminal is restored on
    close() and at exit; if it is ever left without echo, `reset` fixes it.
    """

    def __init__(self, on_key: Callable[[str], None]):
        self._on_key = on_key
        self._running = False
        self._old = None
        self._thread = None
        if (not sys.stdin.isatty()):
            raise RuntimeError(
                "--teleop keyboard precisa de um terminal interativo (stdin nao "
                "e um tty; via ssh use `ssh -t`)")

    def start(self) -> None:
        fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        atexit.register(self.close)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="teleop_keys",
                                        daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if (not ready):
                    continue
                ch = sys.stdin.read(1)
                if (ch == "" or ch == "\x03"):
                    continue
                self._on_key(ch)
            except Exception as e:      # keep the reader alive
                print("\n[teleop] erro no leitor de teclado: {}".format(e))

    def close(self) -> None:
        self._running = False
        if (self._old is not None):
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)
            except Exception:
                pass
            self._old = None


class KeyboardTeleopService:
    """Drop-in for RemoteControlService in BoosterRobotPortal.

    Same interface (get_*_cmd, start_custom_mode, start_rl_gait, hints, close).
    The get_*_cmd values are NORMALIZED by the task's vx_max/vy_max/vyaw_max,
    because BoosterRobotController.update_vel_command multiplies them back and
    adds the --vx baseline - the physical value therefore round-trips exactly.

    The ramp is advanced by its own 50 Hz thread, not by the getters: the
    low-state handler calls all three getters on every /low_state message
    (~500 Hz), and a ramp stepped there would run at the message rate.
    """

    RAMP_DT = 0.02

    def __init__(self, vel_command_cfg, limits: TeleopLimits | None = None,
                 on_damping: Callable[[], None] | None = None):
        self._max = (vel_command_cfg.vx_max, vel_command_cfg.vy_max,
                     vel_command_cfg.vyaw_max)
        self.teleop = KeyboardTeleop(limits, phase=PHASE_WAIT_CUSTOM)
        self.on_damping = on_damping
        self._lock = threading.Lock()
        self._custom = False
        self._rl_gait = False
        self._cmd = (0.0, 0.0, 0.0)
        self._running = True
        self._last_status = None
        self._reader = TerminalKeyReader(self._on_key)
        self._reader.start()
        self._ramp = threading.Thread(target=self._ramp_loop, name="teleop_ramp",
                                      daemon=True)
        self._ramp.start()
        print(HELP, flush=True)
        print(self.teleop.describe(), flush=True)

    def _on_key(self, ch: str) -> None:
        with self._lock:
            event = self.teleop.handle_key(ch)
            if (event == EVENT_CUSTOM):
                self._custom = True
            elif (event == EVENT_RL_GAIT):
                self._rl_gait = True
            elif (event == EVENT_DAMPING):
                self._cmd = (0.0, 0.0, 0.0)
        if (event == EVENT_DAMPING):
            print("\n[teleop] ENTER -> DAMPING", flush=True)
            if (self.on_damping is not None):
                self.on_damping()
        elif (event is None and self.teleop.phase == PHASE_RUNNING):
            print("\r\033[K" + self.teleop.status_line(), end="", flush=True)

    def _ramp_loop(self) -> None:
        while self._running:
            with self._lock:
                self._cmd = self.teleop.step(self.RAMP_DT)
                status = self.teleop.status_line()
            # repaint while the ramp is moving, so the operator sees it settle
            if (self.teleop.phase == PHASE_RUNNING and status != self._last_status):
                print("\r\033[K" + status, end="", flush=True)
            self._last_status = status
            time.sleep(self.RAMP_DT)

    # ---- RemoteControlService interface -----------------------------------
    def get_operation_hint(self) -> str:
        return "[teleop] politica ativa: u i o / j k l / m , . | k parar | ENTER damping"

    def get_custom_mode_operation_hint(self) -> str:
        return "[teleop] pressione 'x' para entrar em custom mode (rampa de prepare)"

    def get_rl_gait_operation_hint(self) -> str:
        return "[teleop] pressione 'r' para iniciar a politica"

    def start_custom_mode(self) -> bool:
        return self._custom

    def start_rl_gait(self) -> bool:
        return self._rl_gait

    def get_vx_cmd(self) -> float:
        with self._lock:
            return self._cmd[0] / self._max[0]

    def get_vy_cmd(self) -> float:
        with self._lock:
            return self._cmd[1] / self._max[1]

    def get_vyaw_cmd(self) -> float:
        with self._lock:
            return self._cmd[2] / self._max[2]

    def close(self) -> None:
        self._running = False
        self._reader.close()


# ---- shared CLI ------------------------------------------------------------
def add_teleop_args(parser) -> None:
    """--safe-* / --teleop-* flags, shared by deploy.py and teleop.py."""
    d = TeleopLimits()
    g = parser.add_argument_group("keyboard teleop (--teleop keyboard)")
    g.add_argument("--safe-vx", type=float, default=d.safe_vx, metavar="M_S",
                   help="teto |vx| do teclado (default %(default)s)")
    g.add_argument("--safe-vy", type=float, default=d.safe_vy, metavar="M_S",
                   help="teto |vy| do teclado (default %(default)s)")
    g.add_argument("--safe-vyaw", type=float, default=d.safe_vyaw, metavar="RAD_S",
                   help="teto |vyaw| do teclado (default %(default)s)")
    g.add_argument("--teleop-speed", type=float, default=d.speed, metavar="M_S",
                   help="escala linear inicial (default %(default)s)")
    g.add_argument("--teleop-turn", type=float, default=d.turn, metavar="RAD_S",
                   help="escala angular inicial (default %(default)s)")
    g.add_argument("--teleop-accel", type=float, default=d.accel, metavar="M_S2",
                   help="rampa de vx/vy (default %(default)s)")
    g.add_argument("--teleop-yaw-accel", type=float, default=d.yaw_accel,
                   metavar="RAD_S2", help="rampa de vyaw (default %(default)s)")
    g.add_argument("--teleop-timeout", type=float, default=None, metavar="S",
                   help="sem tecla por S segundos -> alvo zero (default: "
                        "desligado, o comando fica travado como no teleop_twist)")


def limits_from_args(args, vel_command_cfg) -> TeleopLimits:
    """Build TeleopLimits, refusing an envelope wider than the task's own."""
    lim = TeleopLimits(
        safe_vx=args.safe_vx, safe_vy=args.safe_vy, safe_vyaw=args.safe_vyaw,
        speed=args.teleop_speed, turn=args.teleop_turn,
        accel=args.teleop_accel, yaw_accel=args.teleop_yaw_accel,
        timeout=args.teleop_timeout)
    for name, value, task_max in (("safe-vx", lim.safe_vx, vel_command_cfg.vx_max),
                                  ("safe-vy", lim.safe_vy, vel_command_cfg.vy_max),
                                  ("safe-vyaw", lim.safe_vyaw, vel_command_cfg.vyaw_max)):
        if (value < 0.0 or value > task_max):
            raise ValueError("--{} {} fora de [0, {}] (o *_max da task)".format(
                name, value, task_max))
    for name, value in (("teleop-accel", lim.accel),
                        ("teleop-yaw-accel", lim.yaw_accel)):
        if (value <= 0.0):
            raise ValueError("--{} precisa ser > 0".format(name))
    return lim
