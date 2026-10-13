"""take_log: cmd.csv / events.csv layout, events, monotonic clock.

    python3 tests/unit/test_take_log.py      # or: pytest tests/unit
"""

import csv
import os
import sys
import tempfile
import time
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from booster_deploy.utils.take_log import CommandLog, EventLog  # noqa: E402


def rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_command_log_rows_and_events():
    d = tempfile.mkdtemp()
    log = CommandLog(d)
    cmd = types.SimpleNamespace(lin_vel_x=0.3, lin_vel_y=-0.1, ang_vel_yaw=0.25)
    t0 = time.monotonic()
    log.mark("start")
    log.record(1, cmd, 0.5)
    log.record(2, cmd, None)            # task without heading
    log.mark("fall")
    log.record(3, None, float("nan"))   # task without vel_command
    log.close()
    r = rows(os.path.join(d, "cmd.csv"))
    assert r[0] == CommandLog.HEADER
    assert r[1][1:] == ["1", "0.30000", "-0.10000", "0.25000", "0.50000", "start"]
    assert r[2][5:] == ["", ""]
    assert r[3][2:] == ["", "", "", "", "fall"]
    assert t0 <= float(r[1][0]) <= time.monotonic()


def test_reopen_appends_without_second_header():
    d = tempfile.mkdtemp()
    for _ in range(2):
        log = EventLog(d)
        log.event("custom_mode")
        log.close()
    r = rows(os.path.join(d, "events.csv"))
    assert r[0] == EventLog.HEADER and len(r) == 3


def test_write_after_close_is_ignored():
    d = tempfile.mkdtemp()
    log = EventLog(d)
    log.close()
    log.event("late")          # must not raise: shutdown order is not fixed


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print("{} passed".format(len(tests)))
