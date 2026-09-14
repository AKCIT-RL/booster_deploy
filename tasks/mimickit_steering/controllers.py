"""Backwards-compatible re-exports; the implementation moved to scripts/.

These helpers are task-neutral and are now shared with Booster's own t1_walk
policy (docs/12), so they live in scripts/actuator_models.py. This module stays
so that teleop.py and the commands recorded in docs/08 keep working unchanged.

`MimicKitPDController` is the old name for `ActuatorPDController`; nothing in it
was ever specific to MimicKit.
"""

import os
import sys

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.actuator_models import (  # noqa: F401,E402
    CONTROLLERS,
    DAMPING_EXPLICIT,
    DAMPING_IMPLICIT,
    DAMPING_NONE,
    DEGRADE_MODES,
    DEGRADE_NONE,
    DEGRADE_ZERO_ROOT,
    EFFORT_CATALOG,
    EFFORT_DERATED,
    EFFORT_SOURCES,
    EFFORT_URDF,
    ActuatorPDController,
    ExplicitPDController,
    ImplicitDampingController,
    ZeroDampingController,
    apply_effort_source,
    apply_gains,
    apply_rate,
    apply_tn_curve,
    make_degraded,
)

MimicKitPDController = ActuatorPDController

__all__ = [
    "CONTROLLERS", "DAMPING_EXPLICIT", "DAMPING_IMPLICIT", "DAMPING_NONE",
    "DEGRADE_MODES", "DEGRADE_NONE", "DEGRADE_ZERO_ROOT",
    "EFFORT_CATALOG", "EFFORT_DERATED", "EFFORT_SOURCES", "EFFORT_URDF",
    "ActuatorPDController", "MimicKitPDController", "ExplicitPDController",
    "ImplicitDampingController", "ZeroDampingController",
    "apply_effort_source", "apply_gains", "apply_rate", "apply_tn_curve",
    "make_degraded",
]
