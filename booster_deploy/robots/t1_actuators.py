"""Booster T1 actuator catalogue and the three torque ceilings.

Lives outside ``robots/t1.py`` and ``robots/booster.py`` on purpose: upstream
owns both of those (``booster.py`` is now a pure re-export shim), so editing
them guarantees a conflict on the next release. Everything here is additive.
"""

import math

from .t1 import T1_23DOF_CFG

# ---------------------------------------------------------------------------
# Booster T1 actuator catalogue
#
# Source: docs/info_sources/boosteractuatordata.pdf, table "Booster T1" (data
# from the manufacturer). Per actuator group:
#     (gear_ratio, rated_torque_Nm, peak_torque_Nm, rated_rpm, peak_rpm,
#      rotor_inertia_kg_mm2)
#
# Rated/peak speed are OUTPUT-shaft speeds, not motor speeds. Two checks: at
# gear 18 a motor-side 140 rpm would cap the knee at 7.8 rpm = 0.81 rad/s, which
# no walking gait fits in; and a measured rollout peaks the knee at 19.8 rad/s,
# the same order as the output-side reading of 14.66 rad/s.
#
# Corroboration that this table is what the asset was built from: T1_23dof.xml's
# per-joint `armature` equals rotor_inertia * gear_ratio**2 exactly for five of
# the seven groups -- Neck 18.0e-6*10**2 = 0.0018, Arm 21.8e-6*36**2 = 0.0282528,
# Waist/HipRollYaw 76.5e-6*25**2 = 0.0478125, HipPitch 161.7e-6*18**2 = 0.0523908,
# Knee 196.3e-6*18**2 = 0.0636012.
#
# The two remaining values are ankle, and they close too -- as exact multiples of
# the single-motor armature 26.2e-6*36**2 = 0.0339552:
#     ankle pitch  0.0679104  = 2.0 * 0.0339552   (both motors add in pitch)
#     ankle roll   0.02037312 = 0.6 * 0.0339552   (differential lever ratio)
# Upstream confirms those two factors itself: robots/t1.py writes the ankle
# damping as `5.120323529816263/2` and `1.5360970589448788/0.6`, and dividing
# them out lands both ankle joints on the same motor-side Kd, 2.5601617649.
# See docs/06-runbook-deploy-real.md for the derivation.
#
# NOTE: the Neck row is anomalous -- rated 120 -> peak 400 rpm is a 3.3x jump
# where every other group sits at 1.05-1.27x. Possibly specified differently or a
# typo. It does not affect locomotion.
# ---------------------------------------------------------------------------

T1_ACTUATOR_CATALOG = {
    "Neck":       (10, 3.0, 7.0, 120, 400, 18.0),
    "Arm":        (36, 10.0, 36.0, 75, 89, 21.8),
    "Waist":      (25, 12.0, 40.0, 55, 70, 76.5),
    "HipRollYaw": (25, 12.0, 40.0, 55, 70, 76.5),
    "HipPitch":   (18, 20.0, 55.0, 155, 157, 161.7),
    "Knee":       (18, 25.0, 65.0, 132, 140, 196.3),
    "Ankle":      (36, 15.0, 50.0, 109, 117, 26.2),
}

# Which catalogue group each entry of T1_23DOF_CFG.joint_names belongs to.
T1_JOINT_ACTUATOR_GROUPS = (
    ["Neck"] * 2
    + ["Arm"] * 8
    + ["Waist"]
    + ["HipPitch", "HipRollYaw", "HipRollYaw", "Knee", "Ankle", "Ankle"] * 2
)


def t1_catalog_column(index: int, scale: float = 1.0) -> list[float]:
    """One catalogue column, expanded to the 23 joints in joint_names order."""
    return [T1_ACTUATOR_CATALOG[g][index] * scale
            for g in T1_JOINT_ACTUATOR_GROUPS]


_RPM_TO_RAD_S = math.pi / 30.0

#: Peak output torque per joint [Nm] -- the physical ceiling, not a derating.
T1_CATALOG_PEAK_TORQUE = t1_catalog_column(2)
#: Speed at which available torque reaches zero [rad/s] (catalogue peak speed).
T1_CATALOG_VELOCITY_LIMIT = t1_catalog_column(4, _RPM_TO_RAD_S)
#: Speed below which full torque is available [rad/s] (catalogue rated speed).
T1_CATALOG_KNEE_POINT_VELOCITY = t1_catalog_column(3, _RPM_TO_RAD_S)

# ---------------------------------------------------------------------------
# The three torque ceilings, as explicit named constants.
#
# Upstream release 7bb1462e (2026-09-13) replaced T1_23DOF_CFG.effort_limit with
# the URDF values. Reading the ceilings off that config would silently make the
# "firmware" and "urdf" models identical -- the sweep would still run and still
# plot, just comparing a thing against itself. Hence both are pinned here.
# ---------------------------------------------------------------------------

#: Firmware operating limits [Nm] -- the old T1_23DOF_CFG.effort_limit, in
#: joint_names order. Every value sits at or below the catalogue peak, which is
#: what an operating limit should do.
T1_EFFORT_FIRMWARE = [
    7.0, 7.0,
    18.0, 18.0, 18.0, 18.0,
    18.0, 18.0, 18.0, 18.0,
    25.0,
    45.0, 25.0, 25.0, 60.0, 24.0, 15.0,
    45.0, 25.0, 25.0, 60.0, 24.0, 15.0,
]

#: Ceiling the policy trained against [Nm]; taken from T1_23dof.urdf, which is
#: also what T1_23DOF_CFG.effort_limit became in 7bb1462e.
T1_EFFORT_URDF = [float(v) for v in T1_23DOF_CFG.effort_limit]


def _assert_ceilings_distinct() -> None:
    """Guard against the silent-collapse failure mode described above."""
    if len(T1_EFFORT_FIRMWARE) != len(T1_EFFORT_URDF):
        raise AssertionError(
            "T1_EFFORT_FIRMWARE has {} entries, T1_EFFORT_URDF has {}".format(
                len(T1_EFFORT_FIRMWARE), len(T1_EFFORT_URDF)))
    if T1_EFFORT_FIRMWARE == T1_EFFORT_URDF:
        raise AssertionError(
            "T1_EFFORT_FIRMWARE == T1_EFFORT_URDF: the 'derated' and 'urdf' "
            "actuator models would be the same experiment. Someone probably "
            "re-pointed a ceiling at T1_23DOF_CFG.effort_limit, which carries "
            "the URDF values since upstream 7bb1462e.")


_assert_ceilings_distinct()
