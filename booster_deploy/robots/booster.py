import math

from ..controllers.controller_cfg import PrepareStateCfg, RobotCfg

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
# Knee 196.3e-6*18**2 = 0.0636012. The two remaining values are ankle, whose
# parallel mechanism splits one motor pair across pitch and roll.
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



K1_CFG = RobotCfg(
    name="Booster_K1",
    joint_names=[
        "AAHead_yaw",
        "Head_pitch",
        "ALeft_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "ARight_Shoulder_Pitch",
        "Right_Shoulder_Roll",
        "Right_Elbow_Pitch",
        "Right_Elbow_Yaw",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Knee_Pitch",
        "Right_Ankle_Pitch",
        "Right_Ankle_Roll",
    ],
    body_names=[
        "Trunk",
        "Head_1",
        "Head_2",
        "Left_Arm_1",
        "Left_Arm_2",
        "Left_Arm_3",
        "left_hand_link",
        "Right_Arm_1",
        "Right_Arm_2",
        "Right_Arm_3",
        "right_hand_link",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Shank",
        "Left_Ankle_Cross",
        "left_foot_link",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Shank",
        "Right_Ankle_Cross",
        "right_foot_link",
    ],
    joint_stiffness=[
        4.0, 4.0,
        4.0, 4.0, 4.0, 4.0,
        4.0, 4.0, 4.0, 4.0,
        80., 80.0, 80., 80., 30., 30.,
        80., 80.0, 80., 80., 30., 30.,
    ],
    joint_damping=[
        1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2.,
    ],
    default_joint_pos=[
        0, 0,
        0.0, -1.3, 0, -0.,
        0.0, 1.3, 0, 0.,
        -0.0, 0, 0, 0.0, -0.0, 0.,
        -0.0, 0, 0, 0.0, -0.0, 0.
    ],
    effort_limit=[
        6, 6,
        14, 14, 14, 14,
        14, 14, 14, 14,
        30, 35, 20, 40, 20, 20,
        30, 35, 20, 40, 20, 20,
    ],
    sim_joint_names=[       # joint order in isaacsim/isaaclab
        "AAHead_yaw",
        "ALeft_Shoulder_Pitch",
        "ARight_Shoulder_Pitch",
        "Left_Hip_Pitch",
        "Right_Hip_Pitch",
        "Head_pitch",
        "Left_Shoulder_Roll",
        "Right_Shoulder_Roll",
        "Left_Hip_Roll",
        "Right_Hip_Roll",
        "Left_Elbow_Pitch",
        "Right_Elbow_Pitch",
        "Left_Hip_Yaw",
        "Right_Hip_Yaw",
        "Left_Elbow_Yaw",
        "Right_Elbow_Yaw",
        "Left_Knee_Pitch",
        "Right_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Right_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Ankle_Roll",
    ],
    sim_body_names=[    # body order in isaacsim/isaaclab
        "Trunk",
        "Head_1",
        "Left_Arm_1",
        "Right_Arm_1",
        "Left_Hip_Pitch",
        "Right_Hip_Pitch",
        "Head_2",
        "Left_Arm_2",
        "Right_Arm_2",
        "Left_Hip_Roll",
        "Right_Hip_Roll",
        "Left_Arm_3",
        "Right_Arm_3",
        "Left_Hip_Yaw",
        "Right_Hip_Yaw",
        "left_hand_link",
        "right_hand_link",
        "Left_Shank",
        "Right_Shank",
        "Left_Ankle_Cross",
        "Right_Ankle_Cross",
        "left_foot_link",
        "right_foot_link",
    ],
    # {BOOSTER_ASSETS_DIR} will be replaced with
    # booster_assets.BOOSTER_ASSETS_DIR by MujocoController
    mjcf_path="{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof.xml",
    prepare_state=PrepareStateCfg(
        stiffness=[
            40., 40.,
            40., 50., 20., 20,
            40., 50., 20., 20,
            350., 350., 180., 350., 250., 250.,
            350., 350., 180., 350., 250., 250.,
        ],
        damping=[
            1.5, 1.5,
            0.5, 1.5, 0.2, 0.2,
            0.5, 1.5, 0.2, 0.2,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
        ],
        joint_pos=[
            0, 0,
            0.0, -1.3, 0, -0.,
            0.0, 1.3, 0, 0.,
            -0.0, 0, 0, 0.105, -0.10, 0.,
            -0.0, 0, 0, 0.105, -0.10, 0.
        ],
    ),
)


T1_23DOF_CFG = RobotCfg(
    name="Booster_T1_23DOF",
    joint_names=[
        "AAHead_yaw",
        "Head_pitch",
        "Left_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "Right_Shoulder_Pitch",
        "Right_Shoulder_Roll",
        "Right_Elbow_Pitch",
        "Right_Elbow_Yaw",
        "Waist",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Knee_Pitch",
        "Right_Ankle_Pitch",
        "Right_Ankle_Roll",
    ],
    body_names=[
        "Trunk",
        "H1",
        "H2",
        "AL1",
        "AL2",
        "AL3",
        "left_hand_link",
        "AR1",
        "AR2",
        "AR3",
        "right_hand_link",
        "Waist",
        "Hip_Pitch_Left",
        "Hip_Roll_Left",
        "Hip_Yaw_Left",
        "Shank_Left",
        "Ankle_Cross_Left",
        "left_foot_link",
        "Hip_Pitch_Right",
        "Hip_Roll_Right",
        "Hip_Yaw_Right",
        "Shank_Right",
        "Ankle_Cross_Right",
        "right_foot_link",
    ],

    joint_stiffness=[
        4.0, 4.0,
        4.0, 4.0, 4.0, 4.0,
        4.0, 4.0, 4.0, 4.0,
        80.,
        80., 80.0, 80., 80., 30., 30.,
        80., 80.0, 80., 80., 30., 30.,
    ],
    joint_damping=[
        1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        2.,
        2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2.,
    ],
    default_joint_pos=[
        0, 0,
        0.0, -1.4, 0, -0.,
        0.0, 1.4, 0, 0.,
        0.,
        -0.0, 0, 0, 0.0, -0.0, 0.,
        -0.0, 0, 0, 0.0, -0.0, 0.
    ],
    effort_limit=[
        7, 7,
        18, 18, 18, 18,
        18, 18, 18, 18,
        25.,
        45, 25, 25, 60, 24, 15,
        45, 25, 25, 60, 24, 15,
    ],
    sim_joint_names=[       # joint order in isaacsim/isaaclab
        "AAHead_yaw",
        'Left_Shoulder_Pitch',
        'Right_Shoulder_Pitch',
        'Waist',
        "Head_pitch",
        'Left_Shoulder_Roll',
        'Right_Shoulder_Roll',
        'Left_Hip_Pitch',
        'Right_Hip_Pitch',
        'Left_Elbow_Pitch',
        'Right_Elbow_Pitch',
        'Left_Hip_Roll',
        'Right_Hip_Roll',
        'Left_Elbow_Yaw',
        'Right_Elbow_Yaw',
        'Left_Hip_Yaw',
        'Right_Hip_Yaw',
        'Left_Knee_Pitch',
        'Right_Knee_Pitch',
        'Left_Ankle_Pitch',
        'Right_Ankle_Pitch',
        'Left_Ankle_Roll',
        'Right_Ankle_Roll'
    ],

    sim_body_names=[    # body order in isaacsim/isaaclab

    ],
    # {BOOSTER_ASSETS_DIR} will be replaced with
    # booster_assets.BOOSTER_ASSETS_DIR by MujocoController
    mjcf_path="{BOOSTER_ASSETS_DIR}/robots/T1/T1_23dof.xml",
    prepare_state=PrepareStateCfg(
        stiffness=[
            40., 40.,
            40., 50., 20., 20,
            40., 50., 20., 20,
            350.,
            350., 350., 180., 350., 350., 350.,
            350., 350., 180., 350., 350., 350.,],
        damping=[
            0.65, 0.65,
            0.5, 1.5, 0.2, 0.2,
            0.5, 1.5, 0.2, 0.2,
            5.,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
        ],
        joint_pos=[
            0, 0,
            0.0, -1.4, 0, -0.,
            0.0, 1.4, 0, 0.,
            0.,
            -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,
            -0.1, 0.0, 0.0, 0.2, -0.1, 0.0
        ],
    ),
)
