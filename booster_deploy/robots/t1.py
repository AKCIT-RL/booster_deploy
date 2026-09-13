from ..controllers.controller_cfg import PrepareStateCfg, RobotCfg


T1_23DOF_CFG = RobotCfg(
    name="Booster_T1_23DOF",
    prepare_mode="standing",
    joint_names=[
        "aahead_yaw_joint", "aahead_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_elbow_pitch_joint", "left_elbow_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_elbow_pitch_joint", "right_elbow_yaw_joint",
        "waist_yaw_joint",
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ],
    body_names=[
        "trunk",
        "aahead_yaw_link",
        "aahead_pitch_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_elbow_pitch_link",
        "left_elbow_yaw_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_elbow_pitch_link",
        "right_elbow_yaw_link",
        "waist_yaw_link",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_pitch_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_pitch_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
    ],
    # nf=4,dr=1.5
    joint_stiffness = [
        1.1369784270054941,1.1369784270054941,
        17.846013390278234,17.846013390278234,17.846013390278234,17.846013390278234,
        17.846013390278234,17.846013390278234,17.846013390278234,17.846013390278234,
        30.200989467333436,
        33.09289409642191,30.200989467333436,30.200989467333436,40.17399573981213,42.89592209406327,12.868776628218983,
        33.09289409642191,30.200989467333436,30.200989467333436,40.17399573981213,42.89592209406327,12.868776628218983,
    ],
    joint_damping = [
        0.13571680263507907,0.13571680263507907,
        2.130210934160201,2.130210934160201,2.130210934160201,2.130210934160201,
        2.130210934160201,2.130210934160201,2.130210934160201,2.130210934160201,
        3.6049775699942876,
        3.950173257496611,3.6049775699942876,3.6049775699942876,4.795417504307883,5.120323529816263/2,1.5360970589448788/0.6,
        3.950173257496611,3.6049775699942876,3.6049775699942876,4.795417504307883,5.120323529816263/2,1.5360970589448788/0.6,
    ],
    default_joint_pos=[
        0,      # aahead_yaw_joint (未指定，默认0)
        0,      # aahead_pitch_joint (未指定，默认0)
        0.2,    # left_shoulder_pitch_joint
        -1.3,   # left_shoulder_roll_joint
        0,      # left_elbow_pitch_joint (未指定，默认0)
        -0.5,   # left_elbow_yaw_joint
        0.2,    # right_shoulder_pitch_joint
        1.3,    # right_shoulder_roll_joint
        0,      # right_elbow_pitch_joint (未指定，默认0)
        0.5,    # right_elbow_yaw_joint
        0,      # waist_yaw_joint (未指定，默认0)
        -0.2,   # left_hip_pitch_joint
        0,      # left_hip_roll_joint (未指定，默认0)
        0,      # left_hip_yaw_joint (未指定，默认0)
        0.4,    # left_knee_pitch_joint
        -0.2,   # left_ankle_pitch_joint
        0,      # left_ankle_roll_joint (未指定，默认0)
        -0.2,   # right_hip_pitch_joint
        0,      # right_hip_roll_joint (未指定，默认0)
        0,      # right_hip_yaw_joint (未指定，默认0)
        0.4,    # right_knee_pitch_joint
        -0.2,   # right_ankle_pitch_joint
        0,      # right_ankle_roll_joint (未指定，默认0)
    ],
    effort_limit=[
        7.0, 7.0,
        38.3, 38.3, 38.3, 38.3,
        38.3, 38.3, 38.3, 38.3,
        68,
        98.8, 68, 68, 130.5,
        73.1, 17.2,
        98.8, 68, 68, 130.5,
        73.1, 17.2
    ],
    sim_joint_names=[       # joint order in isaacsim/isaaclab
        "aahead_yaw_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
        "waist_yaw_joint", "aahead_pitch_joint", "left_shoulder_roll_joint",
        "right_shoulder_roll_joint", "left_hip_pitch_joint", "right_hip_pitch_joint",
        "left_elbow_pitch_joint", "right_elbow_pitch_joint", "left_hip_roll_joint",
        "right_hip_roll_joint", "left_elbow_yaw_joint", "right_elbow_yaw_joint",
        "left_hip_yaw_joint", "right_hip_yaw_joint", "left_knee_pitch_joint",
        "right_knee_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
        "left_ankle_roll_joint", "right_ankle_roll_joint"
    ],

    sim_body_names=[    # body order in isaacsim/isaaclab
        "trunk",
        "aahead_yaw_link",
        "left_shoulder_pitch_link",
        "right_shoulder_pitch_link",
        "waist_yaw_link",
        "aahead_pitch_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_hip_pitch_link",
        "right_hip_pitch_link",
        "left_elbow_pitch_link",
        "right_elbow_pitch_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_elbow_yaw_link",
        "right_elbow_yaw_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_knee_pitch_link",
        "right_knee_pitch_link",
        "left_ankle_pitch_link",
        "right_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
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
