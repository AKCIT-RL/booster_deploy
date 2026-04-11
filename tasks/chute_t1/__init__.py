from booster_deploy.utils.registry import register_task
from booster_deploy.utils.isaaclab.configclass import configclass
from .chute_t1 import T1MjDance002ControllerCfg
# from .chute2_t1 import T1Chute2ControllerCfg
# from .chute3_t1 import T1Chute3ControllerCfg


@configclass
class T1MjDance002Cfg(T1MjDance002ControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/paper/booster_t1_trivela_14999.pt"
        self.policy.motion_path = "motions/paper/booster_t1_trivela.npz"
        # self.policy.checkpoint_path = "models/paper/penalti_ney_gmr_model_10500.pt"
        # self.policy.motion_path = "motions/paper/booster_t1_ney_ref.npz"
        self.policy.enable_safety_fallback = True


# @configclass
# class T1Chute2Cfg(T1Chute2ControllerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.checkpoint_path = "models/t1_teste3.pt"
#         self.policy.motion_path = "motions/t1_teste3.npz"


register_task("chute_t1", T1MjDance002Cfg())
# register_task("chute2_t1", T1Chute2Cfg())


# @configclass
# class T1Chute3Cfg(T1Chute3ControllerCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.policy.checkpoint_path = "/home/pettro/Documents/Talos/talos_2026/Motion/booster_deploy/tasks/t1_mj_dance_002/models/t1_chute.pt"


# register_task("chute3_t1", T1Chute3Cfg())
