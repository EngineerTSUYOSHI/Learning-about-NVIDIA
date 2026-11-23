"""ステップC（VLM連携＋方法Aハイブリッド制御・Lv1検証）用の環境設定。

既存の学習用環境（``SO101VialEnvCfg``）を継承し、以下を追加している（既存タスクには一切影響しない）：
1. 箱の中に最初から配置される3個の既存ボール（``box_ball_0〜2``）
2. 箱の上空から真下を見下ろす俯瞰カメラ（``overhead_camera``）
3. 方法Aハイブリッド制御:
   - ボール把持・持ち上げ中は、実際の箱がどこ（逆方向や移動先）にあっても学習時の基準箱座標を観測に渡すことで、
     確実にボールを掴んで持ち上げさせる。
   - 持ち上げ完了後は、VLM（または環境の空きスロット座標）で判定した空きスロットの真上へ誘導して投入する。

公式CLIでそのまま実行可能:
  uv run isaaclab play --rl_library rsl_rl --task IsaacTutorial-Place-Ball-VLM-SO101 --num_envs 1 --checkpoint /path/to/model.pt presets=newton_mjwarp,newton_renderer
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import preset
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_tutorial.assets import BALL_USD
from isaaclab_tutorial.tasks.place_vial import mdp

from .env_cfg import (
    ARM_JOINTS,
    BallSpawnCfg,
    CriticStateGroupCfg,
    PolicyStateGroupCfg,
    SO101SceneCfg,
    SO101VialEnvCfg,
)


def _overhead_renderer_cfg() -> MultiBackendRendererCfg:
    """俯瞰カメラ用のレンダラ設定を返す。"""
    cfg = MultiBackendRendererCfg()
    cfg.newton_renderer.enable_shadows = True
    cfg.default.enable_shadows = True
    cfg.ovrtx.enable_shadows = True
    return cfg


@configclass
class SO101VLMSceneCfg(SO101SceneCfg):
    """箱内ボール3個と俯瞰カメラを追加したVLM検証用シーン設定。"""

    # 箱の中に最初から入っている3個のボール（reset_ball_step1 で4スロットのうち3箇所へランダム配置される）
    box_ball_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoxBall0",
        spawn=BallSpawnCfg(usd_path=str(BALL_USD)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.33, -0.10, 0.066), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    box_ball_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoxBall1",
        spawn=BallSpawnCfg(usd_path=str(BALL_USD)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.33, -0.10, 0.066), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    box_ball_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoxBall2",
        spawn=BallSpawnCfg(usd_path=str(BALL_USD)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.33, -0.10, 0.066), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    # 箱の上空（z=0.45m）から真下を見下ろす俯瞰カメラ（XYZW順: 画像の上=ロボット前方+X, 画像の右=ロボット右側-Y）
    overhead_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverheadCamera",
        spawn=PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(
            pos=(0.33, -0.10, 0.45),
            rot=(0.7071068, -0.7071068, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["rgb"],
        width=320,
        height=240,
        update_period=0.0,
        renderer_cfg=_overhead_renderer_cfg(),
        background_color=(0.3, 0.3, 0.3),
    )


@configclass
class VLMActionsCfg:
    """方法A: 持ち上げまではRLポリシーに従い、持ち上げ後はVLM判定の空きスロット真上へ誘導して放すアクション設定。"""

    arm_action = mdp.VLMGuidedArmActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        preserve_order=True,
        scale=preset(
            default=0.033,
            sim2real={
                "shoulder_pan": 0.033,
                "shoulder_lift": 0.12,
                "elbow_flex": 0.07,
                "wrist_flex": 0.033,
                "wrist_roll": 0.033,
            },
        ),
        use_zero_offset=True,
    )
    gripper_action = mdp.VLMGuidedGripperActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=0.02,
        use_zero_offset=True,
    )


@configclass
class VLMPolicyStateGroupCfg(PolicyStateGroupCfg):
    """箱をどこ（逆方向含む）に動かしてもRLポリシーが確実にボールを掴めるよう、基準箱座標を観測に渡す設定。"""

    box = ObsTerm(
        func=mdp.vlm_box_state_b,
        params={
            "body_asset_cfg": SceneEntityCfg("box"),
            "base_asset_cfg": SceneEntityCfg("robot"),
            "nominal_box_pos": (0.33, -0.10, 0.035),
        },
    )


@configclass
class VLMCriticStateGroupCfg(CriticStateGroupCfg):
    box = ObsTerm(
        func=mdp.vlm_box_state_b,
        params={
            "body_asset_cfg": SceneEntityCfg("box"),
            "base_asset_cfg": SceneEntityCfg("robot"),
            "nominal_box_pos": (0.33, -0.10, 0.035),
        },
    )


@configclass
class VLMObservationsCfg:
    policy: VLMPolicyStateGroupCfg = VLMPolicyStateGroupCfg()
    critic: VLMCriticStateGroupCfg = VLMCriticStateGroupCfg()


@configclass
class SO101VialVLMEnvCfg(SO101VialEnvCfg):
    """VLMによる空きスロット判断と学習済みモデルの推論検証を行うタスク設定。"""

    scene: SO101VLMSceneCfg = SO101VLMSceneCfg(num_envs=1, env_spacing=0.9, replicate_physics=True)
    actions: VLMActionsCfg = VLMActionsCfg()
    observations: VLMObservationsCfg = VLMObservationsCfg()

    def play_mode(self):
        super().play_mode()
        self.scene.num_envs = 1
        # 検証時は100%机上スタートから一連の動作を確認する
        self.events.reset_ball.params["grasp_prob"] = 0.0
