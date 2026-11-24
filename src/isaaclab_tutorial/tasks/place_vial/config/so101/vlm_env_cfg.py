"""ステップC（VLM連携・2カメラハイブリッド制御・Lv1検証）用の環境設定。

既存の学習用環境（``SO101VialEnvCfg``）を継承し、以下を追加している（既存タスクには一切影響しない）：
1. 箱の中に最初から配置される3個の既存ボール（``box_ball_0〜2``）
2. 箱の上空から真下を見下ろす全体俯瞰カメラ（``overhead_camera``）
3. ハイブリッド制御アクション（``VLMActionsCfg``）:
   - 開始直後に全体俯瞰カメラで箱の空きスペースをVLM判定・記憶する。
   - ボール把持・持ち上げまでは学習済みRLモデルの行動をそのまま実行し、
     持ち上げ完了後に記憶した空きスロット真上へ誘導して正確に投入する。

公式CLIでそのまま実行可能:
  uv run isaaclab play --rl_library rsl_rl --task IsaacTutorial-Place-Ball-VLM-SO101 --num_envs 1 --checkpoint /path/to/model.pt presets=newton_mjwarp,newton_renderer
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import preset
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_tutorial.assets import BALL_USD
from isaaclab_tutorial.tasks.place_vial import mdp

from .env_cfg import ARM_JOINTS, BallSpawnCfg, OpenBoxSpawnCfg, SO101SceneCfg, SO101VialEnvCfg


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

    # プリミティブ生成の上開き箱（12cm x 12cm x 高さ5cm、ロボットの少し右前に配置）
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=OpenBoxSpawnCfg(size=0.12, height=0.05, wall_thickness=0.006),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.33, -0.10, 0.035),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

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
    """持ち上げまではRLポリシーに従い、持ち上げ後はVLM判定の空きスロット真上へ誘導して放すアクション設定。"""

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
class SO101VialVLMEnvCfg(SO101VialEnvCfg):
    """VLMによる空きスロット判断と学習済みモデルの推論検証を行うタスク設定（観測は親の body_state_b をそのまま継承）。"""

    scene: SO101VLMSceneCfg = SO101VLMSceneCfg(num_envs=1, env_spacing=0.9, replicate_physics=True)
    actions: VLMActionsCfg = VLMActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        # VLM検証タスクでは、進捗更新（grasped/lifted）を維持しつつ運搬・投入まで完遂できるよう terminate=False に設定
        self.terminations.success.params = {"terminate": False}
        # 把持・持ち上げ・水平移動・降下・リリース・退避まで完遂できるようエピソード長を30秒に設定
        self.episode_length_s = 30.0

    def play_mode(self):
        super().play_mode()
        self.scene.num_envs = 1
        # 検証時は100%机上スタートから一連の動作を確認する
        self.events.reset_ball.params["grasp_prob"] = 0.0
