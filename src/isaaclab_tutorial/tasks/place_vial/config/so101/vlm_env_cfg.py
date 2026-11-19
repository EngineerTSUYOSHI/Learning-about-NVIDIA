"""ステップC（VLM連携・Lv1検証）用の環境設定。

既存の学習用環境（``SO101VialEnvCfg``）を継承し、以下を追加している（既存タスクには一切影響しない）：
1. 箱の中に最初から配置される3個の既存ボール（``box_ball_0〜2``）
2. 箱の上空から真下を見下ろす俯瞰カメラ（``overhead_camera``）
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_tutorial.assets import BALL_USD
from isaaclab_tutorial.tasks.place_vial import mdp

from .env_cfg import BallSpawnCfg, SO101SceneCfg, SO101VialEnvCfg


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
class SO101VialVLMEnvCfg(SO101VialEnvCfg):
    """VLMによる空きスロット判断と学習済みモデルの推論検証を行うタスク設定。"""

    scene: SO101VLMSceneCfg = SO101VLMSceneCfg(num_envs=1, env_spacing=0.9, replicate_physics=True)

    def play_mode(self):
        super().play_mode()
        self.scene.num_envs = 1
        # 検証時は100%机上スタートから一連の動作を確認する
        self.events.reset_ball.params["grasp_prob"] = 0.0
