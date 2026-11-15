"""STEP 1（ボール把持＆持ち上げ）の報酬関数定義。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

from .progress import ball_lift_progress
from .task import (
    BALL_LIFT_TARGET_HEIGHT,
    BALL_REST_HEIGHT,
    ball_grasp_point_w,
    ball_in_jaws,
    ball_is_holding,
    ball_lift_height_progress,
    contact_force,
    grasp_center_w,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class approach_progress(ManagerTermBase):
    """両爪中心とボール中心との距離が縮まるたびに支払われるポテンシャル差分報酬。"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous_distance = torch.full((self.num_envs,), float("nan"), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_distance[slice(None) if env_ids is None else env_ids] = float("nan")

    def __call__(self, env: ManagerBasedRLEnv, scale: float = 0.01, height_offset: float = 0.0) -> torch.Tensor:
        target = ball_grasp_point_w(env)
        if height_offset != 0.0:
            target = target + target.new_tensor((0.0, 0.0, height_offset))
        distance = torch.linalg.vector_norm(grasp_center_w(env) - target, dim=-1)
        progress = ((self.previous_distance - distance) / scale).clamp(-1.0, 1.0)
        self.previous_distance = distance
        # リセット直後の最初のステップは前回距離がNaNなので0を返す。
        # 「一度把持したら以降ずっと0」にすると、把持後に落とした場合に再接近の誘導が無くなるため、
        # 現在ボールを保持している間だけ0にする（差分報酬なので離れる→近づくの往復で得はしない）。
        return torch.where(ball_is_holding(env), 0.0, progress.nan_to_num(0.0))


def lift_progress(
    env: ManagerBasedRLEnv,
    target_height: float = BALL_LIFT_TARGET_HEIGHT,
    pinch_bonus: float = 0.1,
) -> torch.Tensor:
    """爪の間でボールを挟んだ状態（``pinch_bonus``）から、目標高さへ持ち上げるにつれて最大 1.0 まで連続的に加点される密な報酬。

    目標直前で保持し続けるだけの報酬が成功ボーナスに匹敵しないよう、環境設定側の重みは小さめにしている。
    """
    in_jaws = ball_in_jaws(env)
    holding = ball_is_holding(env)
    target = target_height if target_height > 0.0 else BALL_LIFT_TARGET_HEIGHT
    max_travel = max(target - BALL_REST_HEIGHT, 0.01)
    height_ratio = (ball_lift_height_progress(env) / max_travel).clamp(0.0, 1.0)
    # 工程②: 机の上でも爪の間にボールを入れて両爪で挟めばベース報酬（pinch_bonus）を付与
    # 工程③: 机から浮かせている間（holding）は高さに応じて pinch_bonus 〜 1.0 まで連続加点
    lift_score = pinch_bonus + (1.0 - pinch_bonus) * height_ratio
    return torch.where(holding, lift_score, torch.where(in_jaws, pinch_bonus, 0.0))


class milestones(ManagerTermBase):
    """エピソード中に一度だけ支払われるマイルストーン報酬（既定値 把持: 10.0, 目標高さリフト: 20.0）。"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Step 1 用のマイルストーン: 0=grasped(把持), 1=lifted(目標高さ到達)
        self.paid = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.paid[ids] = False
        self.started[ids] = False

    def __call__(self, env: ManagerBasedRLEnv, grasp_bonus: float = 10.0, lift_bonus: float = 20.0) -> torch.Tensor:
        progress = ball_lift_progress(env)
        reached = torch.stack((progress.grasped, progress.lifted), dim=-1)
        bonus = torch.tensor((grasp_bonus, lift_bonus), dtype=torch.float32, device=self.device)
        reward = ((reached & ~self.paid).float() * bonus).sum(-1)
        reward = torch.where(self.started, reward, 0.0)
        self.paid = reached.clone()
        self.started[:] = True
        return reward


def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """目標高さ以上で安定保持してタスク成功となった時の1ステップ報酬。"""
    return ball_lift_progress(env).success.float()


def jaw_support_force(env: ManagerBasedRLEnv, free_force: float = 1.0, force_span: float = 5.0) -> torch.Tensor:
    """爪が机やマットを ``free_force`` [N] 以上で押し付けた際のペナルティ（上から押さえつけ防止）。"""
    force = torch.maximum(contact_force(env, "fixed_jaw_support"), contact_force(env, "moving_jaw_support"))
    return ((force - free_force) / force_span).clamp(0.0, 1.0)
