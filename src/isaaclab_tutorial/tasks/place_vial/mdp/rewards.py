"""ボール把持＆箱投入（Pick & Place）の報酬関数定義。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

from .progress import ball_place_progress
from .task import (
    BALL_LIFT_TARGET_HEIGHT,
    ball_grasp_point_w,
    ball_in_box,
    ball_in_jaws,
    ball_is_holding,
    ball_lift_height_progress,
    ball_rest_height,
    box_target_pos_w,
    contact_force,
    grasp_center_w,
    gripper_openness,
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
        # すでにボールを保持中または箱に入った後は接近報酬を0にする
        progress_state = ball_place_progress(env)
        inactive = ball_is_holding(env) | progress_state.placed
        return torch.where(inactive, 0.0, progress.nan_to_num(0.0))


def lift_progress(
    env: ManagerBasedRLEnv,
    target_height: float = BALL_LIFT_TARGET_HEIGHT,
    pinch_bonus: float = 0.1,
) -> torch.Tensor:
    """爪の間でボールを挟んだ状態（``pinch_bonus``）から、箱の壁を越える目標高さまで持ち上げる密な報酬。

    すでに目標高さへ到達（``lifted``）した後は、箱への運搬（``place_goal``）に集中できるよう加点を止める。
    """
    progress_state = ball_place_progress(env)
    in_jaws = ball_in_jaws(env)
    holding = ball_is_holding(env)
    target = target_height if target_height > 0.0 else BALL_LIFT_TARGET_HEIGHT
    max_travel = (target - ball_rest_height(env)).clamp_min(0.01)
    height_ratio = (ball_lift_height_progress(env) / max_travel).clamp(0.0, 1.0)
    lift_score = pinch_bonus + (1.0 - pinch_bonus) * height_ratio
    raw = torch.where(holding, lift_score, torch.where(in_jaws, pinch_bonus, 0.0))
    return torch.where(progress_state.lifted | progress_state.placed, 0.0, raw)


def place_goal(env: ManagerBasedRLEnv, std: float = 0.12) -> torch.Tensor:
    """持ち上げたボールを箱の中心上空へ運ぶほど大きくなる密な誘導報酬。

    ボールが箱の内側上空に到達して指を開いて放す段階も自然に促すため、箱の目標付近にいる状態も評価する。
    """
    progress_state = ball_place_progress(env)
    holding = ball_is_holding(env)
    # 一度持ち上げた後、箱に投入完了するまで「ボールと箱の目標位置の距離」を縮めるよう誘導する
    active = (progress_state.lifted | holding) & ~progress_state.placed
    dist = torch.linalg.vector_norm(ball_grasp_point_w(env) - box_target_pos_w(env), dim=-1)
    approach_box = torch.exp(-dist / std)
    # ボールが箱の内側に入っている時にグリッパーを開くと追加で小さく加点し、リリースを促す
    in_box_release = torch.where(ball_in_box(env), 0.3 * gripper_openness(env), 0.0)
    return torch.where(active, approach_box + in_box_release, 0.0)


class milestones(ManagerTermBase):
    """エピソード中に一度だけ支払われるマイルストーン報酬（把持: 10.0, リフト: 20.0, 箱投入: 40.0）。"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 0=grasped(把持), 1=lifted(安全高さ到達), 2=placed(箱投入＆リリース)
        self.paid = torch.zeros((self.num_envs, 3), dtype=torch.bool, device=self.device)
        self.started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.paid[ids] = False
        self.started[ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        grasp_bonus: float = 10.0,
        lift_bonus: float = 20.0,
        place_bonus: float = 40.0,
    ) -> torch.Tensor:
        progress = ball_place_progress(env)
        reached = torch.stack((progress.grasped, progress.lifted, progress.placed), dim=-1)
        bonus = torch.tensor((grasp_bonus, lift_bonus, place_bonus), dtype=torch.float32, device=self.device)
        reward = ((reached & ~self.paid).float() * bonus).sum(-1)
        reward = torch.where(self.started, reward, 0.0)
        self.paid = reached.clone()
        self.started[:] = True
        return reward


def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ボールが箱の中で安定静止してタスク成功となった時の1ステップ報酬。"""
    return ball_place_progress(env).success.float()


def jaw_support_force(env: ManagerBasedRLEnv, free_force: float = 1.0, force_span: float = 5.0) -> torch.Tensor:
    """爪が机やマットを ``free_force`` [N] 以上で押し付けた際のペナルティ（上から押さえつけ防止）。"""
    force = torch.maximum(contact_force(env, "fixed_jaw_support"), contact_force(env, "moving_jaw_support"))
    return ((force - free_force) / force_span).clamp(0.0, 1.0)
