"""ボール把持＆箱投入（Pick & Place）の報酬関数定義。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

from .progress import ball_place_progress
from .task import (
    BALL_LIFT_TARGET_HEIGHT,
    BALL_PLACE_APPROACH_HEIGHT,
    BOX_WALL_THICKNESS,
    ball_grasp_point_w,
    ball_height,
    ball_in_box,
    ball_in_jaws,
    ball_is_holding,
    ball_lift_height_progress,
    ball_pick_target_height,
    ball_rest_height,
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
    target_height: float | None = None,
    pinch_bonus: float = 0.1,
) -> torch.Tensor:
    """爪の間でボールを挟んだ状態（``pinch_bonus``）から、目標持ち上げ高さまで持ち上げる密な報酬。

    目標高さに到達（``lifted``）した後は、目標手前でのホバリング局所解を防ぐため加点を停止する。
    """
    progress_state = ball_place_progress(env)
    in_jaws = ball_in_jaws(env)
    holding = ball_is_holding(env)
    target = ball_pick_target_height(env) if target_height is None else torch.full((env.num_envs,), target_height, device=env.device)
    max_travel = (target - ball_rest_height(env)).clamp_min(0.01)
    height_ratio = (ball_lift_height_progress(env) / max_travel).clamp(0.0, 1.0)
    lift_score = pinch_bonus + (1.0 - pinch_bonus) * height_ratio
    raw = torch.where(holding, lift_score, torch.where(in_jaws, pinch_bonus, 0.0))
    return torch.where(progress_state.lifted, 0.0, raw)


def place_goal(env: ManagerBasedRLEnv, std: float = 0.20) -> torch.Tensor:
    """安全高さまで持ち上げた後、壁を越えて箱の上空→箱の内側へ誘導し、リリースを促す報酬。

    1. 目標高さ（``lifted``）または保持したまま高さ ``0.10m`` 以上に達してから発動し、壁への衝突を防ぐ。
    2. 箱のXY範囲に入るまでは壁上空の高さ（``BALL_LIFT_TARGET_HEIGHT``）を維持させ、壁の角への衝突を防ぐ。
    3. 箱のXY範囲に入ったら箱の底付近へ降ろし、指を開く（``gripper_openness``）ことで投入完了を促す。
    """
    progress_state = ball_place_progress(env)
    holding = ball_is_holding(env)
    # 箱の中で指を開いた直後も報酬が切れないよう lifted も条件に含め、placed 完了まで維持する
    active = (progress_state.lifted | (holding & (ball_height(env) >= BALL_PLACE_APPROACH_HEIGHT))) & ~progress_state.placed

    ball_pos = ball_grasp_point_w(env)
    box_pos = env.scene["box"].data.root_pos_w.torch
    in_box_xy = ((ball_pos[:, 0] - box_pos[:, 0]).abs() < 0.045) & ((ball_pos[:, 1] - box_pos[:, 1]).abs() < 0.045)

    target = box_pos.clone()
    above_wall_z = env.scene.env_origins[:, 2] + BALL_LIFT_TARGET_HEIGHT
    inside_box_z = box_pos[:, 2] + BOX_WALL_THICKNESS + 0.035
    target[:, 2] = torch.where(in_box_xy, inside_box_z, above_wall_z)

    dist = torch.linalg.vector_norm(ball_pos - target, dim=-1)
    approach_box = torch.exp(-dist / std)
    # ボールが箱の内側に入っている時にグリッパーを開くと追加で加点し、リリースを促す
    in_box_release = torch.where(ball_in_box(env), 0.5 * gripper_openness(env), 0.0)
    return torch.where(active, approach_box + in_box_release, 0.0)


class milestones(ManagerTermBase):
    """エピソード中に一度だけ支払われるマイルストーン報酬（把持: 10.0, リフト: 20.0）。"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # 0=grasped(把持), 1=lifted(安全高さ到達)
        self.paid = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
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
        place_bonus: float = 0.0,
    ) -> torch.Tensor:
        progress = ball_place_progress(env)
        reached = torch.stack((progress.grasped, progress.lifted), dim=-1)
        bonus = torch.tensor((grasp_bonus, lift_bonus), dtype=torch.float32, device=self.device)
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
