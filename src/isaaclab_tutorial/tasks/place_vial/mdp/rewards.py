"""Reward terms: dense approach and transport shaping, one-time milestones, and success."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

from .geometry import hole_relative_positions, symmetric_axial_keypoint_error
from .progress import placement_progress
from .task import VIAL_AXIS_MAX, VIAL_AXIS_MIN, contact_force, grasp_center_w, vial_grasp_point_w, vial_in_rack

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

HELD_INSERTION_TARGET = (0.0, 0.0, 0.060)
"""Rack-local hole-relative root position at the end of a held insertion; gravity finishes the travel."""


class approach_progress(ManagerTermBase):
    """Pay the decrease in pad-centre-to-cap distance, in units of ``scale``, until the vial is grasped.

    A difference of potentials has no incentive to hover, and gives the same gradient 25 cm from the vial as 2 cm
    away, which is what lets a pixels-only policy discover the approach from the home pose.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous_distance = torch.full((self.num_envs,), float("nan"), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.previous_distance[slice(None) if env_ids is None else env_ids] = float("nan")

    def __call__(self, env: ManagerBasedRLEnv, scale: float = 0.01, height_offset: float = 0.0) -> torch.Tensor:
        # Raising the target above the cap favours a shallower pinch; grasp milestones still need physical proof.
        target = vial_grasp_point_w(env)
        target = target + target.new_tensor((0.0, 0.0, height_offset))
        distance = torch.linalg.vector_norm(grasp_center_w(env) - target, dim=-1)
        progress = ((self.previous_distance - distance) / scale).clamp(-1.0, 1.0)
        self.previous_distance = distance
        # The first step after a reset has no previous distance (NaN) and pays nothing.
        return torch.where(placement_progress(env).grasped, 0.0, progress.nan_to_num(0.0))


# --- 元の処理（ラック挿入用のゴール誘導報酬） ---
# def held_goal(env: ManagerBasedRLEnv, std: float = 0.10) -> torch.Tensor:
#     """小瓶を把持した後、最も近いラック穴の上方へ運ぶための報酬。"""
#     position, axis, _ = vial_in_rack(env)
#     error = symmetric_axial_keypoint_error(...)
#     progress = placement_progress(env)
#     return torch.where(progress.grasped & ~progress.inserted, torch.exp(-error / std), 0.0)


# --- STEP 1: 持ち上げ連続シェーピング報酬（アプローチA） ---
def lift_progress(env: ManagerBasedRLEnv, target_height: float = 0.120) -> torch.Tensor:
    """ボールを両爪で把持して机から浮かせている間、目標高さに近づくほど連続的に加点される密な報酬。"""
    from .task import BALL_LIFT_TARGET_HEIGHT, VIAL_REST_HEIGHT, ball_is_holding, ball_lift_height_progress

    holding = ball_is_holding(env)
    target = target_height if target_height > 0.0 else BALL_LIFT_TARGET_HEIGHT
    max_travel = max(target - VIAL_REST_HEIGHT, 0.01)
    height_ratio = (ball_lift_height_progress(env) / max_travel).clamp(0.0, 1.0)
    # 単に机を押しているだけでは報酬を与えず、実際に把持して浮かせている時のみ高さに比例して報酬を与える
    return torch.where(holding, height_ratio, 0.0)


def held_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """設定ファイルとの後方互換性のためのエイリアス"""
    return lift_progress(env)


class milestones(ManagerTermBase):
    """エピソード中に一度だけ支払われるマイルストーン報酬（把持: 5.0, 目標高さリフト: 10.0）。"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Step 1 用のマイルストーン: 0=grasped(把持), 1=lifted(目標高さ到達)
        self.paid = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self.started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.paid[ids] = False
        self.started[ids] = False

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        progress = placement_progress(env)
        reached = torch.stack((progress.grasped, progress.lifted), dim=-1)
        # 配分比率: 把持=5.0, リフト完了=10.0
        reward = ((reached & ~self.paid).float() * reached.new_tensor((5.0, 10.0), dtype=torch.float32)).sum(-1)
        reward = torch.where(self.started, reward, 0.0)
        self.paid = reached.clone()
        self.started[:] = True
        return reward


def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-step confirmed placement success."""
    return placement_progress(env).success.float()


def jaw_support_force(env: ManagerBasedRLEnv, free_force: float = 1.0, force_span: float = 5.0) -> torch.Tensor:
    """Penalize either jaw pushing on the desk or mat beyond ``free_force`` [N], saturating after ``force_span``."""
    force = torch.maximum(contact_force(env, "fixed_jaw_support"), contact_force(env, "moving_jaw_support"))
    return ((force - free_force) / force_span).clamp(0.0, 1.0)
