"""ボール把持＆箱投入（Pick & Place）の終了判定モジュール。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from .progress import ball_place_progress
from .task import (
    BALL_LIFT_TARGET_HEIGHT,
    ball_grasp_point_w,
    ball_height,
    ball_in_box,
    ball_is_holding,
    ball_lift_min_height,
    ball_radius,
    contact_force,
    grasp_center_w,
    gripper_openness,
    jaw_contacts,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ball_place_success(ManagerTermBase):
    """毎ステップ進捗を更新し、ボールが箱の中で放されて一定ステップ静止したらタスク成功として終了する。"""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.progress = ball_place_progress(env)
        self.max_support_force = torch.zeros(self.num_envs, device=self.device)
        self.support_contact_time = torch.zeros_like(self.max_support_force)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.progress.reset(ids)
        self.max_support_force[ids] = 0.0
        self.support_contact_time[ids] = 0.0
        if hasattr(self._env, "_so101_reset_grasped"):
            grasped = self._env._so101_reset_grasped[ids]
            self.progress.grasped[ids] = grasped
            self.progress.grasp_count[ids] = torch.where(grasped, self.progress.grasp_steps, 0)
            self.progress.lifted[ids] = self._env._so101_reset_lifted[ids]

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        holding = ball_is_holding(env)  # 両爪でボールを挟み、机から浮かせているか
        lifted_to_target = holding & (ball_height(env) >= BALL_LIFT_TARGET_HEIGHT)  # 箱の壁を越える高さに達したか

        # 箱に入っており、爪がボールから離れてグリッパーが開いているか
        # （fixed_jaw_contact は無フィルタのため、爪がボールから離れていれば箱壁等に当たっていても放したとみなす）
        fixed, moving = jaw_contacts(env)
        near_ball = torch.linalg.vector_norm(grasp_center_w(env) - ball_grasp_point_w(env), dim=-1) < (
            ball_radius(env) + 0.015
        )
        touching_ball = (fixed | moving) & near_ball
        released = ~touching_ball & (gripper_openness(env) > 0.25)
        in_box = ball_in_box(env)
        placed = in_box & released

        # 箱の中で速度・角速度が十分に落ちて静止しているか
        ball = env.scene["ball"]
        still = (torch.linalg.vector_norm(ball.data.root_lin_vel_w.torch, dim=-1) < 0.08) & (
            torch.linalg.vector_norm(ball.data.root_ang_vel_w.torch, dim=-1) < 1.0
        )
        settled = placed & still

        success = self.progress.update(holding, lifted_to_target, placed, settled, env.episode_length_buf)

        log = env.extras.setdefault("log", {})
        log["Metrics/grasp_rate"] = self.progress.grasped.float().mean()
        log["Metrics/lift_rate"] = self.progress.lifted.float().mean()
        log["Metrics/place_rate"] = self.progress.placed.float().mean()
        log["Metrics/success_rate"] = success.float().mean()
        if "fixed_jaw_support" in env.scene.sensors:
            support_force = torch.maximum(
                contact_force(env, "fixed_jaw_support"), contact_force(env, "moving_jaw_support")
            )
            self.max_support_force = torch.maximum(self.max_support_force, support_force)
            self.support_contact_time += (support_force > 1.0).float() * env.step_dt
            log["Safety/peak_jaw_support_force_n"] = self.max_support_force.mean()
            env._so101_terminal_max_support_force = self.max_support_force.clone()
            env._so101_terminal_support_contact_time = self.support_contact_time.clone()

        env._so101_terminal_time_to_success_s = self.progress.time_to_success.float() * env.step_dt
        return success


def ball_dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """一度持ち上げたボールが箱の外の机（マット）の上まで落ちた場合に終了判定を返す。"""
    fallen = ball_height(env) < ball_lift_min_height(env)
    return ball_place_progress(env).lifted & fallen & ~ball_in_box(env)
