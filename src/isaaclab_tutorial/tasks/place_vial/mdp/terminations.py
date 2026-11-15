"""STEP 1（ボール把持＆持ち上げ）の終了判定モジュール。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from .progress import ball_lift_progress
from .task import BALL_LIFT_MIN_HEIGHT, BALL_LIFT_TARGET_HEIGHT, ball_height, ball_is_holding, contact_force

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ball_lift_success(ManagerTermBase):
    """毎ステップ進捗を更新し、ボールを目標高さ以上で一定ステップ安定保持できたらエピソード成功として終了する。"""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.progress = ball_lift_progress(env)
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
        lifted_to_target = holding & (ball_height(env) >= BALL_LIFT_TARGET_HEIGHT)  # 把持したまま目標高さに到達したか
        success = self.progress.update(holding, lifted_to_target, env.episode_length_buf)

        log = env.extras.setdefault("log", {})
        log["Metrics/grasp_rate"] = self.progress.grasped.float().mean()
        log["Metrics/lift_rate"] = self.progress.lifted.float().mean()
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
    """一度目標高さまで持ち上げたボールが机（マット）の上まで落ちた場合に終了判定を返す。

    以前はロボット根元（z≈0）との比較だったため、机上（ボール中心 z≈0.055）への落下では発火しなかった。
    机上で静止しているとみなす高さ（``BALL_LIFT_MIN_HEIGHT`` 未満）まで戻ったら落下とする。
    """
    support_height = getattr(env, "_so101_support_height", 0.0)
    fallen = ball_height(env) < BALL_LIFT_MIN_HEIGHT + support_height
    return ball_lift_progress(env).lifted & fallen
