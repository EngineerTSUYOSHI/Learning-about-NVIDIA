"""エピソードごとのタスク進捗（把持・リフト・箱投入・成功）を管理するモジュール。"""

from collections.abc import Sequence

import torch


class BallPlaceProgress:
    """エピソード単位のマイルストーンを保持するクラス。

    * ``grasped``: 両爪でボールを机から浮かせて ``grasp_steps`` 連続保持した状態
    * ``lifted``: ボールが箱の壁を越えられる安全高さに到達した状態
    * ``placed``: ボールを箱の中に投入してグリッパーを放した状態
    * ``success``: 箱の中で ``stable_steps`` 連続静止してタスク成功となった状態
    """

    def __init__(self, num_envs: int, device: str | torch.device, stable_steps: int = 10, grasp_steps: int = 2):
        self.stable_steps = stable_steps
        self.grasp_steps = grasp_steps
        self.grasped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.lifted = torch.zeros_like(self.grasped)
        self.placed = torch.zeros_like(self.grasped)
        self.success = torch.zeros_like(self.grasped)
        self.grasp_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.stable_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.time_to_success = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        """指定された環境IDの進捗のみリセットする。"""
        ids = slice(None) if env_ids is None else env_ids
        self.grasped[ids] = False
        self.lifted[ids] = False
        self.placed[ids] = False
        self.success[ids] = False
        self.grasp_count[ids] = 0
        self.stable_count[ids] = 0
        self.time_to_success[ids] = -1

    def update(
        self,
        holding: torch.Tensor,
        lifted: torch.Tensor,
        placed: torch.Tensor,
        settled: torch.Tensor,
        step: torch.Tensor,
    ) -> torch.Tensor:
        """現在の物理状態からマイルストーンを更新し、成功確定フラグを返す。"""
        self.grasp_count.copy_(torch.where(holding, self.grasp_count + 1, torch.zeros_like(self.grasp_count)))
        self.grasped |= self.grasp_count >= self.grasp_steps
        self.lifted |= lifted
        self.placed |= placed
        self.stable_count.copy_(torch.where(settled, self.stable_count + 1, torch.zeros_like(self.stable_count)))
        newly_successful = ~self.success & (self.stable_count >= self.stable_steps)
        self.time_to_success.copy_(torch.where(newly_successful, step, self.time_to_success))
        self.success |= newly_successful
        return self.success.clone()


def ball_place_progress(env) -> BallPlaceProgress:
    """終了判定・報酬・観測で共有する進捗インスタンスを返す。"""
    if not hasattr(env, "_so101_ball_place_progress"):
        env._so101_ball_place_progress = BallPlaceProgress(env.num_envs, env.device)
    return env._so101_ball_place_progress
