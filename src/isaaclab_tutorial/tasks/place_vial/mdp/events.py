"""イベント処理: ボールのランダム配置リセット、および関節・エンコーダ・カメラのランダマイズ。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    """Event managers pass ``None`` or ``slice(None)`` for all environments."""
    if env_ids is None:
        env_ids = slice(None)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[env_ids]
    return env_ids


class randomize_joint_viscous_friction(ManagerTermBase):
    """Scale passive viscous joint friction per environment, relative to the startup values.

    Isaac Lab randomizes Coulomb friction and armature (``randomize_joint_parameters``) and drive gains
    (``randomize_actuator_gains``) but not viscous friction.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.default_viscous = self.asset.data.joint_viscous_friction_coeff.torch.clone()

    def __call__(
        self, env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, scale_range: tuple
    ) -> None:
        ids = _ids(env, env_ids)
        joint_ids = asset_cfg.joint_ids
        viscous = self.default_viscous[ids][:, joint_ids]
        self.asset.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=self.asset.data.joint_friction_coeff.torch[ids][:, joint_ids],
            joint_viscous_friction_coeff=viscous * torch.empty_like(viscous).uniform_(*scale_range),
            joint_ids=None if joint_ids == slice(None) else joint_ids,
            env_ids=ids,
        )


def randomize_encoder_bias(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, bound: float = 0.01) -> None:
    """Sample an episode-constant joint calibration error [rad], read by the encoder observations."""
    ids = _ids(env, env_ids)
    if not hasattr(env, "_so101_encoder_bias"):
        env._so101_encoder_bias = torch.zeros(env.num_envs, 6, device=env.device)
    env._so101_encoder_bias[ids] = torch.empty(len(ids), 6, device=env.device).uniform_(-bound, bound)


class randomize_wrist_camera_mount(ManagerTermBase):
    """Offset the wrist camera from its nominal mount by a per-episode translation [m] and rotation [rad]."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.nominal_position = self.nominal_orientation = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        position_range: float,
        rotation_range: float,
    ) -> None:
        ids = _ids(env, env_ids)
        camera = env.scene.sensors[sensor_cfg.name]
        robot = env.scene[asset_cfg.name]
        # Dataset resets just wrote joint positions; refresh kinematics before reading the carrier link.
        env.sim.forward()
        carrier_position = robot.data.body_pos_w.torch[:, asset_cfg.body_ids[0]]
        carrier_orientation = robot.data.body_quat_w.torch[:, asset_cfg.body_ids[0]]
        if self.nominal_position is None:
            self.nominal_position, self.nominal_orientation = subtract_frame_transforms(
                carrier_position, carrier_orientation, camera.data.pos_w.torch, camera.data.quat_w_ros.torch
            )
        translation = torch.empty(len(ids), 3, device=env.device).uniform_(-position_range, position_range)
        rotation = quat_from_euler_xyz(*torch.empty_like(translation).uniform_(-rotation_range, rotation_range).T)
        local_position, local_orientation = combine_frame_transforms(
            self.nominal_position[ids], self.nominal_orientation[ids], translation, rotation
        )
        position, orientation = combine_frame_transforms(
            carrier_position[ids], carrier_orientation[ids], local_position, local_orientation
        )
        camera.set_world_poses(position, orientation, env_ids=ids, convention="ros")
        # Reading camera.data above may have rendered the old pose; invalidate it for the new episode.
        camera.reset(env_ids=ids)


# --- STEP 1: ボール把持＆持ち上げ用のリセットイベント ---
class reset_ball_step1(ManagerTermBase):
    """ロボットとボールをリセットする。

    * 通常スタート（確率 ``1 - grasp_prob``）: ロボットをホーム姿勢にリセットし、机の上のランダム位置にボールを配置する。
    * 把持スタート（確率 ``grasp_prob``）: 机のすぐ上で爪がボールを挟んだ状態から開始し、持ち上げ動作（リフト）の学習を促進する。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
            GRASP_GRIPPER_POSITION,
            WORKSHOP_GRASP_JOINT_POSITION,
            WORKSHOP_INITIAL_JOINT_POSITION,
            WORKSHOP_LIFT_MID_JOINT_POSITION,
        )

        self.default_joint_pos = torch.tensor(WORKSHOP_INITIAL_JOINT_POSITION, device=env.device)
        self.grasp_joint_pos = torch.tensor(WORKSHOP_GRASP_JOINT_POSITION, device=env.device)
        self.lift_mid_joint_pos = torch.tensor(WORKSHOP_LIFT_MID_JOINT_POSITION, device=env.device)
        self.grasp_gripper_target = float(GRASP_GRIPPER_POSITION)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        ball_pos_center: tuple[float, float] = (0.23, 0.0),
        ball_pos_noise: tuple[float, float] = (0.03, 0.04),
        grasp_prob: float = 0.3,
    ) -> None:
        from .progress import ball_lift_progress
        from .task import BALL_LIFT_MIN_HEIGHT, BALL_REST_HEIGHT, grasp_center_w

        ids = _ids(env, env_ids)
        n = len(ids)

        # 各環境を「把持スタート」にするか「通常ホームスタート」にするかを確率 grasp_prob で決定
        is_grasped = torch.rand(n, device=env.device) < grasp_prob if grasp_prob > 0.0 else torch.zeros(
            n, dtype=torch.bool, device=env.device
        )

        # 1. ロボット関節角度・目標角度の設定
        robot = env.scene["robot"]
        joint_pos = self.default_joint_pos.unsqueeze(0).expand(n, -1).clone()
        joint_target = joint_pos.clone()

        if is_grasped.any():
            # 把持スタート環境: 机直上の把持ポーズ〜持ち上げ序盤ポーズの間をランダム補間し、少し関節ノイズを加える
            blend = torch.empty((n, 1), device=env.device).uniform_(0.0, 0.35)
            grasp_pose = torch.lerp(self.grasp_joint_pos.unsqueeze(0), self.lift_mid_joint_pos.unsqueeze(0), blend)
            # XY方向の多様性を持たせるため shoulder_pan 等に微小なランダムノイズを付与
            noise = torch.zeros_like(grasp_pose)
            noise[:, 0].uniform_(-0.15, 0.15)  # shoulder_pan
            noise[:, 1].uniform_(-0.03, 0.03)  # shoulder_lift
            noise[:, 2].uniform_(-0.03, 0.03)  # elbow_flex
            grasp_pose = grasp_pose + noise

            # グリッパーの目標角度は閉じ方向（GRASP_GRIPPER_POSITION）に設定してボールをしっかり保持させる
            grasp_target = grasp_pose.clone()
            grasp_target[:, 5] = self.grasp_gripper_target

            joint_pos = torch.where(is_grasped.unsqueeze(-1), grasp_pose, joint_pos)
            joint_target = torch.where(is_grasped.unsqueeze(-1), grasp_target, joint_target)

        zeros = torch.zeros_like(joint_pos)
        robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(velocity=zeros, env_ids=ids)
        robot.set_joint_position_target_index(target=joint_target, env_ids=ids)
        robot.set_joint_velocity_target_index(target=zeros, env_ids=ids)

        # 2. ボールの配置（通常スタートは机の上、把持スタートは両爪の中心座標へ配置）
        ball_pos = torch.zeros((n, 7), device=env.device)
        u_x = torch.empty(n, device=env.device).uniform_(-ball_pos_noise[0], ball_pos_noise[0])
        u_y = torch.empty(n, device=env.device).uniform_(-ball_pos_noise[1], ball_pos_noise[1])
        support_h = getattr(env, "_so101_support_height", 0.0)
        h = support_h[ids] if isinstance(support_h, torch.Tensor) else support_h

        ball_pos[:, 0] = ball_pos_center[0] + u_x
        ball_pos[:, 1] = ball_pos_center[1] + u_y
        ball_pos[:, 2] = BALL_REST_HEIGHT + h
        ball_pos[:, 6] = 1.0  # 単位クォータニオン (XYZW: 0, 0, 0, 1)

        # 環境原点のオフセットを適用
        ball_world_pos = ball_pos.clone()
        ball_world_pos[:, :3] += env.scene.env_origins[ids]

        if is_grasped.any():
            # 順運動学（FK）を更新して、現在の関節角度における両爪先端の中心座標（ワールド座標系）を取得
            env.sim.forward()
            jaw_center_w = grasp_center_w(env)[ids]
            # 机とのめり込みを防ぎ、かつ浮いている判定（BALL_LIFT_MIN_HEIGHT）を満たす最低高さを保証
            min_z_w = env.scene.env_origins[ids, 2] + BALL_LIFT_MIN_HEIGHT + 0.002 + h
            jaw_center_w[:, 2] = torch.maximum(jaw_center_w[:, 2], min_z_w)

            ball_world_pos[is_grasped, :3] = jaw_center_w[is_grasped]
            ball_pos[is_grasped, :3] = ball_world_pos[is_grasped, :3] - env.scene.env_origins[ids[is_grasped]]

        ball = env.scene["ball"]
        ball.write_root_pose_to_sim_index(root_pose=ball_world_pos, env_ids=ids)
        ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros((n, 6), device=env.device), env_ids=ids)

        # 3. 把持・リフト等の追跡フラグを初期化（把持スタート環境は最初から grasped=True に設定）
        for attr, shape, val in (
            ("_so101_reset_grasped", (env.num_envs,), is_grasped),
            ("_so101_reset_lifted", (env.num_envs,), False),
            ("_so101_reset_ball_pose", (env.num_envs, 7), ball_pos),
        ):
            if not hasattr(env, attr):
                dtype = torch.bool if "grasp" in attr or "lift" in attr else torch.float32
                setattr(env, attr, torch.zeros(shape, device=env.device, dtype=dtype))
            getattr(env, attr)[ids] = val

        # 進捗トラッカーにも把持スタート状態を同期
        progress = ball_lift_progress(env)
        progress.grasped[ids] = is_grasped
        progress.grasp_count[ids] = torch.where(is_grasped, progress.grasp_steps, 0)
        progress.lifted[ids] = False
