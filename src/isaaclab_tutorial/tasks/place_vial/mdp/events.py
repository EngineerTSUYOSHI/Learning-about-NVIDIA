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
    """ロボットを初期姿勢にリセットし、机の上のランダムな位置にボールを配置する。"""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import WORKSHOP_INITIAL_JOINT_POSITION
        self.default_joint_pos = torch.tensor(WORKSHOP_INITIAL_JOINT_POSITION, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        ball_pos_center: tuple[float, float] = (0.23, 0.0),
        ball_pos_noise: tuple[float, float] = (0.03, 0.04),
    ) -> None:
        from .task import BALL_REST_HEIGHT

        ids = _ids(env, env_ids)
        n = len(ids)

        # 1. ロボット関節を初期ポーズ（ホーム位置）へリセット
        robot = env.scene["robot"]
        joint_pos = self.default_joint_pos.unsqueeze(0).expand(n, -1)
        zeros = torch.zeros_like(joint_pos)
        robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(velocity=zeros, env_ids=ids)
        robot.set_joint_position_target_index(target=joint_pos, env_ids=ids)
        robot.set_joint_velocity_target_index(target=zeros, env_ids=ids)

        # 2. 机の上の基準位置（±ランダムノイズ）にボールを配置
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

        ball = env.scene["ball"]
        ball.write_root_pose_to_sim_index(root_pose=ball_world_pos, env_ids=ids)
        ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros((n, 6), device=env.device), env_ids=ids)

        # 3. 把持・リフト等の追跡フラグを初期化
        for attr, shape in (
            ("_so101_reset_grasped", (env.num_envs,)),
            ("_so101_reset_lifted", (env.num_envs,)),
            ("_so101_reset_ball_pose", (env.num_envs, 7)),
        ):
            if not hasattr(env, attr):
                setattr(env, attr, torch.zeros(shape, device=env.device, dtype=torch.bool if "grasp" in attr or "lift" in attr else torch.float32))
            getattr(env, attr)[ids] = False if "grasp" in attr or "lift" in attr else ball_pos
