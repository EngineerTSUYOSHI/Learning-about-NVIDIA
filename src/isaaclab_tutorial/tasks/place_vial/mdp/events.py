"""イベント処理: ボールのランダム配置リセット、および関節・エンコーダ・カメラのランダマイズ。"""

from __future__ import annotations

import math
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

    把持スタートのグリッパー角度は、両爪先端の間隔がボール直径 + ``grip_clearance`` になる角度を
    初回リセット時にFKで求めて使う（固定値だとボールと爪がめり込んで弾き飛ばされるか、隙間が空いて落下するため）。
    行動は「現在角度からの相対目標」なので、リセット時に与えた閉じ方向の目標は最初の1ステップしか効かない。
    以降の保持はポリシー自身がグリッパーを閉じ続けることで行う。
    """

    # キャリブレーションで掃引するグリッパー角度範囲 [rad]（直径50mm前後のボールも十分含む範囲）
    _CALIBRATION_RANGE = (math.radians(-12.0), math.radians(75.0))
    _CALIBRATION_SAMPLES = 64

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
            _ENV_BALL_RADII,
            WORKSHOP_GRASP_JOINT_POSITION,
            WORKSHOP_INITIAL_JOINT_POSITION,
            WORKSHOP_LIFT_MID_JOINT_POSITION,
        )
        from .task import BALL_RADIUS

        self.default_joint_pos = torch.tensor(WORKSHOP_INITIAL_JOINT_POSITION, device=env.device)
        self.grasp_joint_pos = torch.tensor(WORKSHOP_GRASP_JOINT_POSITION, device=env.device)
        self.lift_mid_joint_pos = torch.tensor(WORKSHOP_LIFT_MID_JOINT_POSITION, device=env.device)
        self.grip_table: tuple[torch.Tensor, torch.Tensor] | None = None

        # 環境ごとのボール半径を env._so101_ball_radius に登録（サイズ自動判定の基準になる）
        if len(_ENV_BALL_RADII) == env.num_envs:
            env._so101_ball_radius = torch.tensor(_ENV_BALL_RADII, dtype=torch.float32, device=env.device)
        else:
            env._so101_ball_radius = torch.full((env.num_envs,), BALL_RADIUS, dtype=torch.float32, device=env.device)

    def _calibrate_grip_table(self, env: ManagerBasedRLEnv, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """グリッパー角度と両爪先端間隔の対応テーブル（ソート済み）をFKで測定する。"""
        from .task import fingertip_positions_w

        robot = env.scene["robot"]
        angles = torch.linspace(*self._CALIBRATION_RANGE, self._CALIBRATION_SAMPLES, device=env.device)
        gaps = torch.empty_like(angles)
        n = len(ids)
        for start in range(0, len(angles), n):
            chunk = angles[start : start + n]
            k = len(chunk)
            pose = self.grasp_joint_pos.unsqueeze(0).expand(k, -1).clone()
            pose[:, 5] = chunk
            robot.write_joint_position_to_sim_index(position=pose, env_ids=ids[:k])
            env.sim.forward()
            fixed, moving = fingertip_positions_w(env)
            gaps[start : start + k] = torch.linalg.vector_norm(fixed[ids[:k]] - moving[ids[:k]], dim=-1)

        order = torch.argsort(gaps)
        sorted_gaps, sorted_angles = gaps[order], angles[order]
        print(
            f"[reset_ball_step1] 爪間隔キャリブレーション完了: "
            f"間隔 {float(sorted_gaps[0])*1000:.1f}〜{float(sorted_gaps[-1])*1000:.1f} mm "
            f"(角度 {math.degrees(float(sorted_angles[0])):.1f}〜{math.degrees(float(sorted_angles[-1])):.1f} deg)"
        )
        return sorted_gaps, sorted_angles

    def _grip_angles_for_radii(
        self, radii: torch.Tensor, clearance: float
    ) -> torch.Tensor:
        """各環境のボール半径 ``radii`` に対して、爪先端間隔 ``2*r + clearance`` になるグリッパー角度を線形補間で求める。"""
        assert self.grip_table is not None
        sorted_gaps, sorted_angles = self.grip_table
        target_gaps = (2.0 * radii + clearance).clamp(sorted_gaps[0], sorted_gaps[-1])
        idx = torch.searchsorted(sorted_gaps, target_gaps).clamp(1, len(sorted_gaps) - 1)
        g0, g1 = sorted_gaps[idx - 1], sorted_gaps[idx]
        a0, a1 = sorted_angles[idx - 1], sorted_angles[idx]
        denom = (g1 - g0).clamp_min(1.0e-8)
        return a0 + (target_gaps - g0) * (a1 - a0) / denom

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        ball_pos_center: tuple[float, float] = (0.33, 0.08),
        ball_pos_noise: tuple[float, float] = (0.03, 0.04),
        box_pos_center: tuple[float, float] = (0.33, -0.10),
        box_pos_noise: tuple[float, float] = (0.03, 0.04),
        grasp_prob: float = 0.0,
        grip_clearance: float = 0.001,
        grip_squeeze: float = 0.15,
    ) -> None:
        from .task import (
            BALL_LIFT_TARGET_HEIGHT,
            ball_lift_min_height,
            ball_pick_target_height,
            ball_radius,
            ball_rest_height,
            fingertip_positions_w,
            grasp_center_w,
        )

        ids = _ids(env, env_ids)
        n = len(ids)
        robot = env.scene["robot"]
        origins = env.scene.env_origins[ids]
        support_h = getattr(env, "_so101_support_height", 0.0)
        h = support_h[ids] if isinstance(support_h, torch.Tensor) else support_h
        radii = ball_radius(env)[ids]
        rest_h = ball_rest_height(env)[ids]
        min_lift_h = ball_lift_min_height(env)[ids]

        # 0. 箱（box）を先にランダム配置しておく（把持スタート時の箱干渉チェックに使うため）
        box_xy_local = None
        if "box" in env.scene.keys():  # noqa: SIM118
            from .task import BOX_SIZE

            box_pose = torch.zeros((n, 7), device=env.device)
            bx = torch.empty(n, device=env.device).uniform_(-box_pos_noise[0], box_pos_noise[0])
            by = torch.empty(n, device=env.device).uniform_(-box_pos_noise[1], box_pos_noise[1])
            box_pose[:, 0] = box_pos_center[0] + bx
            box_pose[:, 1] = box_pos_center[1] + by
            box_pose[:, 2] = 0.035 + h
            box_pose[:, 6] = 1.0  # 単位クォータニオン (XYZW: 0, 0, 0, 1)
            box_xy_local = box_pose[:, :2].clone()
            box_pose[:, :3] += origins
            box = env.scene["box"]
            box.write_root_pose_to_sim_index(root_pose=box_pose, env_ids=ids)
            box.write_root_velocity_to_sim_index(root_velocity=torch.zeros((n, 6), device=env.device), env_ids=ids)

            # Lv1検証用（VLMタスク等）: シーンに box_ball_0〜2 が存在する場合のみ、箱の中の4スロットのうち3箇所へ自動配置
            if "box_ball_0" in env.scene.keys():  # noqa: SIM118
                from .task import BOX_WALL_THICKNESS

                slots = box_pose.new_tensor(
                    [[-0.027, -0.027], [0.027, -0.027], [-0.027, 0.027], [0.027, 0.027]]
                )
                perms = torch.argsort(torch.rand(n, 4, device=env.device), dim=-1)
                zero_vel = torch.zeros((n, 6), device=env.device)
                for slot_idx, name in enumerate(("box_ball_0", "box_ball_1", "box_ball_2")):
                    if name in env.scene.keys():  # noqa: SIM118
                        chosen = slots[perms[:, slot_idx]]
                        b_pose = box_pose.clone()
                        b_pose[:, :2] += chosen
                        b_pose[:, 2] += BOX_WALL_THICKNESS + radii
                        env.scene[name].write_root_pose_to_sim_index(root_pose=b_pose, env_ids=ids)
                        env.scene[name].write_root_velocity_to_sim_index(root_velocity=zero_vel, env_ids=ids)
                if not hasattr(env, "_so101_empty_slot_offset"):
                    env._so101_empty_slot_offset = torch.zeros((env.num_envs, 2), device=env.device)
                env._so101_empty_slot_offset[ids] = slots[perms[:, 3]]

        # 各環境を「把持スタート」にするか「通常ホームスタート」にするかを確率 grasp_prob で決定
        is_grasped = torch.rand(n, device=env.device) < grasp_prob
        default_pose = self.default_joint_pos.unsqueeze(0).expand(n, -1)

        # 1. 把持スタート姿勢の生成と妥当性チェック
        grasp_pose = default_pose.clone()
        jaw_center_w = None
        if is_grasped.any():
            if self.grip_table is None:
                self.grip_table = self._calibrate_grip_table(env, ids)

            # 机直上の把持ポーズ〜持ち上げ序盤ポーズの間をランダム補間し、少し関節ノイズを加える
            blend = torch.empty((n, 1), device=env.device).uniform_(0.0, 0.20)
            grasp_pose = torch.lerp(self.grasp_joint_pos.unsqueeze(0), self.lift_mid_joint_pos.unsqueeze(0), blend)
            noise = torch.zeros_like(grasp_pose)
            noise[:, 0].uniform_(-0.40, 0.40)  # shoulder_pan（ボール出現側にも届くよう広めに設定）
            noise[:, 1].uniform_(-0.03, 0.03)  # shoulder_lift
            noise[:, 2].uniform_(-0.03, 0.03)  # elbow_flex
            grasp_pose = grasp_pose + noise
            # 各環境のボール直径＋クリアランスになる角度をテーブルから算出
            grasp_pose[:, 5] = self._grip_angles_for_radii(radii, grip_clearance)

            # FKで爪の位置を確認し、机にめり込む姿勢・最初から目標高さ付近にある姿勢・箱と重なる姿勢は通常スタートへ
            robot.write_joint_position_to_sim_index(
                position=torch.where(is_grasped.unsqueeze(-1), grasp_pose, default_pose), env_ids=ids
            )
            env.sim.forward()
            jaw_center_w = grasp_center_w(env)[ids]
            fixed_w, moving_w = fingertip_positions_w(env)
            lowest_tip = torch.minimum(fixed_w[ids, 2], moving_w[ids, 2]) - origins[:, 2]
            center_xy = jaw_center_w[:, :2] - origins[:, :2]
            center_z = jaw_center_w[:, 2] - origins[:, 2]
            desk_top = 0.035 + h
            target_pick_h = ball_pick_target_height(env)[ids]
            valid = (
                (center_z >= min_lift_h + 0.002)
                & (center_z <= target_pick_h - 0.010)
                & (lowest_tip >= desk_top + 0.002)
            )
            if box_xy_local is not None:
                # 爪・ボールが箱の壁と重ならないよう XY クリアランス（箱半径 + ボール半径 + 2cm）を確保
                clear_half = 0.5 * BOX_SIZE + radii + 0.020
                in_box_footprint = (
                    ((center_xy[:, 0] - box_xy_local[:, 0]).abs() < clear_half)
                    & ((center_xy[:, 1] - box_xy_local[:, 1]).abs() < clear_half)
                )
                valid &= ~in_box_footprint
            is_grasped &= valid

        # 2. ロボット関節角度・目標角度の確定
        joint_pos = torch.where(is_grasped.unsqueeze(-1), grasp_pose, default_pose)
        joint_target = joint_pos.clone()
        # 把持スタートは最初の制御周期だけでも握り込むよう、閉じ方向の目標を与える
        # （相対関節位置アクションのため、2ステップ目以降の握り込みはポリシーの出力で決まる）
        joint_target[:, 5] = torch.where(is_grasped, joint_pos[:, 5] - grip_squeeze, joint_pos[:, 5])

        zeros = torch.zeros_like(joint_pos)
        robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(velocity=zeros, env_ids=ids)
        robot.set_joint_position_target_index(target=joint_target, env_ids=ids)
        robot.set_joint_velocity_target_index(target=zeros, env_ids=ids)

        # 3. ボールの配置（通常スタートは机の上、把持スタートは両爪先端の中点へ配置）
        ball_pos = torch.zeros((n, 7), device=env.device)
        u_x = torch.empty(n, device=env.device).uniform_(-ball_pos_noise[0], ball_pos_noise[0])
        u_y = torch.empty(n, device=env.device).uniform_(-ball_pos_noise[1], ball_pos_noise[1])
        ball_pos[:, 0] = ball_pos_center[0] + u_x
        ball_pos[:, 1] = ball_pos_center[1] + u_y
        ball_pos[:, 2] = rest_h
        ball_pos[:, 6] = 1.0  # 単位クォータニオン (XYZW: 0, 0, 0, 1)
        if jaw_center_w is not None:
            ball_pos[is_grasped, :3] = jaw_center_w[is_grasped] - origins[is_grasped]

        # 環境原点のオフセットを適用
        ball_world_pos = ball_pos.clone()
        ball_world_pos[:, :3] += origins

        ball = env.scene["ball"]
        ball.write_root_pose_to_sim_index(root_pose=ball_world_pos, env_ids=ids)
        ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros((n, 6), device=env.device), env_ids=ids)

        # 4. 追跡フラグの初期化
        # 把持スタートでも grasped は最初から立てない。実際に保持できているかは進捗判定（両爪接触＋浮上が
        # 連続 grasp_steps）に任せる。最初から立てると、取り落とした場合に観測フラグが実態と食い違い、
        # 接近報酬や把持マイルストーンも得られなくなるため。
        for attr, shape, dtype, val in (
            ("_so101_reset_grasped", (env.num_envs,), torch.bool, False),
            ("_so101_reset_lifted", (env.num_envs,), torch.bool, False),
            ("_so101_reset_grasp_start", (env.num_envs,), torch.bool, is_grasped),
            ("_so101_reset_ball_pose", (env.num_envs, 7), torch.float32, ball_pos),
        ):
            if not hasattr(env, attr):
                setattr(env, attr, torch.zeros(shape, device=env.device, dtype=dtype))
            getattr(env, attr)[ids] = val
