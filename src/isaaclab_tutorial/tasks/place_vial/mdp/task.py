"""Vial, rack and gripper quantities shared by the observation, reward and termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from .geometry import cylinder_lowest_offset, hole_relative_positions, rack_local_position, vertical_alignment

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# タスクの幾何学定数 [m]
# --- 元の定義（小瓶: Vial） ---
# VIAL_REST_HEIGHT = 0.052
# """机の上に横たわる小瓶のルート高さ"""
# VIAL_AXIS_MIN, VIAL_AXIS_MAX, VIAL_RADIUS = -0.017, 0.100, 0.0177
# """小瓶の軸方向の長さ範囲と衝突判定の半径"""
# VIAL_GRASP_OFFSET = (0.0, 0.0, 0.093145)
# """グリッパーの爪が閉じる小瓶キャップ部のオフセット"""

# --- STEP 1: ボールの把持＆持ち上げタスク用の定義 ---
BALL_RADIUS = 0.020
"""ボールの半径 [m]（直径40mm。現場に合わせて0.015〜0.025に変更可能）"""
VIAL_REST_HEIGHT = 0.035 + BALL_RADIUS
"""35mmの机の上に静止しているボールの中心の高さ [m]"""
VIAL_AXIS_MIN, VIAL_AXIS_MAX, VIAL_RADIUS = -BALL_RADIUS, BALL_RADIUS, BALL_RADIUS
"""球体のサイズ範囲と衝突判定の半径"""
VIAL_GRASP_OFFSET = (0.0, 0.0, 0.0)
"""ボールの掴み目標位置（球体の中心原点）"""
BALL_LIFT_TARGET_HEIGHT = 0.120
"""STEP 1の成功判定となる目標持ち上げ高さ [m]（机から約6.5cm上空）"""
BALL_LIFT_MIN_HEIGHT = VIAL_REST_HEIGHT + 0.010
"""机から確実に浮いていると判定する最小クリアランス高さ [m]（机+1cm）"""
FIXED_FINGERTIP_OFFSET = (-0.01289, -0.00022, -0.0986)
MOVING_FINGERTIP_OFFSET = (-0.0068, -0.0766, 0.0189)
RACK_RIM_HEIGHT = 0.073
RACK_CLEARANCE_HEIGHT = RACK_RIM_HEIGHT + 0.008
"""A vial whose lowest point is this high above the rack base has been lifted clear of the rack."""
INSERTION_RADIUS = 0.012
"""A vial tip below the rim within this distance of a hole centre is in that hole (holes are 60 mm apart)."""
UPRIGHT_ALIGNMENT = 0.90
"""Cosine of the vial axis with vertical; about 26 degrees of tilt."""
SEATED_LOWER, SEATED_UPPER = (-0.015, -0.015, 0.026), (0.015, 0.015, 0.040)
"""Hole-relative box that contains a seated vial root but not a vial resting on the rim."""
HARD_RACK_IMPACT_FORCE = 20.0
"""Light rack guidance is expected during insertion; only larger vial/rack forces [N] count as impacts."""


def contact_force(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """Largest contact force magnitude [N] of a sensor, filtered if the sensor declares filters."""
    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force_matrix_w if sensor.cfg.filter_prim_paths_expr else sensor.data.net_forces_w
    return torch.linalg.vector_norm(forces.torch, dim=-1).reshape(env.num_envs, -1).amax(dim=-1)


def jaw_contacts(env: ManagerBasedRLEnv, threshold: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Whether the fixed and the moving jaw touch the vial."""
    return contact_force(env, "fixed_jaw_contact") > threshold, contact_force(env, "moving_jaw_contact") > threshold


def fingertip_positions_w(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed and moving fingertip pad centres in the world frame [m]."""
    robot = env.scene["robot"]
    pads = []
    for body, offset in (("gripper", FIXED_FINGERTIP_OFFSET), ("moving_jaw_so101_v1", MOVING_FINGERTIP_OFFSET)):
        body_id = robot.find_bodies(body)[0][0]
        pos, quat = robot.data.body_pos_w.torch[:, body_id], robot.data.body_quat_w.torch[:, body_id]
        pads.append(pos + quat_apply(quat, pos.new_tensor(offset).expand_as(pos)))
    return pads[0], pads[1]


def grasp_center_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Midpoint between the two fingertip pads in the world frame [m]."""
    fixed, moving = fingertip_positions_w(env)
    return 0.5 * (fixed + moving)


def vial_grasp_point_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The cap/shoulder point of the vial in the world frame [m]."""
    vial = env.scene["vial"]
    pos, quat = vial.data.root_pos_w.torch, vial.data.root_quat_w.torch
    return pos + quat_apply(quat, pos.new_tensor(VIAL_GRASP_OFFSET).expand_as(pos))


def vial_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vial root height above the environment origin [m]."""
    return env.scene["vial"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]


def vial_in_rack(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rack-local vial root position, rack-local vial axis, and the vial's lowest rack-local height."""
    # --- 元の処理（シーン内にラックが存在することを前提） ---
    # vial, rack = env.scene["vial"], env.scene["rack"]
    # vial_pos, vial_quat = vial.data.root_pos_w.torch, vial.data.root_quat_w.torch
    # rack_pos, rack_quat = rack.data.root_pos_w.torch, rack.data.root_quat_w.torch
    # position = rack_local_position(vial_pos, rack_pos, rack_quat)
    # axis = quat_apply_inverse(
    #     rack_quat, quat_apply(vial_quat, vial_pos.new_tensor((0.0, 0.0, 1.0)).expand_as(vial_pos))
    # )
    # lowest = position[:, 2] + cylinder_lowest_offset(axis[:, 2], VIAL_AXIS_MIN, VIAL_AXIS_MAX, VIAL_RADIUS)
    # return position, axis, lowest

    # --- STEP 1: ラックをシーンから削除した場合の安全なフォールバック ---
    vial = env.scene["vial"]
    vial_pos, vial_quat = vial.data.root_pos_w.torch, vial.data.root_quat_w.torch
    if "rack" in env.scene.keys():  # noqa: SIM118
        rack = env.scene["rack"]
        rack_pos, rack_quat = rack.data.root_pos_w.torch, rack.data.root_quat_w.torch
        position = rack_local_position(vial_pos, rack_pos, rack_quat)
        axis = quat_apply_inverse(
            rack_quat, quat_apply(vial_quat, vial_pos.new_tensor((0.0, 0.0, 1.0)).expand_as(vial_pos))
        )
    else:
        # ラックがない場合はボール自身の座標系をそのまま使用
        position = vial_pos
        axis = quat_apply(vial_quat, vial_pos.new_tensor((0.0, 0.0, 1.0)).expand_as(vial_pos))
    lowest = position[:, 2] + cylinder_lowest_offset(axis[:, 2], VIAL_AXIS_MIN, VIAL_AXIS_MAX, VIAL_RADIUS)
    return position, axis, lowest


def nearest_hole_distance(position: torch.Tensor) -> torch.Tensor:
    """Horizontal distance [m] from a rack-local position to the closest hole centre."""
    return torch.linalg.vector_norm(hole_relative_positions(position)[..., :2], dim=-1).amin(dim=-1)


def gripper_openness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Measured jaw opening normalized to [0, 1] over the soft joint limits."""
    robot = env.scene["robot"]
    joint_id = robot.find_joints("gripper")[0][0]
    position = robot.data.joint_pos.torch[:, joint_id]
    lower, upper = robot.data.soft_joint_pos_limits.torch[:, joint_id].unbind(-1)
    return ((position - lower) / (upper - lower)).clamp(0.0, 1.0)


def vial_upright(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cosine of the vial axis with world vertical, clamped to [0, 1]."""
    return vertical_alignment(env.scene["vial"].data.root_quat_w.torch)


# --- STEP 1: ボールの把持＆持ち上げ判定用ヘルパー関数 ---
def ball_is_holding(env: ManagerBasedRLEnv) -> torch.Tensor:
    """両爪がボールに接触し、かつボールが机のクリアランスを超えて浮いている場合に True を返す。"""
    fixed, moving = jaw_contacts(env)
    support_height = getattr(env, "_so101_support_height", 0.0)
    lifted_off_desk = vial_height(env) > (BALL_LIFT_MIN_HEIGHT + support_height)
    return fixed & moving & lifted_off_desk


def ball_lift_height_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """机の上に静止している基準高さから、どれだけ上に持ち上がったか [m]（0以上）。"""
    support_height = getattr(env, "_so101_support_height", 0.0)
    current_h = vial_height(env)
    rest_h = VIAL_REST_HEIGHT + support_height
    return (current_h - rest_h).clamp_min(0.0)
