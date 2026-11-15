"""観察・報酬・終了判定で共有するボールとグリッパーの物理量・判定ヘルパー。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# タスクの幾何学定数 [m]
BALL_RADIUS = 0.020
"""ボールの半径 [m]（直径40mm。現場に合わせて0.015〜0.025に変更可能）"""
BALL_REST_HEIGHT = 0.035 + BALL_RADIUS
"""35mmの机の上に静止しているボールの中心の高さ [m]"""
BALL_LIFT_TARGET_HEIGHT = 0.120
"""STEP 1の成功判定となる目標持ち上げ高さ [m]（机から約6.5cm上空）"""
BALL_LIFT_MIN_HEIGHT = BALL_REST_HEIGHT + 0.010
"""机から確実に浮いていると判定する最小クリアランス高さ [m]（机+1cm）"""
FIXED_FINGERTIP_OFFSET = (-0.01289, -0.00022, -0.0986)
MOVING_FINGERTIP_OFFSET = (-0.0068, -0.0766, 0.0189)


def contact_force(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """センサーが検知した最大の接触力 [N] を返す。"""
    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force_matrix_w if sensor.cfg.filter_prim_paths_expr else sensor.data.net_forces_w
    return torch.linalg.vector_norm(forces.torch, dim=-1).reshape(env.num_envs, -1).amax(dim=-1)


def jaw_contacts(env: ManagerBasedRLEnv, threshold: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """固定爪・可動爪それぞれがボールに接触しているかを返す。"""
    return contact_force(env, "fixed_jaw_contact") > threshold, contact_force(env, "moving_jaw_contact") > threshold


def fingertip_positions_w(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """固定爪・可動爪の先端パッド中心のワールド座標 [m] を返す。"""
    robot = env.scene["robot"]
    pads = []
    for body, offset in (("gripper", FIXED_FINGERTIP_OFFSET), ("moving_jaw_so101_v1", MOVING_FINGERTIP_OFFSET)):
        body_id = robot.find_bodies(body)[0][0]
        pos, quat = robot.data.body_pos_w.torch[:, body_id], robot.data.body_quat_w.torch[:, body_id]
        pads.append(pos + quat_apply(quat, pos.new_tensor(offset).expand_as(pos)))
    return pads[0], pads[1]


def grasp_center_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """両爪先端の中点（把持中心）のワールド座標 [m] を返す。"""
    fixed, moving = fingertip_positions_w(env)
    return 0.5 * (fixed + moving)


def ball_grasp_point_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ボールの把持目標点（中心座標）をワールド座標系 [m] で返す。"""
    return env.scene["ball"].data.root_pos_w.torch


def ball_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """環境原点に対するボール中心の高さ [m] を返す。"""
    return env.scene["ball"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]


def ball_is_holding(env: ManagerBasedRLEnv) -> torch.Tensor:
    """両爪がボールに接触し、かつボールが机のクリアランスを超えて浮いている場合に True を返す。"""
    fixed, moving = jaw_contacts(env)
    support_height = getattr(env, "_so101_support_height", 0.0)
    lifted_off_desk = ball_height(env) > (BALL_LIFT_MIN_HEIGHT + support_height)
    return fixed & moving & lifted_off_desk


def ball_lift_height_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """机の上に静止している基準高さから、どれだけ上に持ち上がったか [m]（0以上）。"""
    support_height = getattr(env, "_so101_support_height", 0.0)
    current_h = ball_height(env)
    rest_h = BALL_REST_HEIGHT + support_height
    return (current_h - rest_h).clamp_min(0.0)
