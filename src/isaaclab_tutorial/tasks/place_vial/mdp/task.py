"""観察・報酬・終了判定で共有するボール・箱・グリッパーの物理量・判定ヘルパー。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 箱（カゴ）の幾何学定数 [m]（外寸 12cm x 12cm x 高さ5cm）
BOX_SIZE = 0.12
"""箱の外寸（幅・奥行） [m]"""
BOX_HEIGHT = 0.05
"""箱の高さ [m]"""
BOX_WALL_THICKNESS = 0.006
"""箱の壁・底板の厚み [m]"""

# タスクの基準幾何学定数 [m]（環境ごとの動的サイズがある場合は ball_radius(env) 等を優先する）
BALL_RADIUS = 0.025
"""基準のボール半径 [m]（直径50mm = 5cm。env._so101_ball_radius が無い場合のフォールバック値）"""
BALL_REST_HEIGHT = 0.035 + BALL_RADIUS
"""基準の静止高さ [m]（机35mm + 基準半径）"""
BALL_LIFT_TARGET_HEIGHT = 0.035 + BOX_HEIGHT + BALL_RADIUS + 0.010
"""箱の壁（BOX_HEIGHT）を余裕をもって越えられる持ち上げ目標高さ [m]（直径5cm時 0.120m）"""
BALL_PLACE_APPROACH_HEIGHT = 0.035 + BOX_HEIGHT + BALL_RADIUS + 0.005
"""箱への水平移動を開始する最低高さ [m]（ボール下端が壁上端を越える高さ、直径5cm時 0.115m）"""
BALL_LIFT_MIN_HEIGHT = BALL_REST_HEIGHT + 0.004
"""基準の浮上判定高さ [m]（静止高さ+4mm）"""
BALL_PICK_LIFT_DELTA = 0.040
"""Ver4 純粋把持（Pick）モデルにおける目標持ち上げ変位 [m]（静止高さ+4cmで把持完了とみなす）"""

FIXED_FINGERTIP_OFFSET = (-0.01289, -0.00022, -0.0986)
MOVING_FINGERTIP_OFFSET = (-0.0068, -0.0766, 0.0189)


def ball_radius(env: ManagerBasedRLEnv) -> torch.Tensor:
    """各環境のボール半径 [m] を返す（サイズランダム化時は環境ごとに異なる値になる）。"""
    if hasattr(env, "_so101_ball_radius"):
        return env._so101_ball_radius
    return torch.full((env.num_envs,), BALL_RADIUS, device=env.device)


def ball_rest_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """各環境の机（またはマット）上に静止しているボール中心の高さ [m] を返す。"""
    support_height = getattr(env, "_so101_support_height", 0.0)
    return 0.035 + ball_radius(env) + support_height


def ball_lift_min_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """各環境でボールが机から確実に浮いたと判定する最小高さ [m]（静止高さ + 4mm）を返す。"""
    return ball_rest_height(env) + 0.004


def ball_pick_target_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Ver4 純粋把持（Pick）モデルにおける各環境の目標持ち上げ高さ [m]（静止高さ + 4cm）を返す。"""
    return ball_rest_height(env) + BALL_PICK_LIFT_DELTA


def contact_force(env: ManagerBasedRLEnv, sensor_name: str) -> torch.Tensor:
    """センサーが検知した最大の接触力 [N] を返す。"""
    sensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force_matrix_w if sensor.cfg.filter_prim_paths_expr else sensor.data.net_forces_w
    return torch.linalg.vector_norm(forces.torch, dim=-1).reshape(env.num_envs, -1).amax(dim=-1)


def fixed_jaw_ball_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """固定爪（gripperリンク）が机・マット以外から受けている接触力 [N] を返す。

    ``fixed_jaw_contact`` はOV PhysXの制約でフィルタ無し（全接触の合力）のため、そのままだと机に触れただけで
    「ボールに接触」と誤判定する。``fixed_jaw_support`` が有効な場合は机・マットの力ベクトルを差し引く。
    """
    sensor = env.scene.sensors["fixed_jaw_contact"]
    if sensor.cfg.filter_prim_paths_expr:
        return contact_force(env, "fixed_jaw_contact")
    net = sensor.data.net_forces_w.torch.reshape(env.num_envs, -1, 3).sum(dim=1)
    if "fixed_jaw_support" in env.scene.sensors:
        support = env.scene.sensors["fixed_jaw_support"].data.force_matrix_w.torch
        net = net - support.reshape(env.num_envs, -1, 3).sum(dim=1)
    return torch.linalg.vector_norm(net, dim=-1)


def jaw_contacts(env: ManagerBasedRLEnv, threshold: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """固定爪・可動爪それぞれがボールに接触しているかを返す。"""
    return fixed_jaw_ball_force(env) > threshold, contact_force(env, "moving_jaw_contact") > threshold


def gripper_openness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """グリッパーの開き具合を [0, 1] に正規化して返す（0=全閉, 1=全開）。"""
    robot = env.scene["robot"]
    joint_id = robot.find_joints("gripper")[0][0]
    position = robot.data.joint_pos.torch[:, joint_id]
    lower, upper = robot.data.soft_joint_pos_limits.torch[:, joint_id].unbind(-1)
    return ((position - lower) / (upper - lower)).clamp(0.0, 1.0)


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


def ball_in_jaws(env: ManagerBasedRLEnv, margin: float = 0.006) -> torch.Tensor:
    """両爪の中心付近（``ボール半径 + margin`` 以内）にボールがあり、かつ両爪がボールに接触している場合に True を返す。"""
    fixed, moving = jaw_contacts(env)
    max_distance = ball_radius(env) + margin
    centered = torch.linalg.vector_norm(grasp_center_w(env) - ball_grasp_point_w(env), dim=-1) < max_distance
    return fixed & moving & centered


def ball_is_holding(env: ManagerBasedRLEnv) -> torch.Tensor:
    """両爪の中心付近でボールを挟み、かつ各環境のサイズに応じた浮上クリアランスを超えている場合に True を返す。"""
    # fixed_jaw_contact が無フィルタの場合でも机・箱接触だけで誤判定しないよう ball_in_jaws（距離条件付き）を使う
    lifted_off_desk = ball_height(env) > ball_lift_min_height(env)
    return ball_in_jaws(env) & lifted_off_desk


def ball_lift_height_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """各環境の机上静止高さから、どれだけ上に持ち上がったか [m]（0以上）。"""
    return (ball_height(env) - ball_rest_height(env)).clamp_min(0.0)


def box_target_pos_w(env: ManagerBasedRLEnv, height_above_floor: float = 0.045) -> torch.Tensor:
    """箱の中心上空の目標位置（ワールド座標系 [m]）を返す。"""
    box_pos = env.scene["box"].data.root_pos_w.torch.clone()
    box_pos[:, 2] += BOX_WALL_THICKNESS + height_above_floor
    return box_pos


def ball_in_box(env: ManagerBasedRLEnv) -> torch.Tensor:
    """各環境のボールサイズに応じて、ボールが箱の内側（底の上〜壁の高さ以内）に入っている場合に True を返す。"""
    ball_pos = env.scene["ball"].data.root_pos_w.torch
    box_pos = env.scene["box"].data.root_pos_w.torch
    rel = ball_pos - box_pos
    r = ball_radius(env)
    inner_half = 0.5 * BOX_SIZE - BOX_WALL_THICKNESS - 0.5 * r
    in_xy = (rel[:, 0].abs() <= inner_half) & (rel[:, 1].abs() <= inner_half)
    in_z = (rel[:, 2] >= BOX_WALL_THICKNESS + 0.5 * r) & (rel[:, 2] <= BOX_HEIGHT + r)
    return in_xy & in_z
