"""ステップC（VLM連携＋方法Aハイブリッド制御）用の観測・アクション制御モジュール。

方法Aの仕組み:
1. 把持・持ち上げフェーズ（前半）:
   実際の箱がどこ（逆方向や移動先）にあっても、強化学習ポリシーの ``box`` 観測には
   「学習時の基準箱座標 ``(0.33, -0.10, 0.035)``」を渡す。これにより、箱の位置に惑わされず
   100%確実にボールを掴んで持ち上げさせる。
2. 箱（空きスロット）への運搬・リリースフェーズ（後半）:
   ボールが安全高さ（``BALL_PLACE_APPROACH_HEIGHT``）まで持ち上がった瞬間、俯瞰カメラ画像から
   ローカルVLM（またはモック）で空きスロット座標を判定し、事前キャリブレーション済みのFKテーブルで
   空きスロット真上へアームを誘導して指を開き、正確に投入する。
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.core.lift.mdp import body_state_b

from .task import (
    BALL_LIFT_TARGET_HEIGHT,
    BALL_PLACE_APPROACH_HEIGHT,
    BOX_HEIGHT,
    ball_height,
    ball_is_holding,
    ball_radius,
    grasp_center_w,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 俯瞰カメラ画像上の向きに合わせた4スロットのオフセット [m]
# 画像の上 = +X（ロボット前方）、画像の下 = -X（手前）、画像の左 = +Y（左）、画像の右 = -Y（右）
SLOT_OFFSETS: dict[str, tuple[float, float]] = {
    "top_left": (0.027, 0.027),
    "top_right": (0.027, -0.027),
    "bottom_left": (-0.027, 0.027),
    "bottom_right": (-0.027, -0.027),
}


def ask_local_vlm_empty_slot(
    rgb_tensor: torch.Tensor,
    vlm_url: str = "http://localhost:11434/api/chat",
    vlm_model: str = "qwen2.5vl:7b",
) -> str:
    """俯瞰カメラのRGB画像から、箱の中の空いているスロット名をローカルVLMに判定させる。"""
    from PIL import Image

    img_np = rgb_tensor[0, ..., :3].detach().cpu().numpy()
    if img_np.dtype != "uint8":
        img_np = (img_np.clip(0.0, 1.0) * 255).astype("uint8")
    pil_img = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        "画像には四角い緑色の箱があり、中に4つの区画（2x2）があります。\n"
        "現在3つの区画に青いボールが入っており、1つだけ空いています。\n"
        "空いている区画を次の4つから1つだけ選び、JSONのみで答えてください:\n"
        '{"empty_slot": "top_left" | "top_right" | "bottom_left" | "bottom_right"}'
    )
    payload = {
        "model": vlm_model,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
    }
    req = urllib.request.Request(
        vlm_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
    content = res_json["message"]["content"]
    print(f"[VLM応答] {content}")

    match = re.search(r'"empty_slot"\s*:\s*"(top_left|top_right|bottom_left|bottom_right)"', content)
    if match:
        return match.group(1)
    for key in SLOT_OFFSETS:
        if key in content:
            return key
    raise ValueError(f"VLMの応答からスロット名を抽出できませんでした: {content}")


def vlm_box_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
    nominal_box_pos: tuple[float, float, float] = (0.33, -0.10, 0.035),
) -> torch.Tensor:
    """強化学習ポリシーに渡す箱の観測。

    実際の箱を逆方向や離れた位置に動かしてもポリシーが混乱してボールを掴めなくならないよう、
    ポリシーには常に学習時の基準位置 ``nominal_box_pos`` をベース座標系で返す。
    """
    state = body_state_b(env, body_asset_cfg=body_asset_cfg, base_asset_cfg=base_asset_cfg).clone()
    robot = env.scene[base_asset_cfg.name]
    base_pos_w = robot.data.root_pos_w.torch
    base_quat_w = robot.data.root_quat_w.torch
    nominal_w = env.scene.env_origins + base_pos_w.new_tensor(nominal_box_pos)
    state[:, :3] = quat_apply_inverse(base_quat_w, nominal_w - base_pos_w)
    return state


def _ensure_vlm_state(env: ManagerBasedRLEnv) -> None:
    """環境上にVLM運搬制御用の状態テンソルとFKテーブルを初期化する。"""
    if hasattr(env, "_so101_vlm_initialized"):
        return
    n = env.num_envs
    device = env.device
    env._so101_vlm_initialized = True
    env._so101_vlm_transport_active = torch.zeros(n, dtype=torch.bool, device=device)
    env._so101_vlm_queried = torch.zeros(n, dtype=torch.bool, device=device)
    env._so101_vlm_drop_timer = torch.zeros(n, dtype=torch.long, device=device)
    env._so101_vlm_slot_offset = torch.zeros((n, 2), dtype=torch.float32, device=device)
    env._so101_vlm_q_above = torch.zeros((n, 5), dtype=torch.float32, device=device)
    env._so101_vlm_q_drop = torch.zeros((n, 5), dtype=torch.float32, device=device)
    env._so101_vlm_fk_table: tuple[torch.Tensor, torch.Tensor] | None = None


def _calibrate_fk_table(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """机全域（左右両方）をカバーする5関節角度と両爪中心ローカル座標(XYZ)の対応テーブルをFKで構築する。"""
    robot = env.scene["robot"]
    device = env.device
    saved_q = robot.data.joint_pos.torch.clone()

    pans = torch.linspace(-1.45, 1.45, 19, device=device)
    lifts = torch.linspace(-0.30, 0.40, 6, device=device)
    elbows = torch.linspace(-0.45, 0.20, 5, device=device)
    grid_pan, grid_lift, grid_elbow = torch.meshgrid(pans, lifts, elbows, indexing="ij")
    num_pts = grid_pan.numel()

    q_candidates = torch.zeros((num_pts, 6), device=device)
    q_candidates[:, 0] = grid_pan.reshape(-1)
    q_candidates[:, 1] = grid_lift.reshape(-1)
    q_candidates[:, 2] = grid_elbow.reshape(-1)
    q_candidates[:, 3] = 1.20
    q_candidates[:, 4] = -1.64
    q_candidates[:, 5] = 0.0

    tcp_local = torch.zeros((num_pts, 3), device=device)
    n = env.num_envs
    all_ids = torch.arange(n, device=device)
    origins = env.scene.env_origins

    for start in range(0, num_pts, n):
        chunk = q_candidates[start : start + n]
        k = len(chunk)
        robot.write_joint_position_to_sim_index(position=chunk, env_ids=all_ids[:k])
        env.sim.forward()
        tcp_w = grasp_center_w(env)[:k]
        tcp_local[start : start + k] = tcp_w - origins[:k]

    robot.write_joint_position_to_sim_index(position=saved_q, env_ids=all_ids)
    env.sim.forward()
    print(f"[VLM制御] アームFKテーブル構築完了 ({num_pts} サンプル)")
    return q_candidates[:, :5], tcp_local


def _solve_arm_joints_from_table(
    env: ManagerBasedRLEnv, target_local_xyz: torch.Tensor
) -> torch.Tensor:
    """FKテーブルから目標ローカル座標 ``target_local_xyz`` [N, 3] に対応する5関節角度 [N, 5] を補間算出する。"""
    if env._so101_vlm_fk_table is None:
        env._so101_vlm_fk_table = _calibrate_fk_table(env)
    q_table, tcp_table = env._so101_vlm_fk_table
    dists = torch.linalg.vector_norm(tcp_table.unsqueeze(0) - target_local_xyz.unsqueeze(1), dim=-1)
    top_d, top_i = dists.topk(4, largest=False, dim=-1)
    weights = (-top_d / 0.015).softmax(dim=-1)
    return (q_table[top_i] * weights.unsqueeze(-1)).sum(dim=1)


def _update_vlm_transport_state(env: ManagerBasedRLEnv) -> None:
    """毎ステップ把持・持ち上げ状態を監視し、持ち上げ完了時にVLM判定と目標関節角を計算する。"""
    _ensure_vlm_state(env)
    holding = ball_is_holding(env)
    lifted_now = holding & (ball_height(env) >= BALL_PLACE_APPROACH_HEIGHT)
    newly_active = lifted_now & ~env._so101_vlm_transport_active
    env._so101_vlm_transport_active |= lifted_now

    if newly_active.any() and not env._so101_vlm_queried.all():
        vlm_mode = os.environ.get("SO101_VLM_MODE", "auto").lower()
        vlm_url = os.environ.get("SO101_VLM_URL", "http://localhost:11434/api/chat")
        vlm_model = os.environ.get("SO101_VLM_MODEL", "qwen2.5vl:7b")

        default_offset = getattr(
            env, "_so101_empty_slot_offset", torch.zeros((env.num_envs, 2), device=env.device)
        ).clone()
        env._so101_vlm_slot_offset[:] = default_offset

        if vlm_mode != "mock" and "overhead_camera" in env.scene.sensors:
            try:
                env.scene.sensors["overhead_camera"].update(dt=env.step_dt)
                rgb = env.scene.sensors["overhead_camera"].data.output["rgb"].torch
                slot_key = ask_local_vlm_empty_slot(rgb, vlm_url=vlm_url, vlm_model=vlm_model)
                dx, dy = SLOT_OFFSETS[slot_key]
                env._so101_vlm_slot_offset[:, 0] = dx
                env._so101_vlm_slot_offset[:, 1] = dy
                print(f"[VLM制御] VLM判定スロット: {slot_key} -> offset=({dx:+.3f}, {dy:+.3f})")
            except Exception as err:
                print(
                    f"[VLM制御] ローカルVLM未接続のため環境の空きスロット座標を使用します "
                    f"(offset=({float(default_offset[0, 0]):+.3f}, {float(default_offset[0, 1]):+.3f})): {err}"
                )
        else:
            print(
                f"[VLM制御] 空きスロット座標へ運搬開始: "
                f"offset=({float(default_offset[0, 0]):+.3f}, {float(default_offset[0, 1]):+.3f})"
            )

        env._so101_vlm_queried[:] = True

    if env._so101_vlm_transport_active.any():
        origins = env.scene.env_origins
        box_local = env.scene["box"].data.root_pos_w.torch - origins
        target_xy_local = box_local[:, :2] + env._so101_vlm_slot_offset

        above_local = torch.zeros_like(box_local)
        above_local[:, :2] = target_xy_local
        above_local[:, 2] = BALL_LIFT_TARGET_HEIGHT + 0.015

        drop_local = torch.zeros_like(box_local)
        drop_local[:, :2] = target_xy_local
        drop_local[:, 2] = box_local[:, 2] + BOX_HEIGHT + ball_radius(env) + 0.005

        env._so101_vlm_q_above = _solve_arm_joints_from_table(env, above_local)
        env._so101_vlm_q_drop = _solve_arm_joints_from_table(env, drop_local)

        tcp_local = grasp_center_w(env) - origins
        xy_dist = torch.linalg.vector_norm(tcp_local[:, :2] - target_xy_local, dim=-1)
        aligned = env._so101_vlm_transport_active & (xy_dist < 0.025)
        env._so101_vlm_drop_timer = torch.where(
            aligned | (env._so101_vlm_drop_timer > 0),
            env._so101_vlm_drop_timer + 1,
            env._so101_vlm_drop_timer,
        )


class VLMGuidedArmAction(RelativeJointPositionAction):
    """持ち上げまではRLポリシーの行動をそのまま通し、持ち上げ後は空きスロット上空へアームを誘導する。"""

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        _ensure_vlm_state(self._env)
        ids = slice(None) if env_ids is None else env_ids
        self._env._so101_vlm_transport_active[ids] = False
        self._env._so101_vlm_queried[ids] = False
        self._env._so101_vlm_drop_timer[ids] = 0

    def process_action(self, actions: torch.Tensor) -> None:
        _update_vlm_transport_state(self._env)
        active = self._env._so101_vlm_transport_active
        if active.any():
            actions = actions.clone()
            current_q = self._asset.data.joint_pos.torch[:, self._joint_ids]
            timer = self._env._so101_vlm_drop_timer
            # 空きスロット真上に到達後 1〜12 ステップは少し降ろし、12ステップ以降は少し上に逃がす
            lowering = (timer >= 1) & (timer <= 12)
            target_q = torch.where(lowering.unsqueeze(-1), self._env._so101_vlm_q_drop, self._env._so101_vlm_q_above)
            scale = self._scale if isinstance(self._scale, float) else 0.033
            guided_act = ((target_q - current_q) / scale).clamp(-0.85, 0.85)
            actions = torch.where(active.unsqueeze(-1), guided_act, actions)
        super().process_action(actions)


class VLMGuidedGripperAction(RelativeJointPositionAction):
    """運搬中はグリッパーをしっかり閉じ続け、空きスロット真上に到達して降ろした後に開いて放す。"""

    def process_action(self, actions: torch.Tensor) -> None:
        _ensure_vlm_state(self._env)
        active = self._env._so101_vlm_transport_active
        if active.any():
            actions = actions.clone()
            timer = self._env._so101_vlm_drop_timer
            # 降下完了（timer > 10）したら +1.0（開く）、それまでの運搬中は -1.0（閉じて保持）
            grip_cmd = torch.where(timer > 10, 1.0, -1.0).unsqueeze(-1)
            actions = torch.where(active.unsqueeze(-1), grip_cmd, actions)
        super().process_action(actions)


@configclass
class VLMGuidedArmActionCfg(RelativeJointPositionActionCfg):
    class_type: type = VLMGuidedArmAction


@configclass
class VLMGuidedGripperActionCfg(RelativeJointPositionActionCfg):
    class_type: type = VLMGuidedGripperAction
