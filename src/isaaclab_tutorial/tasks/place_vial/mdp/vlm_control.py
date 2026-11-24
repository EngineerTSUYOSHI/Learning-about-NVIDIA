"""ステップC（VLM連携＋2カメラハイブリッド制御）用のアクション制御モジュール。

制御の流れ:
1. 開始直後（アームがホーム姿勢で箱を隠していない時）:
   全体俯瞰カメラ（``overhead_camera``）の映像から箱の領域を切り抜き（クロップ＋2x2十字線描画）、
   ローカルVLM（未接続または ``SO101_VLM_MODE=mock`` 時は環境の空きスロット座標）で空きスペースを記憶する。
2. 前半（把持・持ち上げフェーズ）:
   学習済み強化学習モデル（.pt）の行動をそのまま通し、ボールを掴んで安全高さまで持ち上げさせる。
3. 後半（運搬・投入フェーズ）:
   安全高さ（``BALL_PLACE_APPROACH_HEIGHT``）に達した瞬間、手首の回転角（``wrist_roll``）を維持したまま
   FKテーブル補間で記憶した空きスロットの真上へ誘導し、位置誤差 1.5cm 以内＆静止を確認してから
   降下・リリースして正確に空きスペースへ投入する。
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

from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.utils.configclass import configclass

from .task import (
    BALL_LIFT_TARGET_HEIGHT,
    BALL_PLACE_APPROACH_HEIGHT,
    BOX_HEIGHT,
    BOX_SIZE,
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


def _crop_and_annotate_box_image(
    rgb_tensor: torch.Tensor,
    box_local_xy: tuple[float, float],
    cam_pos: tuple[float, float, float] = (0.33, -0.10, 0.45),
    focal_length: float = 18.0,
    horizontal_aperture: float = 20.955,
):
    """俯瞰カメラ画像から箱の周辺だけを切り抜き、2x2の十字線と象限ラベルを描画したPIL画像を返す。"""
    from PIL import Image, ImageDraw

    img_np = rgb_tensor[0, ..., :3].detach().cpu().numpy()
    if img_np.dtype != "uint8":
        img_np = (img_np.clip(0.0, 1.0) * 255).astype("uint8")
    pil_img = Image.fromarray(img_np)
    w, h = pil_img.size

    # カメラから箱の上面までの距離 [m] と 1m あたりのピクセル数
    depth = max(cam_pos[2] - (0.035 + 0.5 * BOX_HEIGHT), 0.1)
    fov_w_m = depth * (horizontal_aperture / focal_length)
    px_per_m = w / fov_w_m

    # 画像の右 = -Y、画像の上 = +X に対応
    cx = w * 0.5 - (box_local_xy[1] - cam_pos[1]) * px_per_m
    cy = h * 0.5 - (box_local_xy[0] - cam_pos[0]) * px_per_m
    half_crop = max(int(0.5 * BOX_SIZE * 1.35 * px_per_m), 32)

    left = max(0, int(cx - half_crop))
    top = max(0, int(cy - half_crop))
    right = min(w, int(cx + half_crop))
    bottom = min(h, int(cy + half_crop))

    cropped = pil_img.crop((left, top, right, bottom)).resize((256, 256), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(cropped)
    # 箱の中心に2x2の区切り線と象限ラベルを描画（VLMの誤認・先頭バイアス防止）
    draw.line([(128, 0), (128, 256)], fill=(255, 255, 0), width=2)
    draw.line([(0, 128), (256, 128)], fill=(255, 255, 0), width=2)
    draw.text((8, 8), "TL", fill=(255, 255, 0))
    draw.text((228, 8), "TR", fill=(255, 255, 0))
    draw.text((8, 236), "BL", fill=(255, 255, 0))
    draw.text((228, 236), "BR", fill=(255, 255, 0))
    return cropped


def ask_local_vlm_empty_slot(
    rgb_tensor: torch.Tensor,
    box_local_xy: tuple[float, float] = (0.33, -0.10),
    vlm_url: str = "http://localhost:11434/api/chat",
    vlm_model: str = "qwen2.5vl:7b",
) -> str:
    """俯瞰カメラのRGB画像から箱を切り抜いてローカルVLMへ送り、空いているスロット名を判定させる。"""
    pil_img = _crop_and_annotate_box_image(rgb_tensor, box_local_xy=box_local_xy)
    try:
        pil_img.save("/tmp/vlm_last_input.png")
    except OSError:
        pass

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        "画像は緑色の箱を真上から拡大し、黄色い十字線で4つの区画に区切ったものです。\n"
        "左上=TL(top_left)、右上=TR(top_right)、左下=BL(bottom_left)、右下=BR(bottom_right)です。\n"
        "現在3つの区画に青いボールが入っており、1つの区画だけボールが無く空いています。\n"
        "青いボールが入っていない空いている区画を1つだけ選び、JSONのみで答えてください:\n"
        '{"empty_slot": "top_left" | "top_right" | "bottom_left" | "bottom_right"}'
    )
    payload = {
        "model": vlm_model,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    req = urllib.request.Request(
        vlm_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
    content = res_json["message"]["content"]
    print(f"[VLM応答] {content}")

    # 1. 厳密な JSON 形式を最優先で抽出
    match = re.search(r'"empty_slot"\s*:\s*"(top_left|top_right|bottom_left|bottom_right)"', content)
    if match:
        return match.group(1)
    # 2. JSON以外の文章で返った場合、先頭バイアスを避けるため最後に言及された結論を採用
    matches = re.findall(r"\b(top_left|top_right|bottom_left|bottom_right)\b", content)
    if matches:
        return matches[-1]
    # 3. 短縮ラベル（TL/TR/BL/BR）で返った場合のフォールバック
    short_map = {"TL": "top_left", "TR": "top_right", "BL": "bottom_left", "BR": "bottom_right"}
    short_matches = re.findall(r"\b(TL|TR|BL|BR)\b", content)
    if short_matches:
        return short_map[short_matches[-1]]
    raise ValueError(f"VLMの応答からスロット名を抽出できませんでした: {content}")


def compute_safe_lift_z(env: ManagerBasedRLEnv, box_height: float = BOX_HEIGHT, margin: float = 0.025) -> torch.Tensor:
    """AI/幾何計算: 机高さ(0.035) + 箱の壁の高さ + ボール半径 + 安全マージンから必要持ち上げ高さを動的算出する。"""
    r = ball_radius(env)
    return 0.035 + box_height + r + margin


def move_arm_vertical_joints(env: ManagerBasedRLEnv, lift_z: torch.Tensor, wrist_roll: torch.Tensor) -> torch.Tensor:
    """現在のXY座標を保ったまま、指定された高さ lift_z まで真上に垂直上昇する5関節角度を算出する。"""
    origins = env.scene.env_origins
    tcp_local = grasp_center_w(env) - origins
    target_xyz = tcp_local.clone()
    target_xyz[:, 2] = lift_z
    q = _solve_arm_joints_from_table(env, target_xyz)
    q[:, 4] = wrist_roll
    return q


def move_arm_horizontal_joints(
    env: ManagerBasedRLEnv, target_xy_local: torch.Tensor, lift_z: torch.Tensor, wrist_roll: torch.Tensor
) -> torch.Tensor:
    """安全高さ lift_z を維持したまま、目標XY座標の真上へ水平移動する5関節角度を算出する。"""
    target_xyz = torch.zeros((env.num_envs, 3), device=env.device)
    target_xyz[:, :2] = target_xy_local
    target_xyz[:, 2] = lift_z
    q = _solve_arm_joints_from_table(env, target_xyz)
    q[:, 4] = wrist_roll
    return q


def move_arm_drop_joints(
    env: ManagerBasedRLEnv, target_xy_local: torch.Tensor, drop_z: torch.Tensor, wrist_roll: torch.Tensor
) -> torch.Tensor:
    """目標XY座標の上空から、箱の中の投下高さ drop_z まで垂直降下する5関節角度を算出する。"""
    target_xyz = torch.zeros((env.num_envs, 3), device=env.device)
    target_xyz[:, :2] = target_xy_local
    target_xyz[:, 2] = drop_z
    q = _solve_arm_joints_from_table(env, target_xyz)
    q[:, 4] = wrist_roll
    return q


def _ensure_vlm_state(env: ManagerBasedRLEnv) -> None:
    """環境上に3層アーキテクチャ連携（VLM + 決定論的移動メソッド）用の状態テンソルとFKテーブルを初期化する。"""
    if hasattr(env, "_so101_vlm_initialized"):
        return
    n = env.num_envs
    device = env.device
    env._so101_vlm_initialized = True
    # phase: 0=RL_PICK, 1=LIFT_VERTICAL, 2=MOVE_HORIZONTAL, 3=DROP_VERTICAL, 4=RELEASE, 5=RETRACT
    env._so101_vlm_phase = torch.zeros(n, dtype=torch.long, device=device)
    env._so101_vlm_phase_timer = torch.zeros(n, dtype=torch.long, device=device)
    env._so101_vlm_queried = torch.zeros(n, dtype=torch.bool, device=device)
    env._so101_vlm_slot_offset = torch.zeros((n, 2), dtype=torch.float32, device=device)
    env._so101_vlm_saved_wrist_roll = torch.full((n,), -0.8044, dtype=torch.float32, device=device)
    env._so101_vlm_target_q = torch.zeros((n, 5), dtype=torch.float32, device=device)
    env._so101_vlm_fk_table: tuple[torch.Tensor, torch.Tensor] | None = None


def _calibrate_fk_table(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """机全域（左右両方）をカバーする5関節角度と両爪中心ローカル座標(XYZ)の対応テーブルをFKで構築する。"""
    robot = env.scene["robot"]
    device = env.device
    saved_q = robot.data.joint_pos.torch.clone()

    # 通常スタートの可動域に合わせて shoulder_lift / elbow_flex を広めにカバーし、wrist_roll は基準値(-0.8044)を使用
    pans = torch.linspace(-1.45, 1.45, 21, device=device)
    lifts = torch.linspace(-0.65, 0.45, 8, device=device)
    elbows = torch.linspace(-0.55, 0.35, 6, device=device)
    grid_pan, grid_lift, grid_elbow = torch.meshgrid(pans, lifts, elbows, indexing="ij")
    num_pts = grid_pan.numel()

    q_candidates = torch.zeros((num_pts, 6), device=device)
    q_candidates[:, 0] = grid_pan.reshape(-1)
    q_candidates[:, 1] = grid_lift.reshape(-1)
    q_candidates[:, 2] = grid_elbow.reshape(-1)
    q_candidates[:, 3] = 1.25
    q_candidates[:, 4] = -0.8044
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
    """開始直後のVLM空きスロット判定、および把持後の決定論的移動メソッド（縦・横・降下・解放）の状態を更新する。"""
    _ensure_vlm_state(env)
    origins = env.scene.env_origins
    box_local = env.scene["box"].data.root_pos_w.torch - origins

    # 1. エピソード開始直後（アームがまだホーム姿勢で箱を隠していない時）に全体カメラで空きスロットを把握・記憶する
    if (env.episode_length_buf >= 1).any() and not env._so101_vlm_queried.all():
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
                box_xy = (float(box_local[0, 0]), float(box_local[0, 1]))
                slot_key = ask_local_vlm_empty_slot(
                    rgb, box_local_xy=box_xy, vlm_url=vlm_url, vlm_model=vlm_model
                )
                dx, dy = SLOT_OFFSETS[slot_key]
                env._so101_vlm_slot_offset[:, 0] = dx
                env._so101_vlm_slot_offset[:, 1] = dy
                print(
                    f"[VLM判定] 開始時判定スロット: {slot_key} -> offset=({dx:+.3f}, {dy:+.3f}) "
                    f"(正解: ({float(default_offset[0, 0]):+.3f}, {float(default_offset[0, 1]):+.3f}))"
                )
            except Exception as err:
                print(
                    f"[VLM判定] ローカルVLM未接続のため環境の空きスロット座標を使用します "
                    f"(offset=({float(default_offset[0, 0]):+.3f}, {float(default_offset[0, 1]):+.3f})): {err}"
                )
        else:
            print(
                f"[VLM判定] 空きスロット座標を記憶(モック): "
                f"offset=({float(default_offset[0, 0]):+.3f}, {float(default_offset[0, 1]):+.3f})"
            )

        env._so101_vlm_queried[:] = True

    # 2. フェーズ別シーケンス制御
    phase = env._so101_vlm_phase
    holding = ball_is_holding(env)
    lift_z = compute_safe_lift_z(env)
    target_xy_local = box_local[:, :2] + env._so101_vlm_slot_offset
    tcp_local = grasp_center_w(env) - origins
    current_wrist_roll = env.scene["robot"].data.joint_pos.torch[:, 4]

    # Phase 0: RLモデルによる把持＆浮上中 -> 机から浮上＆把持確認でPhase 1（垂直上昇）へ移行
    newly_grasped = (phase == 0) & holding & (ball_height(env) >= (ball_lift_min_height(env) + 0.008))
    if newly_grasped.any():
        env._so101_vlm_saved_wrist_roll = torch.where(
            newly_grasped, current_wrist_roll, env._so101_vlm_saved_wrist_roll
        )
        env._so101_vlm_phase = torch.where(newly_grasped, torch.tensor(1, device=env.device), env._so101_vlm_phase)
        env._so101_vlm_phase_timer = torch.where(newly_grasped, torch.tensor(0, device=env.device), env._so101_vlm_phase_timer)
        print(f"[3層連携] 把持完了を検知 -> プログラム垂直持ち上げフェーズへ移行 (lift_z={float(lift_z[0]):.3f}m)")

    # Phase 1: 垂直上昇（真上に lift_z まで持ち上げる）
    in_phase1 = phase == 1
    if in_phase1.any():
        q_lift = move_arm_vertical_joints(env, lift_z, env._so101_vlm_saved_wrist_roll)
        env._so101_vlm_target_q = torch.where(in_phase1.unsqueeze(-1), q_lift, env._so101_vlm_target_q)
        lift_reached = in_phase1 & (tcp_local[:, 2] >= (lift_z - 0.015))
        if lift_reached.any():
            env._so101_vlm_phase = torch.where(lift_reached, torch.tensor(2, device=env.device), env._so101_vlm_phase)
            env._so101_vlm_phase_timer = torch.where(lift_reached, torch.tensor(0, device=env.device), env._so101_vlm_phase_timer)
            print(f"[3層連携] 垂直持ち上げ完了 -> 空きスロット水平移動フェーズへ移行")

    # Phase 2: 水平移動（lift_z を維持したまま空きスロット直上へ運搬）
    in_phase2 = phase == 2
    if in_phase2.any():
        q_move = move_arm_horizontal_joints(env, target_xy_local, lift_z, env._so101_vlm_saved_wrist_roll)
        env._so101_vlm_target_q = torch.where(in_phase2.unsqueeze(-1), q_move, env._so101_vlm_target_q)
        xy_dist = torch.linalg.vector_norm(tcp_local[:, :2] - target_xy_local, dim=-1)
        ball_xy_speed = torch.linalg.vector_norm(env.scene["ball"].data.root_lin_vel_w.torch[:, :2], dim=-1)
        aligned_and_calm = in_phase2 & (xy_dist < 0.015) & (ball_xy_speed < 0.06)
        if aligned_and_calm.any():
            env._so101_vlm_phase = torch.where(aligned_and_calm, torch.tensor(3, device=env.device), env._so101_vlm_phase)
            env._so101_vlm_phase_timer = torch.where(aligned_and_calm, torch.tensor(0, device=env.device), env._so101_vlm_phase_timer)
            print(f"[3層連携] 空きスロット真上到達＆静止 -> 垂直降下フェーズへ移行")

    # Phase 3: 垂直降下（箱の縁付近 drop_z へ降ろす。爪が隣のボールに挟まれないよう縁の少し下で留める）
    in_phase3 = phase == 3
    if in_phase3.any():
        drop_z = box_local[:, 2] + BOX_HEIGHT - 0.25 * ball_radius(env)
        q_drop = move_arm_drop_joints(env, target_xy_local, drop_z, env._so101_vlm_saved_wrist_roll)
        env._so101_vlm_target_q = torch.where(in_phase3.unsqueeze(-1), q_drop, env._so101_vlm_target_q)
        env._so101_vlm_phase_timer = torch.where(in_phase3, env._so101_vlm_phase_timer + 1, env._so101_vlm_phase_timer)
        # 爪先端高さが降下目標付近（+1.5cm以内）に到達、または最大20ステップ経過でリリースへ
        z_reached = tcp_local[:, 2] <= (drop_z + 0.015)
        dropped_ready = in_phase3 & (z_reached | (env._so101_vlm_phase_timer >= 20))
        if dropped_ready.any():
            env._so101_vlm_phase = torch.where(dropped_ready, torch.tensor(4, device=env.device), env._so101_vlm_phase)
            env._so101_vlm_phase_timer = torch.where(dropped_ready, torch.tensor(0, device=env.device), env._so101_vlm_phase_timer)
            print(f"[3層連携] 降下完了 -> リリースフェーズ（指を開く）へ移行")

    # Phase 4: リリース（指を開く）
    in_phase4 = phase == 4
    if in_phase4.any():
        env._so101_vlm_phase_timer = torch.where(in_phase4, env._so101_vlm_phase_timer + 1, env._so101_vlm_phase_timer)
        release_done = in_phase4 & (env._so101_vlm_phase_timer >= 10)
        if release_done.any():
            env._so101_vlm_phase = torch.where(release_done, torch.tensor(5, device=env.device), env._so101_vlm_phase)
            env._so101_vlm_phase_timer = torch.where(release_done, torch.tensor(0, device=env.device), env._so101_vlm_phase_timer)
            print(f"[3層連携] リリース完了 -> 上空退避フェーズへ移行")

    # Phase 5: 退避（アームを少し上へ引き上げる）
    in_phase5 = phase == 5
    if in_phase5.any():
        q_retract = move_arm_horizontal_joints(env, target_xy_local, lift_z, env._so101_vlm_saved_wrist_roll)
        env._so101_vlm_target_q = torch.where(in_phase5.unsqueeze(-1), q_retract, env._so101_vlm_target_q)


class VLMGuidedArmAction(RelativeJointPositionAction):
    """把持まではRLポリシーの行動を実行し、把持後は決定論的移動メソッド（縦・横・降下・退避）で制御する。"""

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        _ensure_vlm_state(self._env)
        ids = slice(None) if env_ids is None else env_ids
        self._env._so101_vlm_phase[ids] = 0
        self._env._so101_vlm_phase_timer[ids] = 0
        self._env._so101_vlm_queried[ids] = False

    def process_actions(self, actions: torch.Tensor) -> None:
        _update_vlm_transport_state(self._env)
        active = self._env._so101_vlm_phase >= 1
        if active.any():
            actions = actions.clone()
            current_q = self._asset.data.joint_pos.torch[:, self._joint_ids]
            target_q = self._env._so101_vlm_target_q
            scale = self._scale if isinstance(self._scale, float) else 0.033
            guided_act = ((target_q - current_q) / scale).clamp(-0.75, 0.75)
            actions = torch.where(active.unsqueeze(-1), guided_act, actions)
        super().process_actions(actions)


class VLMGuidedGripperAction(RelativeJointPositionAction):
    """運搬中はグリッパーをしっかり閉じ続け、箱の中へ降ろしたリリースフェーズで開いて放す。"""

    def process_actions(self, actions: torch.Tensor) -> None:
        _ensure_vlm_state(self._env)
        phase = self._env._so101_vlm_phase
        active = phase >= 1
        if active.any():
            actions = actions.clone()
            # Phase 4 (RELEASE) または Phase 5 (RETRACT) は指を開く (+1.0)、Phase 1〜3 はしっかり閉じて保持 (-1.0)
            should_open = (phase == 4) | (phase == 5)
            grip_cmd = torch.where(should_open, 1.0, -1.0).unsqueeze(-1)
            actions = torch.where(active.unsqueeze(-1), grip_cmd, actions)
        super().process_actions(actions)


@configclass
class VLMGuidedArmActionCfg(RelativeJointPositionActionCfg):
    class_type: type = VLMGuidedArmAction


@configclass
class VLMGuidedGripperActionCfg(RelativeJointPositionActionCfg):
    class_type: type = VLMGuidedGripperAction
