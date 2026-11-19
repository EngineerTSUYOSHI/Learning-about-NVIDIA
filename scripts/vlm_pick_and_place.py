"""ステップC（VLM連携・Lv1検証）実行スクリプト。

俯瞰カメラで箱の空きスペースをローカルVLM（Ollama等）に判定させ、
学習済み強化学習モデル（--checkpoint）の目標箱位置をその空きスロットに差し替えて実行します。

実行例:
  uv run python scripts/vlm_pick_and_place.py \\
      --checkpoint logs/rsl_rl/so101_vial_state/.../model_xxxx.pt \\
      --vlm_model qwen2.5vl:7b \\
      presets=newton_mjwarp,newton_renderer

  ※ローカルVLMを使わず、手動（CLI入力）や自動（空きスロット正解値）でテストしたい場合:
  uv run python scripts/vlm_pick_and_place.py \\
      --checkpoint logs/rsl_rl/so101_vial_state/.../model_xxxx.pt \\
      --mock_vlm presets=newton_mjwarp,newton_renderer
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import urllib.request

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="VLM連携によるボール箱投入（ステップC）検証スクリプト")
parser.add_argument("--checkpoint", type=str, required=True, help="学習済み強化学習モデル（.pt）のパス")
parser.add_argument("--task", type=str, default="IsaacTutorial-Place-Ball-VLM-SO101", help="タスク名")
parser.add_argument("--vlm_url", type=str, default="http://localhost:11434/api/chat", help="ローカルVLM（Ollama）のAPI URL")
parser.add_argument("--vlm_model", type=str, default="qwen2.5vl:7b", help="使用するローカルVLMモデル名")
parser.add_argument("--mock_vlm", action="store_true", help="VLMを呼ばず、環境が把握している空きスロットで動作テストする")
parser.add_argument("--episodes", type=int, default=5, help="テストするエピソード数")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# AppLauncher起動後にIsaac Labモジュールをインポートする
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_presets  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import isaaclab_tutorial.tasks  # noqa: E402, F401

# 俯瞰カメラ画像上の向きに合わせた4スロットのオフセット [m]
# 画像の上 = +X（ロボット前方）、画像の下 = -X（手前）、画像の左 = +Y（左）、画像の右 = -Y（右）
SLOT_OFFSETS: dict[str, tuple[float, float]] = {
    "top_left": (0.027, 0.027),
    "top_right": (0.027, -0.027),
    "bottom_left": (-0.027, 0.027),
    "bottom_right": (-0.027, -0.027),
}


def ask_local_vlm_empty_slot(rgb_tensor: torch.Tensor, vlm_url: str, vlm_model: str) -> str:
    """俯瞰カメラのRGB画像から、箱の中の空いているスロット名をローカルVLMに判定させる。"""
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
    with urllib.request.urlopen(req, timeout=30) as resp:
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


def get_box_obs_slice(base_env) -> slice:
    """観測グループ 'policy' における 'box' 項（ベース座標系の位置XYZを含む）のスライス範囲を取得する。"""
    terms = base_env.observation_manager.active_terms["policy"]
    dims = base_env.observation_manager.group_obs_term_dim["policy"]
    offset = 0
    for term_name, term_dim in zip(terms, dims, strict=True):
        size = int(torch.tensor(term_dim).prod().item())
        if term_name == "box":
            return slice(offset, offset + size)
        offset += size
    raise RuntimeError("観測グループ 'policy' に 'box' 項が見つかりません。")


def world_pos_to_robot_base(base_env, target_pos_w: torch.Tensor) -> torch.Tensor:
    """ワールド座標 [1, 3] をロボットのベース座標系 [1, 3] へ変換する。"""
    robot = base_env.scene["robot"]
    base_pos_w = robot.data.root_pos_w.torch
    base_quat_w = robot.data.root_quat_w.torch
    return quat_apply_inverse(base_quat_w, target_pos_w - base_pos_w)


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    # コマンドラインの presets=... が指定されていれば反映（デフォルトは newton_mjwarp, newton_renderer）
    presets = ["newton_mjwarp", "newton_renderer"]
    for arg in hydra_args:
        if arg.startswith("presets="):
            presets = [p.strip() for p in arg.split("=", 1)[1].split(",") if p.strip()]
    env_cfg = resolve_presets(env_cfg, presets)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if hasattr(env_cfg, "play_mode"):
        env_cfg.play_mode()

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    base_env = vec_env.unwrapped

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=base_env.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=base_env.device)

    box_slice = get_box_obs_slice(base_env)

    for ep in range(args_cli.episodes):
        obs, _ = vec_env.reset()
        # reset() 直後は未描画のため、sim.render() を呼んでから最新フレームを取得する
        base_env.sim.render()
        base_env.scene.sensors["overhead_camera"].update(dt=base_env.step_dt)
        rgb = base_env.scene.sensors["overhead_camera"].data.output["rgb"].torch

        box_pos_w = base_env.scene["box"].data.root_pos_w.torch.clone()
        if args_cli.mock_vlm:
            offset_xy = getattr(base_env, "_so101_empty_slot_offset", torch.zeros((1, 2), device=base_env.device))[0]
            slot_name = f"mock({float(offset_xy[0]):+.3f}, {float(offset_xy[1]):+.3f})"
        else:
            try:
                slot_key = ask_local_vlm_empty_slot(rgb, args_cli.vlm_url, args_cli.vlm_model)
                dx, dy = SLOT_OFFSETS[slot_key]
                offset_xy = torch.tensor([dx, dy], device=base_env.device)
                slot_name = slot_key
            except Exception as err:
                print(f"[警告] VLM呼び出しに失敗したためフォールバックします: {err}")
                offset_xy = getattr(base_env, "_so101_empty_slot_offset", torch.zeros((1, 2), device=base_env.device))[0]
                slot_name = "fallback"

        target_pos_w = box_pos_w.clone()
        target_pos_w[:, 0] += offset_xy[0]
        target_pos_w[:, 1] += offset_xy[1]
        target_pos_b = world_pos_to_robot_base(base_env, target_pos_w)

        print(f"\n=== エピソード {ep + 1} / 選択空きスロット: {slot_name} ===")

        done = False
        step = 0
        while not done and simulation_app.is_running():
            # 強化学習モデルに渡す観測の 'box' 位置部分を「空きスロット座標」に差し替える
            obs_mod = obs.clone()
            if isinstance(obs_mod, torch.Tensor):
                obs_mod[:, box_slice.start : box_slice.start + 3] = target_pos_b
            else:
                obs_mod["policy"][:, box_slice.start : box_slice.start + 3] = target_pos_b
            with torch.inference_mode():
                actions = policy(obs_mod)
            obs, _, dones, extras = vec_env.step(actions)
            done = bool(dones[0].item())
            step += 1

        print(f"エピソード {ep + 1} 終了 (ステップ数: {step})")

    vec_env.close()


if __name__ == "__main__":
    main()
