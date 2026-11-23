"""ステップC（VLM連携＋方法A・Lv1検証）スタンドアロン実行スクリプト。

Omniverse Isaac Sim（AppLauncher）を使わず、本リポジトリ標準の ``launch_simulation``（Newton MJWarp対応）で起動します。
また、このスクリプトを使わずに普段通りの ``uv run isaaclab play`` コマンドでも同じ動作が実行できます:

【方法1: いつもの isaaclab play コマンドで動かす（一番簡単・確実）】
  uv run isaaclab play --rl_library rsl_rl --task IsaacTutorial-Place-Ball-VLM-SO101 --num_envs 1 --checkpoint /path/to/model.pt presets=newton_mjwarp,newton_renderer

  ※VLMを使わずモック（正解空きスロット）で動かす場合:
  SO101_VLM_MODE=mock uv run isaaclab play --rl_library rsl_rl --task IsaacTutorial-Place-Ball-VLM-SO101 --num_envs 1 --checkpoint /path/to/model.pt presets=newton_mjwarp,newton_renderer

【方法2: このスクリプトでエピソード数を指定して動かす】
  uv run python scripts/vlm_pick_and_place.py --checkpoint /path/to/model.pt --mock_vlm presets=newton_mjwarp,newton_renderer
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.cli import load_external_tasks


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM連携によるボール箱投入（ステップC・方法A）検証スクリプト")
    parser.add_argument("--checkpoint", type=str, required=True, help="学習済み強化学習モデル（.pt）のパス")
    parser.add_argument("--task", type=str, default="IsaacTutorial-Place-Ball-VLM-SO101", help="タスク名")
    parser.add_argument("--vlm_url", type=str, default="http://localhost:11434/api/chat", help="ローカルVLM（Ollama）のAPI URL")
    parser.add_argument("--vlm_model", type=str, default="qwen2.5vl:7b", help="使用するローカルVLMモデル名")
    parser.add_argument("--mock_vlm", action="store_true", help="VLMを呼ばず、環境が把握している空きスロットで動作テストする")
    parser.add_argument("--episodes", type=int, default=5, help="テストするエピソード数")
    parser.add_argument("--presets", default="newton_mjwarp,newton_renderer", help="カンマ区切りのプリセット")
    add_launcher_args(parser)
    args, unknown = parser.parse_known_args(argv)
    for token in unknown:
        if token.startswith("presets="):
            args.presets = token.removeprefix("presets=")
        else:
            parser.error(f"不明な引数です（カンマの前後にスペースを入れないでください）: {token}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.environ["SO101_VLM_MODE"] = "mock" if args.mock_vlm else "auto"
    os.environ["SO101_VLM_URL"] = args.vlm_url
    os.environ["SO101_VLM_MODEL"] = args.vlm_model

    load_external_tasks()
    from isaaclab_tasks.utils.hydra import resolve_presets
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import isaaclab_tutorial.tasks  # noqa: F401

    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    env_cfg = resolve_presets(load_cfg_from_registry(args.task, "env_cfg_entry_point"), presets)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if hasattr(env_cfg, "play_mode"):
        env_cfg.play_mode()
    if getattr(args, "device", None):
        env_cfg.sim.device = args.device

    with launch_simulation(env_cfg, args):
        import torch
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        raw_env = ManagerBasedRLEnv(env_cfg)
        vec_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        try:
            runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=raw_env.device)
            runner.load(args.checkpoint)
            policy = runner.get_inference_policy(device=raw_env.device)

            for ep in range(args.episodes):
                obs, _ = vec_env.reset()
                print(f"\n=== エピソード {ep + 1}/{args.episodes} 開始 ===")
                done = False
                step = 0
                while not done and raw_env.sim.is_running():
                    with torch.inference_mode():
                        actions = policy(obs)
                    obs, _, dones, _ = vec_env.step(actions)
                    done = bool(dones[0].item())
                    step += 1
                print(f"=== エピソード {ep + 1} 終了 (ステップ数: {step}) ===")
        finally:
            vec_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
