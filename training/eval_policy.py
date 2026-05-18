"""Evaluate a saved PPO policy on the Balatro env.

Mirrors `scripts/eval_baselines.py` so PPO numbers are directly
comparable: same seeds, same per-ante reporting.

Usage:
    python3 training/eval_policy.py --run-dir training/runs/sanity \\
        --episodes 100 --max-ante 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from balatro_core.engine import BlindKind, GameStage  # noqa: E402
from balatro_env.gym_env import BalatroEnv  # noqa: E402
from training.ppo import MaskedActorCritic, _build_starting_jokers  # noqa: E402


def load_agent(run_dir: Path, obs_dim: int, num_actions: int) -> MaskedActorCritic:
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    agent = MaskedActorCritic(obs_dim, num_actions, hidden=cfg.get("hidden_size", 256))
    agent.load_state_dict(torch.load(run_dir / "policy.pt", map_location="cpu", weights_only=True))
    agent.eval()
    return agent, cfg


def _highest_beaten(env: BalatroEnv) -> tuple[int, str]:
    run = env._run
    if run is None:
        return 0, "none"
    if run.stage == GameStage.GAME_WON:
        return env.max_ante, BlindKind.BOSS.value
    if run.current_blind == BlindKind.SMALL:
        if run.ante == 1:
            return 0, "none"
        return run.ante - 1, BlindKind.BOSS.value
    if run.current_blind == BlindKind.BIG:
        return run.ante, BlindKind.SMALL.value
    return run.ante, BlindKind.BIG.value


def evaluate(
    run_dir: str,
    episodes: int,
    max_ante: int,
    deterministic: bool,
    seed_base: int = 0,
) -> dict:
    run_path = Path(run_dir)
    with open(run_path / "config.json") as f:
        cfg = json.load(f)
    starting = tuple(cfg.get("starting_jokers", ()))

    env = BalatroEnv(
        max_ante=max_ante,
        starting_jokers=_build_starting_jokers(starting),
    )
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    agent, _ = load_agent(run_path, obs_dim, num_actions)

    results = []
    for ep in range(episodes):
        seed = seed_base + ep
        obs, info = env.reset(seed=seed)
        terminated = False
        steps = 0
        while not terminated and steps < 5000:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(info["action_mask"], dtype=torch.bool).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            steps += 1
        won = env._run is not None and env._run.stage == GameStage.GAME_WON
        ante, blind = _highest_beaten(env)
        results.append({"won": won, "highest_ante": ante, "highest_blind": blind, "steps": steps})

    n = len(results)
    won = sum(r["won"] for r in results)
    print(f"\n=== Trained PPO policy ({run_dir}, {n} eps) ===")
    print(f"  Run-win rate (beat Ante {max_ante} Boss): {won}/{n} = {won / n:.1%}")
    for k in range(1, max_ante + 1):
        beats = sum(
            1 for r in results
            if (r["highest_ante"] > k)
            or (r["highest_ante"] == k and r["highest_blind"] == BlindKind.BOSS.value)
        )
        print(f"  Beat Ante {k} Boss: {beats}/{n} = {beats / n:.1%}")
    return {"results": results, "win_count": won, "n": n}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-ante", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="(no-op currently; PPO samples from masked policy)")
    args = p.parse_args()
    evaluate(args.run_dir, args.episodes, args.max_ante, args.deterministic, args.seed_base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
