"""Evaluate a saved PPO policy (MLP or recurrent) on the Balatro env.

Auto-detects architecture by the keys in policy.pt: a GRU-containing
state_dict triggers the recurrent code path; otherwise the MLP path
is used.

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
from training.ppo_recurrent import MaskedRecurrentActorCritic  # noqa: E402


def _detect_recurrent(state_dict_path: Path) -> bool:
    state = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    return any("gru." in k for k in state.keys())


def _load_agent(run_dir: Path, obs_dim: int, num_actions: int):
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    pol_path = run_dir / "policy.pt"
    is_recurrent = _detect_recurrent(pol_path)
    if is_recurrent:
        agent = MaskedRecurrentActorCritic(
            obs_dim, num_actions,
            encoder_hidden=cfg.get("encoder_hidden", 256),
            gru_hidden=cfg.get("gru_hidden", 128),
        )
    else:
        agent = MaskedActorCritic(obs_dim, num_actions, hidden=cfg.get("hidden_size", 256))
    agent.load_state_dict(torch.load(pol_path, map_location="cpu", weights_only=True))
    agent.eval()
    return agent, cfg, is_recurrent


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
    agent, _, is_recurrent = _load_agent(run_path, obs_dim, num_actions)
    arch_label = "Recurrent" if is_recurrent else "MLP"

    results = []
    for ep in range(episodes):
        seed = seed_base + ep
        obs, info = env.reset(seed=seed)
        if is_recurrent:
            hidden = agent.initial_hidden(1, torch.device("cpu"))
            prev_done = torch.zeros(1)
        terminated = False
        steps = 0
        while not terminated and steps < 5000:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(info["action_mask"], dtype=torch.bool).unsqueeze(0)
            with torch.no_grad():
                if is_recurrent:
                    action, _, _, hidden = agent.act(obs_t, hidden, prev_done, mask_t)
                else:
                    action, _, _, _ = agent.get_action_and_value(obs_t, mask_t)
            obs, reward, terminated, truncated, info = env.step(int(action.item()))
            if is_recurrent:
                prev_done = torch.tensor([1.0 if terminated else 0.0])
            steps += 1
        won = env._run is not None and env._run.stage == GameStage.GAME_WON
        ante, blind = _highest_beaten(env)
        results.append({"won": won, "highest_ante": ante, "highest_blind": blind, "steps": steps})

    n = len(results)
    won = sum(r["won"] for r in results)
    print(f"\n=== {arch_label} PPO ({run_dir}, {n} eps) ===")
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
    args = p.parse_args()
    evaluate(args.run_dir, args.episodes, args.max_ante, args.seed_base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
