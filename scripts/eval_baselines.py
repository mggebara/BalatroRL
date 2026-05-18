"""Evaluate baseline agents on the Balatro env across fixed seeds.

Reports per-ante win rates and the highest blind reached. The Phase 3
DoD target is greedy > 50% Ante 3 — but without a shop layer the
agent cannot accumulate jokers between blinds, so we ALSO run a
"with starter jokers" variant to characterize how far the pipeline
can take us when joker pressure is present.

Usage:
    python3 scripts/eval_baselines.py                # default 100 seeds
    python3 scripts/eval_baselines.py --episodes 50  # fewer
    python3 scripts/eval_baselines.py --max-ante 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.greedy_agent import GreedyAgent  # noqa: E402
from agents.random_agent import RandomAgent  # noqa: E402
from balatro_core.engine import BlindKind, GameStage  # noqa: E402
from balatro_core.jokers import (  # noqa: E402
    Joker,
    JokerJoker,
    LustyJoker,
    SlyJoker,
)
from balatro_env.gym_env import BalatroEnv  # noqa: E402

BLIND_ORDER = [BlindKind.SMALL, BlindKind.BIG, BlindKind.BOSS]


def _level_reached(env: BalatroEnv) -> tuple[int, BlindKind | None]:
    """Highest (ante, blind) the run actually beat — i.e. fully cleared."""
    run = env._run
    if run is None:
        return 0, None
    if run.stage == GameStage.GAME_WON:
        return env.max_ante, BlindKind.BOSS
    # Otherwise GAME_OVER: ante/current_blind hold the blind that killed us.
    # The previous blind is the last beaten.
    if run.current_blind == BlindKind.SMALL:
        if run.ante == 1:
            return 0, None
        return run.ante - 1, BlindKind.BOSS
    if run.current_blind == BlindKind.BIG:
        return run.ante, BlindKind.SMALL
    return run.ante, BlindKind.BIG  # current_blind == BOSS


def run_episode(env: BalatroEnv, agent, seed: int, max_steps: int = 5000) -> dict:
    obs, info = env.reset(seed=seed)
    agent.reset()
    terminated = False
    steps = 0
    while not terminated and steps < max_steps:
        action = agent.act(env, info)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
    won = env._run is not None and env._run.stage == GameStage.GAME_WON
    ante, blind = _level_reached(env)
    return {
        "won": won,
        "highest_ante": ante,
        "highest_blind": blind.value if blind else "none",
        "steps": steps,
    }


def summarize(label: str, results: list[dict], max_ante: int) -> None:
    n = len(results)
    won = sum(r["won"] for r in results)
    print(f"\n=== {label} ({n} eps) ===")
    print(f"  Run-win rate (beat Ante {max_ante} Boss): {won}/{n} = {won / n:.1%}")
    # Per-ante beaten rates: "beat Ante k" = cleared at least one blind of ante k+1
    # OR cleared all of ante k's blinds.
    for k in range(1, max_ante + 1):
        beats = sum(
            1 for r in results
            if (r["highest_ante"] > k)
            or (r["highest_ante"] == k and r["highest_blind"] == BlindKind.BOSS.value)
        )
        print(f"  Beat Ante {k} Boss: {beats}/{n} = {beats / n:.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-ante", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=0)
    args = ap.parse_args()

    seeds = list(range(args.seed_base, args.seed_base + args.episodes))

    # No-joker baselines.
    for label, agent_factory in [
        ("RandomAgent (no jokers)", lambda: RandomAgent(seed=0)),
        ("GreedyAgent (no jokers)", lambda: GreedyAgent()),
    ]:
        env = BalatroEnv(max_ante=args.max_ante)
        agent = agent_factory()
        results = [run_episode(env, agent, s) for s in seeds]
        summarize(label, results, args.max_ante)

    # Greedy with starter jokers (simulates surviving Ante 1 in a shop-having world).
    starters: list[Joker] = [JokerJoker(), LustyJoker(), SlyJoker()]
    env = BalatroEnv(max_ante=args.max_ante, starting_jokers=starters)
    agent = GreedyAgent()
    results = [run_episode(env, agent, s) for s in seeds]
    summarize("GreedyAgent (starter jokers: Joker, Lusty, Sly)", results, args.max_ante)

    return 0


if __name__ == "__main__":
    sys.exit(main())
