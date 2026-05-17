"""Phase 1 smoke test: play a full Ante 1 with a simple scripted agent.

Demonstrates that the engine plays Small + Big + Boss blinds end-to-end.
Strategy: brute-force pick the 1-5 cards from the current hand that
maximize this play's score. Use discards greedily when the best available
hand is High Card only.

Usage:
    python3 scripts/smoke_ante1.py            # default seed 42
    python3 scripts/smoke_ante1.py 7          # use seed 7
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

# Allow running directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balatro_core.cards import Card  # noqa: E402
from balatro_core.engine import BlindKind, Round, Run  # noqa: E402
from balatro_core.hands import HandType, evaluate_hand  # noqa: E402
from balatro_core.scoring import score_played_hand  # noqa: E402


def best_play(hand: list[Card]) -> tuple[list[int], int, HandType]:
    """Brute-force the highest-scoring 1-5 card subset of `hand`."""
    best_score = -1
    best_idx: list[int] = []
    best_type = HandType.HIGH_CARD
    n = len(hand)
    for k in range(1, min(5, n) + 1):
        for combo in itertools.combinations(range(n), k):
            cards = [hand[i] for i in combo]
            r = score_played_hand(cards)
            if r.score > best_score:
                best_score = r.score
                best_idx = list(combo)
                best_type = r.hand_type
    return best_idx, best_score, best_type


def worst_indices(hand: list[Card], k: int) -> list[int]:
    """Indices of the k lowest-rank cards (ties broken by position)."""
    ranked = sorted(range(len(hand)), key=lambda i: (int(hand[i].rank), i))
    return ranked[:k]


def play_blind(run: Run, blind: BlindKind, verbose: bool = True) -> Round:
    rd = run.start_blind(blind)
    if verbose:
        print(f"\n=== Ante {run.ante} {blind.value} (target {rd.state.chip_target}) ===")
    while not rd.state.is_won and rd.state.hands_remaining > 0:
        idx, _, hand_type = best_play(rd.state.hand)
        # If the best available hand is weak and we have discards to spare,
        # fish for something better. Treat High Card and any Pair as weak
        # when we still have time. "Time" = at least one discard, and more
        # than one hand left (so we can afford the round trip).
        weak = hand_type in (HandType.HIGH_CARD, HandType.PAIR)
        if weak and rd.state.discards_remaining > 0 and rd.state.hands_remaining > 1:
            drop = worst_indices(rd.state.hand, 5)
            dropped_cards = [rd.state.hand[i] for i in drop]
            rd.discard(drop)
            if verbose:
                shown = " ".join(str(c) for c in dropped_cards)
                print(
                    f"  Discard [{shown}] "
                    f"(hands {rd.state.hands_remaining}, discards {rd.state.discards_remaining})"
                )
            continue
        cards_to_play = [rd.state.hand[i] for i in idx]
        result = rd.play(idx)
        if verbose:
            shown = " ".join(str(c) for c in cards_to_play)
            print(
                f"  Play    [{shown}] -> {result.hand_type.name}: "
                f"{result.chips} x {result.mult} = {result.score} "
                f"(total {rd.state.total_score}/{rd.state.chip_target}, "
                f"hands left {rd.state.hands_remaining})"
            )
    if verbose:
        outcome = "BEAT" if rd.state.is_won else "LOST"
        print(f"  -> {outcome} ({rd.state.total_score}/{rd.state.chip_target})")
    return rd


def main(seed: int = 42) -> int:
    print(f"Phase 1 smoke test - Ante 1 (seed={seed})")
    run = Run(rng_seed=seed)
    for blind in (BlindKind.SMALL, BlindKind.BIG, BlindKind.BOSS):
        rd = play_blind(run, blind)
        if not rd.state.is_won:
            print(f"\nLOST at {blind.value}.")
            return 1
    print("\nBeat Ante 1.")
    return 0


if __name__ == "__main__":
    arg_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    sys.exit(main(arg_seed))
