"""Smoke test: play a full Ante 1 with a simple scripted agent.

Demonstrates that the engine plays Small + Big + Boss blinds end-to-end.
Strategy: brute-force pick the 1-5 cards from the current hand that
maximize this play's score. Use discards when the best available hand
type is High Card or Pair (and we have discards + hands to spare).

Usage:
    python3 scripts/smoke_ante1.py            # seed 42, no jokers
    python3 scripts/smoke_ante1.py 7          # seed 7, no jokers
    python3 scripts/smoke_ante1.py 42 joker   # seed 42, +4 mult Joker
    python3 scripts/smoke_ante1.py 42 joker,lusty,half   # multiple jokers
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

# Allow running directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from balatro_core.cards import Card  # noqa: E402
from balatro_core.engine import BlindKind, Round, Run  # noqa: E402
from balatro_core.hands import HandType  # noqa: E402
from balatro_core.jokers import JOKER_CLASSES, Joker  # noqa: E402
from balatro_core.scoring import score_played_hand  # noqa: E402


def parse_jokers(spec: str | None) -> list[Joker]:
    """Comma-separated joker names (case-insensitive, partial match)."""
    if not spec:
        return []
    by_lower = {cls().name.lower(): cls for cls in JOKER_CLASSES}
    out: list[Joker] = []
    for token in spec.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key in by_lower:
            out.append(by_lower[key]())
            continue
        matches = [cls for name, cls in by_lower.items() if key in name]
        if not matches:
            raise SystemExit(f"unknown joker: {token!r}")
        if len(matches) > 1:
            names = ", ".join(cls().name for cls in matches)
            raise SystemExit(f"ambiguous joker {token!r} (matches: {names})")
        out.append(matches[0]())
    return out


def best_play(
    hand: list[Card],
    jokers: list[Joker] | None = None,
    discards_remaining: int = 0,
) -> tuple[list[int], int, HandType]:
    """Brute-force the highest-scoring 1-5 card subset of `hand`."""
    best_score = -1
    best_idx: list[int] = []
    best_type = HandType.HIGH_CARD
    n = len(hand)
    for k in range(1, min(5, n) + 1):
        for combo in itertools.combinations(range(n), k):
            cards = [hand[i] for i in combo]
            held = [hand[i] for i in range(n) if i not in combo]
            r = score_played_hand(
                cards,
                jokers=jokers,
                held_cards=held,
                discards_remaining=discards_remaining,
            )
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
        idx, _, hand_type = best_play(
            rd.state.hand,
            jokers=rd.jokers,
            discards_remaining=rd.state.discards_remaining,
        )
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


def main(seed: int = 42, joker_spec: str | None = None) -> int:
    jokers = parse_jokers(joker_spec)
    joker_str = ", ".join(j.name for j in jokers) or "none"
    print(f"Smoke test - Ante 1 (seed={seed}, jokers=[{joker_str}])")
    run = Run(rng_seed=seed, jokers=jokers)
    for blind in (BlindKind.SMALL, BlindKind.BIG, BlindKind.BOSS):
        rd = play_blind(run, blind)
        if not rd.state.is_won:
            print(f"\nLOST at {blind.value}.")
            return 1
    print("\nBeat Ante 1.")
    return 0


if __name__ == "__main__":
    arg_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    arg_jokers = sys.argv[2] if len(sys.argv) > 2 else None
    sys.exit(main(arg_seed, arg_jokers))
