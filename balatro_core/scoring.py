"""Score a played hand through the joker pipeline.

Pipeline:
  1. Detect hand type and scoring-card subset (`evaluate_hand`).
  2. Base (chips, mult) from hand-type level (`hand_chips_mult`).
  3. For each scoring card LR: add chip value, fire per-card joker hooks.
  4. For each held-in-hand card LR: fire per-card joker hooks.
  5. For each joker LR: fire `on_main` hook (additive bonuses, conditionals).
  6. score = chips * mult.

This signature is backward-compatible with Phase 1: passing only `cards`
(no jokers, no held cards, no rng) reproduces the simple hand scoring.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from balatro_core.cards import Card
from balatro_core.hands import HandType, evaluate_hand, hand_chips_mult
from balatro_core.jokers import Joker, ScoringContext


@dataclass
class ScoreResult:
    hand_type: HandType
    chips: int
    mult: int
    score: int
    scoring_cards: list[Card] = field(default_factory=list)
    played_cards: list[Card] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.hand_type.name}: {self.chips} x {self.mult} = {self.score}"


def score_played_hand(
    cards: list[Card],
    hand_levels: dict[HandType, int] | None = None,
    held_cards: list[Card] | None = None,
    jokers: list[Joker] | None = None,
    discards_remaining: int = 0,
    rng: Optional[random.Random] = None,
) -> ScoreResult:
    """Score a played hand of 1-5 cards through the joker pipeline."""
    evaluation = evaluate_hand(cards)
    level = (hand_levels or {}).get(evaluation.hand_type, 1)
    base_c, base_m = hand_chips_mult(evaluation.hand_type, level)

    ctx = ScoringContext(
        chips=base_c,
        mult=base_m,
        hand_type=evaluation.hand_type,
        played_cards=list(cards),
        scoring_cards=list(evaluation.scoring_cards),
        held_cards=list(held_cards or []),
        discards_remaining=discards_remaining,
        rng=rng,
    )
    js = jokers or []

    for card in evaluation.scoring_cards:
        ctx.chips += card.chip_value
        for j in js:
            j.on_scoring_card(ctx, card)

    for card in ctx.held_cards:
        for j in js:
            j.on_held_card(ctx, card)

    for j in js:
        j.on_main(ctx)

    return ScoreResult(
        hand_type=evaluation.hand_type,
        chips=ctx.chips,
        mult=ctx.mult,
        score=ctx.chips * ctx.mult,
        scoring_cards=list(evaluation.scoring_cards),
        played_cards=list(cards),
    )
