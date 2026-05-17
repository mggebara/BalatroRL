"""Score a played hand.

Phase 1 pipeline: hand type -> base (chips, mult) at current level -> add
chip value of each scoring card -> chips * mult. No joker triggers, no
enhancement bonuses, no edition bonuses. Those plug in at named points
in later phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from balatro_core.cards import Card
from balatro_core.hands import HandType, evaluate_hand, hand_chips_mult


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
) -> ScoreResult:
    """Score a played hand of 1-5 cards."""
    evaluation = evaluate_hand(cards)
    levels = hand_levels or {}
    level = levels.get(evaluation.hand_type, 1)
    chips, mult = hand_chips_mult(evaluation.hand_type, level)
    for card in evaluation.scoring_cards:
        chips += card.chip_value
    return ScoreResult(
        hand_type=evaluation.hand_type,
        chips=chips,
        mult=mult,
        score=chips * mult,
        scoring_cards=list(evaluation.scoring_cards),
        played_cards=list(cards),
    )
