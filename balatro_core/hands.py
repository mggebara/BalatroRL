"""Hand type detection and level-based base scoring.

Phase 1 scope: vanilla hand evaluation. No joker effects, no enhancement
edge cases (Stone, Wild, etc.). Those will plug in via the scoring
pipeline in later phases.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum

from balatro_core.cards import Card, Rank


class HandType(IntEnum):
    """Higher integer = stronger hand."""
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    FIVE_OF_A_KIND = 9
    FLUSH_HOUSE = 10
    FLUSH_FIVE = 11


# Level-1 (chips, mult) for each hand type. Values from the in-game table.
BASE_HAND_VALUES: dict[HandType, tuple[int, int]] = {
    HandType.HIGH_CARD:       (5,   1),
    HandType.PAIR:            (10,  2),
    HandType.TWO_PAIR:        (20,  2),
    HandType.THREE_OF_A_KIND: (30,  3),
    HandType.STRAIGHT:        (30,  4),
    HandType.FLUSH:           (35,  4),
    HandType.FULL_HOUSE:      (40,  4),
    HandType.FOUR_OF_A_KIND:  (60,  7),
    HandType.STRAIGHT_FLUSH:  (100, 8),
    HandType.FIVE_OF_A_KIND:  (120, 12),
    HandType.FLUSH_HOUSE:     (140, 14),
    HandType.FLUSH_FIVE:      (160, 16),
}

# Per-level deltas: (+chips, +mult) added when the hand type is leveled up.
HAND_LEVEL_INCREMENTS: dict[HandType, tuple[int, int]] = {
    HandType.HIGH_CARD:       (10, 1),
    HandType.PAIR:            (15, 1),
    HandType.TWO_PAIR:        (20, 1),
    HandType.THREE_OF_A_KIND: (20, 2),
    HandType.STRAIGHT:        (30, 3),
    HandType.FLUSH:           (15, 2),
    HandType.FULL_HOUSE:      (25, 2),
    HandType.FOUR_OF_A_KIND:  (30, 3),
    HandType.STRAIGHT_FLUSH:  (40, 4),
    HandType.FIVE_OF_A_KIND:  (35, 3),
    HandType.FLUSH_HOUSE:     (40, 4),
    HandType.FLUSH_FIVE:      (50, 3),
}


def hand_chips_mult(hand_type: HandType, level: int = 1) -> tuple[int, int]:
    """Base chips and mult for a hand type at a given (1-indexed) level."""
    if level < 1:
        raise ValueError(f"level must be >= 1 (got {level})")
    base_c, base_m = BASE_HAND_VALUES[hand_type]
    inc_c, inc_m = HAND_LEVEL_INCREMENTS[hand_type]
    extra = level - 1
    return base_c + inc_c * extra, base_m + inc_m * extra


@dataclass(frozen=True)
class HandEvaluation:
    hand_type: HandType
    scoring_cards: list[Card]


def _is_straight_ranks(ranks: list[Rank]) -> bool:
    """5 distinct ranks forming a consecutive run. Ace counts high or low."""
    if len(set(ranks)) != 5:
        return False
    values = sorted(int(r) for r in ranks)
    if all(values[i + 1] - values[i] == 1 for i in range(4)):
        return True
    # Wheel: A-2-3-4-5
    return values == [2, 3, 4, 5, 14]


def evaluate_hand(cards: list[Card]) -> HandEvaluation:
    """Detect the best hand type from 1-5 played cards.

    Returns the detected type plus the subset of `cards` that scores (these
    are the cards whose chip values are added to the hand's base chips).
    """
    n = len(cards)
    if not 1 <= n <= 5:
        raise ValueError(f"played hand must have 1-5 cards (got {n})")

    rank_counts = Counter(c.rank for c in cards)
    counts_desc = sorted(rank_counts.values(), reverse=True)

    flush = n == 5 and len({c.suit for c in cards}) == 1
    straight = n == 5 and _is_straight_ranks([c.rank for c in cards])

    if n == 5:
        if counts_desc[0] == 5 and flush:
            return HandEvaluation(HandType.FLUSH_FIVE, list(cards))
        if counts_desc[0] == 5:
            return HandEvaluation(HandType.FIVE_OF_A_KIND, list(cards))
        if counts_desc[:2] == [3, 2] and flush:
            return HandEvaluation(HandType.FLUSH_HOUSE, list(cards))
        if straight and flush:
            return HandEvaluation(HandType.STRAIGHT_FLUSH, list(cards))
        if counts_desc[:2] == [3, 2]:
            return HandEvaluation(HandType.FULL_HOUSE, list(cards))
        if flush:
            return HandEvaluation(HandType.FLUSH, list(cards))
        if straight:
            return HandEvaluation(HandType.STRAIGHT, list(cards))

    if counts_desc and counts_desc[0] >= 4:
        target = max(r for r, c in rank_counts.items() if c >= 4)
        scoring = [c for c in cards if c.rank == target]
        return HandEvaluation(HandType.FOUR_OF_A_KIND, scoring)

    if counts_desc and counts_desc[0] >= 3:
        target = max(r for r, c in rank_counts.items() if c >= 3)
        scoring = [c for c in cards if c.rank == target]
        return HandEvaluation(HandType.THREE_OF_A_KIND, scoring)

    pairs = [r for r, c in rank_counts.items() if c >= 2]
    if len(pairs) >= 2:
        top_two = sorted(pairs, reverse=True)[:2]
        scoring = [c for c in cards if c.rank in top_two]
        return HandEvaluation(HandType.TWO_PAIR, scoring)

    if pairs:
        target = max(pairs)
        scoring = [c for c in cards if c.rank == target]
        return HandEvaluation(HandType.PAIR, scoring)

    top = max(cards, key=lambda c: int(c.rank))
    return HandEvaluation(HandType.HIGH_CARD, [top])
