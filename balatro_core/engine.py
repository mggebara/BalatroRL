"""Run / Ante / Blind / Round state machine.

Phase 1 scope: no shop, no jokers, no consumables. Just play through
blinds with a fixed deck.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

from balatro_core.cards import Card, Deck, standard_deck
from balatro_core.hands import HandType
from balatro_core.scoring import ScoreResult, score_played_hand


# Base small-blind chip target per ante (vanilla, no stake scaling).
ANTE_BASE_CHIPS: dict[int, int] = {
    1: 300,
    2: 800,
    3: 2000,
    4: 5000,
    5: 11000,
    6: 20000,
    7: 35000,
    8: 50000,
}


class BlindKind(Enum):
    SMALL = "Small Blind"
    BIG = "Big Blind"
    BOSS = "Boss Blind"


_BLIND_MULTIPLIERS: dict[BlindKind, float] = {
    BlindKind.SMALL: 1.0,
    BlindKind.BIG: 1.5,
    BlindKind.BOSS: 2.0,
}


def blind_chips(ante: int, kind: BlindKind) -> int:
    if ante not in ANTE_BASE_CHIPS:
        raise ValueError(f"unsupported ante {ante}")
    return int(ANTE_BASE_CHIPS[ante] * _BLIND_MULTIPLIERS[kind])


@dataclass
class RoundState:
    ante: int
    blind: BlindKind
    chip_target: int
    hands_remaining: int
    discards_remaining: int
    hand_size: int
    hand: list[Card] = field(default_factory=list)
    total_score: int = 0
    score_log: list[ScoreResult] = field(default_factory=list)

    @property
    def is_won(self) -> bool:
        return self.total_score >= self.chip_target

    @property
    def is_lost(self) -> bool:
        return not self.is_won and self.hands_remaining <= 0


class Round:
    """One blind. Player picks hand-index subsets to play or discard."""

    def __init__(
        self,
        deck: Deck,
        ante: int,
        blind: BlindKind,
        hand_size: int = 8,
        hands: int = 4,
        discards: int = 4,
        hand_levels: dict[HandType, int] | None = None,
    ) -> None:
        self.deck = deck
        self.hand_levels = hand_levels or {}
        self.state = RoundState(
            ante=ante,
            blind=blind,
            chip_target=blind_chips(ante, blind),
            hands_remaining=hands,
            discards_remaining=discards,
            hand_size=hand_size,
        )
        self.deck.reset_round()
        self._draw_to_full()

    def _draw_to_full(self) -> None:
        needed = self.state.hand_size - len(self.state.hand)
        if needed > 0:
            self.state.hand.extend(self.deck.draw(needed))

    def _remove_indices(self, indices: list[int]) -> list[Card]:
        for i in indices:
            if not 0 <= i < len(self.state.hand):
                raise IndexError(f"hand index {i} out of range")
        if len(set(indices)) != len(indices):
            raise ValueError("duplicate indices")
        cards = [self.state.hand[i] for i in indices]
        for i in sorted(indices, reverse=True):
            self.state.hand.pop(i)
        return cards

    def play(self, indices: list[int]) -> ScoreResult:
        if not 1 <= len(indices) <= 5:
            raise ValueError(f"must play 1-5 cards (got {len(indices)})")
        if self.state.hands_remaining <= 0:
            raise RuntimeError("no hands remaining")
        played = self._remove_indices(indices)
        result = score_played_hand(played, self.hand_levels)
        self.state.total_score += result.score
        self.state.hands_remaining -= 1
        self.state.score_log.append(result)
        if not self.state.is_won and self.state.hands_remaining > 0:
            self._draw_to_full()
        return result

    def discard(self, indices: list[int]) -> list[Card]:
        if not 1 <= len(indices) <= 5:
            raise ValueError(f"must discard 1-5 cards (got {len(indices)})")
        if self.state.discards_remaining <= 0:
            raise RuntimeError("no discards remaining")
        discarded = self._remove_indices(indices)
        self.state.discards_remaining -= 1
        self._draw_to_full()
        return discarded


class Run:
    """Top-level run. Phase 1: deck + ante counter only."""

    def __init__(
        self,
        deck_cards: list[Card] | None = None,
        starting_money: int = 4,
        hand_size: int = 8,
        hands_per_round: int = 4,
        discards_per_round: int = 4,
        starting_ante: int = 1,
        rng_seed: int | None = None,
    ) -> None:
        rng = random.Random(rng_seed)
        cards = deck_cards if deck_cards is not None else standard_deck()
        self.deck = Deck(cards, rng=rng)
        self.money = starting_money
        self.hand_size = hand_size
        self.hands_per_round = hands_per_round
        self.discards_per_round = discards_per_round
        self.ante = starting_ante
        self.hand_levels: dict[HandType, int] = {}

    def start_blind(self, blind: BlindKind) -> Round:
        return Round(
            deck=self.deck,
            ante=self.ante,
            blind=blind,
            hand_size=self.hand_size,
            hands=self.hands_per_round,
            discards=self.discards_per_round,
            hand_levels=self.hand_levels,
        )
