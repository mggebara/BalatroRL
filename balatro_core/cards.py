"""Cards, ranks, suits, modifiers, and the draw deck."""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from enum import Enum, IntEnum


class Suit(Enum):
    SPADES = "Spades"
    HEARTS = "Hearts"
    DIAMONDS = "Diamonds"
    CLUBS = "Clubs"


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def chip_value(self) -> int:
        """Base chip contribution of a card with this rank (no enhancement)."""
        if self <= Rank.TEN:
            return int(self)
        if self == Rank.ACE:
            return 11
        return 10  # J, Q, K


# Modifiers — defined now so the data model is stable. Phase 1 scoring
# treats every card as NONE; later phases activate the effects.
class Enhancement(Enum):
    NONE = "none"
    BONUS = "bonus"
    MULT = "mult"
    WILD = "wild"
    GLASS = "glass"
    STEEL = "steel"
    STONE = "stone"
    GOLD = "gold"
    LUCKY = "lucky"


class Edition(Enum):
    NONE = "none"
    FOIL = "foil"
    HOLOGRAPHIC = "holographic"
    POLYCHROME = "polychrome"
    NEGATIVE = "negative"


class Seal(Enum):
    NONE = "none"
    GOLD = "gold"
    RED = "red"
    BLUE = "blue"
    PURPLE = "purple"


_RANK_GLYPH = {11: "J", 12: "Q", 13: "K", 14: "A"}
_SUIT_GLYPH = {Suit.SPADES: "S", Suit.HEARTS: "H", Suit.DIAMONDS: "D", Suit.CLUBS: "C"}


@dataclass
class Card:
    rank: Rank
    suit: Suit
    enhancement: Enhancement = Enhancement.NONE
    edition: Edition = Edition.NONE
    seal: Seal = Seal.NONE

    def __str__(self) -> str:
        r = _RANK_GLYPH.get(int(self.rank), str(int(self.rank)))
        return f"{r}{_SUIT_GLYPH[self.suit]}"

    @property
    def chip_value(self) -> int:
        """Chip contribution when this card scores (rank + enhancement bonus only)."""
        return self.rank.chip_value


def standard_deck() -> list[Card]:
    """A standard 52-card deck, no enhancements."""
    return [Card(rank=r, suit=s) for s, r in itertools.product(Suit, Rank)]


class Deck:
    """Draw deck. Tracks the full card list and the current draw pile."""

    def __init__(self, cards: list[Card], rng: random.Random | None = None) -> None:
        self._all_cards: list[Card] = list(cards)
        self._remaining: list[Card] = list(cards)
        self._rng = rng or random.Random()

    @property
    def cards(self) -> list[Card]:
        return list(self._all_cards)

    @property
    def remaining(self) -> int:
        return len(self._remaining)

    def shuffle(self) -> None:
        self._rng.shuffle(self._remaining)

    def reset_round(self) -> None:
        """Restore every card to the draw pile and shuffle."""
        self._remaining = list(self._all_cards)
        self.shuffle()

    def draw(self, n: int) -> list[Card]:
        take = min(n, len(self._remaining))
        drawn = self._remaining[:take]
        self._remaining = self._remaining[take:]
        return drawn
