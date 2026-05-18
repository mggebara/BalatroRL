"""Joker base class, scoring context, and Phase 2 joker implementations.

Scoring pipeline (per played hand):
  1. Base (chips, mult) from hand-type level.
  2. For each scoring card LR:
        - chips += card.chip_value (rank + enhancement bonus)
        - each joker's `on_scoring_card(ctx, card)` hook
  3. For each held-in-hand card LR:
        - each joker's `on_held_card(ctx, card)` hook
  4. For each joker LR: `on_main(ctx)` hook.
  5. score = chips * mult.

Phase 2 jokers (20):
  Joker, Greedy/Lusty/Wrathful/Gluttonous, Jolly/Zany/Mad/Crazy/Droll,
  Sly/Wily/Clever/Devious/Crafty, Half, Mime, Credit Card, Banner,
  Misprint.

Note: "hand contains X" jokers trigger on any hand-type that *contains*
the pattern. E.g. a Full House contains both a Pair and a Three of a
Kind, so Jolly Joker (Pair) and Zany Joker (Three) both fire on it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from balatro_core.cards import Card, Suit
from balatro_core.hands import HandType


# Hand-type containment sets. A hand-type belongs in a set if it contains
# the named pattern as a sub-pattern.
_CONTAINS_PAIR = frozenset({
    HandType.PAIR, HandType.TWO_PAIR, HandType.THREE_OF_A_KIND,
    HandType.FULL_HOUSE, HandType.FOUR_OF_A_KIND,
    HandType.FIVE_OF_A_KIND, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE,
})
_CONTAINS_TWO_PAIR = frozenset({
    HandType.TWO_PAIR, HandType.FULL_HOUSE, HandType.FLUSH_HOUSE,
})
_CONTAINS_THREE = frozenset({
    HandType.THREE_OF_A_KIND, HandType.FULL_HOUSE, HandType.FOUR_OF_A_KIND,
    HandType.FIVE_OF_A_KIND, HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE,
})
_CONTAINS_STRAIGHT = frozenset({
    HandType.STRAIGHT, HandType.STRAIGHT_FLUSH,
})
_CONTAINS_FLUSH = frozenset({
    HandType.FLUSH, HandType.STRAIGHT_FLUSH,
    HandType.FLUSH_HOUSE, HandType.FLUSH_FIVE,
})


@dataclass
class ScoringContext:
    """Mutable scoring state passed through joker hooks."""
    chips: int
    mult: int
    hand_type: HandType
    played_cards: list[Card]
    scoring_cards: list[Card]
    held_cards: list[Card] = field(default_factory=list)
    discards_remaining: int = 0
    rng: Optional[random.Random] = None


class Joker:
    """Base class. Subclasses override only the hooks they need."""

    name: str = ""

    def on_scoring_card(self, ctx: ScoringContext, card: Card) -> None:
        """Fires once per scoring card, in left-to-right order."""

    def on_held_card(self, ctx: ScoringContext, card: Card) -> None:
        """Fires once per card held in hand at scoring time."""

    def on_main(self, ctx: ScoringContext) -> None:
        """Fires once per joker after all per-card triggers."""

    def __repr__(self) -> str:
        return f"<{self.name or type(self).__name__}>"


# -------- Independent / always-on jokers --------

class JokerJoker(Joker):
    """+4 Mult."""
    name = "Joker"

    def on_main(self, ctx: ScoringContext) -> None:
        ctx.mult += 4


class Banner(Joker):
    """+30 Chips for each remaining discard."""
    name = "Banner"

    def on_main(self, ctx: ScoringContext) -> None:
        ctx.chips += 30 * ctx.discards_remaining


class Misprint(Joker):
    """+0 to +23 Mult (uniform integer, rerolled per play)."""
    name = "Misprint"

    def on_main(self, ctx: ScoringContext) -> None:
        rng = ctx.rng or random.Random()
        ctx.mult += rng.randint(0, 23)


# -------- Suit-matching jokers (+3 Mult per scoring card of suit) --------

class _SuitJoker(Joker):
    suit: Suit = Suit.SPADES
    bonus: int = 3

    def on_scoring_card(self, ctx: ScoringContext, card: Card) -> None:
        if card.suit == self.suit:
            ctx.mult += self.bonus


class GreedyJoker(_SuitJoker):
    """+3 Mult per scoring Diamond."""
    name = "Greedy Joker"
    suit = Suit.DIAMONDS


class LustyJoker(_SuitJoker):
    """+3 Mult per scoring Heart."""
    name = "Lusty Joker"
    suit = Suit.HEARTS


class WrathfulJoker(_SuitJoker):
    """+3 Mult per scoring Spade."""
    name = "Wrathful Joker"
    suit = Suit.SPADES


class GluttonousJoker(_SuitJoker):
    """+3 Mult per scoring Club."""
    name = "Gluttonous Joker"
    suit = Suit.CLUBS


# -------- "Hand contains" mult jokers --------

class _ContainsMult(Joker):
    types: frozenset[HandType] = frozenset()
    bonus: int = 0

    def on_main(self, ctx: ScoringContext) -> None:
        if ctx.hand_type in self.types:
            ctx.mult += self.bonus


class JollyJoker(_ContainsMult):
    """+8 Mult if played hand contains a Pair."""
    name = "Jolly Joker"
    types = _CONTAINS_PAIR
    bonus = 8


class ZanyJoker(_ContainsMult):
    """+12 Mult if played hand contains a Three of a Kind."""
    name = "Zany Joker"
    types = _CONTAINS_THREE
    bonus = 12


class MadJoker(_ContainsMult):
    """+10 Mult if played hand contains a Two Pair."""
    name = "Mad Joker"
    types = _CONTAINS_TWO_PAIR
    bonus = 10


class CrazyJoker(_ContainsMult):
    """+12 Mult if played hand contains a Straight."""
    name = "Crazy Joker"
    types = _CONTAINS_STRAIGHT
    bonus = 12


class DrollJoker(_ContainsMult):
    """+10 Mult if played hand contains a Flush."""
    name = "Droll Joker"
    types = _CONTAINS_FLUSH
    bonus = 10


# -------- "Hand contains" chips jokers --------

class _ContainsChips(Joker):
    types: frozenset[HandType] = frozenset()
    bonus: int = 0

    def on_main(self, ctx: ScoringContext) -> None:
        if ctx.hand_type in self.types:
            ctx.chips += self.bonus


class SlyJoker(_ContainsChips):
    """+50 Chips if played hand contains a Pair."""
    name = "Sly Joker"
    types = _CONTAINS_PAIR
    bonus = 50


class WilyJoker(_ContainsChips):
    """+100 Chips if played hand contains a Three of a Kind."""
    name = "Wily Joker"
    types = _CONTAINS_THREE
    bonus = 100


class CleverJoker(_ContainsChips):
    """+80 Chips if played hand contains a Two Pair."""
    name = "Clever Joker"
    types = _CONTAINS_TWO_PAIR
    bonus = 80


class DeviousJoker(_ContainsChips):
    """+100 Chips if played hand contains a Straight."""
    name = "Devious Joker"
    types = _CONTAINS_STRAIGHT
    bonus = 100


class CraftyJoker(_ContainsChips):
    """+80 Chips if played hand contains a Flush."""
    name = "Crafty Joker"
    types = _CONTAINS_FLUSH
    bonus = 80


# -------- Conditional / held-in-hand / run-state jokers --------

class HalfJoker(Joker):
    """+20 Mult if played hand contains 3 or fewer cards."""
    name = "Half Joker"

    def on_main(self, ctx: ScoringContext) -> None:
        if len(ctx.played_cards) <= 3:
            ctx.mult += 20


class Mime(Joker):
    """Retriggers held-in-hand card abilities.

    Phase 2 cards have no enhancements/seals, so retrigger is a no-op
    in the current rules. The hook is in place for Phase 3+ when card
    enhancements (Steel, Gold seal, Red seal, etc.) ship.
    """
    name = "Mime"

    def on_held_card(self, ctx: ScoringContext, card: Card) -> None:
        return  # no-op until enhancements ship


class CreditCard(Joker):
    """Allows money to go down to -$20.

    Run-state only — has no scoring hook. The economy layer reads
    `CreditCard.min_money` (or checks `isinstance(j, CreditCard)`) when
    enforcing the money floor.
    """
    name = "Credit Card"
    min_money: int = -20


# Registry of every Phase 2 joker class. Useful for tests, shop generation
# (later phases), and iteration.
JOKER_CLASSES: tuple[type[Joker], ...] = (
    JokerJoker,
    GreedyJoker, LustyJoker, WrathfulJoker, GluttonousJoker,
    JollyJoker, ZanyJoker, MadJoker, CrazyJoker, DrollJoker,
    SlyJoker, WilyJoker, CleverJoker, DeviousJoker, CraftyJoker,
    HalfJoker, Mime, CreditCard, Banner, Misprint,
)
