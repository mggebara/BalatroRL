"""Unit + property tests for the Phase 2 jokers.

Each joker gets a focused test that isolates its trigger on a
hand-crafted board. Property tests cover invariants that must hold
across the full Phase 2 joker pool.
"""
from __future__ import annotations

import itertools
import random

import pytest

from balatro_core.cards import Card, Rank, Suit
from balatro_core.hands import HandType
from balatro_core.jokers import (
    JOKER_CLASSES,
    Banner,
    CleverJoker,
    CraftyJoker,
    CrazyJoker,
    CreditCard,
    DeviousJoker,
    DrollJoker,
    GluttonousJoker,
    GreedyJoker,
    HalfJoker,
    JokerJoker,
    JollyJoker,
    LustyJoker,
    MadJoker,
    Mime,
    Misprint,
    SlyJoker,
    WilyJoker,
    WrathfulJoker,
    ZanyJoker,
)
from balatro_core.scoring import score_played_hand


def C(rank: Rank, suit: Suit = Suit.SPADES) -> Card:
    return Card(rank=rank, suit=suit)


PAIR_OF_SEVENS = [C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS)]
# Pair lvl 1: 10c * 2m, scoring 7+7 -> chips 24, mult 2, score 48.
PAIR_BASE = (24, 2, 48)


def baseline_pair_score() -> int:
    return PAIR_BASE[2]


class TestIndependentJokers:
    def test_joker_plain_adds_four_mult(self):
        # (10+14) * (2+4) = 24 * 6 = 144
        r = score_played_hand(list(PAIR_OF_SEVENS), jokers=[JokerJoker()])
        assert r.chips == 24
        assert r.mult == 6
        assert r.score == 144

    def test_banner_zero_discards_no_chips(self):
        r = score_played_hand(list(PAIR_OF_SEVENS), jokers=[Banner()], discards_remaining=0)
        assert r.score == baseline_pair_score()

    def test_banner_three_discards_adds_90_chips(self):
        # (10 + 14 + 90) * 2 = 114 * 2 = 228
        r = score_played_hand(list(PAIR_OF_SEVENS), jokers=[Banner()], discards_remaining=3)
        assert r.chips == 114
        assert r.score == 228

    def test_misprint_uniform_range(self):
        # Sample many rolls with a fixed RNG; every observed mult bonus
        # must land in [0, 23], and we should observe both endpoints
        # within a few hundred rolls.
        rng = random.Random(0)
        bonuses = set()
        for _ in range(500):
            r = score_played_hand(list(PAIR_OF_SEVENS), jokers=[Misprint()], rng=rng)
            bonus = r.mult - 2  # base mult is 2
            assert 0 <= bonus <= 23
            bonuses.add(bonus)
        assert 0 in bonuses and 23 in bonuses

    def test_misprint_deterministic_given_rng(self):
        # Same seed -> same roll across runs.
        rng_a = random.Random(42)
        rng_b = random.Random(42)
        a = score_played_hand(list(PAIR_OF_SEVENS), jokers=[Misprint()], rng=rng_a)
        b = score_played_hand(list(PAIR_OF_SEVENS), jokers=[Misprint()], rng=rng_b)
        assert a.score == b.score


class TestSuitJokers:
    def test_greedy_per_diamond(self):
        # Pair of 7 diamonds, no kicker. Pair lvl 1: 10c*2m. 7+7=14. +3 mult per diamond.
        # chips=24, mult=2+3+3=8 -> score = 192
        cards = [C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.SEVEN, Suit.DIAMONDS)]
        r = score_played_hand(cards, jokers=[GreedyJoker()])
        assert r.mult == 8
        assert r.score == 192

    def test_greedy_only_diamonds_score(self):
        # Mixed-suit pair: one diamond, one spade. Only diamond scores +3.
        cards = [C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.SEVEN, Suit.SPADES)]
        r = score_played_hand(cards, jokers=[GreedyJoker()])
        assert r.mult == 5  # base 2 + 3

    def test_lusty_per_heart(self):
        cards = [C(Rank.SEVEN, Suit.HEARTS), C(Rank.SEVEN, Suit.HEARTS)]
        r = score_played_hand(cards, jokers=[LustyJoker()])
        assert r.mult == 8

    def test_wrathful_per_spade(self):
        cards = [C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.SPADES)]
        r = score_played_hand(cards, jokers=[WrathfulJoker()])
        assert r.mult == 8

    def test_gluttonous_per_club(self):
        cards = [C(Rank.SEVEN, Suit.CLUBS), C(Rank.SEVEN, Suit.CLUBS)]
        r = score_played_hand(cards, jokers=[GluttonousJoker()])
        assert r.mult == 8

    def test_suit_joker_ignores_non_scoring_kicker(self):
        # Pair of spades + diamond kicker. Wrathful (spade) only counts the two
        # pair cards; kicker doesn't score so no bonus from it.
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.SPADES),
            C(Rank.KING, Suit.SPADES), C(Rank.QUEEN, Suit.SPADES),
            C(Rank.TWO, Suit.DIAMONDS),
        ]
        # Wait — pair of 7 spades + spade kickers means kickers DON'T score for
        # a Pair hand (only the pair cards score). So Wrathful only adds for
        # the two scoring spades. Mult = 2 + 3*2 = 8.
        r = score_played_hand(cards, jokers=[WrathfulJoker()])
        assert r.hand_type == HandType.PAIR
        assert r.mult == 8


class TestContainsMultJokers:
    def test_jolly_fires_on_pair(self):
        r = score_played_hand(list(PAIR_OF_SEVENS), jokers=[JollyJoker()])
        assert r.mult == 2 + 8

    def test_jolly_does_not_fire_on_high_card(self):
        r = score_played_hand([C(Rank.KING)], jokers=[JollyJoker()])
        assert r.mult == 1

    def test_jolly_fires_on_full_house(self):
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS), C(Rank.SEVEN, Suit.DIAMONDS),
            C(Rank.KING, Suit.SPADES), C(Rank.KING, Suit.HEARTS),
        ]
        r = score_played_hand(cards, jokers=[JollyJoker()])
        assert r.hand_type == HandType.FULL_HOUSE
        assert r.mult == 4 + 8

    def test_zany_fires_on_three(self):
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS), C(Rank.SEVEN, Suit.DIAMONDS),
            C(Rank.TWO), C(Rank.THREE),
        ]
        r = score_played_hand(cards, jokers=[ZanyJoker()])
        assert r.hand_type == HandType.THREE_OF_A_KIND
        assert r.mult == 3 + 12

    def test_zany_does_not_fire_on_pair(self):
        r = score_played_hand(list(PAIR_OF_SEVENS), jokers=[ZanyJoker()])
        assert r.mult == 2

    def test_mad_fires_on_two_pair(self):
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS),
            C(Rank.KING, Suit.SPADES), C(Rank.KING, Suit.HEARTS),
            C(Rank.TWO),
        ]
        r = score_played_hand(cards, jokers=[MadJoker()])
        assert r.mult == 2 + 10

    def test_mad_fires_on_full_house(self):
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS), C(Rank.SEVEN, Suit.DIAMONDS),
            C(Rank.KING, Suit.SPADES), C(Rank.KING, Suit.HEARTS),
        ]
        r = score_played_hand(cards, jokers=[MadJoker()])
        assert r.mult == 4 + 10

    def test_crazy_fires_on_straight(self):
        cards = [
            C(Rank.FIVE, Suit.HEARTS), C(Rank.SIX, Suit.SPADES),
            C(Rank.SEVEN, Suit.HEARTS), C(Rank.EIGHT, Suit.SPADES),
            C(Rank.NINE, Suit.HEARTS),
        ]
        r = score_played_hand(cards, jokers=[CrazyJoker()])
        assert r.mult == 4 + 12

    def test_crazy_fires_on_straight_flush(self):
        cards = [Card(r, Suit.SPADES) for r in
                 [Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE]]
        r = score_played_hand(cards, jokers=[CrazyJoker()])
        assert r.mult == 8 + 12

    def test_droll_fires_on_flush(self):
        cards = [Card(r, Suit.SPADES) for r in
                 [Rank.TWO, Rank.FIVE, Rank.SEVEN, Rank.NINE, Rank.JACK]]
        r = score_played_hand(cards, jokers=[DrollJoker()])
        assert r.mult == 4 + 10

    def test_droll_fires_on_straight_flush(self):
        cards = [Card(r, Suit.SPADES) for r in
                 [Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE]]
        r = score_played_hand(cards, jokers=[DrollJoker()])
        assert r.mult == 8 + 10


class TestContainsChipsJokers:
    def test_sly_fires_on_pair(self):
        # Pair: 10c * 2m, +50c, +7+7. (10+50+14)*2 = 148
        r = score_played_hand(list(PAIR_OF_SEVENS), jokers=[SlyJoker()])
        assert r.chips == 74
        assert r.score == 148

    def test_sly_does_not_fire_on_high_card(self):
        r = score_played_hand([C(Rank.KING)], jokers=[SlyJoker()])
        assert r.score == 15

    def test_wily_fires_on_three(self):
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS), C(Rank.SEVEN, Suit.DIAMONDS),
            C(Rank.TWO), C(Rank.THREE),
        ]
        r = score_played_hand(cards, jokers=[WilyJoker()])
        # 3OAK lvl 1: 30c*3m, 7+7+7=21, +100c. (30+21+100)*3 = 453
        assert r.chips == 151
        assert r.score == 453

    def test_clever_fires_on_two_pair(self):
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS),
            C(Rank.KING, Suit.SPADES), C(Rank.KING, Suit.HEARTS),
            C(Rank.TWO),
        ]
        # 2P: 20c*2m, 7+7+10+10=34, +80c. (20+34+80)*2 = 268
        r = score_played_hand(cards, jokers=[CleverJoker()])
        assert r.score == 268

    def test_devious_fires_on_straight(self):
        cards = [
            C(Rank.FIVE, Suit.HEARTS), C(Rank.SIX, Suit.SPADES),
            C(Rank.SEVEN, Suit.HEARTS), C(Rank.EIGHT, Suit.SPADES),
            C(Rank.NINE, Suit.HEARTS),
        ]
        # Straight: 30c*4m, 5+6+7+8+9=35, +100c. (30+35+100)*4 = 660
        r = score_played_hand(cards, jokers=[DeviousJoker()])
        assert r.score == 660

    def test_crafty_fires_on_flush(self):
        cards = [Card(r, Suit.SPADES) for r in
                 [Rank.TWO, Rank.FIVE, Rank.SEVEN, Rank.NINE, Rank.JACK]]
        # Flush: 35c*4m, 2+5+7+9+10=33, +80c. (35+33+80)*4 = 592
        r = score_played_hand(cards, jokers=[CraftyJoker()])
        assert r.score == 592


class TestConditionalJokers:
    def test_half_fires_on_one_card(self):
        # High card K: (5+10)*(1+20) = 15*21 = 315
        r = score_played_hand([C(Rank.KING)], jokers=[HalfJoker()])
        assert r.mult == 21
        assert r.score == 315

    def test_half_fires_on_two_cards(self):
        # (10+14)*(2+20) = 24*22 = 528
        r = score_played_hand(list(PAIR_OF_SEVENS), jokers=[HalfJoker()])
        assert r.mult == 22
        assert r.score == 528

    def test_half_fires_on_three_cards(self):
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS), C(Rank.SEVEN, Suit.DIAMONDS),
        ]
        # 3OAK: 30c*3m, 21 chips. (30+21)*(3+20) = 51*23 = 1173
        r = score_played_hand(cards, jokers=[HalfJoker()])
        assert r.mult == 23
        assert r.score == 1173

    def test_half_does_not_fire_on_four_cards(self):
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS),
            C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.SEVEN, Suit.CLUBS),
        ]
        # 4OAK: 60c*7m, 28 chips. (60+28)*7 = 616 (no half bonus)
        r = score_played_hand(cards, jokers=[HalfJoker()])
        assert r.mult == 7
        assert r.score == 616

    def test_half_does_not_fire_on_five_cards(self):
        cards = [Card(r, Suit.SPADES) for r in
                 [Rank.TWO, Rank.FIVE, Rank.SEVEN, Rank.NINE, Rank.JACK]]
        r = score_played_hand(cards, jokers=[HalfJoker()])
        assert r.mult == 4  # base flush mult, no bonus


class TestHeldInHandJokers:
    def test_mime_no_op_without_enhancements(self):
        # Phase 2 cards have no held-in-hand triggers; Mime is a no-op.
        held = [C(Rank.KING, Suit.SPADES), C(Rank.ACE, Suit.HEARTS)]
        r = score_played_hand(list(PAIR_OF_SEVENS), held_cards=held, jokers=[Mime()])
        assert r.score == baseline_pair_score()


class TestRunStateJokers:
    def test_credit_card_does_not_affect_scoring(self):
        r = score_played_hand(list(PAIR_OF_SEVENS), jokers=[CreditCard()])
        assert r.score == baseline_pair_score()

    def test_credit_card_min_money(self):
        assert CreditCard.min_money == -20


# ---------------- Property tests ----------------

class TestProperties:
    """Invariants that must hold across the full Phase 2 joker pool."""

    def test_score_invariant_under_joker_reorder(self):
        # Every Phase 2 joker is positional-independent (additive). Permuting
        # the joker list must not change the final score.
        # Use a fixed RNG so Misprint is deterministic.
        cards = [
            C(Rank.SEVEN, Suit.HEARTS), C(Rank.SEVEN, Suit.SPADES),
            C(Rank.KING, Suit.HEARTS), C(Rank.KING, Suit.SPADES),
            C(Rank.TWO, Suit.HEARTS),
        ]
        instances = [
            JokerJoker(), LustyJoker(), WrathfulJoker(),
            JollyJoker(), MadJoker(), SlyJoker(), CleverJoker(),
            HalfJoker(), Banner(),
        ]
        # First reference score.
        ref = score_played_hand(
            cards, jokers=list(instances), discards_remaining=2,
        ).score
        # Try a handful of random permutations.
        rng = random.Random(7)
        for _ in range(20):
            shuffled = list(instances)
            rng.shuffle(shuffled)
            got = score_played_hand(
                cards, jokers=shuffled, discards_remaining=2,
            ).score
            assert got == ref, f"score changed under reorder: {got} != {ref}"

    def test_empty_joker_list_matches_phase1(self):
        # The Phase 1 scoring formula must equal Phase 2 with no jokers.
        r = score_played_hand(list(PAIR_OF_SEVENS))
        assert r.score == 48

    def test_every_joker_class_instantiable(self):
        for cls in JOKER_CLASSES:
            j = cls()
            assert isinstance(j, cls)
            assert j.name, f"{cls.__name__} missing name"

    def test_joker_names_unique(self):
        names = [cls().name for cls in JOKER_CLASSES]
        assert len(names) == len(set(names))

    def test_joker_count_is_twenty(self):
        assert len(JOKER_CLASSES) == 20

    def test_independent_jokers_no_per_card_effect(self):
        # JokerJoker / Banner / Half / Credit / Mime have no on_scoring_card
        # behavior. Calling on_scoring_card should leave ctx unchanged.
        # Smoke check: per-suit jokers are the only ones with per-card effect.
        from balatro_core.jokers import ScoringContext
        for cls in (JokerJoker, Banner, HalfJoker, CreditCard, Mime,
                    JollyJoker, ZanyJoker, MadJoker, CrazyJoker, DrollJoker,
                    SlyJoker, WilyJoker, CleverJoker, DeviousJoker, CraftyJoker,
                    Misprint):
            ctx = ScoringContext(
                chips=10, mult=2, hand_type=HandType.PAIR,
                played_cards=[], scoring_cards=[],
            )
            cls().on_scoring_card(ctx, C(Rank.SEVEN, Suit.HEARTS))
            assert ctx.chips == 10 and ctx.mult == 2

    def test_suit_jokers_only_fire_on_matching_suit(self):
        # Sweep a single-card hand across all four suits with each suit joker.
        joker_for_suit = {
            Suit.DIAMONDS: GreedyJoker,
            Suit.HEARTS: LustyJoker,
            Suit.SPADES: WrathfulJoker,
            Suit.CLUBS: GluttonousJoker,
        }
        for card_suit, joker_suit in itertools.product(Suit, Suit):
            cards = [C(Rank.SEVEN, card_suit)]
            r = score_played_hand(cards, jokers=[joker_for_suit[joker_suit]()])
            expected_mult = 1 + (3 if card_suit == joker_suit else 0)
            assert r.mult == expected_mult, (
                f"{joker_for_suit[joker_suit].__name__} on {card_suit}: "
                f"mult={r.mult} expected={expected_mult}"
            )
