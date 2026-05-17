"""Hand-type detection and base-value tests."""
from __future__ import annotations

import pytest

from balatro_core.cards import Card, Rank, Suit
from balatro_core.hands import HandType, evaluate_hand, hand_chips_mult


def C(rank: Rank, suit: Suit = Suit.SPADES) -> Card:
    return Card(rank=rank, suit=suit)


class TestEvaluateHand:
    def test_high_card_single(self):
        e = evaluate_hand([C(Rank.SEVEN)])
        assert e.hand_type == HandType.HIGH_CARD
        assert len(e.scoring_cards) == 1

    def test_high_card_with_kicker(self):
        # 5 unrelated cards, mixed suits, not a straight.
        cards = [
            C(Rank.TWO, Suit.SPADES),
            C(Rank.FOUR, Suit.HEARTS),
            C(Rank.SEVEN, Suit.DIAMONDS),
            C(Rank.NINE, Suit.CLUBS),
            C(Rank.KING, Suit.SPADES),
        ]
        e = evaluate_hand(cards)
        assert e.hand_type == HandType.HIGH_CARD
        # Only the king scores.
        assert e.scoring_cards == [cards[4]]

    def test_pair(self):
        e = evaluate_hand([C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS)])
        assert e.hand_type == HandType.PAIR
        assert len(e.scoring_cards) == 2

    def test_two_pair(self):
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS),
            C(Rank.KING, Suit.SPADES), C(Rank.KING, Suit.HEARTS),
            C(Rank.TWO, Suit.SPADES),
        ]
        e = evaluate_hand(cards)
        assert e.hand_type == HandType.TWO_PAIR
        assert len(e.scoring_cards) == 4  # kicker does not score

    def test_three_of_a_kind(self):
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS), C(Rank.SEVEN, Suit.DIAMONDS),
            C(Rank.TWO, Suit.SPADES), C(Rank.THREE, Suit.SPADES),
        ]
        e = evaluate_hand(cards)
        assert e.hand_type == HandType.THREE_OF_A_KIND
        assert len(e.scoring_cards) == 3

    def test_straight_mid(self):
        cards = [
            C(Rank.FIVE, Suit.HEARTS),
            C(Rank.SIX, Suit.SPADES),
            C(Rank.SEVEN, Suit.HEARTS),
            C(Rank.EIGHT, Suit.SPADES),
            C(Rank.NINE, Suit.HEARTS),
        ]
        e = evaluate_hand(cards)
        assert e.hand_type == HandType.STRAIGHT

    def test_straight_low_wheel(self):
        cards = [
            C(Rank.ACE, Suit.HEARTS),
            C(Rank.TWO, Suit.SPADES),
            C(Rank.THREE, Suit.HEARTS),
            C(Rank.FOUR, Suit.SPADES),
            C(Rank.FIVE, Suit.HEARTS),
        ]
        e = evaluate_hand(cards)
        assert e.hand_type == HandType.STRAIGHT

    def test_straight_high(self):
        cards = [
            C(Rank.TEN, Suit.HEARTS),
            C(Rank.JACK, Suit.SPADES),
            C(Rank.QUEEN, Suit.HEARTS),
            C(Rank.KING, Suit.SPADES),
            C(Rank.ACE, Suit.HEARTS),
        ]
        e = evaluate_hand(cards)
        assert e.hand_type == HandType.STRAIGHT

    def test_flush(self):
        cards = [Card(r, Suit.SPADES) for r in
                 [Rank.TWO, Rank.FIVE, Rank.SEVEN, Rank.NINE, Rank.JACK]]
        e = evaluate_hand(cards)
        assert e.hand_type == HandType.FLUSH

    def test_full_house(self):
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS), C(Rank.SEVEN, Suit.DIAMONDS),
            C(Rank.KING, Suit.SPADES), C(Rank.KING, Suit.HEARTS),
        ]
        e = evaluate_hand(cards)
        assert e.hand_type == HandType.FULL_HOUSE
        assert len(e.scoring_cards) == 5

    def test_four_of_a_kind(self):
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS),
            C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.SEVEN, Suit.CLUBS),
            C(Rank.KING, Suit.SPADES),
        ]
        e = evaluate_hand(cards)
        assert e.hand_type == HandType.FOUR_OF_A_KIND
        assert len(e.scoring_cards) == 4

    def test_straight_flush(self):
        cards = [Card(r, Suit.SPADES) for r in
                 [Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE]]
        e = evaluate_hand(cards)
        assert e.hand_type == HandType.STRAIGHT_FLUSH

    def test_royal_flush_is_straight_flush(self):
        # Royal flush is not a separate scoring type in Balatro.
        cards = [Card(r, Suit.SPADES) for r in
                 [Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE]]
        e = evaluate_hand(cards)
        assert e.hand_type == HandType.STRAIGHT_FLUSH

    def test_five_of_a_kind(self):
        # Requires deck modification IRL; here we just feed five same-rank cards.
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS),
            C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.SEVEN, Suit.CLUBS),
            C(Rank.SEVEN, Suit.SPADES),
        ]
        e = evaluate_hand(cards)
        assert e.hand_type == HandType.FIVE_OF_A_KIND

    def test_flush_house(self):
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.SPADES),
            C(Rank.KING, Suit.SPADES), C(Rank.KING, Suit.SPADES),
        ]
        e = evaluate_hand(cards)
        assert e.hand_type == HandType.FLUSH_HOUSE

    def test_flush_five(self):
        cards = [C(Rank.SEVEN, Suit.SPADES) for _ in range(5)]
        e = evaluate_hand(cards)
        assert e.hand_type == HandType.FLUSH_FIVE

    def test_invalid_zero_cards(self):
        with pytest.raises(ValueError):
            evaluate_hand([])

    def test_invalid_six_cards(self):
        with pytest.raises(ValueError):
            evaluate_hand([C(Rank.TWO)] * 6)


class TestHandValues:
    def test_pair_level_1(self):
        assert hand_chips_mult(HandType.PAIR, 1) == (10, 2)

    def test_pair_level_2(self):
        assert hand_chips_mult(HandType.PAIR, 2) == (25, 3)

    def test_flush_level_3(self):
        # base (35, 4); +15c, +2m per level
        assert hand_chips_mult(HandType.FLUSH, 3) == (65, 8)

    def test_straight_flush_level_5(self):
        # base (100, 8); +40c, +4m per level -> level 5: 100+160, 8+16 -> (260, 24)
        assert hand_chips_mult(HandType.STRAIGHT_FLUSH, 5) == (260, 24)

    def test_invalid_level_zero(self):
        with pytest.raises(ValueError):
            hand_chips_mult(HandType.PAIR, 0)
