"""Hand-crafted scoring scenarios (Phase 1 DoD).

Each test fixes a played hand and asserts the exact final score, matching
the in-game scoring formula:
    score = (base_chips + sum(chip_value of scoring cards)) * base_mult
at hand-type level 1 (unless overridden).
"""
from __future__ import annotations

from balatro_core.cards import Card, Rank, Suit
from balatro_core.hands import HandType
from balatro_core.scoring import score_played_hand


def C(rank: Rank, suit: Suit = Suit.SPADES) -> Card:
    return Card(rank=rank, suit=suit)


class TestScoringScenarios:
    def test_pair_of_sevens(self):
        # Pair lvl 1: 10c x 2m. Cards: 7+7 = 14. (10+14)*2 = 48.
        r = score_played_hand([C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS)])
        assert r.hand_type == HandType.PAIR
        assert r.chips == 24
        assert r.mult == 2
        assert r.score == 48

    def test_two_pair_kings_sevens_kicker(self):
        # Two Pair lvl 1: 20c x 2m. Scoring = 7+7+K+K = 7+7+10+10 = 34.
        # Kicker (2) does not score. (20+34)*2 = 108.
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS),
            C(Rank.KING, Suit.SPADES), C(Rank.KING, Suit.HEARTS),
            C(Rank.TWO, Suit.SPADES),
        ]
        r = score_played_hand(cards)
        assert r.hand_type == HandType.TWO_PAIR
        assert r.chips == 54
        assert r.mult == 2
        assert r.score == 108

    def test_three_aces(self):
        # Three of a Kind lvl 1: 30c x 3m. Scoring = 11+11+11 = 33.
        # Kickers do not score. (30+33)*3 = 189.
        cards = [
            C(Rank.ACE, Suit.SPADES), C(Rank.ACE, Suit.HEARTS), C(Rank.ACE, Suit.DIAMONDS),
            C(Rank.TWO, Suit.SPADES), C(Rank.THREE, Suit.SPADES),
        ]
        r = score_played_hand(cards)
        assert r.hand_type == HandType.THREE_OF_A_KIND
        assert r.chips == 63
        assert r.mult == 3
        assert r.score == 189

    def test_straight_six_to_ten(self):
        # Straight lvl 1: 30c x 4m. 6+7+8+9+10 = 40. (30+40)*4 = 280.
        cards = [
            C(Rank.SIX, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS),
            C(Rank.EIGHT, Suit.SPADES), C(Rank.NINE, Suit.HEARTS),
            C(Rank.TEN, Suit.SPADES),
        ]
        r = score_played_hand(cards)
        assert r.hand_type == HandType.STRAIGHT
        assert r.score == 280

    def test_flush_low_jack(self):
        # Flush lvl 1: 35c x 4m. 2+5+7+9+10(J) = 33. (35+33)*4 = 272.
        cards = [Card(r, Suit.SPADES) for r in
                 [Rank.TWO, Rank.FIVE, Rank.SEVEN, Rank.NINE, Rank.JACK]]
        r = score_played_hand(cards)
        assert r.hand_type == HandType.FLUSH
        assert r.score == 272

    def test_full_house_sevens_over_kings(self):
        # Full House lvl 1: 40c x 4m. 7+7+7+K+K = 21+20 = 41. (40+41)*4 = 324.
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS), C(Rank.SEVEN, Suit.DIAMONDS),
            C(Rank.KING, Suit.SPADES), C(Rank.KING, Suit.HEARTS),
        ]
        r = score_played_hand(cards)
        assert r.hand_type == HandType.FULL_HOUSE
        assert r.score == 324

    def test_four_sevens_with_kicker(self):
        # Four of a Kind lvl 1: 60c x 7m. 7+7+7+7 = 28 (kicker not scored).
        # (60+28)*7 = 616.
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS),
            C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.SEVEN, Suit.CLUBS),
            C(Rank.KING, Suit.SPADES),
        ]
        r = score_played_hand(cards)
        assert r.hand_type == HandType.FOUR_OF_A_KIND
        assert r.score == 616

    def test_straight_flush_five_to_nine(self):
        # Straight Flush lvl 1: 100c x 8m. 5+6+7+8+9 = 35. (100+35)*8 = 1080.
        cards = [Card(r, Suit.SPADES) for r in
                 [Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT, Rank.NINE]]
        r = score_played_hand(cards)
        assert r.hand_type == HandType.STRAIGHT_FLUSH
        assert r.score == 1080

    def test_high_card_king(self):
        # High Card lvl 1: 5c x 1m. K = 10. (5+10)*1 = 15.
        r = score_played_hand([C(Rank.KING)])
        assert r.hand_type == HandType.HIGH_CARD
        assert r.score == 15

    def test_pair_of_sevens_at_level_2(self):
        # Pair lvl 2: 25c x 3m. 7+7 = 14. (25+14)*3 = 117.
        r = score_played_hand(
            [C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS)],
            hand_levels={HandType.PAIR: 2},
        )
        assert r.chips == 39
        assert r.mult == 3
        assert r.score == 117

    def test_five_of_a_kind_sevens(self):
        # Five of a Kind lvl 1: 120c x 12m. 5x7 = 35. (120+35)*12 = 1860.
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.HEARTS),
            C(Rank.SEVEN, Suit.DIAMONDS), C(Rank.SEVEN, Suit.CLUBS),
            C(Rank.SEVEN, Suit.SPADES),
        ]
        r = score_played_hand(cards)
        assert r.hand_type == HandType.FIVE_OF_A_KIND
        assert r.score == 1860

    def test_flush_house_sevens_over_kings(self):
        # Flush House lvl 1: 140c x 14m. 3x7 + 2xK = 21 + 20 = 41.
        # (140+41)*14 = 2534.
        cards = [
            C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.SPADES), C(Rank.SEVEN, Suit.SPADES),
            C(Rank.KING, Suit.SPADES), C(Rank.KING, Suit.SPADES),
        ]
        r = score_played_hand(cards)
        assert r.hand_type == HandType.FLUSH_HOUSE
        assert r.score == 2534

    def test_flush_five_sevens(self):
        # Flush Five lvl 1: 160c x 16m. 5x7 = 35. (160+35)*16 = 3120.
        cards = [C(Rank.SEVEN, Suit.SPADES) for _ in range(5)]
        r = score_played_hand(cards)
        assert r.hand_type == HandType.FLUSH_FIVE
        assert r.score == 3120
