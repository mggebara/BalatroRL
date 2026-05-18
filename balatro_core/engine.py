"""Run / Ante / Blind / Round state machine.

Phase 1 scope: no shop, no jokers, no consumables. Just play through
blinds with a fixed deck.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from balatro_core.cards import Card, Deck, standard_deck
from balatro_core.hands import HandType
from balatro_core.jokers import Joker
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


class GameStage(Enum):
    """Top-level run state, driven by the Run state machine."""
    PRE_BLIND = "pre_blind"     # ready to start the current blind
    IN_ROUND = "in_round"       # round in progress
    ROUND_WON = "round_won"     # round just won, ready to advance (no shop yet)
    GAME_WON = "game_won"       # boss of max_ante beaten
    GAME_OVER = "game_over"     # run lost


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
        jokers: list[Joker] | None = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.deck = deck
        self.hand_levels = hand_levels or {}
        self.jokers: list[Joker] = list(jokers) if jokers else []
        self._rng = rng
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
        result = score_played_hand(
            played,
            hand_levels=self.hand_levels,
            held_cards=list(self.state.hand),
            jokers=self.jokers,
            discards_remaining=self.state.discards_remaining,
            rng=self._rng,
        )
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
    """Top-level run with a multi-blind state machine.

    Phase 1: no shop, no money rewards between blinds — just play
    through Small -> Big -> Boss -> next ante -> ... until either
    the boss of `max_ante` is beaten (GAME_WON) or a round is lost
    (GAME_OVER).
    """

    def __init__(
        self,
        deck_cards: list[Card] | None = None,
        starting_money: int = 4,
        hand_size: int = 8,
        hands_per_round: int = 4,
        discards_per_round: int = 4,
        starting_ante: int = 1,
        max_ante: int = 8,
        rng_seed: int | None = None,
        jokers: list[Joker] | None = None,
    ) -> None:
        self._rng = random.Random(rng_seed)
        cards = deck_cards if deck_cards is not None else standard_deck()
        self.deck = Deck(cards, rng=self._rng)
        self.money = starting_money
        self.hand_size = hand_size
        self.hands_per_round = hands_per_round
        self.discards_per_round = discards_per_round
        self.ante = starting_ante
        self.max_ante = max_ante
        self.hand_levels: dict[HandType, int] = {}
        self.jokers: list[Joker] = list(jokers) if jokers else []
        self.current_blind: BlindKind = BlindKind.SMALL
        self.current_round: Round | None = None
        self.stage: GameStage = GameStage.PRE_BLIND

    def start_blind(self, blind: BlindKind) -> Round:
        """Construct (but do not register) a Round for the given blind.

        Used by the Phase 1 smoke test and by `start_current_blind` below.
        """
        return Round(
            deck=self.deck,
            ante=self.ante,
            blind=blind,
            hand_size=self.hand_size,
            hands=self.hands_per_round,
            discards=self.discards_per_round,
            hand_levels=self.hand_levels,
            jokers=self.jokers,
            rng=self._rng,
        )

    # ---- Run state machine ----

    def start_current_blind(self) -> Round:
        """Begin the round at (ante, current_blind). Transitions to IN_ROUND."""
        if self.stage not in (GameStage.PRE_BLIND, GameStage.ROUND_WON):
            raise RuntimeError(f"cannot start blind from stage {self.stage}")
        self.current_round = self.start_blind(self.current_blind)
        self.stage = GameStage.IN_ROUND
        return self.current_round

    def advance_after_round_win(self) -> None:
        """Move to next blind / next ante / GAME_WON.

        No shop, no money rewards (Phase 3 stub). Callers responsible
        for verifying the current round was actually won.
        """
        if self.current_blind == BlindKind.SMALL:
            self.current_blind = BlindKind.BIG
            self.stage = GameStage.PRE_BLIND
        elif self.current_blind == BlindKind.BIG:
            self.current_blind = BlindKind.BOSS
            self.stage = GameStage.PRE_BLIND
        else:  # BOSS won
            if self.ante >= self.max_ante:
                self.stage = GameStage.GAME_WON
                self.current_round = None
                return
            self.ante += 1
            self.current_blind = BlindKind.SMALL
            self.stage = GameStage.PRE_BLIND

    def handle_round_loss(self) -> None:
        self.stage = GameStage.GAME_OVER
        self.current_round = None

    @property
    def is_terminal(self) -> bool:
        return self.stage in (GameStage.GAME_WON, GameStage.GAME_OVER)
