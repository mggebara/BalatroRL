"""Heuristic greedy baseline.

Strategy:
  - Brute-force the highest-scoring 1-5 card subset of the current hand
    (using the actual scoring pipeline, so jokers are accounted for).
  - If the best available hand is High Card or any Pair AND we have
    discards remaining AND more than one hand left, discard the 5
    lowest-rank cards instead of playing.
  - The agent emits a sequence of TOGGLE actions to build the selection,
    then a PLAY or DISCARD action to commit.

This agent peeks at the underlying engine state via `env._run`. That
is intentional for a baseline — RL agents see only the observation +
info dict.
"""
from __future__ import annotations

import itertools
from typing import Any

from balatro_core.cards import Card
from balatro_core.hands import HandType
from balatro_core.scoring import score_played_hand

from balatro_env.action_space import (
    DISCARD_ACTION,
    MAX_SELECTION,
    PLAY_ACTION,
)
from balatro_env.gym_env import BalatroEnv


class GreedyAgent:
    def __init__(self) -> None:
        self._plan: list[int] = []

    def reset(self) -> None:
        self._plan.clear()

    def act(self, env: BalatroEnv, info: dict[str, Any]) -> int:
        if self._plan:
            return self._plan.pop(0)

        run = env._run
        if run is None or run.current_round is None:
            return 0

        rd = run.current_round
        hand = rd.state.hand
        jokers = run.jokers
        discards_remaining = rd.state.discards_remaining

        best_idx, _, best_type = self._best_play(
            hand, jokers, discards_remaining
        )

        weak = best_type in (HandType.HIGH_CARD, HandType.PAIR)
        if weak and discards_remaining > 0 and rd.state.hands_remaining > 1:
            target_idx = self._worst_indices(hand, min(MAX_SELECTION, len(hand)))
            commit = DISCARD_ACTION
        else:
            target_idx = best_idx
            commit = PLAY_ACTION

        self._plan = [int(i) for i in target_idx] + [int(commit)]
        return self._plan.pop(0)

    @staticmethod
    def _best_play(
        hand: list[Card],
        jokers,
        discards_remaining: int,
    ) -> tuple[list[int], int, HandType]:
        best_score = -1
        best_idx: list[int] = []
        best_type = HandType.HIGH_CARD
        n = len(hand)
        for k in range(1, min(MAX_SELECTION, n) + 1):
            for combo in itertools.combinations(range(n), k):
                cards = [hand[i] for i in combo]
                held = [hand[i] for i in range(n) if i not in combo]
                r = score_played_hand(
                    cards,
                    jokers=jokers,
                    held_cards=held,
                    discards_remaining=discards_remaining,
                )
                if r.score > best_score:
                    best_score = r.score
                    best_idx = list(combo)
                    best_type = r.hand_type
        return best_idx, best_score, best_type

    @staticmethod
    def _worst_indices(hand: list[Card], k: int) -> list[int]:
        ranked = sorted(range(len(hand)), key=lambda i: (int(hand[i].rank), i))
        return ranked[:k]
