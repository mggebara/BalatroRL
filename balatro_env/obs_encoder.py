"""Flat observation encoder for the Balatro env.

Phase 3 takes the deliberately simple route from PLAN.md: pad-to-max
fixed-size tensors with histogram aggregates for variable-length
structure (deck composition). A Transformer encoder over jokers/deck
arrives in Phase 6.

Layout (lengths, sum = OBS_DIM):
  8 * 18  hand: per-slot (rank one-hot[13] + suit one-hot[4] + present[1])
  8        selection mask (0/1 per slot)
  2        (hands_remaining, discards_remaining) / max
  3        (total_score / chip_target, blind_progress_log, chip_target / 50000)
  8        ante one-hot (1..8)
  3        blind one-hot (small/big/boss)
  5 * 21   joker slots: per-slot (id one-hot[20] + present[1])
  12       hand-type level (current level / 10, clipped)
  13       deck rank histogram (count / 52)
  4        deck suit histogram (count / 52)

Total: 144 + 8 + 2 + 3 + 8 + 3 + 105 + 12 + 13 + 4 = 302
"""
from __future__ import annotations

import math

import numpy as np

from balatro_core.cards import Card, Deck, Rank, Suit
from balatro_core.engine import ANTE_BASE_CHIPS, BlindKind, RoundState, Run
from balatro_core.hands import HandType
from balatro_core.jokers import JOKER_CLASSES, Joker

from balatro_env.action_space import MAX_HAND_SIZE

NUM_RANKS = 13
NUM_SUITS = 4
NUM_JOKER_SLOTS = 5
NUM_JOKER_IDS = len(JOKER_CLASSES)  # 20
NUM_HAND_TYPES = len(HandType)      # 12
NUM_BLINDS = len(BlindKind)         # 3
MAX_ANTE = 8

HAND_CARD_DIM = NUM_RANKS + NUM_SUITS + 1                  # 18
HAND_SECTION = MAX_HAND_SIZE * HAND_CARD_DIM               # 144
SELECTION_SECTION = MAX_HAND_SIZE                          # 8
COUNTS_SECTION = 2                                         # hands/discards remaining
SCORE_SECTION = 3                                          # progress, log score, target
ANTE_SECTION = MAX_ANTE                                    # 8
BLIND_SECTION = NUM_BLINDS                                 # 3
JOKER_CARD_DIM = NUM_JOKER_IDS + 1                         # 21
JOKER_SECTION = NUM_JOKER_SLOTS * JOKER_CARD_DIM           # 105
HAND_LEVEL_SECTION = NUM_HAND_TYPES                        # 12
DECK_RANK_HIST = NUM_RANKS                                 # 13
DECK_SUIT_HIST = NUM_SUITS                                 # 4

OBS_DIM = (
    HAND_SECTION + SELECTION_SECTION + COUNTS_SECTION + SCORE_SECTION
    + ANTE_SECTION + BLIND_SECTION + JOKER_SECTION
    + HAND_LEVEL_SECTION + DECK_RANK_HIST + DECK_SUIT_HIST
)

# Stable mapping from joker class -> id index.
_JOKER_ID: dict[type[Joker], int] = {cls: i for i, cls in enumerate(JOKER_CLASSES)}

_RANK_TO_IDX: dict[Rank, int] = {r: i for i, r in enumerate(Rank)}
_SUIT_TO_IDX: dict[Suit, int] = {s: i for i, s in enumerate(Suit)}
_BLIND_TO_IDX: dict[BlindKind, int] = {b: i for i, b in enumerate(BlindKind)}


def _encode_card(card: Card | None) -> np.ndarray:
    v = np.zeros(HAND_CARD_DIM, dtype=np.float32)
    if card is None:
        return v
    v[_RANK_TO_IDX[card.rank]] = 1.0
    v[NUM_RANKS + _SUIT_TO_IDX[card.suit]] = 1.0
    v[NUM_RANKS + NUM_SUITS] = 1.0  # present
    return v


def _encode_joker(joker: Joker | None) -> np.ndarray:
    v = np.zeros(JOKER_CARD_DIM, dtype=np.float32)
    if joker is None:
        return v
    idx = _JOKER_ID.get(type(joker))
    if idx is not None:
        v[idx] = 1.0
    v[NUM_JOKER_IDS] = 1.0  # present
    return v


def encode_observation(
    run: Run,
    round_state: RoundState | None,
    selection: list[bool],
) -> np.ndarray:
    out = np.zeros(OBS_DIM, dtype=np.float32)
    cursor = 0

    # Hand.
    hand: list[Card] = round_state.hand if round_state is not None else []
    for slot in range(MAX_HAND_SIZE):
        card = hand[slot] if slot < len(hand) else None
        out[cursor:cursor + HAND_CARD_DIM] = _encode_card(card)
        cursor += HAND_CARD_DIM

    # Selection mask.
    for slot in range(MAX_HAND_SIZE):
        out[cursor + slot] = 1.0 if selection[slot] else 0.0
    cursor += SELECTION_SECTION

    # Counts.
    if round_state is not None:
        out[cursor] = round_state.hands_remaining / max(run.hands_per_round, 1)
        out[cursor + 1] = round_state.discards_remaining / max(run.discards_per_round, 1)
    cursor += COUNTS_SECTION

    # Score progress.
    if round_state is not None:
        target = max(round_state.chip_target, 1)
        out[cursor] = min(round_state.total_score / target, 4.0)
        out[cursor + 1] = math.log10(max(round_state.total_score, 1)) / 6.0
        out[cursor + 2] = min(round_state.chip_target / 50000.0, 1.0)
    cursor += SCORE_SECTION

    # Ante one-hot.
    ante_idx = max(0, min(run.ante - 1, MAX_ANTE - 1))
    out[cursor + ante_idx] = 1.0
    cursor += ANTE_SECTION

    # Blind one-hot. Use current_blind if round not yet active; otherwise round_state.
    blind = round_state.blind if round_state is not None else run.current_blind
    out[cursor + _BLIND_TO_IDX[blind]] = 1.0
    cursor += BLIND_SECTION

    # Joker slots (pad with empties).
    for slot in range(NUM_JOKER_SLOTS):
        joker = run.jokers[slot] if slot < len(run.jokers) else None
        out[cursor:cursor + JOKER_CARD_DIM] = _encode_joker(joker)
        cursor += JOKER_CARD_DIM

    # Hand-type levels (level/10 clipped).
    for ht in HandType:
        level = run.hand_levels.get(ht, 1)
        out[cursor + int(ht)] = min(level / 10.0, 1.0)
    cursor += HAND_LEVEL_SECTION

    # Deck histograms over the full configured deck (not the depleted draw pile,
    # because the player can see "what's in my deck" but the order is hidden).
    deck_cards: list[Card] = run.deck.cards
    n = max(len(deck_cards), 1)
    for c in deck_cards:
        out[cursor + _RANK_TO_IDX[c.rank]] += 1.0
    for i in range(NUM_RANKS):
        out[cursor + i] /= n
    cursor += DECK_RANK_HIST

    for c in deck_cards:
        out[cursor + _SUIT_TO_IDX[c.suit]] += 1.0
    for i in range(NUM_SUITS):
        out[cursor + i] /= n
    cursor += DECK_SUIT_HIST

    assert cursor == OBS_DIM, f"obs cursor mismatch: {cursor} != {OBS_DIM}"
    return out
