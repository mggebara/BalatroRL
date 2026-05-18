"""Action layout and mask computation for the Balatro env.

Phase 3 action space (Discrete(10)):
    0..7  TOGGLE_CARD_i  — flip selection for card index i in current hand.
    8     PLAY           — play the currently selected cards.
    9     DISCARD        — discard the currently selected cards.

Selection state lives inside the env. PLAY/DISCARD commit it. Both
require 1-5 selected cards (Balatro hand-size rule). The toggle on a
card not currently selected is illegal if selection is already at 5.
"""
from __future__ import annotations

from enum import IntEnum

MAX_HAND_SIZE = 8
MAX_SELECTION = 5

NUM_ACTIONS = MAX_HAND_SIZE + 2  # 8 toggles + play + discard
PLAY_ACTION = MAX_HAND_SIZE
DISCARD_ACTION = MAX_HAND_SIZE + 1


class ActionKind(IntEnum):
    TOGGLE = 0
    PLAY = 1
    DISCARD = 2


def decode_action(action: int) -> tuple[ActionKind, int]:
    """Return (kind, hand_index). For PLAY/DISCARD the index is -1."""
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError(f"action {action} out of range [0,{NUM_ACTIONS})")
    if action == PLAY_ACTION:
        return ActionKind.PLAY, -1
    if action == DISCARD_ACTION:
        return ActionKind.DISCARD, -1
    return ActionKind.TOGGLE, int(action)


def compute_mask(
    hand_size: int,
    selection: list[bool],
    selection_count: int,
    hands_remaining: int,
    discards_remaining: int,
) -> list[bool]:
    """Return a length-NUM_ACTIONS list of legality flags.

    Rules:
      - TOGGLE_i legal if i < hand_size AND (selection[i] OR selection_count < MAX_SELECTION).
        I.e. you can always deselect; you can only select if room remains.
      - PLAY legal iff hands_remaining > 0 AND 1 <= selection_count <= MAX_SELECTION.
      - DISCARD legal iff discards_remaining > 0 AND 1 <= selection_count <= MAX_SELECTION.
    """
    mask = [False] * NUM_ACTIONS
    for i in range(MAX_HAND_SIZE):
        if i >= hand_size:
            mask[i] = False
            continue
        if selection[i]:
            mask[i] = True  # always allowed to deselect
        else:
            mask[i] = selection_count < MAX_SELECTION
    has_selection = 1 <= selection_count <= MAX_SELECTION
    mask[PLAY_ACTION] = hands_remaining > 0 and has_selection
    mask[DISCARD_ACTION] = discards_remaining > 0 and has_selection
    return mask
