"""Action layout and mask computation for the Balatro env.

Action space (Discrete(14)):
    0..7   TOGGLE_CARD_i  — flip selection for hand slot i (in-round only).
    8      PLAY           — play the currently selected cards (in-round only).
    9      DISCARD        — discard the currently selected cards (in-round only).
    10     LEAVE_SHOP     — close the shop, advance to the next blind.
    11     REROLL_SHOP    — pay to reroll unsold shop slots.
    12..13 BUY_SHOP_i     — buy shop slot i (Phase 4 has 2 slots).

Masking is stage-dependent:
  - In-round (stage == IN_ROUND): only 0..9 may be legal.
  - In-shop (stage == IN_SHOP): only 10..13 may be legal.
"""
from __future__ import annotations

from enum import IntEnum

MAX_HAND_SIZE = 8
MAX_SELECTION = 5
MAX_SHOP_SLOTS = 2

PLAY_ACTION = MAX_HAND_SIZE          # 8
DISCARD_ACTION = MAX_HAND_SIZE + 1   # 9
LEAVE_SHOP_ACTION = MAX_HAND_SIZE + 2  # 10
REROLL_SHOP_ACTION = MAX_HAND_SIZE + 3  # 11
BUY_SHOP_ACTION_BASE = MAX_HAND_SIZE + 4  # 12

NUM_ACTIONS = BUY_SHOP_ACTION_BASE + MAX_SHOP_SLOTS  # 14


class ActionKind(IntEnum):
    TOGGLE = 0
    PLAY = 1
    DISCARD = 2
    LEAVE_SHOP = 3
    REROLL_SHOP = 4
    BUY_SHOP = 5


def decode_action(action: int) -> tuple[ActionKind, int]:
    """Return (kind, payload).

    payload meaning by kind:
      TOGGLE          -> hand index
      BUY_SHOP        -> shop slot index
      PLAY / DISCARD / LEAVE_SHOP / REROLL_SHOP -> -1
    """
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError(f"action {action} out of range [0,{NUM_ACTIONS})")
    if action < MAX_HAND_SIZE:
        return ActionKind.TOGGLE, int(action)
    if action == PLAY_ACTION:
        return ActionKind.PLAY, -1
    if action == DISCARD_ACTION:
        return ActionKind.DISCARD, -1
    if action == LEAVE_SHOP_ACTION:
        return ActionKind.LEAVE_SHOP, -1
    if action == REROLL_SHOP_ACTION:
        return ActionKind.REROLL_SHOP, -1
    return ActionKind.BUY_SHOP, int(action - BUY_SHOP_ACTION_BASE)


def compute_round_mask(
    hand_size: int,
    selection: list[bool],
    selection_count: int,
    hands_remaining: int,
    discards_remaining: int,
) -> list[bool]:
    """In-round mask. Shop actions are uniformly disabled here."""
    mask = [False] * NUM_ACTIONS
    for i in range(MAX_HAND_SIZE):
        if i >= hand_size:
            mask[i] = False
            continue
        if selection[i]:
            mask[i] = True  # can always deselect
        else:
            mask[i] = selection_count < MAX_SELECTION
    has_selection = 1 <= selection_count <= MAX_SELECTION
    mask[PLAY_ACTION] = hands_remaining > 0 and has_selection
    mask[DISCARD_ACTION] = discards_remaining > 0 and has_selection
    return mask


def compute_shop_mask(
    money: int,
    joker_slots_free: int,
    shop_slot_filled: list[bool],
    shop_slot_prices: list[int],
    reroll_cost: int,
) -> list[bool]:
    """In-shop mask. Round actions are uniformly disabled here."""
    mask = [False] * NUM_ACTIONS
    mask[LEAVE_SHOP_ACTION] = True
    mask[REROLL_SHOP_ACTION] = money >= reroll_cost
    for i in range(MAX_SHOP_SLOTS):
        slot_action = BUY_SHOP_ACTION_BASE + i
        if i >= len(shop_slot_filled):
            mask[slot_action] = False
            continue
        if not shop_slot_filled[i]:
            mask[slot_action] = False
            continue
        if joker_slots_free <= 0:
            mask[slot_action] = False
            continue
        mask[slot_action] = money >= shop_slot_prices[i]
    return mask
