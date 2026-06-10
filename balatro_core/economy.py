"""Money / economy rules.

Phase 4 scope: end-of-round payout (base + hand bonus + interest).
Skip rewards and voucher upgrades come later.

Real game numbers used:
  - Base reward per blind: $3 / $4 / $5 (Small / Big / Boss).
  - Hand bonus: $1 per unused hand at the moment the blind is beaten.
  - Interest: $1 per $5 owned at end of round, cap $5
    (vanilla pre-voucher cap; Money Tree etc. raise it later).
  - Credit Card joker lets money go to -$20 (otherwise floor is $0).
"""
from __future__ import annotations

from balatro_core.engine import BlindKind


BLIND_REWARDS: dict[BlindKind, int] = {
    BlindKind.SMALL: 3,
    BlindKind.BIG: 4,
    BlindKind.BOSS: 5,
}

DEFAULT_INTEREST_RATE = 5
DEFAULT_INTEREST_CAP = 5
DEFAULT_HAND_BONUS = 1


def interest_payout(
    money: int,
    rate: int = DEFAULT_INTEREST_RATE,
    cap: int = DEFAULT_INTEREST_CAP,
) -> int:
    if money <= 0:
        return 0
    return min(cap, money // rate)


def end_of_round_payout(
    money_at_end: int,
    blind: BlindKind,
    hands_remaining: int,
    hand_bonus: int = DEFAULT_HAND_BONUS,
) -> int:
    """Total dollars earned when a blind is beaten."""
    return (
        BLIND_REWARDS[blind]
        + hands_remaining * hand_bonus
        + interest_payout(money_at_end)
    )
