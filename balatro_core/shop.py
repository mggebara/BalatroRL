"""Minimal shop layer for Phase 4.

Scope: joker-only shop, 2 slots, fixed price per joker. Booster packs,
tarot/planet/spectral cards, vouchers, and playing cards are deferred
to a later phase. Joker price is flat $4 — the real game varies prices
by rarity ($4 common / $6 uncommon / $8 rare), TODO to model when
joker rarity ships.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from balatro_core.jokers import JOKER_CLASSES, Joker

DEFAULT_NUM_SLOTS = 2
DEFAULT_JOKER_PRICE = 4
DEFAULT_REROLL_BASE_COST = 5


@dataclass
class ShopSlot:
    """One purchasable item. None joker means slot already bought."""
    joker: Optional[Joker]
    price: int


class Shop:
    """A single shop visit. Reroll cost grows by $1 per reroll."""

    def __init__(
        self,
        slots: list[ShopSlot],
        reroll_cost: int = DEFAULT_REROLL_BASE_COST,
    ) -> None:
        self.slots: list[ShopSlot] = slots
        self.reroll_cost: int = reroll_cost
        self.rerolls_used: int = 0

    @property
    def slot_count(self) -> int:
        return len(self.slots)

    def buy(self, idx: int, money: int) -> tuple[Joker, int]:
        """Purchase slot idx. Returns (joker, money_after_purchase).

        Raises if the slot is empty or money is insufficient.
        """
        if not 0 <= idx < len(self.slots):
            raise IndexError(f"shop slot {idx} out of range")
        slot = self.slots[idx]
        if slot.joker is None:
            raise RuntimeError(f"slot {idx} already sold")
        if money < slot.price:
            raise RuntimeError(
                f"cannot afford slot {idx}: ${money} < ${slot.price}"
            )
        joker = slot.joker
        slot.joker = None
        return joker, money - slot.price

    def reroll(self, new_jokers: list[Joker], money: int) -> int:
        """Replace UNSOLD slot contents with `new_jokers`. Returns new money.

        Sold slots remain empty after reroll (matches real game behavior).
        """
        if money < self.reroll_cost:
            raise RuntimeError(
                f"cannot afford reroll: ${money} < ${self.reroll_cost}"
            )
        if len(new_jokers) < len(self.slots):
            raise ValueError("not enough new jokers for reroll")
        for i, slot in enumerate(self.slots):
            if slot.joker is not None:
                slot.joker = new_jokers[i]
        money -= self.reroll_cost
        self.reroll_cost += 1
        self.rerolls_used += 1
        return money


def _pick_jokers(
    rng: random.Random,
    n: int,
    pool: list[type[Joker]] | None = None,
) -> list[Joker]:
    pool = pool if pool is not None else list(JOKER_CLASSES)
    return [rng.choice(pool)() for _ in range(n)]


def generate_shop(
    rng: random.Random,
    num_slots: int = DEFAULT_NUM_SLOTS,
    price: int = DEFAULT_JOKER_PRICE,
    pool: list[type[Joker]] | None = None,
) -> Shop:
    jokers = _pick_jokers(rng, num_slots, pool)
    slots = [ShopSlot(joker=j, price=price) for j in jokers]
    return Shop(slots=slots)


def reroll_jokers(
    rng: random.Random,
    n: int,
    pool: list[type[Joker]] | None = None,
) -> list[Joker]:
    return _pick_jokers(rng, n, pool)
