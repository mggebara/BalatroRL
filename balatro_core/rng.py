"""Random number generation for Balatro.

The real game uses LÖVE2D's `love.math.random` seeded by a hash of
(run_seed, context_string). Each random event in Balatro queries a
specific context (e.g. "Joker1", "Tarot2", "shop_pack_ante_1") so that
re-runs of the same seed reproduce the same shop contents, joker
rolls, etc.

**Status: NOT bit-parity verified.** Bit-exact parity with the real game
requires capturing golden traces from Balatrobot (the Lua mod exposing
the game's PRNG state) and reproducing the exact hash + step function.
That cannot happen in this environment — flagged in PLAN.md.

This module provides a structurally-correct stand-in: deterministic
given (run_seed, context), produces uniform random numbers, suitable
for testing joker effects and the agent pipeline. Outputs WILL NOT
match the real game's stream.

When parity work begins, swap `BalatroRNG` internals while keeping the
public API identical so callers don't change.
"""
from __future__ import annotations

import hashlib
import random
from typing import Sequence, TypeVar

T = TypeVar("T")


class BalatroRNG:
    """Deterministic PRNG keyed by (run_seed, context).

    Public API is intended to match Balatro's needs once parity lands.
    """

    def __init__(self, run_seed: str | int, context: str) -> None:
        self.run_seed = str(run_seed)
        self.context = context
        digest = hashlib.sha256(f"{self.run_seed}|{context}".encode()).digest()
        int_seed = int.from_bytes(digest[:8], "big")
        self._rng = random.Random(int_seed)

    def randint(self, a: int, b: int) -> int:
        """Uniform integer in [a, b]."""
        return self._rng.randint(a, b)

    def random(self) -> float:
        """Uniform float in [0.0, 1.0)."""
        return self._rng.random()

    def choice(self, seq: Sequence[T]) -> T:
        return self._rng.choice(seq)

    def shuffle(self, seq: list) -> None:
        self._rng.shuffle(seq)
