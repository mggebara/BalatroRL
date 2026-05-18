"""Uniform-random-over-legal-actions agent."""
from __future__ import annotations

import random
from typing import Any

import numpy as np


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> None:
        return

    def act(self, env: Any, info: dict[str, Any]) -> int:
        del env  # unused; mask is in info
        mask = np.asarray(info["action_mask"], dtype=bool)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return 0
        return int(self._rng.choice(legal.tolist()))
