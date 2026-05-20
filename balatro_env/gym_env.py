"""Gymnasium adapter for Balatro.

Phase 3 scope (no shop, no money rewards): the env steps the agent
through Small -> Big -> Boss for each ante until either GAME_WON
(max_ante boss beaten) or GAME_OVER (any round lost).

Reward shape:
  - Terminal: +1.0 on GAME_WON, 0.0 on GAME_OVER.
  - Per-play shaping (small, decayed later): a clipped log of
    score_delta / chip_target.

Action masking: callers should consult `info["action_mask"]` after
reset/step, or the `action_masks()` method (compatible with
sb3-contrib MaskablePPO).
"""
from __future__ import annotations

import math
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from balatro_core.engine import BlindKind, GameStage, Run
from balatro_core.jokers import Joker

from balatro_env.action_space import (
    BUY_SHOP_ACTION_BASE,
    DISCARD_ACTION,
    LEAVE_SHOP_ACTION,
    MAX_HAND_SIZE,
    MAX_SHOP_SLOTS,
    NUM_ACTIONS,
    PLAY_ACTION,
    REROLL_SHOP_ACTION,
    ActionKind,
    compute_round_mask,
    compute_shop_mask,
    decode_action,
)
from balatro_env.obs_encoder import OBS_DIM, encode_observation


class BalatroEnv(gym.Env):
    metadata = {"render_modes": ["human"], "name": "Balatro-v0"}

    observation_space = spaces.Box(
        low=0.0, high=4.0, shape=(OBS_DIM,), dtype=np.float32,
    )
    action_space = spaces.Discrete(NUM_ACTIONS)

    def __init__(
        self,
        max_ante: int = 8,
        starting_jokers: Optional[list[Joker]] = None,
        shaping_coef: float = 0.05,
        blind_win_bonus: float = 0.0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_ante = max_ante
        self._starting_jokers = starting_jokers or []
        self.shaping_coef = shaping_coef
        self.blind_win_bonus = blind_win_bonus
        self.render_mode = render_mode

        self._run: Optional[Run] = None
        self._selection: list[bool] = [False] * MAX_HAND_SIZE
        self._terminated = False
        self._steps = 0

    # ---- gym API ----

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        rng_seed = seed
        if options and "rng_seed" in options:
            rng_seed = options["rng_seed"]
        self._run = Run(
            rng_seed=rng_seed,
            jokers=[type(j)() for j in self._starting_jokers],  # fresh instances
            max_ante=self.max_ante,
        )
        self._run.start_current_blind()
        self._selection = [False] * MAX_HAND_SIZE
        self._terminated = False
        self._steps = 0
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("step() called on a terminated env; call reset() first.")
        self._steps += 1

        kind, idx = decode_action(int(action))
        mask = self._current_mask()
        if not mask[int(action)]:
            # Illegal action — terminate with 0 reward to surface the bug.
            self._terminated = True
            info = self._info(reason="illegal_action")
            return self._obs(), 0.0, True, False, info

        reward = 0.0
        info: dict[str, Any] = {}

        if kind == ActionKind.TOGGLE:
            self._selection[idx] = not self._selection[idx]
        elif kind == ActionKind.PLAY:
            reward, info_play = self._do_play()
            info.update(info_play)
        elif kind == ActionKind.DISCARD:
            self._do_discard()
        elif kind == ActionKind.LEAVE_SHOP:
            self._do_leave_shop()
        elif kind == ActionKind.REROLL_SHOP:
            self._do_reroll_shop()
        elif kind == ActionKind.BUY_SHOP:
            self._do_buy_shop(idx)

        if self._run is not None and self._run.is_terminal:
            self._terminated = True
            if self._run.stage == GameStage.GAME_WON:
                reward += 1.0
                info["outcome"] = "game_won"
            else:
                info["outcome"] = "game_over"

        full_info = self._info()
        full_info.update(info)
        return self._obs(), float(reward), self._terminated, False, full_info

    def render(self) -> None:
        if self.render_mode != "human" or self._run is None:
            return
        rd = self._run.current_round
        if rd is None:
            print(f"[Run stage={self._run.stage.value} ante={self._run.ante}]")
            return
        sel = [i for i, s in enumerate(self._selection) if s]
        print(
            f"Ante {self._run.ante} {rd.state.blind.value}: "
            f"{rd.state.total_score}/{rd.state.chip_target} "
            f"hands={rd.state.hands_remaining} discards={rd.state.discards_remaining} "
            f"hand={[str(c) for c in rd.state.hand]} selected={sel}"
        )

    # ---- masking ----

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO calls this to retrieve the legal-action mask."""
        return np.array(self._current_mask(), dtype=bool)

    def _current_mask(self) -> list[bool]:
        if self._run is None or self._terminated:
            return [False] * NUM_ACTIONS
        if self._run.stage == GameStage.IN_ROUND and self._run.current_round is not None:
            rd = self._run.current_round
            return compute_round_mask(
                hand_size=len(rd.state.hand),
                selection=list(self._selection),
                selection_count=sum(self._selection[: len(rd.state.hand)]),
                hands_remaining=rd.state.hands_remaining,
                discards_remaining=rd.state.discards_remaining,
            )
        if self._run.stage == GameStage.IN_SHOP and self._run.current_shop is not None:
            shop = self._run.current_shop
            filled = [
                slot.joker is not None
                for slot in shop.slots[:MAX_SHOP_SLOTS]
            ]
            # Pad to MAX_SHOP_SLOTS if shop is smaller.
            filled = filled + [False] * (MAX_SHOP_SLOTS - len(filled))
            prices = [
                slot.price for slot in shop.slots[:MAX_SHOP_SLOTS]
            ]
            prices = prices + [0] * (MAX_SHOP_SLOTS - len(prices))
            return compute_shop_mask(
                money=self._run.money,
                joker_slots_free=self._run.max_joker_slots - len(self._run.jokers),
                shop_slot_filled=filled,
                shop_slot_prices=prices,
                reroll_cost=shop.reroll_cost,
            )
        return [False] * NUM_ACTIONS

    # ---- internals ----

    def _selected_indices(self) -> list[int]:
        if self._run is None or self._run.current_round is None:
            return []
        hand_size = len(self._run.current_round.state.hand)
        return [i for i in range(hand_size) if self._selection[i]]

    def _do_play(self) -> tuple[float, dict[str, Any]]:
        assert self._run is not None and self._run.current_round is not None
        idx = self._selected_indices()
        rd = self._run.current_round
        target = max(rd.state.chip_target, 1)
        result = rd.play(idx)
        self._selection = [False] * MAX_HAND_SIZE
        shaped = self.shaping_coef * math.log1p(result.score) / math.log1p(target)
        reward = float(min(shaped, 1.0))
        info = {
            "hand_type": result.hand_type.name,
            "play_score": result.score,
            "play_chips": result.chips,
            "play_mult": result.mult,
        }
        # Round transition: won (-> shop), lost (-> game over), or continue.
        if rd.state.is_won:
            reward += self.blind_win_bonus
            info["blind_won"] = rd.state.blind.value
            self._run.advance_after_round_win()
        elif rd.state.is_lost:
            self._run.handle_round_loss()
        return reward, info

    def _do_discard(self) -> None:
        assert self._run is not None and self._run.current_round is not None
        idx = self._selected_indices()
        self._run.current_round.discard(idx)
        self._selection = [False] * MAX_HAND_SIZE

    def _do_leave_shop(self) -> None:
        assert self._run is not None
        self._run.leave_shop()
        if self._run.stage == GameStage.PRE_BLIND:
            self._run.start_current_blind()

    def _do_reroll_shop(self) -> None:
        assert self._run is not None
        self._run.reroll_shop()

    def _do_buy_shop(self, idx: int) -> None:
        assert self._run is not None
        self._run.buy_shop_slot(idx)

    def _obs(self) -> np.ndarray:
        assert self._run is not None
        rd = self._run.current_round
        rs = rd.state if rd is not None else None
        return encode_observation(self._run, rs, list(self._selection))

    def _info(self, **extra: Any) -> dict[str, Any]:
        if self._run is None:
            return {"action_mask": np.zeros(NUM_ACTIONS, dtype=bool), **extra}
        rd = self._run.current_round
        info: dict[str, Any] = {
            "action_mask": self.action_masks(),
            "ante": self._run.ante,
            "stage": self._run.stage.value,
        }
        if rd is not None:
            info["blind"] = rd.state.blind.value
            info["chip_target"] = rd.state.chip_target
            info["total_score"] = rd.state.total_score
            info["hands_remaining"] = rd.state.hands_remaining
            info["discards_remaining"] = rd.state.discards_remaining
        info.update(extra)
        return info
