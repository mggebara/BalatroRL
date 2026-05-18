"""Gymnasium env tests: API conformance, masking, episode termination."""
from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from balatro_core.engine import GameStage
from balatro_env.action_space import (
    BUY_SHOP_ACTION_BASE,
    DISCARD_ACTION,
    LEAVE_SHOP_ACTION,
    MAX_HAND_SIZE,
    MAX_SHOP_SLOTS,
    NUM_ACTIONS,
    PLAY_ACTION,
    REROLL_SHOP_ACTION,
    compute_round_mask,
    compute_shop_mask,
    decode_action,
)
from balatro_env.gym_env import BalatroEnv
from balatro_env.obs_encoder import OBS_DIM


class TestEnvAPI:
    def test_check_env_passes(self):
        env = BalatroEnv(max_ante=1)
        check_env(env, skip_render_check=True)

    def test_observation_shape(self):
        env = BalatroEnv()
        obs, _ = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_action_space_size(self):
        env = BalatroEnv()
        assert env.action_space.n == NUM_ACTIONS == 14

    def test_reset_returns_action_mask(self):
        env = BalatroEnv()
        _, info = env.reset(seed=0)
        mask = info["action_mask"]
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == bool

    def test_mask_at_reset_is_all_toggles_legal(self):
        # Fresh round: 8 cards in hand, no selection, full hands/discards.
        # Every toggle slot must be legal; PLAY/DISCARD illegal (empty selection).
        env = BalatroEnv()
        _, info = env.reset(seed=0)
        mask = info["action_mask"]
        assert mask[:MAX_HAND_SIZE].all()
        assert not mask[PLAY_ACTION]
        assert not mask[DISCARD_ACTION]

    def test_toggle_then_play(self):
        env = BalatroEnv()
        env.reset(seed=0)
        # Toggle card 0 then play.
        _, _, _, _, info = env.step(0)
        assert info["action_mask"][PLAY_ACTION]
        assert info["action_mask"][DISCARD_ACTION]
        obs, reward, terminated, truncated, info = env.step(PLAY_ACTION)
        assert "hand_type" in info
        assert reward >= 0.0

    def test_terminates_on_loss_or_win(self):
        # Play a forced losing strategy: always play one card. With max_ante=1
        # this should usually lose Boss (target 600); single high card scores
        # ~15 chips per play.
        env = BalatroEnv(max_ante=1)
        env.reset(seed=1)
        terminated = False
        steps = 0
        while not terminated and steps < 1000:
            # Toggle card 0 (or deselect if selected) and try to play one card.
            _, _, terminated, truncated, info = env.step(0)
            steps += 1
            if terminated:
                break
            if info["action_mask"][PLAY_ACTION]:
                _, _, terminated, truncated, info = env.step(PLAY_ACTION)
                steps += 1
        assert terminated, "env did not terminate within 1000 steps"

    def test_illegal_action_terminates(self):
        env = BalatroEnv()
        env.reset(seed=0)
        # PLAY with no selection is illegal -> terminate.
        _, _, terminated, _, info = env.step(PLAY_ACTION)
        assert terminated
        assert info.get("reason") == "illegal_action"


class TestActionDecoding:
    def test_decode_toggle(self):
        for i in range(MAX_HAND_SIZE):
            kind, idx = decode_action(i)
            assert kind.name == "TOGGLE"
            assert idx == i

    def test_decode_play_discard(self):
        kind, _ = decode_action(PLAY_ACTION)
        assert kind.name == "PLAY"
        kind, _ = decode_action(DISCARD_ACTION)
        assert kind.name == "DISCARD"

    def test_decode_out_of_range(self):
        with pytest.raises(ValueError):
            decode_action(NUM_ACTIONS)


class TestMaskComputation:
    def test_round_mask_disables_shop_actions(self):
        sel = [True, False] + [False] * 6
        m = compute_round_mask(
            hand_size=8, selection=sel, selection_count=1,
            hands_remaining=4, discards_remaining=4,
        )
        assert not m[LEAVE_SHOP_ACTION]
        assert not m[REROLL_SHOP_ACTION]
        for i in range(MAX_SHOP_SLOTS):
            assert not m[BUY_SHOP_ACTION_BASE + i]

    def test_shop_mask_disables_round_actions(self):
        m = compute_shop_mask(
            money=20, joker_slots_free=3,
            shop_slot_filled=[True, True],
            shop_slot_prices=[4, 4],
            reroll_cost=5,
        )
        for i in range(MAX_HAND_SIZE):
            assert not m[i]
        assert not m[PLAY_ACTION]
        assert not m[DISCARD_ACTION]
        assert m[LEAVE_SHOP_ACTION]
        assert m[REROLL_SHOP_ACTION]
        assert m[BUY_SHOP_ACTION_BASE]
        assert m[BUY_SHOP_ACTION_BASE + 1]

    def test_shop_mask_too_poor_to_buy(self):
        m = compute_shop_mask(
            money=3, joker_slots_free=3,
            shop_slot_filled=[True, True],
            shop_slot_prices=[4, 4],
            reroll_cost=5,
        )
        assert not m[REROLL_SHOP_ACTION]
        assert not m[BUY_SHOP_ACTION_BASE]
        assert not m[BUY_SHOP_ACTION_BASE + 1]
        assert m[LEAVE_SHOP_ACTION]  # leave always legal

    def test_shop_mask_no_joker_slots(self):
        m = compute_shop_mask(
            money=20, joker_slots_free=0,
            shop_slot_filled=[True, True],
            shop_slot_prices=[4, 4],
            reroll_cost=5,
        )
        assert not m[BUY_SHOP_ACTION_BASE]
        assert not m[BUY_SHOP_ACTION_BASE + 1]
        assert m[LEAVE_SHOP_ACTION]
        assert m[REROLL_SHOP_ACTION]  # still legal even if pointless

    def test_empty_selection_blocks_commit(self):
        sel = [False] * MAX_HAND_SIZE
        m = compute_round_mask(
            hand_size=8, selection=sel, selection_count=0,
            hands_remaining=4, discards_remaining=4,
        )
        assert not m[PLAY_ACTION]
        assert not m[DISCARD_ACTION]

    def test_full_selection_blocks_extra_toggles(self):
        sel = [True] * 5 + [False] * 3
        m = compute_round_mask(
            hand_size=8, selection=sel, selection_count=5,
            hands_remaining=4, discards_remaining=4,
        )
        # Five-selected: can deselect (toggle 0..4) but not select more.
        for i in range(5):
            assert m[i]
        for i in range(5, MAX_HAND_SIZE):
            assert not m[i]
        assert m[PLAY_ACTION]
        assert m[DISCARD_ACTION]

    def test_no_hands_blocks_play(self):
        sel = [True, False] + [False] * 6
        m = compute_round_mask(
            hand_size=8, selection=sel, selection_count=1,
            hands_remaining=0, discards_remaining=4,
        )
        assert not m[PLAY_ACTION]
        assert m[DISCARD_ACTION]

    def test_no_discards_blocks_discard(self):
        sel = [True, False] + [False] * 6
        m = compute_round_mask(
            hand_size=8, selection=sel, selection_count=1,
            hands_remaining=4, discards_remaining=0,
        )
        assert m[PLAY_ACTION]
        assert not m[DISCARD_ACTION]

    def test_shorter_hand_disables_unused_toggles(self):
        sel = [False] * MAX_HAND_SIZE
        m = compute_round_mask(
            hand_size=5, selection=sel, selection_count=0,
            hands_remaining=4, discards_remaining=4,
        )
        for i in range(5):
            assert m[i]
        for i in range(5, MAX_HAND_SIZE):
            assert not m[i]


class TestBaselines:
    """Smoke checks: baselines run end-to-end and complete episodes."""

    def test_random_agent_completes_episode(self):
        from agents.random_agent import RandomAgent
        env = BalatroEnv(max_ante=1)
        obs, info = env.reset(seed=0)
        agent = RandomAgent(seed=0)
        terminated = False
        steps = 0
        while not terminated and steps < 5000:
            action = agent.act(env, info)
            obs, _, terminated, _, info = env.step(action)
            steps += 1
        assert terminated

    def test_greedy_agent_completes_episode(self):
        from agents.greedy_agent import GreedyAgent
        env = BalatroEnv(max_ante=1)
        obs, info = env.reset(seed=42)
        agent = GreedyAgent()
        terminated = False
        steps = 0
        while not terminated and steps < 5000:
            action = agent.act(env, info)
            obs, _, terminated, _, info = env.step(action)
            steps += 1
        assert terminated
