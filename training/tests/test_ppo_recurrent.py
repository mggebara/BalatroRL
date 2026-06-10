"""Tests for the recurrent (GRU) PPO.

Verifies:
  - Masked logits still produce zero probability on illegal actions.
  - Hidden state resets correctly on episode boundaries (prev_done=1
    zeros the incoming state; prev_done=0 preserves it).
  - The training loop runs end-to-end without error on a tiny config.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch

from training.ppo_recurrent import (
    MaskedRecurrentActorCritic,
    RecurrentPPOConfig,
    train,
)


class TestMasking:
    def test_illegal_action_zero_probability(self):
        torch.manual_seed(0)
        agent = MaskedRecurrentActorCritic(obs_dim=8, num_actions=5, gru_hidden=16)
        obs = torch.zeros(1, 8)
        mask = torch.tensor([[True, False, True, False, False]])
        h = agent.initial_hidden(1, torch.device("cpu"))
        prev_done = torch.zeros(1)
        seen = set()
        for _ in range(200):
            a, _, _, _ = agent.act(obs, h, prev_done, mask)
            seen.add(int(a.item()))
        assert seen.issubset({0, 2})


class TestHiddenStateGating:
    def test_prev_done_resets_hidden(self):
        torch.manual_seed(0)
        agent = MaskedRecurrentActorCritic(obs_dim=4, num_actions=2, gru_hidden=8)
        obs = torch.randn(1, 4)
        mask = torch.tensor([[True, True]])

        # Step 1: build up some hidden state.
        h0 = agent.initial_hidden(1, torch.device("cpu"))
        _, _, _, h1 = agent.act(obs, h0, torch.zeros(1), mask)
        # h1 should be non-zero (encoder + GRU push it off the origin).
        assert h1.abs().sum().item() > 0

        # Step 2 with prev_done=1: incoming hidden should be zeroed
        # before the GRU cell runs. We can verify by running act() twice
        # — once with the real h1, once with zeros — and confirming both
        # produce the SAME new hidden (because the gate zeros h1 anyway).
        _, _, _, h_after_done = agent.act(obs, h1, torch.ones(1), mask)
        _, _, _, h_from_zero = agent.act(obs, h0, torch.zeros(1), mask)
        assert torch.allclose(h_after_done, h_from_zero)

    def test_prev_done_zero_preserves_hidden(self):
        torch.manual_seed(0)
        agent = MaskedRecurrentActorCritic(obs_dim=4, num_actions=2, gru_hidden=8)
        obs = torch.randn(1, 4)
        mask = torch.tensor([[True, True]])
        h0 = agent.initial_hidden(1, torch.device("cpu"))
        _, _, _, h1 = agent.act(obs, h0, torch.zeros(1), mask)

        # Step 2 with prev_done=0: h1 should be preserved across the gate.
        # Two runs with prev_done=0 must give different results from
        # prev_done=1 (otherwise the gate is broken).
        _, _, _, h_preserved = agent.act(obs, h1, torch.zeros(1), mask)
        _, _, _, h_reset = agent.act(obs, h1, torch.ones(1), mask)
        assert not torch.allclose(h_preserved, h_reset)


class TestEvaluateSequence:
    def test_evaluate_shapes(self):
        torch.manual_seed(0)
        agent = MaskedRecurrentActorCritic(obs_dim=4, num_actions=3, gru_hidden=8)
        T, B = 5, 2
        obs_seq = torch.randn(T, B, 4)
        prev_done = torch.zeros(T, B)
        prev_done[2, 0] = 1.0  # one episode boundary
        mask_seq = torch.ones(T, B, 3, dtype=torch.bool)
        actions = torch.zeros(T, B, dtype=torch.long)
        h0 = agent.initial_hidden(B, torch.device("cpu"))
        logp, ent, val = agent.evaluate(obs_seq, prev_done, h0, mask_seq, actions)
        assert logp.shape == (T, B)
        assert ent.shape == (T, B)
        assert val.shape == (T, B)


class TestTrainSmoke:
    @pytest.mark.slow
    def test_short_train_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = RecurrentPPOConfig(
                total_timesteps=512,
                num_envs=2,
                num_steps=64,
                num_minibatches=2,
                update_epochs=1,
                max_ante=1,
                save_dir=tmp,
                log_every=99,
            )
            result = train(cfg)
            assert "history" in result
