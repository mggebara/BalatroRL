"""Tests for the PPO masking and short-training smoke."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from training.ppo import (
    MaskedActorCritic,
    PPOConfig,
    _info_to_mask,
    train,
)


class TestMasking:
    def test_illegal_action_zero_probability(self):
        torch.manual_seed(0)
        agent = MaskedActorCritic(obs_dim=8, num_actions=5)
        obs = torch.zeros(1, 8)
        mask = torch.tensor([[True, False, True, False, False]])
        # Sample many times; we should never see an illegal action.
        actions = []
        for _ in range(200):
            a, _, _, _ = agent.get_action_and_value(obs, mask)
            actions.append(int(a.item()))
        legal = {0, 2}
        assert set(actions).issubset(legal)

    def test_log_prob_only_over_legal(self):
        torch.manual_seed(0)
        agent = MaskedActorCritic(obs_dim=8, num_actions=4)
        obs = torch.randn(1, 8)
        mask = torch.tensor([[True, True, False, False]])
        a = torch.tensor([0])
        _, logp, ent, _ = agent.get_action_and_value(obs, mask, action=a)
        # Entropy upper bound for 2-legal-action categorical is log(2).
        assert ent.item() <= float(np.log(2)) + 1e-5
        assert torch.isfinite(logp).all()

    def test_single_legal_action_entropy_zero(self):
        agent = MaskedActorCritic(obs_dim=8, num_actions=4)
        obs = torch.randn(1, 8)
        mask = torch.tensor([[False, True, False, False]])
        _, _, ent, _ = agent.get_action_and_value(obs, mask)
        assert ent.item() < 1e-4


class TestInfoToMask:
    def test_array_input(self):
        info = {"action_mask": np.array([[True, False], [False, True]])}
        m = _info_to_mask(info, num_envs=2, num_actions=2)
        assert m.shape == (2, 2)
        assert m.dtype == bool

    def test_missing_key_returns_all_legal(self):
        m = _info_to_mask({}, num_envs=3, num_actions=4)
        assert m.shape == (3, 4)
        assert m.all()


class TestTrainSmoke:
    @pytest.mark.slow
    def test_short_train_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PPOConfig(
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
            assert os.path.exists(os.path.join(tmp, "policy.pt"))
            assert os.path.exists(os.path.join(tmp, "config.json"))
