"""CleanRL-style PPO with action masking for the Balatro env.

Single-file PPO closely following CleanRL's `ppo.py` reference, adapted
for:
  - Gymnasium 1.x vector envs.
  - Invalid-action masking via -inf logits on the Categorical
    distribution. Both action sampling AND log-prob / entropy
    computation operate on the masked logits, which is critical for
    correctness (otherwise PPO ratios drift on illegal actions).
  - `info["action_mask"]` is read from vector-env step/reset and stored
    in the rollout buffer alongside observations.

Phase 4b scope: MLP policy, no recurrence, no LR scheduler beyond
linear anneal, no observation normalization. Recurrent ablation and
obs-norm are listed in PLAN.md as next-iteration items.

Usage:
    python3 training/ppo.py --total-timesteps 200000 --num-envs 8 \\
        --max-ante 3 --save-dir training/runs/sanity
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure repo root is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from balatro_env.gym_env import BalatroEnv  # noqa: E402


@dataclass
class PPOConfig:
    exp_name: str = "ppo_balatro"
    seed: int = 0
    num_envs: int = 8
    num_steps: int = 128
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 256
    max_ante: int = 3
    log_every: int = 1
    save_dir: str = "training/runs/default"
    starting_jokers: tuple[str, ...] = field(default_factory=tuple)


def _build_starting_jokers(names: tuple[str, ...]):
    if not names:
        return []
    from balatro_core.jokers import JOKER_CLASSES
    by_name = {cls().name.lower(): cls for cls in JOKER_CLASSES}
    out = []
    for n in names:
        cls = by_name.get(n.lower())
        if cls is None:
            matches = [c for k, c in by_name.items() if n.lower() in k]
            if len(matches) != 1:
                raise SystemExit(f"unknown / ambiguous joker name: {n!r}")
            cls = matches[0]
        out.append(cls())
    return out


def make_env(seed: int, idx: int, max_ante: int, starting_jokers_names: tuple[str, ...]):
    def thunk():
        env = BalatroEnv(
            max_ante=max_ante,
            starting_jokers=_build_starting_jokers(starting_jokers_names),
        )
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MaskedActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        self.policy_head = _layer_init(nn.Linear(hidden, num_actions), std=0.01)
        self.value_head = _layer_init(nn.Linear(hidden, 1), std=1.0)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.trunk(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.features(obs))

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        h = self.features(obs)
        logits = self.policy_head(h)
        big_neg = torch.finfo(logits.dtype).min
        masked_logits = torch.where(mask, logits, torch.full_like(logits, big_neg))
        dist = Categorical(logits=masked_logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_head(h)
        return action, logprob, entropy, value


def _info_to_mask(info: dict, num_envs: int, num_actions: int) -> np.ndarray:
    """Robustly extract per-env action_mask from a vector env info dict."""
    if "action_mask" not in info:
        # SyncVectorEnv may drop the key if a sub-env returned non-stackable.
        return np.ones((num_envs, num_actions), dtype=bool)
    am = info["action_mask"]
    if isinstance(am, np.ndarray) and am.shape == (num_envs, num_actions):
        return am.astype(bool, copy=False)
    # Fallback: list-of-arrays.
    return np.stack([np.asarray(m, dtype=bool) for m in am])


def parse_args() -> PPOConfig:
    d = PPOConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", type=str, default=d.exp_name)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--num-envs", type=int, default=d.num_envs)
    p.add_argument("--num-steps", type=int, default=d.num_steps)
    p.add_argument("--total-timesteps", type=int, default=d.total_timesteps)
    p.add_argument("--learning-rate", type=float, default=d.learning_rate)
    p.add_argument("--anneal-lr", type=lambda x: x.lower() == "true", default=d.anneal_lr)
    p.add_argument("--gamma", type=float, default=d.gamma)
    p.add_argument("--gae-lambda", type=float, default=d.gae_lambda)
    p.add_argument("--num-minibatches", type=int, default=d.num_minibatches)
    p.add_argument("--update-epochs", type=int, default=d.update_epochs)
    p.add_argument("--norm-adv", type=lambda x: x.lower() == "true", default=d.norm_adv)
    p.add_argument("--clip-coef", type=float, default=d.clip_coef)
    p.add_argument("--clip-vloss", type=lambda x: x.lower() == "true", default=d.clip_vloss)
    p.add_argument("--ent-coef", type=float, default=d.ent_coef)
    p.add_argument("--vf-coef", type=float, default=d.vf_coef)
    p.add_argument("--max-grad-norm", type=float, default=d.max_grad_norm)
    p.add_argument("--hidden-size", type=int, default=d.hidden_size)
    p.add_argument("--max-ante", type=int, default=d.max_ante)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--save-dir", type=str, default=d.save_dir)
    p.add_argument("--starting-jokers", nargs="*", default=list(d.starting_jokers))
    args = vars(p.parse_args())
    args["starting_jokers"] = tuple(args["starting_jokers"])
    return PPOConfig(**args)


def train(cfg: PPOConfig) -> dict:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    envs = gym.vector.SyncVectorEnv(
        [make_env(cfg.seed, i, cfg.max_ante, cfg.starting_jokers) for i in range(cfg.num_envs)]
    )
    obs_dim = envs.single_observation_space.shape[0]
    num_actions = envs.single_action_space.n

    agent = MaskedActorCritic(obs_dim, num_actions, hidden=cfg.hidden_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # Storage
    obs_buf = torch.zeros((cfg.num_steps, cfg.num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    rewards_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    dones_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    values_buf = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    masks_buf = torch.zeros((cfg.num_steps, cfg.num_envs, num_actions), dtype=torch.bool, device=device)

    next_obs, next_info = envs.reset(seed=cfg.seed)
    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_mask_t = torch.as_tensor(
        _info_to_mask(next_info, cfg.num_envs, num_actions), dtype=torch.bool, device=device
    )
    next_done = torch.zeros(cfg.num_envs, device=device)

    num_updates = max(1, cfg.total_timesteps // (cfg.num_envs * cfg.num_steps))

    # Episode tracking
    ep_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(cfg.num_envs, dtype=np.int32)
    recent_returns: list[float] = []
    recent_wins: list[int] = []  # 1 if terminal reward >= 1 (game won)
    recent_max_ante: list[int] = []

    global_step = 0
    start = time.time()
    history: list[dict] = []

    for update in range(1, num_updates + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs_buf[step] = next_obs_t
            dones_buf[step] = next_done
            masks_buf[step] = next_mask_t

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs_t, next_mask_t)

            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value.flatten()

            np_action = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = envs.step(np_action)
            done = np.logical_or(terminated, truncated)

            rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)

            ep_returns += reward
            ep_lengths += 1
            for env_idx in range(cfg.num_envs):
                if done[env_idx]:
                    recent_returns.append(float(ep_returns[env_idx]))
                    # Win = terminal reward shows the +1 win bonus; we recover
                    # it by checking the final reward + the env's outcome info.
                    last_r = float(reward[env_idx])
                    recent_wins.append(1 if last_r >= 0.95 else 0)
                    # Best-effort ante reach: env auto-resets; we don't have
                    # final_info reliably across versions, so log return only.
                    ep_returns[env_idx] = 0.0
                    ep_lengths[env_idx] = 0
            # Keep buffer bounded
            if len(recent_returns) > 1000:
                recent_returns = recent_returns[-1000:]
                recent_wins = recent_wins[-1000:]

            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            next_mask_t = torch.as_tensor(
                _info_to_mask(info, cfg.num_envs, num_actions), dtype=torch.bool, device=device
            )
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs_t).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = torch.zeros(cfg.num_envs, device=device)
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value.squeeze(0)
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + cfg.gamma * nextvalues * nextnonterminal - values_buf[t]
                lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values_buf

        # Flatten batch
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_masks = masks_buf.reshape(-1, num_actions)

        batch_size = cfg.num_envs * cfg.num_steps
        minibatch_size = batch_size // cfg.num_minibatches
        b_inds = np.arange(batch_size)

        approx_kl_acc = 0.0
        pg_loss_acc = 0.0
        v_loss_acc = 0.0
        ent_acc = 0.0
        n_updates = 0

        for _ in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start_idx in range(0, batch_size, minibatch_size):
                end_idx = start_idx + minibatch_size
                mb = b_inds[start_idx:end_idx]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], b_masks[mb], b_actions[mb]
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()

                mb_advantages = b_advantages[mb]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg1 = -mb_advantages * ratio
                pg2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_unclip = (newvalue - b_returns[mb]) ** 2
                    v_clip = b_values[mb] + torch.clamp(
                        newvalue - b_values[mb], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss = 0.5 * torch.max(v_unclip, (v_clip - b_returns[mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

                approx_kl_acc += approx_kl
                pg_loss_acc += pg_loss.item()
                v_loss_acc += v_loss.item()
                ent_acc += entropy_loss.item()
                n_updates += 1

        if update % cfg.log_every == 0 or update == num_updates:
            elapsed = time.time() - start
            sps = global_step / max(elapsed, 1e-6)
            mean_ret = float(np.mean(recent_returns[-100:])) if recent_returns else 0.0
            win_rate = float(np.mean(recent_wins[-100:])) if recent_wins else 0.0
            print(
                f"upd {update:>4}/{num_updates} step={global_step:>8d} "
                f"sps={sps:>6.0f} "
                f"ret(100)={mean_ret:+.3f} win(100)={win_rate:.1%} "
                f"pg={pg_loss_acc/n_updates:+.3f} v={v_loss_acc/n_updates:.3f} "
                f"ent={ent_acc/n_updates:.3f} kl={approx_kl_acc/n_updates:.4f}",
                flush=True,
            )
            history.append({
                "update": update, "step": global_step, "sps": sps,
                "return_mean100": mean_ret, "win_rate100": win_rate,
                "pg_loss": pg_loss_acc / n_updates,
                "v_loss": v_loss_acc / n_updates,
                "entropy": ent_acc / n_updates,
                "approx_kl": approx_kl_acc / n_updates,
            })

    envs.close()

    torch.save(agent.state_dict(), save_dir / "policy.pt")
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining done. Saved policy + history to {save_dir}/")
    return {
        "history": history,
        "save_dir": str(save_dir),
        "final_win_rate": history[-1]["win_rate100"] if history else 0.0,
    }


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
