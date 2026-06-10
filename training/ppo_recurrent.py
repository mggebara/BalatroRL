"""Recurrent (GRU) PPO with action masking for the Balatro env.

Companion to `training/ppo.py`. Same training loop and CLI surface
except for the model + minibatch sampling.

Why GRU here:
  Phase 4b results show PPO converging to a deterministic policy
  that captures per-blind shaped reward but doesn't link multi-shop
  / multi-blind decisions to the terminal +1 win signal. A
  recurrent policy gives the agent an explicit memory across the
  ~50 env-steps of an Ante-3 episode, which the flat obs vector
  alone leaves implicit.

Implementation follows CleanRL's recurrent-PPO reference: hidden
state is gated by `done` at each replay step so episode boundaries
correctly zero memory both during rollout and during minibatch
updates. Minibatches are env-major (sample whole envs, preserve
time order within each env) so truncated BPTT across `num_steps`
chunks is well-defined.

Usage:
    python3 training/ppo_recurrent.py --total-timesteps 500000 \\
        --num-envs 8 --max-ante 1 --save-dir training/runs/rec_ante1 \\
        --starting-jokers joker lusty sly wrathful clever
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from balatro_env.gym_env import BalatroEnv  # noqa: E402
from training.ppo import (  # noqa: E402
    _build_starting_jokers,
    _info_to_mask,
    _layer_init,
    make_env,
)


@dataclass
class RecurrentPPOConfig:
    exp_name: str = "ppo_recurrent_balatro"
    seed: int = 0
    num_envs: int = 8
    num_steps: int = 128
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4   # env-major: must divide num_envs
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    encoder_hidden: int = 256
    gru_hidden: int = 128
    max_ante: int = 3
    blind_win_bonus: float = 0.0
    log_every: int = 1
    save_dir: str = "training/runs/recurrent_default"
    starting_jokers: tuple[str, ...] = field(default_factory=tuple)
    load_policy: str = ""


class MaskedRecurrentActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        encoder_hidden: int = 256,
        gru_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, encoder_hidden)), nn.Tanh(),
            _layer_init(nn.Linear(encoder_hidden, encoder_hidden)), nn.Tanh(),
        )
        self.gru = nn.GRU(encoder_hidden, gru_hidden, num_layers=1, batch_first=False)
        for name, param in self.gru.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.policy_head = _layer_init(nn.Linear(gru_hidden, num_actions), std=0.01)
        self.value_head = _layer_init(nn.Linear(gru_hidden, 1), std=1.0)
        self.gru_hidden = gru_hidden

    def initial_hidden(self, num_envs: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, num_envs, self.gru_hidden, device=device)

    def _gru_replay(
        self,
        feats: torch.Tensor,   # (T, B, encoder_hidden)
        prev_done: torch.Tensor,  # (T, B) — done arriving INTO step t
        h0: torch.Tensor,         # (1, B, gru_hidden)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequential replay with hidden-state gating at episode boundaries.

        prev_done[t] = 1 means step t starts a new episode (the previous
        step was terminal), so we zero the hidden state before running
        the GRU cell at t.
        """
        T, B, _ = feats.shape
        h = h0
        outs = []
        for t in range(T):
            gate = (1.0 - prev_done[t]).view(1, B, 1)
            h = h * gate
            out, h = self.gru(feats[t : t + 1], h)
            outs.append(out)
        out_seq = torch.cat(outs, dim=0)  # (T, B, gru_hidden)
        return out_seq, h

    def act(
        self,
        obs: torch.Tensor,     # (B, obs_dim)
        hidden: torch.Tensor,  # (1, B, gru_hidden)
        prev_done: torch.Tensor,  # (B,) — terminal at previous env step
        mask: torch.Tensor,    # (B, num_actions)
    ):
        gate = (1.0 - prev_done).view(1, -1, 1)
        h = hidden * gate
        x = self.encoder(obs).unsqueeze(0)  # (1, B, encoder_hidden)
        out, new_h = self.gru(x, h)
        feats = out.squeeze(0)              # (B, gru_hidden)
        logits = self.policy_head(feats)
        big_neg = torch.finfo(logits.dtype).min
        masked = torch.where(mask, logits, torch.full_like(logits, big_neg))
        dist = Categorical(logits=masked)
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.value_head(feats).squeeze(-1)
        return action, logprob, value, new_h

    def evaluate(
        self,
        obs_seq: torch.Tensor,    # (T, B, obs_dim)
        prev_done_seq: torch.Tensor,  # (T, B)
        h0: torch.Tensor,         # (1, B, gru_hidden)
        mask_seq: torch.Tensor,   # (T, B, num_actions)
        action_seq: torch.Tensor,  # (T, B)
    ):
        T, B, _ = obs_seq.shape
        feats = self.encoder(obs_seq.reshape(T * B, -1)).reshape(T, B, -1)
        out_seq, _ = self._gru_replay(feats, prev_done_seq, h0)
        logits = self.policy_head(out_seq.reshape(T * B, -1))
        big_neg = torch.finfo(logits.dtype).min
        flat_mask = mask_seq.reshape(T * B, -1)
        masked = torch.where(flat_mask, logits, torch.full_like(logits, big_neg))
        dist = Categorical(logits=masked)
        flat_action = action_seq.reshape(T * B)
        logprob = dist.log_prob(flat_action)
        entropy = dist.entropy()
        value = self.value_head(out_seq.reshape(T * B, -1)).squeeze(-1)
        return logprob.reshape(T, B), entropy.reshape(T, B), value.reshape(T, B)


def parse_args() -> RecurrentPPOConfig:
    d = RecurrentPPOConfig()
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
    p.add_argument("--encoder-hidden", type=int, default=d.encoder_hidden)
    p.add_argument("--gru-hidden", type=int, default=d.gru_hidden)
    p.add_argument("--max-ante", type=int, default=d.max_ante)
    p.add_argument("--blind-win-bonus", type=float, default=d.blind_win_bonus)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--save-dir", type=str, default=d.save_dir)
    p.add_argument("--starting-jokers", nargs="*", default=list(d.starting_jokers))
    p.add_argument("--load-policy", type=str, default=d.load_policy)
    args = vars(p.parse_args())
    args["starting_jokers"] = tuple(args["starting_jokers"])
    return RecurrentPPOConfig(**args)


def train(cfg: RecurrentPPOConfig) -> dict:
    if cfg.num_envs % cfg.num_minibatches != 0:
        raise ValueError(
            f"num_envs ({cfg.num_envs}) must be divisible by num_minibatches "
            f"({cfg.num_minibatches}) for env-major minibatches"
        )

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
        [
            make_env(cfg.seed, i, cfg.max_ante, cfg.starting_jokers, cfg.blind_win_bonus)
            for i in range(cfg.num_envs)
        ]
    )
    obs_dim = envs.single_observation_space.shape[0]
    num_actions = envs.single_action_space.n

    agent = MaskedRecurrentActorCritic(
        obs_dim, num_actions,
        encoder_hidden=cfg.encoder_hidden,
        gru_hidden=cfg.gru_hidden,
    ).to(device)
    if cfg.load_policy:
        state = torch.load(cfg.load_policy, map_location=device, weights_only=True)
        agent.load_state_dict(state)
        print(f"Loaded initial weights from {cfg.load_policy}")
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

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
    next_hidden = agent.initial_hidden(cfg.num_envs, device)

    num_updates = max(1, cfg.total_timesteps // (cfg.num_envs * cfg.num_steps))

    ep_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    recent_returns: list[float] = []
    recent_wins: list[int] = []

    global_step = 0
    start = time.time()
    history: list[dict] = []

    for update in range(1, num_updates + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1) / num_updates
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        # Snapshot hidden state at the start of this rollout — needed for replay.
        initial_hidden = next_hidden.detach().clone()

        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs_buf[step] = next_obs_t
            dones_buf[step] = next_done
            masks_buf[step] = next_mask_t

            with torch.no_grad():
                action, logprob, value, next_hidden = agent.act(
                    next_obs_t, next_hidden, next_done, next_mask_t
                )

            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value

            np_action = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = envs.step(np_action)
            done = np.logical_or(terminated, truncated)

            rewards_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            ep_returns += reward
            for env_idx in range(cfg.num_envs):
                if done[env_idx]:
                    recent_returns.append(float(ep_returns[env_idx]))
                    last_r = float(reward[env_idx])
                    recent_wins.append(1 if last_r >= 0.95 else 0)
                    ep_returns[env_idx] = 0.0
            if len(recent_returns) > 1000:
                recent_returns = recent_returns[-1000:]
                recent_wins = recent_wins[-1000:]

            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            next_mask_t = torch.as_tensor(
                _info_to_mask(info, cfg.num_envs, num_actions), dtype=torch.bool, device=device
            )
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

        # Bootstrap value at end of rollout (gated by `next_done`).
        with torch.no_grad():
            gate = (1.0 - next_done).view(1, -1, 1)
            x = agent.encoder(next_obs_t).unsqueeze(0)
            _, h_end = agent.gru(x, next_hidden * gate)
            next_value = agent.value_head(h_end.squeeze(0)).squeeze(-1)

            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = torch.zeros(cfg.num_envs, device=device)
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + cfg.gamma * nextvalues * nextnonterminal - values_buf[t]
                lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values_buf

        # Env-major minibatches: shuffle env indices, slice each into chunks.
        env_indices = np.arange(cfg.num_envs)
        envs_per_minibatch = cfg.num_envs // cfg.num_minibatches

        approx_kl_acc = 0.0
        pg_loss_acc = 0.0
        v_loss_acc = 0.0
        ent_acc = 0.0
        n_updates = 0

        for _ in range(cfg.update_epochs):
            np.random.shuffle(env_indices)
            for start_e in range(0, cfg.num_envs, envs_per_minibatch):
                end_e = start_e + envs_per_minibatch
                mb_envs = env_indices[start_e:end_e]
                mb_envs_t = torch.as_tensor(mb_envs, dtype=torch.long, device=device)

                mb_obs = obs_buf.index_select(1, mb_envs_t)
                mb_dones = dones_buf.index_select(1, mb_envs_t)
                mb_masks = masks_buf.index_select(1, mb_envs_t)
                mb_actions = actions_buf.index_select(1, mb_envs_t)
                mb_logprobs = logprobs_buf.index_select(1, mb_envs_t)
                mb_advantages = advantages.index_select(1, mb_envs_t)
                mb_returns = returns.index_select(1, mb_envs_t)
                mb_values = values_buf.index_select(1, mb_envs_t)
                mb_h0 = initial_hidden.index_select(1, mb_envs_t)

                newlogprob, entropy, newvalue = agent.evaluate(
                    mb_obs, mb_dones, mb_h0, mb_masks, mb_actions
                )
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()

                adv = mb_advantages
                if cfg.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                if cfg.clip_vloss:
                    v_unclip = (newvalue - mb_returns) ** 2
                    v_clip = mb_values + torch.clamp(
                        newvalue - mb_values, -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss = 0.5 * torch.max(v_unclip, (v_clip - mb_returns) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

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
    return {"history": history, "save_dir": str(save_dir)}


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
