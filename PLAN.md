# BalatroRL — Master Plan

> **North star:** train an RL agent that consistently beats Boss Ante 8 of Balatro.
> "Consistently" = ≥70% win rate on Red Deck / White Stake across unseen seeds.

This document is the source of truth for the project. It is meant to be read at
the start of every working session so we can pick up where we left off.
Update the **Status** section as phases complete.

---

## Status

- **Current phase:** Phase 4b — Recurrent PPO infrastructure
  shipped. Ante 3 win-rate DoD still not met; per-architecture
  characterization continues.
- **Last updated:** 2026-05-19 (recurrent added later same day)
- **New in this session (recurrent thread):**
  - `training/ppo_recurrent.py`: `MaskedRecurrentActorCritic` with a
    GRU layer on top of the MLP trunk. Hidden state is gated by
    `done` at each step both during rollout (in `act`) and during
    minibatch replay (in `evaluate`), so episode boundaries
    correctly zero memory in both passes. Minibatches are env-major
    (whole envs preserved as time sequences) so truncated BPTT
    across `num_steps` chunks is well-defined.
  - 4 new unit tests in `training/tests/test_ppo_recurrent.py`:
    illegal-action masking holds with GRU output, `prev_done=1`
    zeros incoming hidden state (verified by equivalence to a
    zero-initialized run), `prev_done=0` preserves it (verified by
    inequivalence), and `evaluate` returns the expected shapes.
  - `training/eval_policy.py` auto-detects MLP vs recurrent from
    the state-dict keys, so the same CLI runs both architectures.
  - Trained one 500k-step recurrent run on the same config used for
    the MLP Stage 1 (max_ante=1, 5 starter jokers, bonus=0.1).
    **Result:** 76% Ante 1. The MLP under the same config and
    step count hit 85%.
- **Recurrent vs MLP scorecard (100 seeds, max_ante=1, 5 jokers):**
  | Model | Steps | Wall-clock | Ante 1 |
  |---|---|---|---|
  | MLP | 500k | ~3 min | **85%** |
  | Recurrent (GRU 128) | 500k | ~18 min | 76% |
- **Honest read of the recurrent result:**
  - Recurrent did NOT beat MLP on Ante 1 at matched step count.
    Plausible because the Ante-1 horizon (≤ ~25 env steps,
    Markov-enough obs) doesn't reward memory.
  - Throughput dropped 6.5× — sequential GRU replay is the cost.
    Per wall-clock minute, MLP gets ~7× more training steps.
  - The thesis "recurrent helps Ante 3 long-horizon credit
    assignment" remains untested. A fair recurrent-vs-MLP comparison
    on Ante 3 would need either (a) 3M+ recurrent steps (~2 hours)
    or (b) a head-to-head wall-clock-matched experiment showing
    recurrent reaches higher win rate per minute.
- **Phase 4 DoD scorecard (unchanged):**
  - Training reproducible from a single config + seed: **green.**
  - PPO beats greedy baseline by ≥20pp on Ante 3 win rate: **NOT MET.**
  - >90% Ante 3 win rate with the 20-joker pool: **NOT MET.**
- **Updated priority order for next session:**
  1. **Greedy-imitation bootstrap** (now top, leapfrogs recurrent).
     Collect ~10k greedy trajectories, pretrain the MLP policy via
     behavioral cloning (cross-entropy on greedy's masked actions),
     then PPO fine-tune. Bypasses the cold-start exploration cliff
     and gives the value head a sensible starting point.
  2. **Recurrent on Ante 3 with 3M+ steps** — only worth doing
     after the MLP imitation-bootstrap path establishes a working
     baseline at Ante 3.
  3. **Entropy floor + KL-target** to prevent the Stage 3
     determinism collapse.
  4. **Reward shaping ablation** — does `blind_win_bonus` actually
     improve policy or just inflate returns?
- **Honest reminder (unchanged):** all numbers come from our Python
  simulator. RNG bit-parity gap (Phase 2) is still open.

When resuming, read this section first, then the **Phase Plan** section
to find the active phase.
- **New in this session:**
  - `BalatroEnv` gained `blind_win_bonus` (default 0.0) — per-blind
    intermediate reward to densify the signal across the ~50-step
    Ante-3 horizon.
  - `training/ppo.py` gained `--load-policy` for weight transfer
    between training stages.
  - Ran a 3-stage curriculum (Ante 1 → 2 → 3, weights transferred
    forward, bonus=0.1, 5 starter jokers). 100-seed eval after each
    stage:
    - **Stage 1** (max_ante=1, 500k steps): 85% Ante 1.
    - **Stage 2** (max_ante=2, +1M steps, transfer): 90 / 37% (Ante 1/2).
    - **Stage 3** (max_ante=3, +1.5M steps, transfer): 83 / 23 / 0%.
  - **Outcome:** staging clearly helped Ante 2 (0% → 37%) and
    Ante 1 over baseline. Ante 3 still at 0%. Stage 3 entropy
    collapsed to 0.41 and value loss to 0.003 — a deterministic
    local-optimum policy that isn't the winning one, and that
    catastrophically forgot a few percentage points on Ante 1/2
    while reaching for Ante 3.

- **Comparison snapshot (max_ante=3 + 5 starter jokers, 100 seeds):**
  | Agent | Ante 1 | Ante 2 | Ante 3 |
  |---|---|---|---|
  | Random | ~2% | ~0% | 0% |
  | PPO direct (1M) | 67% | 10% | 0% |
  | PPO staged (3M total) | 83% | 23% | 0% |
  | Greedy | 100% | 100% | 82% |

- **Phase 4 DoD scorecard (unchanged):**
  - Training reproducible from a single config + seed: **green.**
  - PPO beats greedy baseline by ≥20pp on Ante 3 win rate:
    **NOT MET** — gap is in the wrong direction (~80pp behind).
  - >90% Ante 3 win rate with the 20-joker pool: **NOT MET.**

- **What this session ruled out:** "just train longer / add a
  per-blind bonus / stage the curriculum" is necessary but not
  sufficient. The agent finds a deterministic policy that captures
  most per-blind shaped reward but fails to invest in the
  long-horizon decisions (shop purchases, discard timing across
  multiple blinds) that make Ante 3 winnable.

- **What's next (in priority order):**
  1. **Recurrent policy** (GRU on the trunk + truncated BPTT across
     `num_steps` chunks). Highest expected impact: the env genuinely
     needs memory of past plays / past shops / accumulated jokers
     beyond what one obs vector encodes.
  2. **Entropy floor / KL-target restart** to prevent the entropy
     collapse seen at Stage 3.
  3. **Imitation bootstrap from greedy demos** — collect ~10k
     greedy trajectories, pre-train the policy with behavioral
     cloning, then fine-tune with PPO. Bypasses the cold-start
     exploration cliff at Ante 3.
  4. **Reward shaping ablation** — measure if `blind_win_bonus`
     is actually helping or just inflating returns without driving
     policy improvement. Earlier 80% Ante-1 result used bonus=0,
     this session's 85% used bonus=0.1 — small delta.
  5. Possibly move to vector envs at 32+ instances and 10M steps;
     CPU sim runs at ~3k sps so a 10M run is ~1 hour.

- **Honest reminder:** all numbers come from our Python simulator,
  not the real Balatro game. RNG bit-parity gap (Phase 2) is still
  open.

When resuming, read this section first, then the **Phase Plan** section
to find the active phase.
- **Completed in Phase 4b:**
  - `training/ppo.py`: single-file CleanRL-style PPO. MLP actor-critic
    (2 hidden tanh layers, default 256 units), invalid-action masking
    via −∞ logits applied uniformly to sampling, log-prob, and
    entropy. SyncVectorEnv with N parallel envs. Linear LR anneal,
    GAE, clipped value loss, gradient clipping. Full config via CLI
    args. Saves `policy.pt`, `config.json`, `history.json` per run.
  - `training/eval_policy.py`: loads a saved policy, runs N seeds with
    the same per-ante reporting as the baseline eval script — direct
    PPO-vs-baseline comparison.
  - 5 PPO unit tests (`training/tests/test_ppo.py`): illegal actions
    have zero sampled probability over 200 samples; masked log-probs
    finite and entropy ≤ log(legal); single-legal-action entropy ~0;
    `_info_to_mask` handles array and missing-key inputs.
  - **Training pipeline verified at ~3k steps/sec on CPU.** Two
    training runs executed:
    - max_ante=1 + 5 starter jokers, 500k steps → 80% Ante-1 win rate.
    - max_ante=3 + 5 starter jokers, 1M steps → 67 / 10 / 0%
      (Ante 1 / 2 / 3).
  - **Reproducibility:** all hparams in `config.json`; seeded RNG
    everywhere; same command + seed reproduces the run.
- **Phase 4 DoD scorecard:**
  - Training reproducible from a single config + seed: **green.**
  - PPO beats greedy baseline by ≥20pp on Ante 3 win rate: **NOT
    MET.** At 1M steps PPO hits 0% Ante 3 vs greedy's 82% (with same
    3 starter jokers; greedy is brute-force optimal per turn).
  - >90% Ante 3 win rate with the 20-joker pool: **NOT MET.**
- **Comparison on max_ante=1 + 5 starter jokers, 100 seeds:**
  - Random: **2%**
  - PPO (500k steps): **80%**
  - Greedy: **100%**
  - PPO beats random by +78pp; trails greedy by 20pp.
- **Why PPO trails greedy:** the greedy peeks at engine state and
  brute-forces the per-turn optimal play, while PPO sees only the obs
  vector and must learn card→hand value through gradient updates. The
  gap should close with (a) ~10× more training steps, (b) recurrent
  policy for long-horizon credit assignment across 9 blinds and 3
  shops per Ante-3 run, (c) sharper reward shaping that rewards
  joker accumulation explicitly, (d) curriculum (start Ante 1, then
  unlock 2, then 3).
- **Honest reminder about all numbers in this doc:** they come from
  our Python simulator (`balatro_core/` + `balatro_env/`), not the
  real game. RNG bit-parity with Balatro is still pending Balatrobot
  access (Phase 2 gap).
- **Next concrete task:** Phase 4b convergence work. Three threads
  in parallel: (1) staged curriculum (max_ante=1 → 2 → 3, transfer
  weights), (2) recurrent policy (GRU on the trunk), (3) reward
  shaping ablation — log-score relative to a moving baseline rather
  than raw chip-target. Target a single 10M-step run that beats
  greedy on Ante 3 by ≥20pp. Once green, advance to Phase 5
  (expand to 60 jokers + full Ante 1-8).

When resuming, read this section first, then the **Phase Plan** section
to find the active phase.
- **Completed in Phase 4a:**
  - `balatro_core/economy.py`: end-of-round payout (blind reward + $1
    per unused hand + interest at $1 per $5, cap $5).
  - `balatro_core/shop.py`: `Shop` with N slots (default 2), flat $4
    joker price, reroll cost starting at $5 with +$1 per reroll.
  - `balatro_core/engine.py`: `GameStage.IN_SHOP`, `Run.current_shop`,
    `Run.advance_after_round_win` now generates a shop instead of
    advancing directly. New methods `buy_shop_slot`, `reroll_shop`,
    `leave_shop`. Game flow is now PRE_BLIND -> IN_ROUND -> IN_SHOP
    -> PRE_BLIND ... until GAME_WON / GAME_OVER.
  - `balatro_env/action_space.py`: action space grows from
    Discrete(10) to Discrete(14). New actions: LEAVE_SHOP, REROLL,
    BUY_SLOT_0, BUY_SLOT_1. Mask is stage-dependent
    (`compute_round_mask` / `compute_shop_mask`).
  - `balatro_env/obs_encoder.py`: observation grows to 353 dims;
    additions are stage one-hot[5], money[1], shop-slot tensor (2 x
    22), reroll cost[1].
  - `balatro_env/gym_env.py`: routes shop actions to engine methods;
    transparently auto-starts the next blind on LEAVE_SHOP.
  - `agents/greedy_agent.py`: in-shop strategy = buy first affordable
    slot with a free joker slot, otherwise leave; no rerolls.
  - 21 new tests (`balatro_core/tests/test_shop.py` + extended
    `test_env.py`): interest math, shop buy/reroll mechanics, full
    ante-loop state machine, shop-mask correctness across money /
    slot / reroll conditions. **118 total tests passing.**
- **Phase 3 DoD scorecard (revisited with shop in place):**
  - `check_env`: **green.**
  - Random + greedy baselines: **green.**
  - Greedy beats Ante 3 > 50% on Red Deck: **GREEN — 66% with no
    starter jokers, 82% with 3 starter jokers.** The shop layer was
    the gating dependency for this DoD.
- **Headroom signal (50 seeds each):**
  - Random: 0% across all antes.
  - Greedy no-jokers: 98 / 94 / 66 / 12 / 0 / 0 / 0 / 0% (Ante 1-8).
  - Greedy + 3 starter jokers: 100 / 100 / 82 / 10 / 0 / 0 / 0 / 0%.
  - The cliff at Ante 4-5 reflects exponential chip-target growth;
    closing it requires Planet cards (hand-leveling), joker synergy,
    rarity-aware shop decisions, and selective buying — exactly the
    space RL should win in.
- **Known gaps still open:**
  - RNG bit-parity with the real game (Phase 2 gap, needs Balatrobot).
  - Joker rarity / variable prices (currently flat $4).
  - Booster packs, tarot / planet / spectral cards, vouchers, deck
    modifications, skip-blind rewards. All deferred to later phases.
- **Next concrete task:** Phase 4b — PPO MVP. Recommended setup:
  CleanRL-style PPO with action masking via logit masking,
  64 parallel envs, ~10M steps target. Start training on Ante 1-3
  curriculum (matches the original Phase 4 DoD). Verify the pipeline
  learns within a short (~100k step) sanity run, then queue the long
  training run for offline execution.

When resuming, read this section first, then the **Phase Plan** section
to find the active phase.

---

## Architectural decisions (with rationale)

These decisions are not final, but reversing one is expensive. Revisit
deliberately, not in passing.

### 1. Simulator: build in Python, from scratch, in two layers

- `balatro_core/` — pure rules engine. Deterministic given a seed. Pickleable
  state. No RL concepts. Target ≥5k steps/sec/core.
- `balatro_env/` — Gymnasium adapter. Observation encoding, action masking,
  reward shaping.

**Why not wrap the real game?** LÖVE2D/Lua via screen-scrape or a mod bridge
runs at ~10–30 steps/sec/instance. We need ~10⁷–10⁹ training steps. Wrong
order of magnitude.

**Use Balatrobot (Lua mod with socket API) only as a *fidelity oracle*** —
not as the training env. Record golden traces from the real game, replay
them through our sim, assert digest equality.

### 2. RNG parity with the real game

Balatro seeds its PRNG with `(run_seed, context_string)` (e.g. `"Joker1"`,
`"Tarot2"`). Reproducing this hash exactly lets us replay any
community-shared seed. **This is the single biggest fidelity win** and a
hard requirement before scaling joker coverage.

### 3. Action space: flat discrete with masking

One policy, ~250–400 discrete actions, invalid-action masking via
−∞-logits. Factored along natural phases (in-round / shop / pack / blind
select). In-round uses a **toggle-then-commit** pattern (8 "toggle card i"
+ 1 "play" + 1 "discard") instead of enumerating card subsets, which
would explode with targeting.

**Why not hierarchical RL?** Phases share state heavily — joker decisions
in the shop depend on hand-type plans in-round. One policy is simpler and
sufficient.

### 4. Observation space: padded tensors + small Transformer encoder

Per-entity-type linear embed → 2-layer 4-head Transformer (d_model=128)
over tokens (jokers, hand, shop) with type embeddings → pool + global
scalars MLP → action head with mask.

Deck encoding: start with **aggregate stats** (rank/suit/enhancement
histograms, ~60 dims). Move to full set-encoder over deck tokens only if
ablations show deck composition matters.

### 5. Algorithm: PPO with masking first; reserve MuZero for v2

PPO + invalid-action masking + (recurrent OR frame-stack) policy.

**Why not start with AlphaZero/MuZero?** Hidden information (deck order)
breaks vanilla AlphaZero. Branching factor is moderate post-masking.
Reward isn't that sparse — per-hand chips, per-shop money, per-blind
bonus all give dense signal. PPO has battle-tested distributed
implementations (CleanRL, RLlib).

If PPO plateaus below target win rate, layer **Stochastic MuZero** on the
same env. Don't start there.

### 6. Reward shaping: terminal win, small decaying shaped bonuses

- **Terminal:** +1 on Boss Ante 8 win, 0 on loss. The only reward that
  matters long-term.
- **Shaped (decayed to 0 over first ~30% of training):**
  - `0.01 * log(score / blind_target)` per hand played
  - `0.001 * money_delta` per shop step
- **Do NOT reward** "jokers acquired", "hand level increased", or "raw
  chips scored". Verified reward-hacking failure modes in similar
  deckbuilder RL: the agent will buy garbage jokers, over-level Pair, and
  play just-above-blind hands to burn discards.
- Final fine-tune with shaping fully off.

### 7. Curriculum: ante-restricted → joker-restricted → full game

1. Ante 1, fixed seed, no shop, hand-picked joker loadout. Smoke test.
2. Ante 1–3, restricted joker pool (top 20), full shop.
3. Ante 1–8, restricted joker pool (top 60), Red Deck only. **Main regime.**
4. Full joker pool, all decks, stakes up to Black.

Use **prioritized seed replay** from phase 3: oversample seeds where the
current policy fails near Ante 6–8 — that's where learning signal lives.

---

## Phase Plan

Each phase ends with something runnable and a clear Definition of Done.

### Phase 1 — Rules core skeleton — *est. 1–2 wk*

Deck, Card, hand-type enumeration and evaluation, scoring for all 12 hand
types (incl. Flush House, Flush Five), blind progression, money, basic
shop stubs (purchase but no joker effects yet).

**DoD:**
- CLI that plays a scripted Ante 1 against a fixed deck.
- Score matches real game on ≥5 hand-crafted scenarios (covers each
  hand type from Pair upward).
- Unit tests on hand evaluation pass.

### Phase 2 — RNG parity + first 20 jokers — *est. 2–3 wk*

Reverse-engineer the (seed, context) → PRNG hash. Implement the most-played
jokers: Joker, the four suit jokers (Greedy/Lusty/Wrathful/Gluttonous),
the four pair jokers (Jolly/Zany/Mad/Crazy/Droll), the five suit-flush
jokers (Sly/Wily/Clever/Devious/Crafty), Half, Mime, Credit Card, Banner,
Misprint.

**DoD:**
- 20 Balatrobot golden traces replay bit-identical (score + money + RNG
  state digest at each step).
- Per-joker unit tests with hand-crafted board states.
- Property tests pass (scoring invariant under reorder when no positional
  triggers; discard never increases round score; etc.).

### Phase 3 — Gym env + random/heuristic baselines — *est. 1 wk*

`balatro_env/` Gymnasium adapter. Action masking. Observation encoder
(start simple — flat with histograms — Transformer arrives in Phase 6).
Two baselines: uniform-random-over-legal and a hand-coded greedy
(always play highest-scoring legal hand, never discard, buy any joker if
affordable).

**DoD:**
- `gym.make("Balatro-v0")` returns a working env that passes
  `check_env`.
- Random and greedy baselines run end-to-end.
- Greedy beats Ante 3 on Red Deck >50% of fixed-seed evals.

### Phase 4 — PPO MVP, Ante 1–3 curriculum — *est. 2 wk*

CleanRL-based PPO, 64 parallel envs, ~10M steps. Action masking via
logit masking. Recurrent policy ablation.

**DoD:**
- PPO beats greedy baseline by ≥20pp on Ante 3 win rate.
- >90% Ante 3 win rate with the 20-joker pool.
- Training reproducible via single config + seed.

### Phase 5 — Expand to 60 jokers + full Ante 1–8 — *est. 4–6 wk*

Implement the next 40 priority jokers. **Careful** state machines for
Blueprint, Brainstorm, Showman, DNA, Burglar, Yorick, Ride the Bus, etc.
Scale training to 256 envs, 100M+ steps.

**DoD:**
- >30% Ante 8 win rate on Red Deck across 1000 unseen seeds.
- All 60 jokers have golden-trace coverage.

### Phase 6 — Architecture and curriculum tuning — *est. 2–3 wk*

Set/Transformer encoder for jokers (replaces flat). Prioritized seed
replay. Shaped-reward decay schedule. Recurrent vs. frame-stack
ablation.

**DoD:**
- >60% Ante 8 win rate on Red Deck.
- Ablation report justifying architecture choices.

### Phase 7 — Long-tail jokers + decks/stakes — *est. 3–4 wk*

Remaining ~70 jokers. All 15 decks. Stakes up to Black.

**DoD:**
- >70% Ante 8 across decks on White stake.
- Documented per-deck win rates.

### Phase 8 — *Optional* — MuZero pass — *est. 4–6 wk*

Stochastic MuZero on same env. Compare to PPO at fixed compute.

**DoD:**
- Statistically significant lift over PPO at matched compute, OR a
  documented null result we can move on from.

---

## Engineering scope summary

| Component                              | Engineer-weeks | Risk   |
|----------------------------------------|----------------|--------|
| Core rules engine (no jokers)          | 2–3            | Low    |
| RNG parity with real game              | 1–2            | Medium |
| Top 20 jokers + tests                  | 2              | Low    |
| Remaining ~60 priority jokers          | 4–6            | Medium |
| Long tail (~70 jokers)                 | 4–8            | Medium |
| Gym wrapper + masking                  | 1              | Low    |
| Obs encoder + PPO baseline             | 1–2            | Low    |
| Distributed training infra             | 1–2            | Low    |
| Fidelity test suite + Balatrobot bridge| 2              | Medium |
| Training to Ante 8 consistently        | 4–12           | **High** |
| **First Ante 8 win**                   | **~16–24 wk**  |        |
| **>80% Ante 8 win rate**               | **~30–45 wk**  |        |

---

## Top risks and mitigations

1. **RNG parity bugs surface only after 10M training steps.** Mitigation:
   golden-trace CI from day one; hash-state checkpoint every step in
   debug builds.
2. **Positional/conditional joker triggers** (Blueprint copies, Brainstorm,
   Showman duplicate prevention, DNA hand mutation). Mitigation: dedicated
   state-machine tests; implement these *after* the simple jokers so the
   harness is mature.
3. **Reward hacking.** Mitigation: terminal-reward-dominant schedule;
   monitor "weird" stats (hand-types-leveled, jokers-bought-and-sold).
4. **Credit assignment over ~30-step antes.** Mitigation: GAE with high
   λ; recurrent policy; consider value-function pretraining on
   heuristic-agent trajectories.
5. **Scope creep on joker count.** Mitigation: keep a frozen "tier list"
   of jokers per phase; refuse to implement Tier-N+1 until Tier-N is
   green in CI.

---

## Repository layout (target)

```
BalatroRL/
├── PLAN.md                          # this file
├── README.md
├── pyproject.toml
├── balatro_core/
│   ├── __init__.py
│   ├── cards.py                     # Card, Deck, enhancements/editions/seals
│   ├── hands.py                     # hand-type detection + base scoring
│   ├── engine.py                    # Run, Round, Ante, Blind state machine
│   ├── jokers.py                    # joker definitions + trigger order
│   ├── shop.py                      # shop generation, packs, vouchers
│   ├── rng.py                       # seed → context PRNG (real-game parity)
│   └── tests/
├── balatro_env/
│   ├── __init__.py
│   ├── gym_env.py                   # Gymnasium adapter
│   ├── obs_encoder.py
│   ├── action_space.py              # flat discrete + masking
│   └── reward.py
├── agents/
│   ├── random_agent.py
│   ├── greedy_agent.py
│   └── ppo/                         # CleanRL-style PPO + masking
├── training/
│   ├── configs/
│   └── train.py
├── fidelity/
│   ├── balatrobot_bridge.py         # capture traces from real game
│   ├── golden_traces/               # checked-in trace files
│   └── replay_check.py
└── scripts/
```

---

## Reference

- **Balatro** game by LocalThunk (Playstack publisher). Steam release Feb 2024.
- **Gymnasium** — the maintained fork of OpenAI Gym. Use this, not legacy `gym`.
- **CleanRL** — minimal-deps PPO implementation; good starting point.
- **Stable-Baselines3** — alternative with `MaskablePPO`.
- **Balatrobot** — community Lua mod exposing a socket API to the real game;
  used as a fidelity oracle, NOT as the training environment.
- Hand-type scoring base values: see in-game `hand_type` table.
  Reproduced in `balatro_core/hands.py` when implemented.

---

## Working agreement (for future sessions)

- Update the **Status** section every session that changes phase progress.
- Don't start Phase N+1 until Phase N's DoD is green in CI.
- New architectural decisions go in **Architectural decisions** with a
  one-paragraph rationale. Don't quietly change defaults.
- Joker implementations land with golden-trace tests, not just unit tests.
- Reward shaping changes require an ablation note in the commit message.
