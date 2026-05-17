# BalatroRL — Master Plan

> **North star:** train an RL agent that consistently beats Boss Ante 8 of Balatro.
> "Consistently" = ≥70% win rate on Red Deck / White Stake across unseen seeds.

This document is the source of truth for the project. It is meant to be read at
the start of every working session so we can pick up where we left off.
Update the **Status** section as phases complete.

---

## Status

- **Current phase:** Phase 1 — Rules core skeleton, in progress.
- **Last updated:** 2026-05-17
- **Completed:**
  - `balatro_core/` package scaffolded: cards, hands, scoring, engine.
  - All 12 hand types detected and scored at level 1; level scaling
    implemented and tested at levels 2, 3, 5.
  - Engine: `Run` / `Round` / `Blind` with play/discard/draw and per-ante
    chip targets (Ante 1-8).
  - 36 pytest tests passing (18 hand-evaluation, 13 scoring scenarios
    covering every hand type, 5 hand-level value tests).
  - `scripts/smoke_ante1.py` plays a scripted Ante 1 end-to-end with a
    brute-force greedy + simple discard heuristic. Beats Ante 1 on
    several test seeds (e.g. 1, 42); loses on others (e.g. 7, 13, 100) —
    which is expected for a non-RL agent and confirms the engine handles
    both win and loss paths.
- **Next concrete task:** Phase 1 wrap-up — decide whether to expand the
  rules core (booster/shop stubs, money/interest) before moving to
  Phase 2 (RNG parity + first 20 jokers). The current core is a solid
  base for Phase 2; the shop stubs are deferrable until Phase 3 when the
  Gym adapter starts needing them.

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
