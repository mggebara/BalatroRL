"""Shop / economy / Run state-machine tests."""
from __future__ import annotations

import random

import pytest

from balatro_core.economy import (
    BLIND_REWARDS,
    end_of_round_payout,
    interest_payout,
)
from balatro_core.engine import BlindKind, GameStage, Run
from balatro_core.jokers import JokerJoker
from balatro_core.shop import (
    DEFAULT_REROLL_BASE_COST,
    Shop,
    ShopSlot,
    generate_shop,
    reroll_jokers,
)


class TestEconomy:
    def test_interest_floors_below_5(self):
        assert interest_payout(0) == 0
        assert interest_payout(4) == 0
        assert interest_payout(5) == 1
        assert interest_payout(9) == 1
        assert interest_payout(10) == 2

    def test_interest_caps_at_5(self):
        assert interest_payout(25) == 5
        assert interest_payout(100) == 5

    def test_interest_ignores_negative_money(self):
        assert interest_payout(-10) == 0

    def test_end_of_round_payout_components(self):
        # $10 in pocket + Big Blind ($4) + 3 hands left ($3) + interest ($2) = $9
        # (interest is computed on money_at_end before payout)
        out = end_of_round_payout(
            money_at_end=10,
            blind=BlindKind.BIG,
            hands_remaining=3,
        )
        assert out == BLIND_REWARDS[BlindKind.BIG] + 3 + 2

    def test_blind_rewards_known_values(self):
        assert BLIND_REWARDS[BlindKind.SMALL] == 3
        assert BLIND_REWARDS[BlindKind.BIG] == 4
        assert BLIND_REWARDS[BlindKind.BOSS] == 5


class TestShop:
    def test_generate_shop_two_slots(self):
        rng = random.Random(0)
        shop = generate_shop(rng, num_slots=2)
        assert shop.slot_count == 2
        assert all(slot.joker is not None for slot in shop.slots)
        assert all(slot.price == 4 for slot in shop.slots)

    def test_buy_returns_joker_and_new_money(self):
        rng = random.Random(0)
        shop = generate_shop(rng)
        original = shop.slots[0].joker
        joker, new_money = shop.buy(0, money=10)
        assert joker is original
        assert new_money == 6
        assert shop.slots[0].joker is None

    def test_buy_insufficient_money_raises(self):
        rng = random.Random(0)
        shop = generate_shop(rng)
        with pytest.raises(RuntimeError):
            shop.buy(0, money=2)

    def test_buy_empty_slot_raises(self):
        rng = random.Random(0)
        shop = generate_shop(rng)
        shop.buy(0, money=10)
        with pytest.raises(RuntimeError):
            shop.buy(0, money=10)

    def test_reroll_replaces_unsold_slots_only(self):
        rng = random.Random(0)
        shop = generate_shop(rng)
        # Buy slot 0, sold slot becomes None.
        shop.buy(0, money=10)
        replacements = reroll_jokers(rng, shop.slot_count)
        original_slot1 = shop.slots[1].joker
        new_money = shop.reroll(replacements, money=20)
        assert new_money == 20 - DEFAULT_REROLL_BASE_COST
        assert shop.slots[0].joker is None  # still sold
        # Slot 1 was replaced.
        assert shop.slots[1].joker is not original_slot1

    def test_reroll_cost_increments(self):
        rng = random.Random(0)
        shop = generate_shop(rng)
        assert shop.reroll_cost == 5
        shop.reroll(reroll_jokers(rng, shop.slot_count), money=20)
        assert shop.reroll_cost == 6
        shop.reroll(reroll_jokers(rng, shop.slot_count), money=20)
        assert shop.reroll_cost == 7


class TestRunStateMachine:
    def test_pristine_run_starts_in_pre_blind(self):
        run = Run(rng_seed=0)
        assert run.stage == GameStage.PRE_BLIND
        assert run.ante == 1
        assert run.current_blind == BlindKind.SMALL
        assert run.current_shop is None

    def test_round_win_enters_shop(self):
        run = Run(rng_seed=0)
        rd = run.start_current_blind()
        # Force a win without actually scoring: set total_score above target.
        rd.state.total_score = rd.state.chip_target
        # Reduce hands_remaining to a known value so payout is deterministic.
        rd.state.hands_remaining = 2
        money_before = run.money
        run.advance_after_round_win()
        assert run.stage == GameStage.IN_SHOP
        assert run.current_shop is not None
        # 3 (small reward) + 2 (hands left) + interest($4) = 0 -> 5.
        expected_pay = 3 + 2 + 0
        assert run.money == money_before + expected_pay

    def test_buy_then_leave_advances_blind(self):
        run = Run(rng_seed=0, starting_money=20)
        rd = run.start_current_blind()
        rd.state.total_score = rd.state.chip_target
        run.advance_after_round_win()
        assert run.stage == GameStage.IN_SHOP
        # Buy slot 0.
        run.buy_shop_slot(0)
        assert len(run.jokers) == 1
        # Leave shop -> next blind PRE_BLIND.
        run.leave_shop()
        assert run.stage == GameStage.PRE_BLIND
        assert run.current_blind == BlindKind.BIG

    def test_full_ante_loop(self):
        # Small -> shop -> Big -> shop -> Boss -> shop -> next ante.
        run = Run(rng_seed=0, starting_money=0, max_ante=2)
        for expected_blind in (BlindKind.SMALL, BlindKind.BIG, BlindKind.BOSS):
            assert run.current_blind == expected_blind
            rd = run.start_current_blind()
            rd.state.total_score = rd.state.chip_target
            run.advance_after_round_win()
            assert run.stage == GameStage.IN_SHOP
            run.leave_shop()
        # After Boss + leave: should advance to next ante, Small.
        assert run.ante == 2
        assert run.current_blind == BlindKind.SMALL

    def test_game_won_at_max_ante_boss(self):
        run = Run(rng_seed=0, max_ante=1)
        # Win Small.
        rd = run.start_current_blind()
        rd.state.total_score = rd.state.chip_target
        run.advance_after_round_win()
        run.leave_shop()
        # Win Big.
        rd = run.start_current_blind()
        rd.state.total_score = rd.state.chip_target
        run.advance_after_round_win()
        run.leave_shop()
        # Win Boss.
        rd = run.start_current_blind()
        rd.state.total_score = rd.state.chip_target
        run.advance_after_round_win()
        run.leave_shop()
        assert run.stage == GameStage.GAME_WON

    def test_buy_blocked_when_slots_full(self):
        run = Run(rng_seed=0, starting_money=100, jokers=[JokerJoker()] * 5)
        rd = run.start_current_blind()
        rd.state.total_score = rd.state.chip_target
        run.advance_after_round_win()
        with pytest.raises(RuntimeError):
            run.buy_shop_slot(0)
