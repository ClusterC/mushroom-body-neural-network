"""
Unit tests for StockTradingEnv.
Validates market data generation, action masking, trade executions, fees, and accounting.
"""

import unittest
import numpy as np

from src.envs.stock_trading_env import StockTradingEnv, HOLD, BUY, SELL


class TestStockTradingEnv(unittest.TestCase):
    def setUp(self):
        self.env = StockTradingEnv(initial_cash=10000.0, max_steps=50, seed=123)

    def test_initial_state(self):
        self.assertEqual(self.env.cash, 10000.0)
        self.assertEqual(self.env.shares, 0)
        self.assertEqual(self.env.total_trades, 0)
        self.assertFalse(self.env.done)

        obs = self.env.get_observation()
        self.assertEqual(obs.shape, (18,))
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_action_masking_initial(self):
        mask = self.env.get_action_mask()
        # In cash: can HOLD (0), can BUY (1), CANNOT SELL (2)
        self.assertTrue(mask[HOLD])
        self.assertTrue(mask[BUY])
        self.assertFalse(mask[SELL])

    def test_buy_execution(self):
        curr_price = self.env.prices[self.env.current_step]
        obs, reward, done, info = self.env.step(BUY)

        self.assertGreater(self.env.shares, 0)
        self.assertLess(self.env.cash, curr_price)
        self.assertEqual(info["trade_event"], "BUY")

        # After buying: can HOLD, CANNOT BUY again, CAN SELL
        mask = self.env.get_action_mask()
        self.assertTrue(mask[HOLD])
        self.assertFalse(mask[BUY])
        self.assertTrue(mask[SELL])

    def test_sell_execution(self):
        # First buy
        self.env.step(BUY)
        shares_bought = self.env.shares

        # Advance a few steps
        self.env.step(HOLD)
        self.env.step(HOLD)

        # Then sell
        obs, reward, done, info = self.env.step(SELL)
        self.assertEqual(self.env.shares, 0)
        self.assertGreater(self.env.cash, 0.0)
        self.assertEqual(self.env.total_trades, 1)
        self.assertEqual(info["trade_event"], "SELL")

    def test_illegal_action_fallback_to_hold(self):
        # Selling when having no shares is illegal, should fallback to HOLD
        self.assertEqual(self.env.shares, 0)
        obs, reward, done, info = self.env.step(SELL)
        self.assertEqual(self.env.shares, 0)
        self.assertEqual(info["action"], HOLD)

    def test_reset(self):
        self.env.step(BUY)
        self.env.step(HOLD)
        obs = self.env.reset()

        self.assertEqual(self.env.cash, 10000.0)
        self.assertEqual(self.env.shares, 0)
        self.assertEqual(self.env.total_trades, 0)
        self.assertFalse(self.env.done)
        self.assertEqual(obs.shape, (18,))

    def test_episode_termination(self):
        while not self.env.done:
            self.env.step(HOLD)
        self.assertTrue(self.env.done)
        self.assertGreaterEqual(self.env.current_step, 50)

    def test_asset_profiles_generation(self):
        profiles = ["TECH_MOMENTUM", "INDEX_ETF", "CRYPTO_VOLATILE", "DEFENSIVE_VALUE", "CYCLICAL_COMMODITY"]
        for prof in profiles:
            env = StockTradingEnv(initial_cash=10000.0, max_steps=40, asset_profile=prof, seed=42)
            obs = env.reset()
            self.assertEqual(obs.shape, (18,))
            self.assertTrue(np.all(np.isfinite(obs)))
            self.assertGreater(len(env.prices), 40)
            self.assertTrue(np.all(env.prices > 0))

    def test_dynamic_set_asset_profile(self):
        self.env.set_asset_profile("CRYPTO_VOLATILE")
        self.assertEqual(self.env.asset_profile, "CRYPTO_VOLATILE")
        self.assertEqual(self.env.cash, 10000.0)
        self.assertEqual(self.env.shares, 0)
        self.assertGreater(self.env.prices[0], 1000.0)

    def test_real_csv_loading(self):
        import os
        csv_path = "data/assets/SPY_historical.csv"
        if os.path.exists(csv_path):
            env = StockTradingEnv(initial_cash=10000.0, csv_path=csv_path)
            obs = env.reset()
            self.assertEqual(obs.shape, (18,))
            self.assertGreater(len(env.prices), 50)
            self.assertFalse(env.done)

    def test_inaction_tracking_and_penalty(self):
        # Step HOLD while in cash: inaction_bars should increment
        self.assertEqual(self.env.inaction_bars, 0)
        obs, reward, done, info = self.env.step(HOLD)
        self.assertEqual(self.env.inaction_bars, 1)
        self.assertEqual(info["inaction_bars"], 1)

        # Buying should reset inaction_bars to 0
        obs, reward, done, info = self.env.step(BUY)
        self.assertEqual(self.env.inaction_bars, 0)
        self.assertEqual(info["inaction_bars"], 0)


if __name__ == "__main__":
    unittest.main()
