"""
Unit tests for TradingVisualizerApp.
Validates headless state transitions, stepping, resetting, and training routine.
"""

import unittest

from src.visualizer.trading_app import TradingVisualizerApp


class TestTradingVisualizerApp(unittest.TestCase):
    def setUp(self):
        self.app = TradingVisualizerApp(headless=True)

    def test_initial_headless_state(self):
        self.assertIsNotNone(self.app.env)
        self.assertIsNotNone(self.app.mb)
        self.assertEqual(self.app.obs.shape, (18,))
        self.assertEqual(self.app.env.cash, 10000.0)
        self.assertEqual(len(self.app.executed_trades), 0)

    def test_step_simulation(self):
        initial_step = self.app.env.current_step
        self.app.step_simulation()
        self.assertGreater(self.app.env.current_step, initial_step)
        self.assertIn(self.app.last_action, [0, 1, 2])

    def test_reset_simulation(self):
        self.app.step_simulation()
        self.app.step_simulation()
        self.app.reset_simulation()

        self.assertEqual(self.app.env.cash, 10000.0)
        self.assertEqual(len(self.app.executed_trades), 0)
        self.assertFalse(self.app.env.done)

    def test_training_routine(self):
        self.app.train_episodes(n_episodes=2)
        self.assertEqual(self.app.total_trained_episodes, 2)
        self.assertEqual(self.app.training_current_ep, 2)
        self.assertAlmostEqual(self.app.training_progress, 1.0)
        self.assertFalse(self.app.is_training)

    def test_hardware_telemetry(self):
        self.assertIsInstance(self.app.has_cuda, bool)
        self.assertIsInstance(self.app.gpu_device_name, str)
        self.assertGreater(len(self.app.gpu_device_name), 0)
        self.assertFalse(self.app.gpu_active)

    def test_async_training_thread(self):
        self.app.train_episodes(n_episodes=2, async_mode=True)
        self.assertIsNotNone(self.app.training_thread)
        self.app.training_thread.join(timeout=5.0)
        self.assertFalse(self.app.is_training)
        self.assertEqual(self.app.total_trained_episodes, 2)

    def test_asset_profile_switching(self):
        initial_idx = self.app.asset_idx
        self.app.asset_idx = (self.app.asset_idx + 1) % len(self.app.asset_profiles_list)
        prof_key, prof_label, csv_f = self.app.asset_profiles_list[self.app.asset_idx]
        self.app.env.set_asset_profile(prof_key, csv_path=csv_f)
        self.app.reset_simulation()

        self.assertNotEqual(self.app.asset_idx, initial_idx)
        self.assertEqual(self.app.env.asset_profile, prof_key)
        self.assertEqual(len(self.app.executed_trades), 0)


if __name__ == "__main__":
    unittest.main()
