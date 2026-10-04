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
        self.assertEqual(self.app.obs.shape, (16,))
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


if __name__ == "__main__":
    unittest.main()
