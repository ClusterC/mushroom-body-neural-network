"""
Unit tests for HippocampalTradingMB architecture.
Validates EC encoding, Dentate Gyrus 2.44% ultra-sparsity, CA3 permutation,
SWR episodic replay, and CPG Risk Reflex stop-loss.
"""

import unittest
import numpy as np

from src.models.hippocampal_trading_mb import HippocampalTradingMB, HOLD, BUY, SELL


class TestHippocampalTradingMB(unittest.TestCase):
    def setUp(self):
        self.agent = HippocampalTradingMB(
            dim=2048,
            k_dg=50,
            k_ca3=120,
            stop_loss_pct=-0.03,
            seed=42
        )

    def test_entorhinal_cortex_encoding(self):
        obs = np.random.uniform(-1.0, 1.0, size=16).astype(np.float32)
        ec_vec = self.agent.encode_entorhinal_cortex(obs)

        self.assertEqual(ec_vec.shape, (2048,))
        self.assertTrue(np.all(np.isin(ec_vec, [-1.0, 1.0])))

    def test_dentate_gyrus_sparsity(self):
        ec_vec = np.random.choice([-1.0, 1.0], size=2048).astype(np.float32)
        dg_sparse = self.agent.dentate_gyrus_separation(ec_vec)

        active_count = int(np.count_nonzero(dg_sparse))
        self.assertEqual(active_count, 50)
        # Sparsity check: 50 / 2048 = 0.024414 (approx 2.44%)
        sparsity = active_count / 2048.0
        self.assertAlmostEqual(sparsity, 0.0244, places=3)

    def test_ca3_temporal_sequence_memory(self):
        dg_vec = np.zeros(2048, dtype=np.float32)
        dg_vec[:50] = 1.0

        ca3_1 = self.agent.ca3_recurrent_sequence(dg_vec)
        self.assertEqual(self.agent.last_ca3_depth, 1)

        ca3_2 = self.agent.ca3_recurrent_sequence(dg_vec)
        self.assertEqual(self.agent.last_ca3_depth, 2)
        self.assertLessEqual(np.count_nonzero(ca3_2), 120)

    def test_cpg_stop_loss_reflex(self):
        # Scenario 1: Normal holding (unrealized loss -1.0%, not reaching -3.0%)
        obs_normal = np.zeros(16, dtype=np.float32)
        obs_normal[8] = 1.0   # is_holding = True
        obs_normal[9] = -0.1  # unrealized_pnl = -1.0%
        act, triggered, reason = self.agent.check_cpg_risk_reflex(obs_normal, HOLD)
        self.assertEqual(act, HOLD)
        self.assertFalse(triggered)

        # Scenario 2: Severe drawdown (unrealized loss -3.5%, exceeds -3.0% stop-loss)
        obs_severe = np.zeros(16, dtype=np.float32)
        obs_severe[8] = 1.0    # is_holding = True
        obs_severe[9] = -0.35  # unrealized_pnl = -3.5%
        act, triggered, reason = self.agent.check_cpg_risk_reflex(obs_severe, HOLD)
        self.assertEqual(act, SELL)
        self.assertTrue(triggered)
        self.assertIn("STOP-LOSS", reason)

    def test_select_action_with_mask(self):
        obs = np.random.uniform(-0.5, 0.5, size=16).astype(np.float32)
        # Mask only HOLD allowed
        mask = np.array([True, False, False], dtype=bool)
        act, probs, dg, ca3 = self.agent.select_action(obs, mask, training=False)

        self.assertEqual(act, HOLD)
        self.assertEqual(probs.shape, (3,))
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)

    def test_swr_episodic_replay(self):
        obs = np.random.uniform(-0.5, 0.5, size=16).astype(np.float32)
        mask = np.array([True, True, False], dtype=bool)

        # Record 3 action steps
        for _ in range(3):
            self.agent.select_action(obs, mask, training=True)

        self.assertEqual(len(self.agent.episode_experiences), 3)

        w_before = self.agent.action_prototypes.copy()
        # Trigger SWR with profitable reward (+1.0)
        self.agent.trigger_swr_episodic_replay(final_reward=1.0)

        self.assertTrue(self.agent.last_swr_active)
        self.assertEqual(len(self.agent.episode_experiences), 0)
        self.assertFalse(np.array_equal(self.agent.action_prototypes, w_before))
        # Dale's Bound check
        self.assertTrue(np.all(self.agent.action_prototypes >= 0.0))


if __name__ == "__main__":
    unittest.main()
