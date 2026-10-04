"""
Unit tests for GPUTradingTrainer (PyTorch CUDA acceleration for Hippocampus Trading MB).
"""

import unittest
import numpy as np
import torch

from src.models.hippocampal_trading_mb import HippocampalTradingMB
from src.training.gpu_trading_trainer import GPUTradingTrainer


class TestGPUTradingTrainer(unittest.TestCase):
    def test_gpu_trading_trainer_init(self):
        """Verify GPUTradingTrainer initializes properly with tensors."""
        model = HippocampalTradingMB()
        trainer = GPUTradingTrainer(model, batch_size=32)

        self.assertEqual(trainer.role_vectors.shape, (18, 2048))
        self.assertEqual(trainer.level_vectors.shape, (25, 2048))
        self.assertEqual(trainer.action_prototypes.shape, (2048, 3))
        self.assertEqual(trainer.regime_prototypes.shape, (2048, 4))
        self.assertEqual(trainer.traces.shape, (32, 2048, 3))

    def test_gpu_trading_trainer_forward_pass(self):
        """Verify VSA encoding, DG separation, CA3 attractor, and Action selection."""
        model = HippocampalTradingMB()
        batch_size = 16
        trainer = GPUTradingTrainer(model, batch_size=batch_size)

        # Random observations [-2.0, 2.0]
        obs = torch.randn((batch_size, 18), dtype=torch.float32, device=trainer.device).clamp_(-2.0, 2.0)
        ec = trainer.encode_vsa_batch(obs)
        self.assertEqual(ec.shape, (batch_size, 2048))
        # Check bipolar values (+1 or -1)
        self.assertTrue(torch.all((ec == 1.0) | (ec == -1.0)))

        # DG k-WTA
        dg = trainer.dentate_gyrus_batch(ec)
        self.assertEqual(dg.shape, (batch_size, 2048))
        active_counts = torch.sum(dg, dim=-1)
        self.assertTrue(torch.all(active_counts == trainer.k_dg))

        # CA3 Recurrent
        ca3 = trainer.ca3_recurrent_batch(dg)
        self.assertEqual(ca3.shape, (batch_size, 2048))

        # Regime Classification
        regimes = trainer.classify_regimes_batch(dg)
        self.assertEqual(regimes.shape, (batch_size,))
        self.assertTrue(torch.all((regimes >= 0) & (regimes < 4)))

        # Action Selection
        masks = torch.ones((batch_size, 3), dtype=torch.bool, device=trainer.device)
        holding = torch.zeros(batch_size, dtype=torch.float32, device=trainer.device)
        actions = trainer.select_actions_batch(ca3, regimes, masks, holding)
        self.assertEqual(actions.shape, (batch_size,))
        self.assertTrue(torch.all((actions >= 0) & (actions < 3)))

    def test_gpu_trading_trainer_train_short(self):
        """Verify quick training loop completes and synchronizes weights."""
        model = HippocampalTradingMB()
        initial_weights = model.action_prototypes.copy()

        trainer = GPUTradingTrainer(model, batch_size=32)
        result = trainer.train(total_episodes=32)

        self.assertGreaterEqual(result["episodes"], 32)
        self.assertIn("duration", result)
        self.assertIn("fps", result)
        self.assertIn("win_rate", result)

        # Check weights were updated and synced back
        self.assertFalse(np.array_equal(model.action_prototypes, initial_weights))


if __name__ == "__main__":
    unittest.main()
