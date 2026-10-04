import unittest
import numpy as np
from src.models.mushroom_body import MushroomBodyNet
from src.training.self_play import SelfPlayTrainer, train_self_play

class TestSelfPlay(unittest.TestCase):
    def setUp(self):
        self.mb = MushroomBodyNet(num_kc=500, k_active=35, seed=42)

    def test_self_play_single_episode(self):
        trainer = SelfPlayTrainer(self.mb, seed=123)
        w_init = self.mb.w_kc_mbon.copy()
        res = trainer.train_episode()

        self.assertIn("winner", res)
        self.assertIn("active_player", res)
        # ตรวจสอบว่าน้ำหนัก Synapse มีการอัปเดต
        self.assertFalse(np.array_equal(self.mb.w_kc_mbon, w_init))

    def test_train_self_play_multi_episodes(self):
        stats = train_self_play(self.mb, episodes=30, snapshot_interval=10)
        self.assertEqual(stats["episodes"], 30)
        self.assertIn("draw_rate", stats)
        self.assertIn("weight_delta", stats)
        self.assertTrue(0.0 <= stats["draw_rate"] <= 1.0)
        self.assertTrue(np.all(self.mb.w_kc_mbon >= 0.0))

if __name__ == "__main__":
    unittest.main()
