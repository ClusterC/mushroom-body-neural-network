import unittest
import numpy as np
from src.envs.snake_env import SnakeEnv
from src.models.visual_mushroom_body import VisualMushroomBody

class TestVisualMushroomBody(unittest.TestCase):
    def setUp(self):
        self.env = SnakeEnv(width=10, height=10, seed=42)
        self.vmb = VisualMushroomBody(channels=3, grid_h=10, grid_w=10, num_kc=2000, k_active=100, seed=42)

    def test_visual_pn_and_kc_sparsity(self):
        obs = self.env.reset()
        self.assertEqual(self.vmb.num_pn, 300)
        kc_act = self.vmb.encode_kc(obs)
        self.assertEqual(kc_act.shape, (2000,))

        # ต้องมี Active เซลล์ตรง 100 เซลล์ (5.0% Sparsity)
        active_count = int(np.sum(kc_act == 1.0))
        self.assertEqual(active_count, 100)
        self.assertEqual(active_count / 2000.0, 0.05)

    def test_forward_and_action_masking(self):
        obs = self.env.reset()
        mask = self.env.get_action_mask()
        probs, _, mbon_act = self.vmb.forward(obs, legal_mask=mask)

        self.assertEqual(probs.shape, (4,))
        # ผลรวมความน่าจะเป็นต้องเท่ากับ 1.0
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        # ทิศทางที่ถูก Mask (ห้ามเลี้ยว 180) ต้องมีความน่าจะเป็น 0
        forbidden_idx = np.where(~mask)[0]
        for f_idx in forbidden_idx:
            self.assertAlmostEqual(float(probs[f_idx]), 0.0, places=5)

    def test_step_plasticity_update(self):
        self.env.reset()
        w_init = self.vmb.w_kc_mbon.copy()

        action = self.vmb.select_action(self.env, training=True)
        _, reward, done, _ = self.env.step(action)

        self.vmb.update_synapses(reward)
        # ตรวจสอบว่าน้ำหนัก Synapse มีการเปลี่ยนแปลงและอยู่ในขอบเขต [0, w_max]
        self.assertFalse(np.array_equal(self.vmb.w_kc_mbon, w_init))
        self.assertTrue(np.all(self.vmb.w_kc_mbon >= 0.0))
        self.assertTrue(np.all(self.vmb.w_kc_mbon <= self.vmb.w_max))

if __name__ == "__main__":
    unittest.main()
