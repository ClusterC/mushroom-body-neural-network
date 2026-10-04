import unittest
import numpy as np
from src.envs.tic_tac_toe import TicTacToeEnv
from src.models.mushroom_body import MushroomBodyNet

class TestMushroomBody(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()
        self.mb = MushroomBodyNet(num_kc=1000, k_active=75, seed=42)

    def test_kc_sparsity_k_wta(self):
        obs = self.env.reset()
        kc_act = self.mb.encode_kc(obs)
        self.assertEqual(kc_act.shape, (1000,))
        active_count = np.sum(kc_act > 0)
        self.assertEqual(active_count, 75)
        # ตรวจสอบว่าอยู่ในช่วง 5% - 10%
        density = active_count / 1000.0
        self.assertTrue(0.05 <= density <= 0.10)

    def test_action_masking_probabilities(self):
        self.env.reset()
        self.env.step(4)  # ช่อง 4 ถูกครอบครอง
        obs = self.env.get_observation()
        mask = self.env.get_action_mask()
        probs, _, _ = self.mb.forward(obs, legal_mask=mask)

        # ความน่าจะเป็นของช่อง 4 ต้องเป็น 0
        self.assertAlmostEqual(probs[4], 0.0, places=5)
        self.assertAlmostEqual(np.sum(probs), 1.0, places=5)

    def test_plasticity_update(self):
        self.env.reset()
        initial_weights = self.mb.w_kc_mbon.copy()

        # ทำการเลือกก้าวเดิน 1 ครั้ง
        action = self.mb.select_action(self.env, training=True)
        self.assertTrue(np.any(self.mb.eligibility_trace > 0))

        # ส่งสัญญาณ Dopamine เป็นบวก (+1.0)
        self.mb.update_synapses(dopamine_signal=1.0)

        # ตรวจสอบว่าน้ำหนัก Synapse ของคู่ที่ Active ถูกปรับเพิ่มขึ้น และไม่มีค่าน้อยกว่า 0
        self.assertTrue(np.all(self.mb.w_kc_mbon >= 0.0))
        self.assertTrue(np.all(self.mb.w_kc_mbon <= self.mb.w_max))
        self.assertTrue(np.any(self.mb.w_kc_mbon > initial_weights))
        # Eligibility trace ต้องถูกรีเซ็ต
        self.assertTrue(np.all(self.mb.eligibility_trace == 0.0))

if __name__ == "__main__":
    unittest.main()
