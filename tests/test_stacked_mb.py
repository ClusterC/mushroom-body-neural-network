import unittest
import numpy as np
from src.models.stacked_visual_mb import StackedVisualMushroomBody

class TestStackedVisualMushroomBody(unittest.TestCase):
    def setUp(self):
        self.model = StackedVisualMushroomBody(
            channels=3,
            grid_h=10,
            grid_w=10,
            num_kc1=1200,
            k_active1=60,
            num_concepts=12,
            num_kc2=800,
            k_active2=40,
            num_mbon=4,
            seed=42
        )

    def test_initialization(self):
        """ตรวจสอบขนาดและมิติของโครงข่ายทั้งสองชั้น"""
        self.assertEqual(self.model.w_pn1_kc1.shape, (300, 1200))
        self.assertEqual(self.model.w_kc1_mbon1.shape, (1200, 12))
        self.assertEqual(self.model.w_pn2_kc2.shape, (16, 800))
        self.assertEqual(self.model.w_kc2_mbon2.shape, (800, 4))
        self.assertEqual(len(self.model.CONCEPT_NAMES), 12)

    def test_forward_and_dual_sparsity(self):
        """ตรวจสอบว่าทั้งสองชั้นรักษาระดับความเบาบาง 5% (KC1=60, KC2=40)"""
        dummy_obs = np.random.uniform(0.0, 1.0, size=(3, 10, 10)).astype(np.float32)
        action, probs, kc1, concepts, kc2 = self.model.forward(dummy_obs)

        # ตรวจสอบการกระทำและความน่าจะเป็น
        self.assertIn(action, [0, 1, 2, 3])
        self.assertEqual(len(probs), 4)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)

        # ตรวจสอบ Sparsity ชั้นที่ 1 (60 เซลล์ หรือ 5.0%)
        active_kc1 = np.sum(kc1 > 0)
        self.assertEqual(active_kc1, 60, f"KC1 ควรทำงาน 60 เซลล์ แต่ได้ {active_kc1}")

        # ตรวจสอบ Intermediate Concepts (12 มิติ ในช่วง [0, 1])
        self.assertEqual(len(concepts), 12)
        self.assertTrue(np.all(concepts >= 0.0) and np.all(concepts <= 1.0))

        # ตรวจสอบ Sparsity ชั้นที่ 2 (40 เซลล์ หรือ 5.0%)
        active_kc2 = np.sum(kc2 > 0)
        self.assertEqual(active_kc2, 40, f"KC2 ควรทำงาน 40 เซลล์ แต่ได้ {active_kc2}")

    def test_action_masking(self):
        """ตรวจสอบว่า Action Masking สามารถบล็อกตาเดินต้องห้ามได้สมบูรณ์"""
        dummy_obs = np.random.uniform(0.0, 1.0, size=(3, 10, 10)).astype(np.float32)
        # บล็อกทุกทาง ยกเว้น Action 1 (RIGHT)
        mask = np.array([0, 1, 0, 0], dtype=np.int32)
        action, probs, _, _, _ = self.model.forward(dummy_obs, action_mask=mask)

        self.assertEqual(action, 1)
        self.assertAlmostEqual(probs[1], 1.0, places=4)
        self.assertAlmostEqual(probs[0], 0.0, places=4)
        self.assertAlmostEqual(probs[2], 0.0, places=4)
        self.assertAlmostEqual(probs[3], 0.0, places=4)

    def test_dual_plasticity_update(self):
        """ตรวจสอบการอัปเดตค่าน้ำหนัก Synapse ทั้งสองชั้นตามรางวัล"""
        dummy_obs = np.random.uniform(0.0, 1.0, size=(3, 10, 10)).astype(np.float32)
        self.model.forward(dummy_obs)

        w1_before = float(np.sum(self.model.w_kc1_mbon1))
        w2_before = float(np.sum(self.model.w_kc2_mbon2))

        # อัปเดตด้วยรางวัลบวก
        self.model.update_plasticity(reward=1.0)

        w1_after = float(np.sum(self.model.w_kc1_mbon1))
        w2_after = float(np.sum(self.model.w_kc2_mbon2))

        self.assertGreater(w1_after, w1_before, "Synapse ชั้นที่ 1 ควรเพิ่มขึ้นตามรางวัล")
        self.assertGreater(w2_after, w2_before, "Synapse ชั้นที่ 2 ควรเพิ่มขึ้นตามรางวัล")
        
        # ตรวจสอบ Dale's Principle (ไม่ติดลบ)
        self.assertTrue(np.all(self.model.w_kc1_mbon1 >= 0.0))
        self.assertTrue(np.all(self.model.w_kc2_mbon2 >= 0.0))

    def test_egocentric_and_cpg_reflex(self):
        """ตรวจสอบ Egocentric Whiskers 12 มิติ และ CPG Reflex Safe Mask"""
        from src.envs.snake_env import SnakeEnv, UP, RIGHT, DOWN, LEFT
        env = SnakeEnv(width=10, height=10, seed=123)
        ego = env.get_egocentric_observation()
        self.assertEqual(len(ego), 12)
        self.assertTrue(np.all(ego >= 0.0) and np.all(ego <= 1.0))

        # ตรวจสอบ Safe Action Mask
        safe_mask = env.get_safe_action_mask()
        self.assertEqual(len(safe_mask), 4)
        self.assertIsInstance(safe_mask[0], (bool, np.bool_))

    def test_innate_concept_grounding(self):
        """ตรวจสอบว่า Innate Concept Grounding มีการกำหนดน้ำหนักเริ่มต้นให้กับ Concepts อย่างมีความหมาย"""
        # น้ำหนักใน w_kc1_mbon1 ต้องมีค่าเฉลี่ยมากกว่า 0.1 และไม่แบนราบ
        self.assertGreater(float(np.mean(self.model.w_kc1_mbon1)), 0.15)
        # ตรวจสอบว่า w_kc2_mbon2 มีค่าเชื่อมโยงสัญชาตญาณ
        self.assertGreater(float(np.mean(self.model.w_kc2_mbon2)), 0.15)

    def test_export_and_load_weights(self):
        """ตรวจสอบการส่งออกและโหลด Synaptic Weights"""
        weights = self.model.export_weights()
        self.assertIn("w_pn1_kc1", weights)
        self.assertIn("w_kc1_mbon1", weights)
        self.assertIn("w_pn2_kc2", weights)
        self.assertIn("w_kc2_mbon2", weights)

        # โคลนโมเดลใหม่แล้วโหลด weights
        new_model = StackedVisualMushroomBody(seed=999)
        new_model.load_weights(weights)
        np.testing.assert_array_almost_equal(new_model.w_kc1_mbon1, self.model.w_kc1_mbon1)
        np.testing.assert_array_almost_equal(new_model.w_kc2_mbon2, self.model.w_kc2_mbon2)

    def test_gpu_trainer(self):
        """ตรวจสอบการทำงานของ GPUMushroomBodyTrainer"""
        from src.training.gpu_snake_trainer import GPUMushroomBodyTrainer
        trainer = GPUMushroomBodyTrainer(self.model, batch_size=32, device="cpu")
        res = trainer.train(total_episodes=50, cpg_reflex=True, use_whiskers=True)
        self.assertIn("episodes", res)
        self.assertIn("fps", res)
        self.assertGreaterEqual(res["episodes"], 50)

if __name__ == "__main__":
    unittest.main()
