import unittest
import numpy as np
from src.models.hdc_visual_mb import HDCVisualMushroomBody
from src.envs.snake_env import SnakeEnv, UP, RIGHT, DOWN, LEFT

class TestHDCVisualMushroomBody(unittest.TestCase):
    def setUp(self):
        self.model = HDCVisualMushroomBody(
            dim=2048,
            k_active=100,
            num_mbon=4,
            seed=42
        )

    def test_item_memory_orthogonality(self):
        """ตรวจสอบคุณสมบัติ High-Dimensional Orthogonality ของ Basis Hypervectors"""
        keys = list(self.model.item_memory.keys())
        # สุ่มเปรียบเทียบคู่เวกเตอร์ใน Item Memory
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                v1 = self.model.item_memory[keys[i]]
                v2 = self.model.item_memory[keys[j]]
                cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                # ในมิติ 2,048 ค่า Cosine Similarity ระหว่างเวกเตอร์สุ่มควรเกือบเป็น 0 (|cos| < 0.12)
                self.assertLess(
                    abs(cosine_sim), 0.12,
                    f"เวกเตอร์ {keys[i]} และ {keys[j]} ควรตั้งฉากกัน (|cos| < 0.12) แต่ได้ {abs(cosine_sim)}"
                )

    def test_binding_and_bundling(self):
        """ตรวจสอบคุณสมบัติการ Binding (⊗) และ Bundling (+)"""
        v_ahead = self.model.item_memory["DIR_AHEAD"]
        v_food = self.model.item_memory["ENT_FOOD"]
        
        # Binding (Hadamard product)
        bound = self.model.bind(v_ahead, v_food)
        self.assertEqual(bound.shape, (2048,))
        
        # ผลลัพธ์จากการ Binding ต้องตั้งฉากกับเวกเตอร์เดิมทั้งสอง
        sim_ahead = np.dot(bound, v_ahead) / (np.linalg.norm(bound) * np.linalg.norm(v_ahead))
        sim_food = np.dot(bound, v_food) / (np.linalg.norm(bound) * np.linalg.norm(v_food))
        self.assertLess(abs(sim_ahead), 0.12)
        self.assertLess(abs(sim_food), 0.12)

    def test_kc_sparsity_k_wta(self):
        """ตรวจสอบ APL Lateral Inhibition (k-WTA: Top 5% Firing)"""
        dummy_ego = np.zeros(12, dtype=np.float32)
        dummy_ego[6] = 1.0  # Food Ahead
        scene = self.model.encode_scene(dummy_ego)
        sparse_kc = self.model.sparsify_kc(scene)

        active_count = np.sum(sparse_kc > 0)
        self.assertEqual(active_count, 100, f"Kenyon Cells ควรทำงาน 100 จุด (5.0%) แต่ได้ {active_count}")

    def test_forward_action_selection(self):
        """ตรวจสอบ Forward Pass และ Clean-up Memory Decision"""
        env = SnakeEnv(width=10, height=10, seed=42)
        ego = env.get_egocentric_observation()
        mask = env.get_action_mask()
        safe_mask = env.get_safe_action_mask()

        action, probs, kc, sims = self.model.forward(
            ego_obs=ego,
            action_mask=mask,
            cpg_safe_mask=safe_mask,
            direction=env.direction
        )

        self.assertIn(action, [0, 1, 2, 3])
        self.assertEqual(len(probs), 4)
        self.assertEqual(len(sims), 4)
        # ตาเดินที่เลือกต้องอยู่ใน Safe Mask
        self.assertTrue(safe_mask[action])

    def test_three_factor_plasticity(self):
        """ตรวจสอบ Three-Factor Plasticity (Dopamine Bundling)"""
        proto_before = self.model.mbon_prototypes.copy()
        
        # จำลองการเดินและให้รางวัล
        dummy_ego = np.zeros(12, dtype=np.float32)
        dummy_ego[6] = 1.0
        self.model.forward(dummy_ego, direction=RIGHT)
        
        # ปรับปรุงด้วยรางวัลโดปามีน
        self.model.update_plasticity(reward=1.0)
        
        proto_after = self.model.mbon_prototypes.copy()
        diff = np.linalg.norm(proto_after - proto_before)
        self.assertGreater(diff, 0.0, "Prototypes ควรเปลี่ยนแปลงเมื่อได้รับรางวัล Dopamine")

if __name__ == "__main__":
    unittest.main()
