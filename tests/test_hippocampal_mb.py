import unittest
import numpy as np
from src.models.hippocampal_hdc_mb import HippocampalHDCVisualMB, UP, RIGHT, DOWN, LEFT

class TestHippocampalHDCVisualMB(unittest.TestCase):
    def setUp(self):
        self.dim = 2048
        self.k_dg = 50
        self.k_ca3 = 120
        self.model = HippocampalHDCVisualMB(
            dim=self.dim,
            k_dg=self.k_dg,
            k_ca3=self.k_ca3,
            seed=42
        )

    def test_initialization(self):
        """ตรวจสอบมิติและการเตรียม Basis Hypervectors"""
        self.assertEqual(self.model.dim, 2048)
        self.assertEqual(self.model.k_dg, 50)
        self.assertEqual(self.model.mbon_prototypes.shape, (4, 2048))
        self.assertIn("DIR_AHEAD", self.model.item_memory)
        self.assertIn("ENT_FOOD", self.model.item_memory)
        self.assertEqual(len(self.model.item_memory["DIR_AHEAD"]), 2048)

    def test_dentate_gyrus_pattern_separation(self):
        """
        ทดสอบกลไก Dentate Gyrus Pattern Separation:
        สถานะที่คล้ายกันในระดับ Input เมื่อผ่าน DG Ultra-sparse k-WTA
        จะต้องมี Cosine Similarity ลดลงอย่างชัดเจน (Orthogonalization)
        """
        # สถานะที่ 1: อาหารอยู่ข้างหน้า มีกำแพงขวางอยู่ประชิดด้านหน้า (Wall Threat)
        ego_1 = np.zeros(12, dtype=np.float32)
        ego_1[0] = 0.1  # Wall ahead very close (threat = 1.0 - 0.1 = 0.9 > 0.6)
        ego_1[6] = 1.0  # Food ahead

        # สถานะที่ 2: อาหารอยู่ข้างหน้าเหมือนกัน แต่พื้นที่ข้างหน้าโล่ง (ไม่มีกำแพง)
        ego_2 = np.zeros(12, dtype=np.float32)
        ego_2[0] = 0.9  # Wall ahead far away (threat = 1.0 - 0.9 = 0.1 <= 0.6)
        ego_2[6] = 1.0  # Food ahead

        # เวกเตอร์ก่อนเข้า DG (Entorhinal EC)
        ec_1 = self.model.encode_entorhinal(ego_1)
        ec_2 = self.model.encode_entorhinal(ego_2)
        
        sim_ec = np.dot(ec_1, ec_2) / (np.linalg.norm(ec_1) * np.linalg.norm(ec_2))

        # เวกเตอร์หลังผ่าน Dentate Gyrus
        dg_1 = self.model.dentate_gyrus_pattern_separation(ec_1)
        dg_2 = self.model.dentate_gyrus_pattern_separation(ec_2)

        sim_dg = np.dot(dg_1, dg_2) / (np.linalg.norm(dg_1) * np.linalg.norm(dg_2))

        # ค่าความคล้ายคลึงใน DG ต้องลดลง หรืออยู่ในระดับตั้งฉาก
        self.assertLess(sim_dg, sim_ec)
        # ตรวจสอบความเบาบาง (Sparsity) ของ DG ต้องมีจำนวนเซลล์ที่ยิงสัญญาณเท่ากับ k_dg พอดี
        active_cells = np.count_nonzero(dg_1)
        self.assertEqual(active_cells, self.k_dg)

    def test_ca3_temporal_permutation_sequence(self):
        """
        ทดสอบ Temporal Permutation Sequence Memory (Π):
        การมีประวัติศาสตร์การเดิน จะทำให้ Trajectory Vector ต่างจากสถานะโดดเดี่ยว
        """
        ego = np.zeros(12, dtype=np.float32)
        ego[6] = 1.0  # Food ahead

        ec = self.model.encode_entorhinal(ego)
        dg = self.model.dentate_gyrus_pattern_separation(ec)
        
        # ก้าวที่ 1 (ยังไม่มีประวัติศาสตร์)
        ca3_1, traj_1 = self.model.ca3_pattern_completion_and_sequence(dg)

        # ก้าวที่ 2 (มีประวัติศาสตร์ 1 ก้าว)
        ca3_2, traj_2 = self.model.ca3_pattern_completion_and_sequence(dg)

        # traj_1 กับ traj_2 ต้องไม่เหมือนกันเป๊ะ เพราะ traj_2 รวมข้อมูลอดีตผ่าน Π
        sim = np.dot(traj_1, traj_2)
        self.assertLess(sim, 0.99)

    def test_sharp_wave_ripple_replay(self):
        """
        ทดสอบกลไก Sharp-Wave Ripple (SWR) Episodic Replay:
        เมื่อจบ Episode และมีรางวัล การ Replay ย้อนหลังต้องส่งผลให้ Prototypes ปรับปรุง
        """
        ego = np.zeros(12, dtype=np.float32)
        ego[6] = 1.0  # Food ahead

        initial_proto = self.model.mbon_prototypes.copy()

        # เดิน 3 ก้าว
        for _ in range(3):
            self.model.forward(ego, direction=RIGHT)
            self.model.update(reward=0.1, done=False)

        # ก้าวสุดท้ายได้กินอาหาร (Reward = +5.0)
        self.model.forward(ego, direction=RIGHT)
        self.model.update(reward=5.0, done=True)

        # น้ำหนักต้องเปลี่ยนไปจากจุดเริ่มต้นอย่างมีนัยสำคัญ
        diff = np.linalg.norm(self.model.mbon_prototypes - initial_proto)
        self.assertGreater(diff, 0.05)
        # ตรวจสอบว่า Episode Buffer และ History ถูกรีเซ็ตหลัง SWR ทำงาน
        self.assertEqual(len(self.model.episode_buffer), 0)
        self.assertEqual(len(self.model.temporal_history), 0)

    def test_forward_with_cpg_mask(self):
        """
        ทดสอบการเลือก Action ร่วมกับ CPG Safe Mask
        """
        ego = np.zeros(12, dtype=np.float32)
        ego[6] = 1.0  # Food ahead (ถ้าเดินหน้าจะได้กิน)

        # สั่ง Mask ห้ามเดินตรง (UP)
        cpg_mask = np.array([False, True, True, True], dtype=bool)

        action, probs, combined, similarities = self.model.forward(
            ego,
            cpg_safe_mask=cpg_mask,
            direction=UP,
            deterministic=True
        )

        # ต้องไม่เลือก Action 0 (UP)
        self.assertNotEqual(action, 0)
        self.assertEqual(probs[0], 0.0)

    def test_export_and_load_weights(self):
        """ทดสอบการบันทึกและโหลดน้ำหนัก"""
        data = self.model.export_weights()
        self.assertIn("mbon_prototypes", data)
        self.assertIn("item_memory", data)

        new_model = HippocampalHDCVisualMB(dim=2048, seed=999)
        new_model.load_weights(data)

        np.testing.assert_allclose(
            new_model.mbon_prototypes,
            self.model.mbon_prototypes,
            atol=1e-5
        )

if __name__ == "__main__":
    unittest.main()
