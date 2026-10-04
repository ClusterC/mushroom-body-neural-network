import unittest
import numpy as np
from src.models.bee_mushroom_body import BeeMushroomBody

class TestBeeMushroomBody(unittest.TestCase):
    def setUp(self):
        self.mb = BeeMushroomBody(n_pn=36, n_kc=2500, n_mbon=5, sparsity_ratio=0.05, seed=42)

    def test_calyx_zoning_and_projections(self):
        """ตรวจสอบสัดส่วนการแบ่งโซน Calyx 3 โซนของผึ้ง"""
        self.assertEqual(self.mb.n_lip, 800)
        self.assertEqual(self.mb.n_collar, 1000)
        self.assertEqual(self.mb.n_basal, 700)
        self.assertEqual(self.mb.W_pn_kc.shape, (36, 2500))
        self.assertEqual(self.mb.W_kc_mbon.shape, (2500, 5))

    def test_sparsity_and_k_wta(self):
        """ตรวจสอบความเบาบางของสัญญาณ 5% (125 เซลล์ที่ทำงาน)"""
        obs = np.random.uniform(0.1, 0.9, size=36).astype(np.float32)
        action, probs, kc_act = self.mb.forward(obs)
        
        active_count = np.sum(kc_act > 0)
        self.assertEqual(active_count, 125, f"จำนวน KC ที่ทำงานควรเป็น 125 แต่ได้ {active_count}")
        
        # ตรวจสอบการแบ่งตามโซน (Lip: 40, Collar: 50, Basal: 35)
        lip_active = np.sum(kc_act[0:800] > 0)
        collar_active = np.sum(kc_act[800:1800] > 0)
        basal_active = np.sum(kc_act[1800:2500] > 0)
        
        self.assertEqual(lip_active, 40)
        self.assertEqual(collar_active, 50)
        self.assertEqual(basal_active, 35)

    def test_mbon_probability_distribution(self):
        """ตรวจสอบการแจกแจงความน่าจะเป็นของ MBONs 5 การกระทำ"""
        obs = np.random.uniform(0.1, 0.9, size=36).astype(np.float32)
        action, probs, _ = self.mb.forward(obs, temperature=0.5)
        
        self.assertEqual(len(probs), 5)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)
        self.assertIn(action, [0, 1, 2, 3, 4])

    def test_octopamine_positive_plasticity(self):
        """ทดสอบการเพิ่มขึ้นของค่าน้ำหนัก Synapse เมื่อได้รับรางวัล Octopamine"""
        obs = np.random.uniform(0.1, 0.9, size=36).astype(np.float32)
        action, _, kc_act = self.mb.forward(obs)
        
        w_before = float(np.sum(self.mb.W_kc_mbon))
        self.mb.update_plasticity(reward=1.5)  # Octopamine surge!
        w_after = float(np.sum(self.mb.W_kc_mbon))
        
        self.assertGreater(w_after, w_before)
        self.assertGreater(self.mb.last_octopamine, 0.0)
        self.assertEqual(self.mb.last_dopamine, 0.0)

    def test_dopamine_negative_plasticity(self):
        """ทดสอบการลดลงของค่าน้ำหนัก Synapse เมื่อถูกลงโทษด้วย Dopamine (Aversive)"""
        obs = np.random.uniform(0.1, 0.9, size=36).astype(np.float32)
        action, _, kc_act = self.mb.forward(obs)
        
        # เพิ่มค่าน้ำหนักให้สูงก่อน
        self.mb.W_kc_mbon += 0.5
        w_before = float(np.sum(self.mb.W_kc_mbon))
        
        self.mb.update_plasticity(reward=-1.0)  # Dopamine depression
        w_after = float(np.sum(self.mb.W_kc_mbon))
        
        self.assertLess(w_after, w_before)
        self.assertEqual(self.mb.last_octopamine, 0.0)
        self.assertGreater(self.mb.last_dopamine, 0.0)
        
        # ตรวจสอบ Dale's Principle: ค่าน้ำหนักต้องไม่ติดลบ
        self.assertTrue(np.all(self.mb.W_kc_mbon >= 0.0))

if __name__ == "__main__":
    unittest.main()
