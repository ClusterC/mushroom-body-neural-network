import unittest
import numpy as np
from src.envs.bee_foraging_env import BeeForagingEnv, Flower

class TestBeeForagingEnv(unittest.TestCase):
    def setUp(self):
        self.env = BeeForagingEnv(meadow_size=20.0, max_steps=100, seed=42)

    def test_initialization(self):
        """ตรวจสอบสถานะเริ่มต้นของทุ่งดอกไม้และตัวผึ้ง"""
        self.assertEqual(len(self.env.flowers), 24)
        self.assertEqual(self.env.meadow_size, 20.0)
        self.assertAlmostEqual(self.env.hive_pos[0], 10.0)
        self.assertAlmostEqual(self.env.hive_pos[1], 10.0)
        
        # ตรวจสอบการกระจายสปีชีส์ดอกไม้
        species_counts = [0, 0, 0, 0]
        for f in self.env.flowers:
            species_counts[f.species] += 1
        self.assertEqual(species_counts, [6, 6, 6, 6])

    def test_observation_shape_and_bounds(self):
        """ตรวจสอบขนาดและขอบเขตของเวกเตอร์สังเกตการณ์พหุสัมผัส (36 มิติ)"""
        obs = self.env.reset(seed=42)
        self.assertEqual(obs.shape, (36,))
        self.assertTrue(np.all(obs >= 0.0), f"พบค่าติดลบใน obs: {obs[obs < 0.0]}")
        self.assertTrue(np.all(obs <= 1.0), f"พบค่าเกิน 1.0 ใน obs: {obs[obs > 1.0]}")

    def test_forward_movement(self):
        """ทดสอบการบินไปข้างหน้า (Action 0: FORWARD)"""
        self.env.reset(seed=42)
        initial_pos = np.copy(self.env.bee_pos)
        self.env.bee_heading = 0.0  # หันหน้าไปทางทิศตะวันออก (แกน +X)
        
        obs, reward, done, info = self.env.step(0)
        self.assertGreater(self.env.bee_pos[0], initial_pos[0])
        self.assertAlmostEqual(self.env.bee_pos[1], initial_pos[1], delta=0.01)
        self.assertEqual(info["action_name"], "FORWARD")

    def test_turn_angles(self):
        """ทดสอบการหันเลี้ยวซ้ายและขวา"""
        self.env.reset(seed=42)
        self.env.bee_heading = 90.0
        
        # เลี้ยวซ้าย (+45°)
        self.env.step(1)
        self.assertAlmostEqual(self.env.bee_heading, 135.0)
        
        # เลี้ยวขวา (-45°)
        self.env.step(2)
        self.assertAlmostEqual(self.env.bee_heading, 90.0)

    def test_land_and_feed_mechanism(self):
        """ทดสอบการลงตอมดูดน้ำหวานจากดอกไม้"""
        self.env.reset(seed=42)
        target_flower = self.env.flowers[0]  # Lavender
        self.env.bee_pos = np.array([target_flower.x, target_flower.y], dtype=np.float32)
        
        initial_crop = self.env.crop_nectar
        obs, reward, done, info = self.env.step(3)  # LAND_AND_FEED
        
        self.assertGreater(self.env.crop_nectar, initial_crop)
        self.assertGreater(reward, 0.0)  # ได้รับรางวัล Octopamine
        self.assertIn("FED_", info["event"])

    def test_toxic_flower_punishment(self):
        """ทดสอบการได้รับบทลงโทษเมื่อลงตอมดอกไม้พิษ (Species 3)"""
        self.env.reset(seed=42)
        # หาดอกไม้พิษ
        toxic_flower = next(f for f in self.env.flowers if f.species == 3)
        self.env.bee_pos = np.array([toxic_flower.x, toxic_flower.y], dtype=np.float32)
        
        obs, reward, done, info = self.env.step(3)  # LAND_AND_FEED
        self.assertLess(reward, -0.5)  # ถูกลงโทษ
        self.assertEqual(info["event"], "POISONED")

    def test_return_to_hive_and_unload(self):
        """ทดสอบการนำน้ำหวานกลับมาถ่ายทอดสู่รังผึ้ง"""
        self.env.reset(seed=42)
        self.env.crop_nectar = 3.0  # มีน้ำหวานในกระเพาะ
        self.env.bee_pos = np.array([10.5, 10.5], dtype=np.float32)  # ใกล้รัง
        
        obs, reward, done, info = self.env.step(4)  # RETURN_TO_HIVE
        self.assertEqual(self.env.crop_nectar, 0.0)  # ถ่ายทอดน้ำหวานหมดกระเพาะ
        self.assertEqual(self.env.total_hive_nectar, 3.0)
        self.assertEqual(self.env.trips_completed, 1)
        self.assertGreater(reward, 5.0)  # รางวัลมหาศาลจากการนำอาหารกลับรัง

    def test_flower_finite_depletion(self):

        """ทดสอบว่าดอกไม้มีน้ำหวานจำกัด เมื่อดูดจนหมดจะกลายเป็น Depleted และไม่ได้น้ำหวานอีก"""
        self.env.reset(seed=42)
        target_flower = self.env.flowers[0]  # Lavender (มี 3.0 หน่วย)
        self.env.bee_pos = np.array([target_flower.x, target_flower.y], dtype=np.float32)
        
        # ดูดครั้งที่ 1 (ได้ 1.5 หน่วย)
        self.env.step(3)
        self.assertFalse(target_flower.is_depleted)
        
        # ถ่ายน้ำหวานออกชั่วคราวเพื่อดูดต่อ
        self.env.crop_nectar = 0.0
        
        # ดูดครั้งที่ 2 (ได้อีก 1.5 หน่วย -> หมดเกลี้ยง 3.0 หน่วย)
        self.env.step(3)
        self.assertTrue(target_flower.is_depleted)
        self.assertAlmostEqual(target_flower.current_nectar, 0.0)
        
        # ดูดครั้งที่ 3 (ดอกไม้หมดแล้ว ต้องไม่ได้น้ำหวานและเกิดเหตุการณ์ EMPTY_FLOWER)
        prev_crop = self.env.crop_nectar
        obs, reward, done, info = self.env.step(3)
        self.assertEqual(self.env.crop_nectar, prev_crop)
        self.assertEqual(info["event"], "EMPTY_FLOWER")
        self.assertLess(reward, 0.0)

if __name__ == "__main__":
    unittest.main()

