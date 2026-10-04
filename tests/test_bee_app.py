import unittest
from src.visualizer.bee_app import BeeVisualizerApp

class TestBeeVisualizerApp(unittest.TestCase):
    def setUp(self):
        self.app = BeeVisualizerApp(headless=True)

    def test_initialization(self):
        """ตรวจสอบการกำหนดค่าเริ่มต้นในโหมด Headless"""
        self.assertIsNotNone(self.app.env)
        self.assertIsNotNone(self.app.mb)
        self.assertEqual(len(self.app.kc_points), 2500)
        self.assertEqual(self.app.env.meadow_size, 20.0)

    def test_step_simulation(self):
        """ทดสอบการทำงานของ step_simulation ใน Visualizer"""
        initial_step = self.app.env.step_count
        self.app.step_simulation()
        self.assertEqual(self.app.env.step_count, initial_step + 1)
        self.assertIn(self.app.last_action, [0, 1, 2, 3, 4])
        self.assertIsNotNone(self.app.obs)

    def test_quick_train_episodes(self):
        """ทดสอบการเรียกใช้งาน train_episodes"""
        initial_nectar = self.app.env.total_hive_nectar
        # ทดสอบฝึก 2 เที่ยวบิน
        self.app.train_episodes(n_trips=2)
        self.assertIsNotNone(self.app.mb.get_diagnostics())

if __name__ == "__main__":
    unittest.main()
