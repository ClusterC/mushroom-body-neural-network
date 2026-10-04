import unittest
from src.visualizer.snake_app import SnakeVisualizerApp

class TestSnakeVisualizer(unittest.TestCase):
    def setUp(self):
        self.app = SnakeVisualizerApp(headless=True)

    def test_initialization(self):
        self.assertIsNotNone(self.app.env)
        self.assertIsNotNone(self.app.mb)
        self.assertEqual(self.app.total_trained_episodes, 300)
        self.assertEqual(len(self.app.kc_points), 2000)

    def test_execute_step(self):
        self.app.reset_game()
        self.app.execute_step()
        self.assertIsNotNone(self.app.sparse_kc)
        self.assertIsNotNone(self.app.mbon_probs)
        self.assertIsNotNone(self.app.selected_action)

    def test_train_batch(self):
        res = self.app._train_episodes(episodes=20)
        self.assertEqual(res["episodes"], 20)
        self.assertIn("mean_weight", res)

if __name__ == "__main__":
    unittest.main()
