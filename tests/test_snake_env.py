import unittest
import numpy as np
from src.envs.snake_env import SnakeEnv, UP, RIGHT, DOWN, LEFT

class TestSnakeEnv(unittest.TestCase):
    def setUp(self):
        self.env = SnakeEnv(width=10, height=10, seed=42)

    def test_initial_state_and_visual_shape(self):
        obs = self.env.reset()
        self.assertEqual(obs.shape, (3, 10, 10))
        self.assertEqual(len(self.env.snake), 3)
        self.assertEqual(self.env.score, 0)
        self.assertFalse(self.env.done)

        # Head channel ต้องมี 1.0 จุดเดียว
        self.assertEqual(np.sum(obs[0] == 1.0), 1)
        # Food channel ต้องมี 1.0 จุดเดียว
        self.assertEqual(np.sum(obs[2] == 1.0), 1)
        # Body channel ต้องมี 2 จุด (งูยาว 3 = หัว 1 + ลำตัว 2)
        self.assertEqual(np.sum(obs[1] > 0), 2)

    def test_action_mask_prevents_reverse(self):
        self.env.reset()
        self.env.direction = RIGHT
        mask = self.env.get_action_mask()
        # ถ้าหันขวา ห้ามเลี้ยวซ้าย (180 องศา)
        self.assertTrue(mask[UP])
        self.assertTrue(mask[RIGHT])
        self.assertTrue(mask[DOWN])
        self.assertFalse(mask[LEFT])

    def test_step_and_food_consumption(self):
        self.env.reset()
        # บังคับวางอาหารไว้ข้างหน้าหัวงู 1 ช่อง
        head_r, head_c = self.env.snake[0]
        self.env.food = (head_r, head_c + 1)
        self.env.direction = RIGHT

        obs, reward, done, info = self.env.step(RIGHT)
        self.assertEqual(reward, 1.0)
        self.assertEqual(self.env.score, 1)
        self.assertEqual(len(self.env.snake), 4)  # ความยาวเพิ่มขึ้นเป็น 4

    def test_wall_collision(self):
        self.env.reset()
        # ขยับไปชนกำแพงขวา
        for _ in range(10):
            if self.env.done:
                break
            _, reward, done, info = self.env.step(RIGHT)

        self.assertTrue(self.env.done)
        self.assertEqual(reward, -1.0)
        self.assertEqual(info["cause"], "wall")

if __name__ == "__main__":
    unittest.main()
