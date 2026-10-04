import unittest
import numpy as np
from src.envs.tic_tac_toe import TicTacToeEnv, EMPTY, PLAYER_X, PLAYER_O

class TestTicTacToeEnv(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()

    def test_initial_state(self):
        obs = self.env.reset()
        self.assertEqual(len(self.env.get_legal_actions()), 9)
        self.assertTrue(np.all(self.env.get_action_mask()))
        self.assertEqual(obs.shape, (27,))
        # ตรวจสอบว่าทุกช่องเริ่มต้นเป็นสถานะว่าง [1, 0, 0]
        obs_reshaped = obs.reshape(9, 3)
        for i in range(9):
            np.testing.assert_array_equal(obs_reshaped[i], [1.0, 0.0, 0.0])

    def test_step_and_masking(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(4)  # X เดินกลาง
        self.assertFalse(done)
        self.assertEqual(reward, 0.0)
        self.assertNotIn(4, self.env.get_legal_actions())
        mask = self.env.get_action_mask()
        self.assertFalse(mask[4])
        self.assertEqual(np.sum(mask), 8)

    def test_invalid_move(self):
        self.env.reset()
        self.env.step(0)
        with self.assertRaises(ValueError):
            self.env.step(0)  # เดินซ้ำช่องเดิม

    def test_horizontal_win(self):
        self.env.reset()
        # X: 0, O: 3, X: 1, O: 4, X: 2 (X ชนะแถวบน)
        self.env.step(0)
        self.env.step(3)
        self.env.step(1)
        self.env.step(4)
        _, reward, done, info = self.env.step(2)
        self.assertTrue(done)
        self.assertEqual(self.env.winner, PLAYER_X)
        self.assertEqual(reward, 1.0)

    def test_draw_condition(self):
        self.env.reset()
        # กระดานเสมอ:
        # X O X
        # X O O
        # O X X
        moves = [0, 1, 2, 4, 3, 5, 7, 6, 8]
        for m in moves[:-1]:
            self.env.step(m)
        _, reward, done, info = self.env.step(moves[-1])
        self.assertTrue(done)
        self.assertEqual(self.env.winner, "DRAW")
        self.assertEqual(reward, 0.0)

if __name__ == "__main__":
    unittest.main()
