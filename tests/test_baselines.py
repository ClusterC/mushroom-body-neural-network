import unittest
import numpy as np
from src.envs.tic_tac_toe import TicTacToeEnv, PLAYER_X
from src.baselines.q_learning import TabularQLearningAgent

class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()
        self.agent = TabularQLearningAgent(seed=42)

    def test_state_key_and_action_mask(self):
        self.env.reset()
        self.env.step(4)  # ช่อง 4 โดนครอบครอง
        action = self.agent.select_action(self.env, training=False, player=PLAYER_X)
        self.assertNotEqual(action, 4)
        self.assertIn(action, self.env.get_legal_actions())

    def test_q_table_update(self):
        self.env.reset()
        self.agent.reset_episode()
        action = self.agent.select_action(self.env, training=True, player=PLAYER_X)
        self.assertEqual(len(self.agent.trajectory), 1)

        self.agent.update_q_values(final_reward=1.0)
        # ตรวจสอบว่าใน Q-table มีการอัปเดตค่ามากกว่า 0
        state_key = self.agent.get_state_key(self.env.board, PLAYER_X)
        self.assertIn(state_key, self.agent.q_table)
        self.assertTrue(self.agent.q_table[state_key][action] > 0.0)

if __name__ == "__main__":
    unittest.main()
