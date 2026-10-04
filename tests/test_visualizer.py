import unittest
from src.visualizer.app import VisualizerApp, AVAILABLE_AGENTS
from src.envs.tic_tac_toe import PLAYER_X, PLAYER_O

class TestVisualizer(unittest.TestCase):
    def setUp(self):
        self.app = VisualizerApp(headless=True)

    def test_initialization(self):
        self.assertEqual(len(AVAILABLE_AGENTS), 7)
        self.assertEqual(self.app.get_agent_name(PLAYER_X), "Hippocampal MB")
        self.assertEqual(self.app.get_agent_name(PLAYER_O), "Heuristic")
        self.assertEqual(self.app.stats["total"], 0)

    def test_step_execution(self):
        self.app.reset_game()
        # X เป็น Mushroom Body เดินก้าวแรก
        self.app.execute_step()
        self.assertIsNotNone(self.app.last_move)
        self.assertEqual(self.app.env.board[self.app.last_move], PLAYER_X)
        self.assertIsNotNone(self.app.mb_neural_data["sparse_kc"])
        self.assertIsNotNone(self.app.mb_neural_data["mbon_probs"])

    def test_agent_switching(self):
        # สลับ X เป็น Minimax
        minimax_idx = AVAILABLE_AGENTS.index("Minimax")
        self.app.agent_x_idx = minimax_idx
        self.assertEqual(self.app.get_agent_name(PLAYER_X), "Minimax")

    def test_full_game_playout(self):
        self.app.reset_game()
        for _ in range(9):
            if self.app.env.done:
                break
            self.app.execute_step()

        self.assertTrue(self.app.env.done)
        self.assertEqual(self.app.stats["total"], 1)

    def test_train_button_feedback(self):
        initial_episodes = self.app.total_trained_episodes
        self.app._on_button_click('train_500')
        self.assertEqual(self.app.total_trained_episodes, initial_episodes + 500)
        self.assertIsNotNone(self.app.training_toast)
        self.assertTrue(self.app.toast_timer > 0)
        self.assertNotEqual(self.app.mean_synapse_weight, 0.0)

    def test_self_play_button_feedback(self):
        initial_episodes = self.app.total_trained_episodes
        self.app._on_button_click('train_self_play')
        self.assertEqual(self.app.total_trained_episodes, initial_episodes + 500)
        self.assertIn("SELF-PLAY", self.app.training_toast)

if __name__ == "__main__":
    unittest.main()
