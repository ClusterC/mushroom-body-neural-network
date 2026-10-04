import unittest
from src.envs.tic_tac_toe import TicTacToeEnv, PLAYER_X, PLAYER_O
from src.opponents.random_agent import RandomAgent
from src.opponents.heuristic_agent import HeuristicAgent
from src.opponents.minimax_agent import MinimaxAgent

class TestOpponents(unittest.TestCase):
    def setUp(self):
        self.env = TicTacToeEnv()

    def test_random_agent_legal_actions(self):
        agent = RandomAgent(seed=42)
        self.env.reset()
        for _ in range(5):
            if self.env.done:
                break
            action = agent.select_action(self.env)
            self.assertIn(action, self.env.get_legal_actions())
            self.env.step(action)

    def test_heuristic_immediate_win(self):
        agent = HeuristicAgent(seed=42)
        self.env.reset()
        # จัดกระดานให้ X ชนะได้ที่ช่อง 2:
        # X X .
        # O O .
        # . . .
        self.env.board[0] = PLAYER_X
        self.env.board[1] = PLAYER_X
        self.env.board[3] = PLAYER_O
        self.env.board[4] = PLAYER_O
        self.env.current_player = PLAYER_X

        action = agent.select_action(self.env, player=PLAYER_X)
        self.assertEqual(action, 2)

    def test_heuristic_block_opponent(self):
        agent = HeuristicAgent(seed=42)
        self.env.reset()
        # O กำลังจะชนะที่ช่อง 2, X ต้องบล็อกที่ช่อง 2:
        # O O .
        # X . .
        # . . .
        self.env.board[0] = PLAYER_O
        self.env.board[1] = PLAYER_O
        self.env.board[3] = PLAYER_X
        self.env.current_player = PLAYER_X

        action = agent.select_action(self.env, player=PLAYER_X)
        self.assertEqual(action, 2)

    def test_minimax_never_loses_to_random(self):
        minimax = MinimaxAgent()
        random_opponent = RandomAgent(seed=42)
        losses = 0

        for _ in range(20):
            self.env.reset()
            while not self.env.done:
                if self.env.current_player == PLAYER_X:
                    action = minimax.select_action(self.env, player=PLAYER_X)
                else:
                    action = random_opponent.select_action(self.env, player=PLAYER_O)
                self.env.step(action)

            if self.env.winner == PLAYER_O:
                losses += 1

        self.assertEqual(losses, 0)

if __name__ == "__main__":
    unittest.main()
