import numpy as np
from src.envs.tic_tac_toe import EMPTY, PLAYER_X, PLAYER_O, WINNING_COMBINATIONS

class HeuristicAgent:
    """
    คู่ต่อสู้แบบ Rule-based Heuristic:
    1. หากมีตาเดินที่ทำให้ชนะทันที -> เดินช่องนั้น
    2. หากคู่ต่อสู้กำลังจะชนะในตาถัดไป -> เดินบล็อกช่องนั้น
    3. หากช่องกึ่งกลาง (4) ว่าง -> เดินช่องกึ่งกลาง
    4. หากมีช่องมุม (0, 2, 6, 8) ว่าง -> สุ่มเดินช่องมุม
    5. เดินช่องขอบที่เหลือ (1, 3, 5, 7)
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _check_winning_move(self, board, player, legal_actions):
        for action in legal_actions:
            board[action] = player
            is_win = False
            for a, b, c in WINNING_COMBINATIONS:
                if board[a] == board[b] == board[c] == player:
                    is_win = True
                    break
            board[action] = EMPTY
            if is_win:
                return action
        return None

    def select_action(self, env, player=None):
        if player is None:
            player = env.current_player
        opponent = PLAYER_O if player == PLAYER_X else PLAYER_X

        legal_actions = env.get_legal_actions()
        if not legal_actions:
            raise RuntimeError("ไม่มีช่องที่สามารถเดินได้")

        board_copy = env.board.copy()

        # 1. จังหวะชนะทันที
        win_action = self._check_winning_move(board_copy, player, legal_actions)
        if win_action is not None:
            return win_action

        # 2. จังหวะบล็อกคู่ต่อสู้
        block_action = self._check_winning_move(board_copy, opponent, legal_actions)
        if block_action is not None:
            return block_action

        # 3. เลือกช่องกึ่งกลาง
        if 4 in legal_actions:
            return 4

        # 4. เลือกช่องมุม
        corners = [c for c in [0, 2, 6, 8] if c in legal_actions]
        if corners:
            return int(self.rng.choice(corners))

        # 5. เลือกช่องขอบที่เหลือ
        edges = [e for e in [1, 3, 5, 7] if e in legal_actions]
        if edges:
            return int(self.rng.choice(edges))

        return int(self.rng.choice(legal_actions))
