from src.envs.tic_tac_toe import EMPTY, PLAYER_X, PLAYER_O, WINNING_COMBINATIONS

class MinimaxAgent:
    """
    คู่ต่อสู้แบบ Minimax (Optimal Play):
    คำนวณคะแนนทุกกิ่งของเกมต้นไม้ รับประกันการเล่นที่ดีที่สุด ไม่มีวันแพ้
    ใช้ Memoization เพื่อความรวดเร็วระดับมิลลิวินาที
    """
    def __init__(self):
        self.memo = {}

    def _check_terminal(self, board):
        for a, b, c in WINNING_COMBINATIONS:
            if board[a] != EMPTY and board[a] == board[b] == board[c]:
                return board[a]
        if all(cell != EMPTY for cell in board):
            return "DRAW"
        return None

    def _minimax(self, board, player, maximizing_player):
        state_key = (tuple(board), player, maximizing_player)
        if state_key in self.memo:
            return self.memo[state_key]

        outcome = self._check_terminal(board)
        opponent = PLAYER_O if player == PLAYER_X else PLAYER_X
        maximizer_opponent = PLAYER_O if maximizing_player == PLAYER_X else PLAYER_X

        if outcome is not None:
            if outcome == maximizing_player:
                score = 10
            elif outcome == maximizer_opponent:
                score = -10
            else:
                score = 0
            return score, None

        legal_actions = [i for i in range(9) if board[i] == EMPTY]

        if player == maximizing_player:
            best_score = -999
            best_action = legal_actions[0]
            for action in legal_actions:
                board[action] = player
                score, _ = self._minimax(board, opponent, maximizing_player)
                board[action] = EMPTY
                if score > best_score:
                    best_score = score
                    best_action = action
            self.memo[state_key] = (best_score, best_action)
            return best_score, best_action
        else:
            best_score = 999
            best_action = legal_actions[0]
            for action in legal_actions:
                board[action] = player
                score, _ = self._minimax(board, opponent, maximizing_player)
                board[action] = EMPTY
                if score < best_score:
                    best_score = score
                    best_action = action
            self.memo[state_key] = (best_score, best_action)
            return best_score, best_action

    def select_action(self, env, player=None):
        if player is None:
            player = env.current_player
        board_list = list(env.board)
        _, action = self._minimax(board_list, player, player)
        return action
