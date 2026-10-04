import numpy as np

EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2

WINNING_COMBINATIONS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
    (0, 4, 8), (2, 4, 6)             # Diagonals
]

class TicTacToeEnv:
    """
    สภาพแวดล้อมจำลองเกม XO (3x3 Tic-Tac-Toe)
    รองรับ Action Masking และ 27-dim One-Hot Observation จากมุมมองของผู้เล่นปัจจุบัน
    """
    def __init__(self):
        self.board = np.zeros(9, dtype=np.int32)
        self.current_player = PLAYER_X
        self.winner = None
        self.done = False

    def reset(self, starting_player=PLAYER_X):
        """
        รีเซ็ตสถานะกระดานสู่จุดเริ่มต้น
        """
        self.board.fill(EMPTY)
        self.current_player = starting_player
        self.winner = None
        self.done = False
        return self.get_observation()

    def get_legal_actions(self):
        """
        ส่งคืนรายการ Index ของช่องที่ยังว่าง (0 ถึง 8)
        """
        if self.done:
            return []
        return [i for i in range(9) if self.board[i] == EMPTY]

    def get_action_mask(self):
        """
        ส่งคืน Boolean mask ขนาด 9 (True = เดินได้, False = เดินไม่ได้)
        """
        if self.done:
            return np.zeros(9, dtype=bool)
        return self.board == EMPTY

    def get_observation(self, player=None):
        """
        ส่งคืน 27-dimensional One-Hot Vector จากมุมมองของ player
        ถ้าไม่ระบุ player จะใช้ self.current_player
        แต่ละช่องประกอบด้วย 3 มิติ: [ช่องว่าง, ตัวเรา, คู่ต่อสู้]
        """
        if player is None:
            player = self.current_player

        opponent = PLAYER_O if player == PLAYER_X else PLAYER_X
        obs = np.zeros((9, 3), dtype=np.float32)

        for i in range(9):
            cell = self.board[i]
            if cell == EMPTY:
                obs[i, 0] = 1.0
            elif cell == player:
                obs[i, 1] = 1.0
            elif cell == opponent:
                obs[i, 2] = 1.0

        return obs.flatten()

    def check_winner(self):
        """
        ตรวจสอบผู้ชนะจากกระดานปัจจุบัน
        ส่งคืน PLAYER_X, PLAYER_O, 'DRAW', หรือ None (ยังไม่จบ)
        """
        for a, b, c in WINNING_COMBINATIONS:
            if self.board[a] != EMPTY and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]

        if np.all(self.board != EMPTY):
            return "DRAW"

        return None

    def step(self, action):
        """
        ดำเนินการเดินในช่อง action (0-8)
        ส่งคืน (observation, reward, done, info)
        """
        if self.done:
            raise RuntimeError("เกมจบแล้ว ต้องเรียก reset() ก่อนเริ่มเกมใหม่")

        if action < 0 or action > 8 or self.board[action] != EMPTY:
            raise ValueError(f"Action {action} ไม่ถูกต้องหรือไม่ใช่ช่องว่าง")

        mover = self.current_player
        self.board[action] = mover

        outcome = self.check_winner()
        reward = 0.0

        if outcome is not None:
            self.done = True
            self.winner = outcome
            if outcome == "DRAW":
                reward = 0.0
            elif outcome == mover:
                reward = 1.0
        else:
            self.current_player = PLAYER_O if mover == PLAYER_X else PLAYER_X

        info = {
            "mover": mover,
            "winner": self.winner,
            "legal_actions": self.get_legal_actions()
        }

        return self.get_observation(), reward, self.done, info

    def render(self):
        """
        แสดงผลกระดานในรูปแบบข้อความ
        """
        symbols = {EMPTY: ".", PLAYER_X: "X", PLAYER_O: "O"}
        rows = []
        for r in range(3):
            row_str = " ".join(symbols[self.board[r * 3 + c]] for c in range(3))
            rows.append(row_str)
        return "\n".join(rows)
