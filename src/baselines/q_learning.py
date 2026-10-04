import numpy as np
from src.envs.tic_tac_toe import EMPTY, PLAYER_X, PLAYER_O

class TabularQLearningAgent:
    """
    Standard Tabular Q-Learning Agent สำหรับเปรียบเทียบเป็น Baseline:
    - เก็บ Q-Table ของสถานะกระดานจากมุมมองของตัวแทน
    - ใช้ ε-greedy Action Selection ร่วมกับ Legal Action Masking
    - อัปเดต Q-values ด้วย Bellman Equation
    """
    def __init__(
        self,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
        seed=None
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        # Q-table: key -> np.zeros(9)
        self.q_table = {}
        # Trajectory history สำหรับอัปเดตย้อนหลังในเทิร์นของตนเอง
        self.trajectory = []

    def get_state_key(self, board, player):
        """
        แปลงสถานะกระดานเป็น Tuple เพื่อใช้เป็น Key ใน Q-table จากมุมมองของ player
        """
        opponent = PLAYER_O if player == PLAYER_X else PLAYER_X
        mapped = []
        for cell in board:
            if cell == EMPTY:
                mapped.append(0)
            elif cell == player:
                mapped.append(1)
            else:
                mapped.append(2)
        return tuple(mapped)

    def get_q_values(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(9, dtype=np.float32)
        return self.q_table[state_key]

    def reset_episode(self):
        self.trajectory.clear()

    def select_action(self, env, training=True, player=None):
        if player is None:
            player = env.current_player

        legal_actions = env.get_legal_actions()
        if not legal_actions:
            raise RuntimeError("ไม่มีช่องที่สามารถเดินได้")

        state_key = self.get_state_key(env.board, player)
        q_vals = self.get_q_values(state_key)

        if training and self.rng.random() < self.epsilon:
            action = int(self.rng.choice(legal_actions))
        else:
            # เลือก Action ที่มีค่า Q สูงสุดเฉพาะใน legal actions
            best_q = -1e9
            best_actions = []
            for a in legal_actions:
                val = q_vals[a]
                if val > best_q:
                    best_q = val
                    best_actions = [a]
                elif np.isclose(val, best_q):
                    best_actions.append(a)
            action = int(self.rng.choice(best_actions))

        if training:
            self.trajectory.append((state_key, action))

        return action

    def update_q_values(self, final_reward):
        """
        ทำการอัปเดต Bellman ย้อนหลังตลอด Trajectory ใน Episode นั้นๆ
        """
        if not self.trajectory:
            return

        # Terminal state value
        next_max_q = 0.0
        target = final_reward

        for i in reversed(range(len(self.trajectory))):
            s_key, a = self.trajectory[i]
            q_vals = self.get_q_values(s_key)
            # Q(s, a) = Q(s, a) + alpha * (target + gamma * next_max_q - Q(s, a))
            q_vals[a] += self.learning_rate * (target + self.gamma * next_max_q - q_vals[a])
            next_max_q = np.max(q_vals)
            target = 0.0  # intermediate rewards are 0

        self.reset_episode()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
