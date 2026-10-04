import copy
import numpy as np
from src.envs.tic_tac_toe import TicTacToeEnv, PLAYER_X, PLAYER_O
from src.opponents.minimax_agent import MinimaxAgent

class SelfPlayTrainer:
    """
    ระบบการฝึกฝนแบบ Self-Play ร่วมกับ Adversarial Curriculum สำหรับ Mushroom Body:
    - ให้ Mushroom Body เล่นแข่งกับตนเอง (หรือ Snapshot ในอดีต)
    - สะสม Eligibility Traces แยกกันระหว่างผู้เล่นเดินก่อน (X) และเดินทีหลัง (O)
    - ผสมผสานคู่ต่อสู้แบบ Noisy Minimax เพื่อปลูกฝังพื้นฐาน Game Theory ที่ถูกต้อง
    """
    def __init__(self, mb_agent, snapshot_pool_size=5, minimax_mix_ratio=0.25, seed=None):
        self.mb_agent = mb_agent
        self.snapshot_pool_size = snapshot_pool_size
        self.minimax_mix_ratio = minimax_mix_ratio
        self.rng = np.random.default_rng(seed)

        # เก็บ Pool ของน้ำหนัก Synapse ในอดีต
        self.snapshot_pool = [copy.deepcopy(mb_agent.w_kc_mbon)]
        self.minimax = MinimaxAgent()
        self.env = TicTacToeEnv()

    def add_snapshot(self):
        """
        บันทึก Snapshot ค่าน้ำหนักปัจจุบันเข้าสู่ Pool
        """
        if len(self.snapshot_pool) >= self.snapshot_pool_size:
            self.snapshot_pool.pop(0)
        self.snapshot_pool.append(copy.deepcopy(self.mb_agent.w_kc_mbon))

    def _select_action_from_weights(self, weights, obs, legal_mask, temperature):
        sparse_kc = self.mb_agent.encode_kc(obs)
        mbon_activation = np.dot(sparse_kc, weights)
        masked_activation = np.where(legal_mask, mbon_activation, -1e9)

        shift_act = (masked_activation - np.max(masked_activation)) / max(temperature, 1e-4)
        exp_act = np.exp(shift_act)
        probs = exp_act / np.sum(exp_act)

        legal_indices = np.where(legal_mask)[0]
        legal_probs = probs[legal_indices]
        prob_sum = np.sum(legal_probs)
        if prob_sum <= 0 or np.isnan(prob_sum):
            legal_probs = np.ones(len(legal_indices)) / len(legal_indices)
        else:
            legal_probs = legal_probs / prob_sum

        action = int(self.rng.choice(legal_indices, p=legal_probs))
        return action, sparse_kc

    def train_episode(self):
        """
        จำลองการแข่งขัน 1 Episode และอัปเดต Synaptic Weights
        """
        self.env.reset()
        use_minimax = (self.rng.random() < self.minimax_mix_ratio)

        # สุ่มเลือกว่าโมเดลหลัก (Active MB) จะเล่นเป็น Player X หรือ Player O
        mb_is_x = bool(self.rng.random() < 0.5)
        active_player = PLAYER_X if mb_is_x else PLAYER_O
        opponent_player = PLAYER_O if mb_is_x else PLAYER_X

        # หากเล่นกับ Snapshot ให้สุ่มน้ำหนักจาก Pool
        opp_weights = self.snapshot_pool[self.rng.integers(0, len(self.snapshot_pool))]

        # Traces แยกสำหรับ Player X และ Player O
        trace_x = np.zeros((self.mb_agent.num_kc, self.mb_agent.num_mbon), dtype=np.float32)
        trace_o = np.zeros((self.mb_agent.num_kc, self.mb_agent.num_mbon), dtype=np.float32)

        while not self.env.done:
            current = self.env.current_player
            legal_mask = self.env.get_action_mask()
            obs = self.env.get_observation(current)

            if current == active_player:
                # Active MB ทำการเลือกก้าวเดิน
                action, sparse_kc = self._select_action_from_weights(
                    self.mb_agent.w_kc_mbon, obs, legal_mask, self.mb_agent.temperature
                )
                y_mbon = np.zeros(self.mb_agent.num_mbon, dtype=np.float32)
                y_mbon[action] = 1.0

                if current == PLAYER_X:
                    trace_x = self.mb_agent.gamma * self.mb_agent.lambda_trace * trace_x + np.outer(sparse_kc, y_mbon)
                else:
                    trace_o = self.mb_agent.gamma * self.mb_agent.lambda_trace * trace_o + np.outer(sparse_kc, y_mbon)

            else:
                # คู่ต่อสู้ทำการเลือกก้าวเดิน
                if use_minimax:
                    # Noisy Minimax (สุ่ม 20% เพื่อเปิดโอกาสให้เกิดสถานการณ์หลากหลาย)
                    if self.rng.random() < 0.20:
                        actions = self.env.get_legal_actions()
                        action = int(self.rng.choice(actions))
                    else:
                        action = self.minimax.select_action(self.env, player=opponent_player)
                else:
                    # Snapshot MB
                    action, sparse_kc = self._select_action_from_weights(
                        opp_weights, obs, legal_mask, self.mb_agent.temperature
                    )
                    y_mbon = np.zeros(self.mb_agent.num_mbon, dtype=np.float32)
                    y_mbon[action] = 1.0

                    if current == PLAYER_X:
                        trace_x = self.mb_agent.gamma * self.mb_agent.lambda_trace * trace_x + np.outer(sparse_kc, y_mbon)
                    else:
                        trace_o = self.mb_agent.gamma * self.mb_agent.lambda_trace * trace_o + np.outer(sparse_kc, y_mbon)

            self.env.step(action)

        # ประเมิน Dopamine Rewards สำหรับทั้งสองมุมมอง
        winner = self.env.winner
        if winner == PLAYER_X:
            reward_x = 1.0
            reward_o = -1.0
        elif winner == PLAYER_O:
            reward_x = -1.0
            reward_o = 1.0
        else:
            # Draw: ให้รางวัลความสามารถในการป้องกัน +0.3
            reward_x = 0.3
            reward_o = 0.3

        # อัปเดตค่าน้ำหนัก Synapse ของ Active MB
        if active_player == PLAYER_X:
            active_trace = trace_x
            active_reward = reward_x
        else:
            active_trace = trace_o
            active_reward = reward_o

        delta_w = self.mb_agent.learning_rate * active_trace * active_reward
        self.mb_agent.w_kc_mbon = np.clip(self.mb_agent.w_kc_mbon + delta_w, 0.0, self.mb_agent.w_max)

        # หากเล่นแบบ Self-Play (ไม่ใช่ Minimax) อัปเดตน้ำหนักของฝั่งตรงข้ามด้วยเพื่อเพิ่มความเข้มข้นของการเรียนรู้ร่วม
        if not use_minimax:
            opp_trace = trace_o if active_player == PLAYER_X else trace_x
            opp_reward = reward_o if active_player == PLAYER_X else reward_x
            delta_opp = self.mb_agent.learning_rate * opp_trace * opp_reward
            self.mb_agent.w_kc_mbon = np.clip(self.mb_agent.w_kc_mbon + delta_opp, 0.0, self.mb_agent.w_max)

        self.mb_agent.decay_temperature()

        return {
            "winner": winner,
            "active_player": active_player,
            "active_won": (winner == active_player),
            "is_draw": (winner == "DRAW")
        }

def train_self_play(mb_agent, episodes=500, snapshot_interval=100, minimax_mix_ratio=0.25):
    """
    ฟังก์ชันหลักสำหรับดำเนินการฝึกฝนแบบ Self-Play ในจำนวน episodes ที่กำหนด
    """
    w_before = float(np.mean(mb_agent.w_kc_mbon))
    trainer = SelfPlayTrainer(mb_agent, minimax_mix_ratio=minimax_mix_ratio)

    draw_count = 0
    active_win_count = 0

    for ep in range(episodes):
        res = trainer.train_episode()
        if res["is_draw"]:
            draw_count += 1
        elif res["active_won"]:
            active_win_count += 1

        if (ep + 1) % snapshot_interval == 0:
            trainer.add_snapshot()

    w_after = float(np.mean(mb_agent.w_kc_mbon))

    return {
        "episodes": episodes,
        "draws": draw_count,
        "draw_rate": draw_count / episodes,
        "active_wins": active_win_count,
        "active_win_rate": active_win_count / episodes,
        "weight_before": w_before,
        "weight_after": w_after,
        "weight_delta": w_after - w_before
    }
