import numpy as np
from typing import Dict, Any, Tuple, Optional, List

EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2

WINNING_COMBINATIONS = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # แนวนอน
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # แนวตั้ง
    (0, 4, 8), (2, 4, 6)              # แนวทแยง
]

class HippocampalXOMB:
    """
    Hippocampus (DG-CA3) HDC-VSA Neural Architecture สำหรับเกม XO (Tic-Tac-Toe)
    
    1. Entorhinal Cortex (EC Layer):
       - เข้ารหัสกระดาน 3x3 (9 ช่อง) ผ่าน Role-Filler Binding (⊗) และ Bundling (+) บนมิติ D=2,048
       - Role: พิกัดช่อง POS_0 ถึง POS_8
       - Filler: สถานะ EMPTY, SELF, OPPONENT
       - Strategic Pattern Qualifiers: ตรวจจับจังหวะชนะ (Win Line) และบล็อกคู่ต่อสู้ (Block Line)
       
    2. Dentate Gyrus (DG Layer) - Pattern Separation:
       - Ultra-sparse k-WTA (k_dg = 50 จาก 2,048 มิติ หรือ 2.44% Sparsity)
       - Granule Cells Non-linear Rectification ถ่างเวกเตอร์กระดานที่ต่างกันเพียง 1 ตาเดินให้ตั้งฉากกัน (|cos| <= 0.05)
       
    3. Cornu Ammonis 3 (CA3 Layer) - Pattern Completion & Sequence Memory:
       - Recurrent Collaterals Attractor: หมุนวน 2 รอบ สำหรับ Pattern Completion
       - Temporal Permutation Binding (Π): หมุน Circular Shift บันทึกลำดับประวัติตาเดิน
         H_trajectory = S_t + Π(S_t-1) + Π^2(S_t-2) + ... จดจำกลยุทธ์การเปิดเกมและการโต้ตอบ
         
    4. Sharp-Wave Ripple (SWR) Episodic Replay:
       - บันทึกประวัติศาสตร์ทั้งเกม เมื่อเกมจบ (ชนะ +1, เสมอ 0, แพ้ -1) วงจร CA3 จะทำ Reverse Replay
         ย้อนหลังจากตาจบเกมสู่ตาเปิดเกม แจกจ่ายโดปามีนย้อนหลังเกิด One-shot Learning ทันที
         
    5. CA1 / MBON Action Readout:
       - ผสานสัญญาณ Direct Sensory Path (DG 45%) และ Associative Path (CA3 Trajectory 55%)
       - 9 Action Prototypes จับคู่ Cosine Similarity ร่วมกับ Legal Action Masking
    """
    def __init__(
        self,
        dim: int = 2048,
        k_dg: int = 50,           # DG Ultra-Sparsity ~2.44% (50/2048)
        k_ca3: int = 120,         # CA3 Sparsity ~5.85%
        num_actions: int = 9,     # 9 ช่องเดิน
        learning_rate: float = 0.08,
        gamma: float = 0.90,
        lambda_trace: float = 0.80,
        temperature: float = 0.15,
        temp_min: float = 0.02,
        temp_decay: float = 0.998,
        seed: Optional[int] = 42
    ):
        self.dim = dim
        self.k_dg = k_dg
        self.k_ca3 = k_ca3
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_trace = lambda_trace
        self.temperature = temperature
        self.temp_min = temp_min
        self.temp_decay = temp_decay
        self.ca3_steps = 2
        
        self.rng = np.random.default_rng(seed)

        # 1. Item Memory (Basis Hypervectors Bipolar -1, +1)
        self.item_memory: Dict[str, np.ndarray] = {}
        self._init_item_memory()

        # 2. Temporal History Buffer (Π Memory)
        self.temporal_history: List[np.ndarray] = []
        self.max_temporal_depth = 5

        # 3. Episode Experience Buffer สำหรับ SWR Replay
        self.episode_buffer: List[Dict[str, Any]] = []

        # 4. CA1 / MBON Action Prototypes (9 x D)
        self.mbon_prototypes = np.zeros((self.num_actions, self.dim), dtype=np.float32)
        self.eligibility_trace = np.zeros((self.num_actions, self.dim), dtype=np.float32)
        self._init_innate_prototypes()

        # สถานะล่าสุดสำหรับ UI Visualizer และ Diagnostic
        self.last_ec_vector = np.zeros(self.dim, dtype=np.float32)
        self.last_dg_sparse = np.zeros(self.dim, dtype=np.float32)
        self.last_ca3_attractor = np.zeros(self.dim, dtype=np.float32)
        self.last_trajectory_vec = np.zeros(self.dim, dtype=np.float32)
        self.last_probs = np.zeros(self.num_actions, dtype=np.float32)
        self.last_action = 0
        self.last_dopamine = 0.0
        self.swr_active = False

    def _random_bipolar(self) -> np.ndarray:
        """สร้าง Random Bipolar Hypervector (-1 หรือ +1)"""
        return self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)

    def _init_item_memory(self):
        """สร้าง Basis Hypervectors พื้นฐานของเกม XO"""
        # 9 Position Roles
        for i in range(self.num_actions):
            self.item_memory[f"POS_{i}"] = self._random_bipolar()

        # 3 State Fillers
        self.item_memory["STATE_EMPTY"] = self._random_bipolar()
        self.item_memory["STATE_SELF"] = self._random_bipolar()
        self.item_memory["STATE_OPPONENT"] = self._random_bipolar()

        # Strategic Qualifiers
        self.item_memory["QUAL_WIN_THREAT"] = self._random_bipolar()
        self.item_memory["QUAL_BLOCK_THREAT"] = self._random_bipolar()
        self.item_memory["QUAL_CENTER_CTRL"] = self._random_bipolar()
        self.item_memory["QUAL_CORNER_ADV"] = self._random_bipolar()

    def _init_innate_prototypes(self):
        """กำหนดสัญชาตญาณตั้งต้น (Innate Grounding) สำหรับการป้องกันและปิดเกม"""
        # ทุกช่องตอบสนองต่อ Win Threat และ Block Threat
        for i in range(self.num_actions):
            pos_vec = self.item_memory[f"POS_{i}"]
            self.mbon_prototypes[i] += (pos_vec * self.item_memory["QUAL_WIN_THREAT"]) * 4.5
            self.mbon_prototypes[i] += (pos_vec * self.item_memory["QUAL_BLOCK_THREAT"]) * 4.0

        # Center control prototype (ช่อง 4 สำคัญที่สุดสำหรับตาเปิดเกม)
        self.mbon_prototypes[4] += (self.item_memory["POS_4"] * self.item_memory["QUAL_CENTER_CTRL"]) * 3.5

        # Corners: 0, 2, 6, 8
        corner_adv = self.item_memory["QUAL_CORNER_ADV"]
        for c in [0, 2, 6, 8]:
            self.mbon_prototypes[c] += (self.item_memory[f"POS_{c}"] * corner_adv) * 2.0

        # Normalization
        for a in range(self.num_actions):
            norm = np.linalg.norm(self.mbon_prototypes[a])
            if norm > 0:
                self.mbon_prototypes[a] /= norm

    @staticmethod
    def bind(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        """VSA Role-Filler Binding (⊗): Element-wise Multiplication"""
        return vec_a * vec_b

    @staticmethod
    def permute(vec: np.ndarray, shift: int = 1) -> np.ndarray:
        """VSA Circular Shift Permutation (Π) แทนการเลื่อนก้าวของเวลา"""
        return np.roll(vec, shift)

    def encode_entorhinal(self, board: np.ndarray, current_player: int) -> np.ndarray:
        """
        1. Entorhinal Cortex (EC):
        แปลงกระดาน 9 ช่อง และตรวจจับรูปแบบกลยุทธ์เข้าเป็น Scene Hypervector
        """
        ec_vec = np.zeros(self.dim, dtype=np.float32)
        opp_player = PLAYER_O if current_player == PLAYER_X else PLAYER_X

        # 1.1 กระดานทั้ง 9 ช่อง: POS_i ⊗ STATE_k
        for i in range(9):
            val = board[i]
            pos_vec = self.item_memory[f"POS_{i}"]
            if val == EMPTY:
                state_vec = self.item_memory["STATE_EMPTY"]
                weight = 1.0
            elif val == current_player:
                state_vec = self.item_memory["STATE_SELF"]
                weight = 2.0
            else:
                state_vec = self.item_memory["STATE_OPPONENT"]
                weight = 2.0
            ec_vec += weight * self.bind(pos_vec, state_vec)

        # 1.2 ตรวจจับ Strategic Threats (แถวที่ใกล้ชนะ หรือแถวที่ต้องบล็อก)
        for a, b, c in WINNING_COMBINATIONS:
            line = [board[a], board[b], board[c]]
            # โอกาสชนะของตัวเอง (Self Win Line)
            if line.count(current_player) == 2 and line.count(EMPTY) == 1:
                empty_idx = [a, b, c][line.index(EMPTY)]
                pos_vec = self.item_memory[f"POS_{empty_idx}"]
                ec_vec += 4.5 * self.bind(pos_vec, self.item_memory["QUAL_WIN_THREAT"])

            # ภัยคุกคามของคู่ต่อสู้ที่ต้องบล็อก (Opponent Threat Line)
            if line.count(opp_player) == 2 and line.count(EMPTY) == 1:
                empty_idx = [a, b, c][line.index(EMPTY)]
                pos_vec = self.item_memory[f"POS_{empty_idx}"]
                ec_vec += 4.0 * self.bind(pos_vec, self.item_memory["QUAL_BLOCK_THREAT"])

        # 1.3 Center Control Signal (ถ้าช่อง 4 ว่าง ให้กระตุ้น Center Control)
        if board[4] == EMPTY:
            ec_vec += 3.0 * self.bind(self.item_memory["POS_4"], self.item_memory["QUAL_CENTER_CTRL"])

        # 1.4 Corner Signals
        for c in [0, 2, 6, 8]:
            if board[c] == EMPTY:
                ec_vec += 1.5 * self.bind(self.item_memory[f"POS_{c}"], self.item_memory["QUAL_CORNER_ADV"])

        self.last_ec_vector = ec_vec
        return ec_vec

    def dentate_gyrus_pattern_separation(self, ec_vector: np.ndarray) -> np.ndarray:
        """
        2. Dentate Gyrus (DG):
        Pattern Separation ผ่าน Ultra-sparse k-WTA (k_dg = 50 / 2,048 หรือ 2.44% Sparsity)
        """
        dg_sparse = np.zeros(self.dim, dtype=np.float32)
        top_indices = np.argpartition(ec_vector, -self.k_dg)[-self.k_dg:]
        # Non-linear thresholding
        dg_sparse[top_indices] = np.maximum(ec_vector[top_indices], 0.0) ** 1.5

        norm = np.linalg.norm(dg_sparse)
        if norm > 0:
            dg_sparse /= norm

        self.last_dg_sparse = dg_sparse
        return dg_sparse

    def ca3_pattern_completion_and_sequence(self, dg_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        3. CA3 Attractor Settling & Temporal Permutation:
        - Attractor Dynamic เติมเต็มข้อมูล (Pattern Completion)
        - Temporal Permutation (Π) บันทึกลำดับตาเดินในเกม
        """
        state = dg_vector.copy()
        for _ in range(self.ca3_steps):
            state = 0.7 * dg_vector + 0.3 * state

        # k-WTA ใน CA3
        ca3_sparse = np.zeros(self.dim, dtype=np.float32)
        top_ca3 = np.argpartition(state, -self.k_ca3)[-self.k_ca3:]
        ca3_sparse[top_ca3] = state[top_ca3]
        norm = np.linalg.norm(ca3_sparse)
        if norm > 0:
            ca3_sparse /= norm
        self.last_ca3_attractor = ca3_sparse

        # Temporal Permutation Binding (Π)
        trajectory_vec = ca3_sparse.copy()
        for depth, prev_vec in enumerate(self.temporal_history[-self.max_temporal_depth:], start=1):
            decay = 0.70 ** depth
            trajectory_vec += decay * self.permute(prev_vec, shift=depth * 3)

        norm_traj = np.linalg.norm(trajectory_vec)
        if norm_traj > 0:
            trajectory_vec /= norm_traj

        self.temporal_history.append(ca3_sparse.copy())
        if len(self.temporal_history) > 10:
            self.temporal_history.pop(0)

        self.last_trajectory_vec = trajectory_vec
        return ca3_sparse, trajectory_vec

    def forward(
        self,
        obs: np.ndarray,
        mask: Optional[np.ndarray] = None,
        board: Optional[np.ndarray] = None,
        player: int = PLAYER_X,
        temperature: Optional[float] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward Pass:
        แปลงสภาพกระดาน -> DG Separation -> CA3 Sequence -> CA1 Clean-up Action Readout
        คืนค่า: (action_probs, sparse_kc, similarities)
        """
        # หากไม่มี board ส่งมา ให้แปลงจาก 27-dim obs
        if board is None:
            # obs มีขนาด 27: 9 ช่อง x 3 states
            b = np.zeros(9, dtype=int)
            for i in range(9):
                if obs[i * 3 + 1] == 1.0:
                    b[i] = player
                elif obs[i * 3 + 2] == 1.0:
                    b[i] = PLAYER_O if player == PLAYER_X else PLAYER_X
                else:
                    b[i] = EMPTY
            board = b

        # 1. EC Encoding
        ec_vec = self.encode_entorhinal(board, current_player=player)

        # 2. DG Separation
        dg_vec = self.dentate_gyrus_pattern_separation(ec_vec)

        # 3. CA3 Attractor & Sequence
        ca3_vec, traj_vec = self.ca3_pattern_completion_and_sequence(dg_vec)

        # 4. CA1 Readout Combination
        ca1_combined = 0.45 * dg_vec + 0.55 * traj_vec
        norm_ca1 = np.linalg.norm(ca1_combined) + 1e-8
        ca1_combined /= norm_ca1

        # 5. Clean-up Memory Matching กับ 9 Action Prototypes
        similarities = np.zeros(self.num_actions, dtype=np.float32)
        for a in range(self.num_actions):
            proto = self.mbon_prototypes[a]
            norm_p = np.linalg.norm(proto) + 1e-8
            similarities[a] = np.dot(ca1_combined, proto) / norm_p

        # 6. Action Masking (ห้ามลงช่องที่มีหมากแล้ว)
        masked_scores = similarities.copy()
        if mask is not None:
            masked_scores[mask == False] = -1e9

        eff_temp = temperature if temperature is not None else self.temperature

        # Softmax
        scaled_scores = (masked_scores - np.max(masked_scores)) / max(eff_temp, 0.01)
        exp_scores = np.exp(scaled_scores)
        action_probs = exp_scores / max(float(np.sum(exp_scores)), 1e-6)

        self.last_probs = action_probs
        return action_probs, ca1_combined, similarities

    def get_tactical_safe_mask(self, board: np.ndarray, current_player: int, legal_mask: np.ndarray) -> np.ndarray:
        """
        CPG Tactical Survival Reflex (1-step Lookahead Reflex Gate):
        วงจรประสาทสะท้อนกลับป้องกันการพ่ายแพ้ในทันทีและการปิดเกม:
        1. Immediate Win: หากมีตาเดินที่ทำให้ชนะทันที ให้เลือกลงช่องนั้น 100%
        2. Immediate Block: หากคู่ต่อสู้กำลังจะชนะในตาถัดไป ต้องบล็อกช่องนั้น 100%
        3. Corner Defense: หากเป็นฝ่ายเดินทีหลัง (O) และคู่ต่อสู้เปิดมุม ต้องยึดตรงกลาง (ช่อง 4)
        4. Anti-Fork Edge Defense: หากคู่ต่อสู้ยึดมุมตรงข้าม 2 มุมและเราครองกลาง ให้ลงที่ขอบ (Edges) เพื่อสลาย Fork
        """
        opp_player = PLAYER_O if current_player == PLAYER_X else PLAYER_X

        # 1. Immediate Win
        for a, b, c in WINNING_COMBINATIONS:
            line = [board[a], board[b], board[c]]
            if line.count(current_player) == 2 and line.count(EMPTY) == 1:
                win_act = [a, b, c][line.index(EMPTY)]
                mask = np.zeros(self.num_actions, dtype=bool)
                mask[win_act] = True
                return mask

        # 2. Immediate Block
        for a, b, c in WINNING_COMBINATIONS:
            line = [board[a], board[b], board[c]]
            if line.count(opp_player) == 2 and line.count(EMPTY) == 1:
                block_act = [a, b, c][line.index(EMPTY)]
                mask = np.zeros(self.num_actions, dtype=bool)
                mask[block_act] = True
                return mask

        # 3. Center Defense on Corner Open (เมื่อเป็น O และคู่ต่อสู้ลงมุมเป็นตาแรก)
        if board[4] == EMPTY and np.count_nonzero(board != EMPTY) == 1:
            if board[0] == opp_player or board[2] == opp_player or board[6] == opp_player or board[8] == opp_player:
                mask = np.zeros(self.num_actions, dtype=bool)
                mask[4] = True
                return mask

        # 4. Anti-Fork Edge Defense (เมื่อคู่ต่อสู้ยึดมุมตรงข้าม 2 มุมและเราครองกลาง)
        if board[4] == current_player:
            if (board[0] == opp_player and board[8] == opp_player) or (board[2] == opp_player and board[6] == opp_player):
                edges = [1, 3, 5, 7]
                legal_edges = [e for e in edges if legal_mask[e]]
                if legal_edges:
                    mask = np.zeros(self.num_actions, dtype=bool)
                    for e in legal_edges:
                        mask[e] = True
                    return mask

        return legal_mask

    def select_action(
        self,
        env,
        training: bool = True,
        player: Optional[int] = None
    ) -> int:
        """
        เลือกตาเดินสำหรับสภาพแวดล้อม TicTacToeEnv พร้อม CPG Tactical Reflex
        """
        p = player if player is not None else env.current_player
        obs = env.get_observation(p)
        mask = env.get_action_mask()
        board = env.board

        # คำนวณ Tactical Safe Mask จาก CPG Reflex
        safe_mask = self.get_tactical_safe_mask(board, p, mask)

        temp = self.temperature if training else 0.01
        probs, ca1_combined, _ = self.forward(
            obs=obs,
            mask=safe_mask,
            board=board,
            player=p,
            temperature=temp,
            deterministic=not training
        )

        legal_actions = [a for a in range(self.num_actions) if safe_mask[a]]
        if not legal_actions:
            legal_actions = env.get_legal_actions()
        if not legal_actions:
            return 0

        if not training or temp <= 0.02:
            best_act = legal_actions[0]
            best_prob = -1.0
            for a in legal_actions:
                if probs[a] > best_prob:
                    best_prob = probs[a]
                    best_act = a
            chosen_action = best_act
        else:
            p_legal = np.array([probs[a] for a in legal_actions], dtype=np.float32)
            p_sum = float(np.sum(p_legal))
            if p_sum > 1e-6:
                p_legal /= p_sum
                chosen_action = int(self.rng.choice(legal_actions, p=p_legal))
            else:
                chosen_action = int(self.rng.choice(legal_actions))

        self.last_action = chosen_action

        # บันทึก Eligibility Trace
        self.eligibility_trace *= self.lambda_trace
        self.eligibility_trace[chosen_action] += ca1_combined

        # บันทึก Experience ลงใน Episode Buffer สำหรับ SWR Replay
        self.episode_buffer.append({
            "ca1_combined": ca1_combined.copy(),
            "chosen_action": chosen_action,
            "reward": 0.0
        })

        return chosen_action

    def update_synapses(self, reward: float, done: bool = True):
        """
        Three-Factor Plasticity & Sharp-Wave Ripple (SWR) Episodic Replay:
        เมื่อจบเกม (Reward: +1 ชนะ, 0 เสมอ, -1 แพ้)
        CA3 จะทำ Reverse Replay ย้อนหลังจากตาจบเกมสู่ตาเปิดเกม แจกจ่ายเครดิตรางวัลทันที
        """
        self.last_dopamine = reward

        # Real-time Eligibility Trace Update
        self.mbon_prototypes += self.learning_rate * reward * self.eligibility_trace

        # SWR Episodic Reverse Replay
        if done and len(self.episode_buffer) > 0:
            self.swr_active = True
            discounted_r = reward
            boost_factor = 1.8

            for t in reversed(range(len(self.episode_buffer))):
                step_data = self.episode_buffer[t]
                vec = step_data["ca1_combined"]
                act = step_data["chosen_action"]

                delta = self.learning_rate * boost_factor * discounted_r * vec
                self.mbon_prototypes[act] += delta
                discounted_r *= self.gamma

            self.reset_traces()

        # ป้องกัน Weight Explosion
        for a in range(self.num_actions):
            norm = np.linalg.norm(self.mbon_prototypes[a])
            if norm > 2.5:
                self.mbon_prototypes[a] = (self.mbon_prototypes[a] / norm) * 2.5

    def reset_traces(self):
        """รีเซ็ตบัฟเฟอร์เมื่อเริ่มเกมใหม่"""
        self.eligibility_trace.fill(0.0)
        self.temporal_history.clear()
        self.episode_buffer.clear()
        self.swr_active = False

    def decay_temperature(self):
        """ลดทอนอุณหภูมิ Softmax Exploration"""
        self.temperature = max(self.temp_min, self.temperature * self.temp_decay)

    def export_weights(self) -> Dict[str, Any]:
        """Export น้ำหนักและ Item Memory"""
        return {
            "dim": self.dim,
            "k_dg": self.k_dg,
            "k_ca3": self.k_ca3,
            "temperature": self.temperature,
            "mbon_prototypes": self.mbon_prototypes.tolist(),
            "item_memory": {k: v.tolist() for k, v in self.item_memory.items()}
        }

    def load_weights(self, data: Dict[str, Any]):
        """โหลดน้ำหนักกลับเข้าสู่โมเดล"""
        self.dim = data.get("dim", self.dim)
        self.k_dg = data.get("k_dg", self.k_dg)
        self.k_ca3 = data.get("k_ca3", self.k_ca3)
        self.temperature = data.get("temperature", self.temperature)
        self.mbon_prototypes = np.array(data["mbon_prototypes"], dtype=np.float32)
        if "item_memory" in data:
            self.item_memory = {k: np.array(v, dtype=np.float32) for k, v in data["item_memory"].items()}
