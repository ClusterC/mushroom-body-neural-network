import numpy as np

class MushroomBodyNet:
    """
    Bio-Inspired Mushroom Body Neural Network สำหรับเกม XO (Tic-Tac-Toe):
    - Input: Projection Neurons (PN) ขนาด 27 มิติ (One-hot 9 ช่อง x 3 สถานะ)
    - Expansion: Kenyon Cells (KC) ขนาด 1,000 เซลล์ ผ่าน Fixed Sparse Projection
    - Sparsification: k-Winner-Take-All (k-WTA) คัดเลือก 5% - 10% Active Neurons
    - Output: Mushroom Body Output Neurons (MBON) ขนาด 9 เซลล์
    - Learning: Three-Factor Hebbian Plasticity ควบคุมด้วย Dopamine และ Eligibility Traces
    """
    def __init__(
        self,
        num_pn=27,
        num_kc=1000,
        num_mbon=9,
        k_active=75,
        pn_connectivity=6,
        learning_rate=0.05,
        gamma=0.95,
        lambda_trace=0.8,
        w_max=5.0,
        temperature=1.0,
        temp_min=0.1,
        temp_decay=0.9995,
        seed=None
    ):
        self.num_pn = num_pn
        self.num_kc = num_kc
        self.num_mbon = num_mbon
        self.k_active = k_active
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_trace = lambda_trace
        self.w_max = w_max
        self.temperature = temperature
        self.temp_min = temp_min
        self.temp_decay = temp_decay

        self.rng = np.random.default_rng(seed)

        # 1. PN -> KC Fixed Sparse Random Projections (ไม่ปรับค่าน้ำหนัก)
        # แต่ละ KC จะสุ่มเชื่อมต่อกับ PN จำนวน pn_connectivity เส้น
        self.w_pn_kc = np.zeros((num_pn, num_kc), dtype=np.float32)
        for j in range(num_kc):
            connected_pns = self.rng.choice(num_pn, size=pn_connectivity, replace=False)
            self.w_pn_kc[connected_pns, j] = self.rng.uniform(0.5, 1.5, size=pn_connectivity)

        # 2. KC -> MBON Plastic Synaptic Weights (ปรับค่าน้ำหนักผ่าน Three-factor rule)
        # ค่าน้ำหนักเริ่มต้นเป็น Non-negative สม่ำเสมอ [0.1, 0.5]
        self.w_kc_mbon = self.rng.uniform(0.1, 0.5, size=(num_kc, num_mbon)).astype(np.float32)

        # 3. Eligibility Traces สำหรับคู่ประสาท (KC, MBON)
        self.eligibility_trace = np.zeros((num_kc, num_mbon), dtype=np.float32)

    def reset_traces(self):
        """
        รีเซ็ต Eligibility Trace เมื่อเริ่มเกม/Episode ใหม่
        """
        self.eligibility_trace.fill(0.0)

    def encode_kc(self, pn_input):
        """
        แปลงสัญญาณจาก PN สู่ชั้นขยายมิติ KC และใช้กลไก k-WTA ในการสร้าง Sparse Coding
        ส่งคืนเวกเตอร์ขนาด num_kc ที่มีค่า 1.0 เฉพาะตำแหน่ง Top-k
        """
        # Linear excitation
        raw_kc = np.dot(pn_input, self.w_pn_kc)  # shape (num_kc,)

        # k-WTA (APL Global Lateral Inhibition)
        top_k_indices = np.argpartition(raw_kc, -self.k_active)[-self.k_active:]

        sparse_kc = np.zeros(self.num_kc, dtype=np.float32)
        sparse_kc[top_k_indices] = 1.0
        return sparse_kc

    def forward(self, pn_input, legal_mask=None):
        """
        Forward Pass: PN -> KC (Sparse) -> MBON
        ส่งคืนความน่าจะเป็น (Action Probabilities) หลังผ่าน Action Masking และ Softmax
        """
        sparse_kc = self.encode_kc(pn_input)
        mbon_activation = np.dot(sparse_kc, self.w_kc_mbon)  # shape (9,)

        if legal_mask is not None:
            # Action Masking: ให้ช่องที่ไม่ถูกต้องมีค่าเป็นลบอนันต์
            masked_activation = np.where(legal_mask, mbon_activation, -1e9)
        else:
            masked_activation = mbon_activation

        # Softmax with temperature
        shift_act = (masked_activation - np.max(masked_activation)) / max(self.temperature, 1e-4)
        exp_act = np.exp(shift_act)
        probs = exp_act / np.sum(exp_act)

        return probs, sparse_kc, mbon_activation

    def select_action(self, env, training=True, player=None):
        """
        เลือกตาเดินจากสภาพแวดล้อม:
        - training=True: สุ่มตาม Softmax Action Distribution และบันทึก Eligibility Trace
        - training=False: เลือกตาเดินที่มีความน่าจะเป็นสูงสุด (Greedy) จาก legal actions
        """
        if player is None:
            player = env.current_player

        obs = env.get_observation(player)
        legal_mask = env.get_action_mask()
        probs, sparse_kc, _ = self.forward(obs, legal_mask)

        legal_indices = np.where(legal_mask)[0]
        if len(legal_indices) == 0:
            raise RuntimeError("ไม่มีช่องที่สามารถเดินได้")

        if training:
            # กรองความน่าจะเป็นเฉพาะช่องที่ถูกต้อง ป้องกันข้อผิดพลาดจากการปัดเศษตัวเลข
            legal_probs = probs[legal_indices]
            prob_sum = np.sum(legal_probs)
            if prob_sum <= 0 or np.isnan(prob_sum):
                legal_probs = np.ones(len(legal_indices)) / len(legal_indices)
            else:
                legal_probs = legal_probs / prob_sum

            action = int(self.rng.choice(legal_indices, p=legal_probs))

            # อัปเดต Eligibility Trace สะสมคู่ประสาท (KC, MBON ที่เลือก)
            # E_t = gamma * lambda * E_{t-1} + x_kc (outer) y_mbon
            y_mbon = np.zeros(self.num_mbon, dtype=np.float32)
            y_mbon[action] = 1.0

            self.eligibility_trace = (
                self.gamma * self.lambda_trace * self.eligibility_trace
                + np.outer(sparse_kc, y_mbon)
            )
        else:
            # Greedy Argmax
            masked_probs = np.where(legal_mask, probs, -1.0)
            action = int(np.argmax(masked_probs))

        return action

    def update_synapses(self, dopamine_signal):
        """
        Three-Factor Hebbian Plasticity Rule:
        Delta W = eta * EligibilityTrace * DopamineSignal
        พร้อม Clip ค่าน้ำหนักให้อยู่ในช่วง [0, w_max] ตามกฎ Dale's Principle เชิงชีววิทยา
        """
        delta_w = self.learning_rate * self.eligibility_trace * float(dopamine_signal)
        self.w_kc_mbon = np.clip(self.w_kc_mbon + delta_w, 0.0, self.w_max)
        self.reset_traces()

    def decay_temperature(self):
        """
        ลดอุณหภูมิ Softmax ตาม Annealing Schedule
        """
        self.temperature = max(self.temp_min, self.temperature * self.temp_decay)
