import numpy as np

class VisualMushroomBody:
    """
    Visual Mushroom Body Neural Network สำหรับประมวลผลภาพเกมงู (Visual Pixel Input):
    - Input: Visual Projection Neurons (300 Visual PNs จากภาพ 3 แชนแนล 10x10)
    - Receptive Fields: เชื่อมต่อตามตำแหน่งเชิงพื้นที่ (Spatial Receptive Patches)
    - Expansion: Kenyon Cells (KC 2,000 เซลล์)
    - Sparsification: k-Winner-Take-All (k-WTA 5.0% Sparsity: คัดเลือก Top-100 เซลล์)
    - Output: MBON 4 เซลล์ สำหรับทิศทาง [UP, RIGHT, DOWN, LEFT]
    - Plasticity: Three-Factor Hebbian Learning พร้อม Eligibility Traces และ Dopamine Step Modulation
    """
    def __init__(
        self,
        channels=3,
        grid_h=10,
        grid_w=10,
        num_kc=2000,
        k_active=100,
        num_mbon=4,
        learning_rate=0.04,
        gamma=0.90,
        lambda_trace=0.75,
        w_max=5.0,
        temperature=0.8,
        temp_min=0.05,
        temp_decay=0.9997,
        seed=None
    ):
        self.channels = channels
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_pn = channels * grid_h * grid_w  # 300 PNs
        self.num_kc = num_kc
        self.k_active = k_active
        self.num_mbon = num_mbon

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_trace = lambda_trace
        self.w_max = w_max
        self.temperature = temperature
        self.temp_min = temp_min
        self.temp_decay = temp_decay

        self.rng = np.random.default_rng(seed)

        # 1. Visual PN -> KC Sparse Receptive Field Projections
        # จำลองเซลล์ประสาทสายตาของแมลง (Optic Lobe) โดยให้แต่ละ KC สุ่มจับคู่กับ Local Patch 3x3 บนกระดาน
        self.w_pn_kc = np.zeros((self.num_pn, self.num_kc), dtype=np.float32)
        self._init_receptive_fields()

        # 2. KC -> MBON Synaptic Weights (เริ่มต้นสม่ำเสมอ [0.1, 0.4])
        self.w_kc_mbon = self.rng.uniform(0.1, 0.4, size=(num_kc, num_mbon)).astype(np.float32)

        # 3. Eligibility Traces สำหรับคู่ประสาท (KC, MBON)
        self.eligibility_trace = np.zeros((num_kc, num_mbon), dtype=np.float32)

    def _init_receptive_fields(self):
        """
        สร้างการเชื่อมต่อแบบ Local Receptive Field:
        ให้ KC แต่ละตัวรับสัญญาณจากพื้นที่บริเวณ 3x3 พิกเซล และสุ่มข้ามแชนแนล (Head, Body, Food)
        """
        for j in range(self.num_kc):
            # สุ่มจุดศูนย์กลาง Patch บนตาราง 10x10
            center_r = self.rng.integers(0, self.grid_h)
            center_c = self.rng.integers(0, self.grid_w)

            # คัดเลือกพิกเซลรอบจุดศูนย์กลางระยะรัศมี 1 ช่อง
            patch_indices = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    pr = center_r + dr
                    pc = center_c + dc
                    if 0 <= pr < self.grid_h and 0 <= pc < self.grid_w:
                        # กระจายข้ามแชนแนล Head, Body, Food
                        for ch in range(self.channels):
                            flat_idx = ch * (self.grid_h * self.grid_w) + pr * self.grid_w + pc
                            patch_indices.append(flat_idx)

            # สุ่มเลือก 8-12 เส้นประสาทจาก Patch นี้
            k_conn = min(len(patch_indices), int(self.rng.integers(8, 13)))
            chosen_pns = self.rng.choice(patch_indices, size=k_conn, replace=False)
            self.w_pn_kc[chosen_pns, j] = self.rng.uniform(0.6, 1.4, size=k_conn)

    def reset_traces(self):
        self.eligibility_trace.fill(0.0)

    def encode_kc(self, visual_obs):
        """
        แปลงภาพ 3x10x10 สู่สัญญาณ Kenyon Cells และทำ k-WTA Sparsification
        """
        pn_flat = visual_obs.flatten()
        raw_kc = np.dot(pn_flat, self.w_pn_kc)

        # k-WTA: คัดเลือกเฉพาะ Top-k (Top-100 หรือ 5.0% Sparsity)
        top_k_indices = np.argpartition(raw_kc, -self.k_active)[-self.k_active:]
        sparse_kc = np.zeros(self.num_kc, dtype=np.float32)
        sparse_kc[top_k_indices] = 1.0
        return sparse_kc

    def forward(self, visual_obs, legal_mask=None):
        """
        Forward pass: Visual Input -> KC (Sparse) -> MBON (4 Actions)
        """
        sparse_kc = self.encode_kc(visual_obs)
        mbon_activation = np.dot(sparse_kc, self.w_kc_mbon)

        if legal_mask is not None:
            masked_activation = np.where(legal_mask, mbon_activation, -1e9)
        else:
            masked_activation = mbon_activation

        # Softmax with temperature
        shift_act = (masked_activation - np.max(masked_activation)) / max(self.temperature, 1e-4)
        exp_act = np.exp(shift_act)
        probs = exp_act / np.sum(exp_act)

        return probs, sparse_kc, mbon_activation

    def select_action(self, env, training=True):
        """
        เลือกทิศทางเดิน (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
        """
        obs = env.get_visual_observation()
        legal_mask = env.get_action_mask()
        probs, sparse_kc, _ = self.forward(obs, legal_mask)

        legal_indices = np.where(legal_mask)[0]
        if len(legal_indices) == 0:
            return 0

        if training:
            legal_probs = probs[legal_indices]
            p_sum = np.sum(legal_probs)
            if p_sum <= 0 or np.isnan(p_sum):
                legal_probs = np.ones(len(legal_indices)) / len(legal_indices)
            else:
                legal_probs = legal_probs / p_sum

            action = int(self.rng.choice(legal_indices, p=legal_probs))

            # อัปเดต Eligibility Trace
            y_mbon = np.zeros(self.num_mbon, dtype=np.float32)
            y_mbon[action] = 1.0

            self.eligibility_trace = (
                self.gamma * self.lambda_trace * self.eligibility_trace
                + np.outer(sparse_kc, y_mbon)
            )
        else:
            masked_probs = np.where(legal_mask, probs, -1.0)
            action = int(np.argmax(masked_probs))

        return action

    def update_synapses(self, dopamine_reward):
        """
        Three-Factor Plasticity Rule:
        Delta W = eta * EligibilityTrace * DopamineReward
        """
        delta_w = self.learning_rate * self.eligibility_trace * float(dopamine_reward)
        self.w_kc_mbon = np.clip(self.w_kc_mbon + delta_w, 0.0, self.w_max)

    def decay_temperature(self):
        self.temperature = max(self.temp_min, self.temperature * self.temp_decay)
