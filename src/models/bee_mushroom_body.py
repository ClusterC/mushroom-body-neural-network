import numpy as np
from typing import Optional, Tuple, Dict, Any

class BeeMushroomBody:
    """
    แบบจำลองวงจรประสาท Mushroom Body ของผึ้งน้ำหวาน (Apis mellifera)
    - รองรับสัญญาณพหุสัมผัส 36 มิติ (กลิ่น, แสงสี, เซนเซอร์ระยะ, ทิศทางรังผึ้ง)
    - ชั้นขยายมิติ Kenyon Cells 2,500 เซลล์ แบ่งตามโซนชีววิทยาจริง:
      1. Lip Region (800 KCs): รับสัญญาณกลิ่น (Olfactory)
      2. Collar Region (1,000 KCs): รับสัญญาณแสงสีและภาพ (Visual)
      3. Basal Ring (700 KCs): ผสานข้อมูลพหุสัมผัส (Multisensory Integration)
    - Lateral Inhibition ด้วย Partitioned k-WTA ควบคุม Sparsity ให้คงที่ที่ 5.0% (125 KCs ทำงาน)
    - 5 MBONs ควบคุมพฤติกรรม (FORWARD, TURN_LEFT, TURN_RIGHT, LAND_AND_FEED, RETURN_TO_HIVE)
    - ปรับค่าน้ำหนัก Synapse ด้วย Three-Factor Plasticity (Octopamine / Dopamine Modulation)
    """
    def __init__(
        self,
        n_pn: int = 36,
        n_kc: int = 2500,
        n_mbon: int = 5,
        sparsity_ratio: float = 0.05,  # 5% Sparsity (125 active KCs)
        lr_octopamine: float = 0.04,   # อัตราการเรียนรู้เมื่อได้รับรางวัล (Octopamine)
        lr_dopamine: float = 0.02,     # อัตราการเรียนรู้เมื่อถูกลงโทษ (Dopamine)
        gamma: float = 0.92,           # อัตราส่วนลด
        trace_decay: float = 0.85,     # อัตราการเสื่อมของ Eligibility Trace
        seed: Optional[int] = None
    ):
        self.n_pn = n_pn
        self.n_kc = n_kc
        self.n_mbon = n_mbon
        self.k_active = int(n_kc * sparsity_ratio)  # 125 เซลล์
        self.lr_octopamine = lr_octopamine
        self.lr_dopamine = lr_dopamine
        self.gamma = gamma
        self.trace_decay = trace_decay
        self.rng = np.random.default_rng(seed)

        # สัดส่วนการแบ่งโซน Calyx ของผึ้ง
        self.n_lip = 800       # Olfactory KCs (0..799)
        self.n_collar = 1000   # Visual KCs (800..1799)
        self.n_basal = 700     # Multisensory KCs (1800..2499)
        assert self.n_lip + self.n_collar + self.n_basal == self.n_kc

        # จำนวนเซลล์ที่ยอมให้ทำงานในแต่ละโซน (Partitioned k-WTA รวมเป็น 125)
        self.k_lip = 40        # 40 / 800 = 5.0%
        self.k_collar = 50     # 50 / 1000 = 5.0%
        self.k_basal = 35      # 35 / 700 = 5.0%

        # 1. สร้างโครงข่ายการเชื่อมต่อแบบสุ่มเบาบาง PN -> KC (Non-trainable)
        self.W_pn_kc = np.zeros((self.n_pn, self.n_kc), dtype=np.float32)
        self._init_pn_to_kc_connections()

        # 2. ค่าน้ำหนัก Synapse ระหว่าง KC -> MBON (Trainable Plastic Weights)
        # ตามหลัก Dale's Principle: ค่าเป็นบวกเสมอ [0.0, 2.0]
        self.W_kc_mbon = self.rng.uniform(0.05, 0.15, size=(self.n_kc, self.n_mbon)).astype(np.float32)
        self.max_weight = 2.0

        # 3. ร่องรอยการกระตุ้น (Eligibility Traces)
        self.eligibility_traces = np.zeros((self.n_kc, self.n_mbon), dtype=np.float32)

        # ตัวแปรสถานะสำหรับการติดตามผลและการแสดงผล
        self.last_pn_input = np.zeros(self.n_pn, dtype=np.float32)
        self.last_kc_activation = np.zeros(self.n_kc, dtype=np.float32)
        self.last_mbon_probs = np.ones(self.n_mbon, dtype=np.float32) / self.n_mbon
        self.last_action = 0
        self.last_octopamine = 0.0
        self.last_dopamine = 0.0

    def _init_pn_to_kc_connections(self):
        """เชื่อมต่อเซนเซอร์สู่โซนสมองตามชีววิทยาของผึ้ง"""
        # Lip Region: รับเฉพาะกลิ่น (PN 0..7) + เข็มทิศ/พลังงานบางส่วน (PN 28..35)
        lip_pns = list(range(0, 8)) + list(range(28, 36))
        for kc in range(0, self.n_lip):
            selected_pns = self.rng.choice(lip_pns, size=4, replace=False)
            self.W_pn_kc[selected_pns, kc] = self.rng.uniform(0.5, 1.0, size=4).astype(np.float32)

        # Collar Region: รับเฉพาะสายตาตรวจจับสี (PN 8..19) + เซนเซอร์ระยะ (PN 20..27)
        collar_pns = list(range(8, 20)) + list(range(20, 28))
        for kc in range(self.n_lip, self.n_lip + self.n_collar):
            selected_pns = self.rng.choice(collar_pns, size=5, replace=False)
            self.W_pn_kc[selected_pns, kc] = self.rng.uniform(0.5, 1.0, size=5).astype(np.float32)

        # Basal Ring: รับสัญญาณผสมทุกชนิด (PN 0..35)
        all_pns = list(range(0, self.n_pn))
        for kc in range(self.n_lip + self.n_collar, self.n_kc):
            selected_pns = self.rng.choice(all_pns, size=6, replace=False)
            self.W_pn_kc[selected_pns, kc] = self.rng.uniform(0.4, 0.9, size=6).astype(np.float32)

    def _apply_k_wta(self, raw_kc: np.ndarray) -> np.ndarray:
        """
        ประยุกต์ใช้ Partitioned k-WTA ในแต่ละโซนของ Calyx เพื่อรักษาความเบาบางของสัญญาณ 5%
        """
        sparse_kc = np.zeros_like(raw_kc)

        # 1. โซน Lip (กลิ่น)
        lip_slice = raw_kc[0 : self.n_lip]
        if np.any(lip_slice > 0):
            top_indices = np.argpartition(lip_slice, -self.k_lip)[-self.k_lip:]
            sparse_kc[top_indices] = np.clip(lip_slice[top_indices], 0.1, 1.0)

        # 2. โซน Collar (สายตา)
        collar_offset = self.n_lip
        collar_slice = raw_kc[collar_offset : collar_offset + self.n_collar]
        if np.any(collar_slice > 0):
            top_indices = np.argpartition(collar_slice, -self.k_collar)[-self.k_collar:]
            sparse_kc[collar_offset + top_indices] = np.clip(collar_slice[top_indices], 0.1, 1.0)

        # 3. โซน Basal Ring (พหุสัมผัส)
        basal_offset = self.n_lip + self.n_collar
        basal_slice = raw_kc[basal_offset : self.n_kc]
        if np.any(basal_slice > 0):
            top_indices = np.argpartition(basal_slice, -self.k_basal)[-self.k_basal:]
            sparse_kc[basal_offset + top_indices] = np.clip(basal_slice[top_indices], 0.1, 1.0)

        return sparse_kc

    def forward(
        self,
        obs: np.ndarray,
        temperature: float = 0.5,
        action_mask: Optional[np.ndarray] = None
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Forward Pass:
        1. PN -> KC Activation
        2. Lateral Inhibition (Partitioned k-WTA: 5% Sparsity)
        3. KC -> MBON Readout
        4. Softmax Policy Selection
        """
        self.last_pn_input = np.copy(obs)

        # คำนวณสัญญาณดิบในชั้น KC ผ่าน Sparse Projections
        raw_kc = np.dot(obs, self.W_pn_kc)
        # แปลงเป็นบวก (ReLU-like non-linearity)
        raw_kc = np.maximum(0.0, raw_kc)

        # ใช้ Partitioned k-WTA
        self.last_kc_activation = self._apply_k_wta(raw_kc)

        # MBON Firing Readout: W_kc_mbon.T * x_kc
        mbon_logits = np.dot(self.last_kc_activation, self.W_kc_mbon)

        # ปรับอุณหภูมิความสุ่ม (Temperature Scaling)
        scaled_logits = mbon_logits / max(temperature, 0.05)

        # Action Masking
        if action_mask is not None:
            scaled_logits[action_mask == 0] = -1e9

        # Softmax Policy
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)
        self.last_mbon_probs = probs

        # สุ่มเลือกการกระทำตามการแจกแจงความน่าจะเป็น
        action = int(self.rng.choice(self.n_mbon, p=probs))
        self.last_action = action

        # อัปเดต Eligibility Traces ข้ามเวลา
        action_onehot = np.zeros(self.n_mbon, dtype=np.float32)
        action_onehot[action] = 1.0
        
        # E_t = gamma * lambda * E_{t-1} + (x_kc (outer) y_action)
        self.eligibility_traces = (self.gamma * self.trace_decay * self.eligibility_traces) + \
                                  np.outer(self.last_kc_activation, action_onehot)

        return action, probs, self.last_kc_activation

    def update_plasticity(self, reward: float):
        """
        ปรับค่าน้ำหนัก Synapse ด้วย Three-Factor Local Plasticity:
        - Reward > 0: Octopaminergic Reinforcement (เสริมแรงการเรียนรู้เมื่อเจอน้ำหวาน)
        - Reward < 0: Dopaminergic Aversive Depression (ลดค่าน้ำหนักเมื่อดูดโดนพิษหรือเสียพลังงาน)
        """
        if reward > 0:
            # เสริมแรงด้วย Octopamine
            self.last_octopamine = reward
            self.last_dopamine = 0.0
            delta_w = self.lr_octopamine * reward * self.eligibility_traces
            self.W_kc_mbon += delta_w
        elif reward < 0:
            # กดค่าน้ำหนักด้วย Dopamine (Aversive)
            self.last_octopamine = 0.0
            self.last_dopamine = abs(reward)
            delta_w = self.lr_dopamine * reward * self.eligibility_traces
            self.W_kc_mbon += delta_w

        # คงหลัก Dale's Principle: ค่าน้ำหนักห้ามติดลบ และไม่เกินเพดานสูงสุด
        np.clip(self.W_kc_mbon, 0.0, self.max_weight, out=self.W_kc_mbon)

    def reset_traces(self):
        """รีเซ็ต Eligibility Traces เมื่อเริ่มต้นเที่ยวบินใหม่จากรัง"""
        self.eligibility_traces.fill(0.0)
        self.last_octopamine = 0.0
        self.last_dopamine = 0.0

    def get_diagnostics(self) -> Dict[str, Any]:
        """ดึงข้อมูลสถิติของวงจรประสาทสำหรับการแสดงผล"""
        active_kc = int(np.sum(self.last_kc_activation > 0))
        lip_active = int(np.sum(self.last_kc_activation[0 : self.n_lip] > 0))
        collar_active = int(np.sum(self.last_kc_activation[self.n_lip : self.n_lip + self.n_collar] > 0))
        basal_active = int(np.sum(self.last_kc_activation[self.n_lip + self.n_collar :] > 0))

        return {
            "active_kc": active_kc,
            "sparsity_pct": (active_kc / self.n_kc) * 100.0,
            "lip_active": lip_active,
            "collar_active": collar_active,
            "basal_active": basal_active,
            "mean_weight": float(np.mean(self.W_kc_mbon)),
            "max_weight": float(np.max(self.W_kc_mbon)),
            "octopamine": self.last_octopamine,
            "dopamine": self.last_dopamine,
            "mbon_probs": list(self.last_mbon_probs)
        }
