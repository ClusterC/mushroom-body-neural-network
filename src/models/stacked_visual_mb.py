import numpy as np
from typing import Tuple, Optional, Dict, Any, List

class StackedVisualMushroomBody:
    """
    สถาปัตยกรรมสมองซ้อน 2 ลำดับขั้น (Hierarchical Deep Mushroom Body):
    - Layer 1: Perceptual Abstraction MB (สิ่งเร้าดิบ 300 Visual PNs -> KC1 1,200 เซลล์ -> 12 Intermediate Concepts)
      ทำหน้าที่กลั่นกรองและจำแนกบริบทสถานการณ์ (เช่น ทางโล่ง, ทางตัน, ติดกำแพง, อาหารอยู่ใกล้, ความเสี่ยงติดกับดัก)
    - Layer 2: Strategic Executive MB (12 Concepts + Context -> KC2 800 เซลล์ -> 4 Executive Actions)
      ทำหน้าที่ตัดสินใจระดับยุทธศาสตร์ วางแผนและเลือกทิศทางการเคลื่อนที่ [UP, RIGHT, DOWN, LEFT]
    - Non-backpropagation: ทั้งสองชั้นเรียนรู้ผ่าน Dual Three-Factor Local Plasticity ด้วย Eligibility Traces ของตนเอง
    """
    CONCEPT_NAMES = [
        "CLEAR_AHEAD",
        "WALL_UP",
        "WALL_RIGHT",
        "WALL_DOWN",
        "WALL_LEFT",
        "FOOD_NEAR",
        "FOOD_FAR",
        "BODY_DANGER",
        "POCKET_TRAP",
        "SAFE_OPENING",
        "AXIS_HORIZ",
        "AXIS_VERT"
    ]

    def __init__(
        self,
        channels: int = 3,
        grid_h: int = 10,
        grid_w: int = 10,
        num_kc1: int = 1200,
        k_active1: int = 60,       # 5% ของ 1200
        num_concepts: int = 12,
        num_kc2: int = 800,
        k_active2: int = 40,       # 5% ของ 800
        num_mbon: int = 4,         # UP, RIGHT, DOWN, LEFT
        lr1: float = 0.03,         # Learning rate Layer 1
        lr2: float = 0.04,         # Learning rate Layer 2
        gamma: float = 0.90,
        lambda1: float = 0.75,     # Trace decay Layer 1
        lambda2: float = 0.80,     # Trace decay Layer 2
        w_max1: float = 3.0,
        w_max2: float = 5.0,
        temperature: float = 0.8,
        temp_min: float = 0.05,
        temp_decay: float = 0.9997,
        seed: Optional[int] = None
    ):
        self.channels = channels
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_pn1 = channels * grid_h * grid_w  # 300 Visual PNs
        self.num_kc1 = num_kc1
        self.k_active1 = k_active1
        self.num_concepts = num_concepts

        self.num_pn2 = num_concepts + 4            # 12 Concepts + 4 Context PNs = 16
        self.num_kc2 = num_kc2
        self.k_active2 = k_active2
        self.num_mbon = num_mbon

        self.lr1 = lr1
        self.lr2 = lr2
        self.gamma = gamma
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.w_max1 = w_max1
        self.w_max2 = w_max2
        self.temperature = temperature
        self.temp_min = temp_min
        self.temp_decay = temp_decay

        self.rng = np.random.default_rng(seed)

        # ==================== LAYER 1 (PERCEPTUAL MB) ====================
        # 1.1 Local Receptive Fields: 300 PNs -> 1,200 KC1
        self.w_pn1_kc1 = np.zeros((self.num_pn1, self.num_kc1), dtype=np.float32)
        self.kc1_centers = []
        self._init_layer1_receptive_fields()

        # 1.2 KC1 -> Intermediate MBON1 Synapses (1200 x 12) พร้อม Innate Concept Grounding
        self.w_kc1_mbon1 = self.rng.uniform(0.1, 0.25, size=(self.num_kc1, self.num_concepts)).astype(np.float32)
        self._ground_innate_concepts()
        self.eligibility_trace1 = np.zeros((self.num_kc1, self.num_concepts), dtype=np.float32)

        # ==================== LAYER 2 (STRATEGIC EXECUTIVE MB) ====================
        # 2.1 Sparse Random Projections: 16 Inputs -> 800 KC2
        self.w_pn2_kc2 = np.zeros((self.num_pn2, self.num_kc2), dtype=np.float32)
        self._init_layer2_projections()

        # 2.2 KC2 -> Executive MBON2 Synapses (800 x 4)
        self.w_kc2_mbon2 = self.rng.uniform(0.1, 0.35, size=(self.num_kc2, self.num_mbon)).astype(np.float32)
        self._ground_executive_reflexes()
        self.eligibility_trace2 = np.zeros((self.num_kc2, self.num_mbon), dtype=np.float32)

        # สถานะล่าสุดสำหรับการติดตามและการแสดงผล
        self.last_pn1 = np.zeros(self.num_pn1, dtype=np.float32)
        self.last_kc1 = np.zeros(self.num_kc1, dtype=np.float32)
        self.last_concepts = np.zeros(self.num_concepts, dtype=np.float32)
        self.last_kc2 = np.zeros(self.num_kc2, dtype=np.float32)
        self.last_mbon2_probs = np.ones(self.num_mbon, dtype=np.float32) / self.num_mbon
        self.last_action = 0
        self.last_reward = 0.0

    def _init_layer1_receptive_fields(self):
        """สร้าง Receptive Fields 3x3 บนกระดาน เชื่อมต่อไปยัง KC1 แต่ละตัว"""
        self.kc1_centers = []
        for j in range(self.num_kc1):
            center_r = int(self.rng.integers(0, self.grid_h))
            center_c = int(self.rng.integers(0, self.grid_w))
            self.kc1_centers.append((center_r, center_c))
            patch_indices = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    pr = center_r + dr
                    pc = center_c + dc
                    if 0 <= pr < self.grid_h and 0 <= pc < self.grid_w:
                        for ch in range(self.channels):
                            flat_idx = ch * (self.grid_h * self.grid_w) + pr * self.grid_w + pc
                            patch_indices.append(flat_idx)

            k_conn = min(len(patch_indices), int(self.rng.integers(8, 13)))
            chosen_pns = self.rng.choice(patch_indices, size=k_conn, replace=False)
            self.w_pn1_kc1[chosen_pns, j] = self.rng.uniform(0.6, 1.4, size=k_conn)

    def _ground_innate_concepts(self):
        """
        Innate Concept Grounding: กำหนดสัญชาตญาณตรวจจับสถานการณ์เบื้องต้น
        ผูกตำแหน่งทางกายภาพของ KC1 เข้ากับความหมายของ 12 Situational Concepts
        """
        for j, (cr, cc) in enumerate(self.kc1_centers):
            # 1. ทิศทางอาหาร (Food UP, RIGHT, DOWN, LEFT)
            if cr < 4:
                self.w_kc1_mbon1[j, 0] += 0.35  # Food UP
            if cc > 5:
                self.w_kc1_mbon1[j, 1] += 0.35  # Food RIGHT
            if cr > 5:
                self.w_kc1_mbon1[j, 2] += 0.35  # Food DOWN
            if cc < 4:
                self.w_kc1_mbon1[j, 3] += 0.35  # Food LEFT

            # 2. สิ่งกีดขวางขอบกำแพง (Wall Danger)
            if cr == 0 or cr == self.grid_h - 1 or cc == 0 or cc == self.grid_w - 1:
                self.w_kc1_mbon1[j, 4] += 0.40  # Wall NEAR
                self.w_kc1_mbon1[j, 5] += 0.30  # Wall CRITICAL
            elif cr == 1 or cr == self.grid_h - 2 or cc == 1 or cc == self.grid_w - 2:
                self.w_kc1_mbon1[j, 4] += 0.20  # Wall NEAR

            # 3. สภาวะเปิดโล่ง (Open Space)
            if 3 <= cr <= 6 and 3 <= cc <= 6:
                self.w_kc1_mbon1[j, 9] += 0.30  # Open SPACE

    def _ground_executive_reflexes(self):
        """
        Innate Executive Grounding: เสริมสัญชาตญาณเบื้องต้นใน Layer 2
        เชื่อมโยงสัญญาณมโนทัศน์ไปยังทิศทางเดินที่สอดคล้อง
        """
        # สุ่มเพิ่มน้ำหนักให้กับ KC2 ที่ต่อกับ Concept ทิศทางอาหาร
        for j in range(self.num_kc2):
            active_inputs = np.where(self.w_pn2_kc2[:, j] > 0)[0]
            for inp in active_inputs:
                if inp == 0:   # Concept: Food UP
                    self.w_kc2_mbon2[j, 0] += 0.45  # Action UP
                elif inp == 1: # Concept: Food RIGHT
                    self.w_kc2_mbon2[j, 1] += 0.45  # Action RIGHT
                elif inp == 2: # Concept: Food DOWN
                    self.w_kc2_mbon2[j, 2] += 0.45  # Action DOWN
                elif inp == 3: # Concept: Food LEFT
                    self.w_kc2_mbon2[j, 3] += 0.45  # Action LEFT

    def _init_layer2_projections(self):
        """สุ่มเชื่อมต่อ 16 มโนทัศน์ขั้นสูงไปยัง KC2 แต่ละตัว (3-6 connections)"""
        for j in range(self.num_kc2):
            k_conn = int(self.rng.integers(3, 7))
            chosen = self.rng.choice(self.num_pn2, size=k_conn, replace=False)
            self.w_pn2_kc2[chosen, j] = self.rng.uniform(0.5, 1.2, size=k_conn)

    def forward(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        prev_action: Optional[int] = None,
        temperature: Optional[float] = None,
        ego_obs: Optional[np.ndarray] = None,
        cpg_safe_mask: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward Pass ผ่านสมองซ้อน 2 ชั้น:
        1. Layer 1: Visual PN -> KC1 -> k-WTA1 (5%) -> Intermediate Concepts (MBON1)
        2. Layer 2: Concepts + Context + Ego Whiskers -> KC2 -> k-WTA2 (5%) -> Motor Actions (MBON2)
        """
        flat_obs = obs.flatten().astype(np.float32)
        self.last_pn1 = flat_obs

        # ================= 1. LAYER 1 (PERCEPTUAL ABSTRACTION) =================
        raw_kc1 = np.dot(flat_obs, self.w_pn1_kc1)
        raw_kc1 = np.maximum(0.0, raw_kc1)

        # k-WTA 1 (5% Sparsity: คัดเลือก Top 60 จาก 1,200)
        kc1_sparse = np.zeros(self.num_kc1, dtype=np.float32)
        top_k1_idx = np.argpartition(raw_kc1, -self.k_active1)[-self.k_active1:]
        kc1_sparse[top_k1_idx] = raw_kc1[top_k1_idx]
        self.last_kc1 = kc1_sparse

        # MBON1 Readout: ผลลัพธ์ 12 Situational Concepts
        concept_logits = np.dot(kc1_sparse, self.w_kc1_mbon1)
        concepts = 1.0 / (1.0 + np.exp(-np.clip(concept_logits, -6.0, 6.0)))

        # ผสาน Egocentric Whiskers เพื่อเสริมความแหลมคมของมโนทัศน์
        if ego_obs is not None:
            # ego_obs[6..9]: Food Ahead, Food Left, Food Right, Food Behind
            concepts[0] = np.clip(concepts[0] + 0.4 * ego_obs[6], 0.0, 1.0)
            concepts[1] = np.clip(concepts[1] + 0.4 * ego_obs[8], 0.0, 1.0)
            concepts[2] = np.clip(concepts[2] + 0.4 * ego_obs[9], 0.0, 1.0)
            concepts[3] = np.clip(concepts[3] + 0.4 * ego_obs[7], 0.0, 1.0)
            wall_threat = 1.0 - ego_obs[0]
            concepts[4] = np.clip(concepts[4] + 0.5 * wall_threat, 0.0, 1.0)
            concepts[6] = np.clip(concepts[6] + 0.5 * ego_obs[3], 0.0, 1.0)
            concepts[8] = np.clip(concepts[8] + 0.5 * ego_obs[11], 0.0, 1.0)

        self.last_concepts = concepts

        # อัปเดต Eligibility Trace 1 (จับคู่ KC1 กับ Active Concepts)
        self.eligibility_trace1 = (self.gamma * self.lambda1 * self.eligibility_trace1) + \
                                  np.outer(kc1_sparse, concepts)

        # ================= 2. LAYER 2 (STRATEGIC EXECUTIVE MB) =================
        # ประกอบ Sensory Input ของ Layer 2 (12 Concepts + 4 Context ทิศทางก่อนหน้า)
        context_vec = np.zeros(4, dtype=np.float32)
        if prev_action is not None and 0 <= prev_action < 4:
            context_vec[prev_action] = 1.0
        elif self.last_action is not None and 0 <= self.last_action < 4:
            context_vec[self.last_action] = 1.0
            
        pn2_input = np.concatenate([concepts, context_vec])

        # สัญญาณในชั้น KC2
        raw_kc2 = np.dot(pn2_input, self.w_pn2_kc2)
        raw_kc2 = np.maximum(0.0, raw_kc2)

        # k-WTA 2 (5% Sparsity: คัดเลือก Top 40 จาก 800)
        kc2_sparse = np.zeros(self.num_kc2, dtype=np.float32)
        top_k2_idx = np.argpartition(raw_kc2, -self.k_active2)[-self.k_active2:]
        kc2_sparse[top_k2_idx] = raw_kc2[top_k2_idx]
        self.last_kc2 = kc2_sparse

        # MBON2 Readout (4 Motor Actions)
        motor_logits = np.dot(kc2_sparse, self.w_kc2_mbon2)
        eff_temp = temperature if temperature is not None else self.temperature
        scaled_logits = motor_logits / max(eff_temp, 0.02)

        # นำ CPG Reflex Mask หรือ Action Mask มาคัดกรองทิศทาง
        effective_mask = cpg_safe_mask if cpg_safe_mask is not None else action_mask
        if effective_mask is not None:
            scaled_logits[effective_mask == False] = -1e9

        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        action_probs = exp_logits / max(float(np.sum(exp_logits)), 1e-6)
        self.last_mbon2_probs = action_probs

        # สุ่มหรือเลือกทิศทางที่มีคะแนนสูงสุด
        if deterministic or (eff_temp < 0.05):
            chosen_action = int(np.argmax(scaled_logits))
        else:
            chosen_action = int(self.rng.choice(self.num_mbon, p=action_probs))
        self.last_action = chosen_action

        # อัปเดต Eligibility Trace 2 (จับคู่ KC2 กับการกระทำที่เลือก)
        action_onehot = np.zeros(self.num_mbon, dtype=np.float32)
        action_onehot[chosen_action] = 1.0
        self.eligibility_trace2 = (self.gamma * self.lambda2 * self.eligibility_trace2) + \
                                  np.outer(kc2_sparse, action_onehot)

        return chosen_action, action_probs, kc1_sparse, concepts, kc2_sparse

    def export_weights(self) -> Dict[str, np.ndarray]:
        """ส่งออกค่าน้ำหนัก Synapse ทั้งสองชั้น"""
        return {
            "w_pn1_kc1": self.w_pn1_kc1.copy(),
            "w_kc1_mbon1": self.w_kc1_mbon1.copy(),
            "w_pn2_kc2": self.w_pn2_kc2.copy(),
            "w_kc2_mbon2": self.w_kc2_mbon2.copy(),
        }

    def load_weights(self, weights: Dict[str, np.ndarray]):
        """โหลดค่าน้ำหนัก Synapse ที่ฝึกฝนจาก GPU เข้าสู่โมเดล"""
        if "w_pn1_kc1" in weights:
            self.w_pn1_kc1 = weights["w_pn1_kc1"].astype(np.float32)
        if "w_kc1_mbon1" in weights:
            self.w_kc1_mbon1 = weights["w_kc1_mbon1"].astype(np.float32)
        if "w_pn2_kc2" in weights:
            self.w_pn2_kc2 = weights["w_pn2_kc2"].astype(np.float32)
        if "w_kc2_mbon2" in weights:
            self.w_kc2_mbon2 = weights["w_kc2_mbon2"].astype(np.float32)

    def update_plasticity(self, reward: float):
        """
        ปรับค่าน้ำหนัก Synapse ทั้งสองชั้นด้วย Dual Three-Factor Plasticity:
        - Layer 2: ปรับปรุงการเลือกทิศทางเดินตามผลลัพธ์
        - Layer 1: ปรับปรุงการจัดหมวดหมู่สถานการณ์ให้สอดคล้องกับคุณค่าที่ได้รับ
        """
        self.last_reward = reward

        # 1. อัปเดต Synapse ใน Layer 2 (Strategic Executive)
        delta_w2 = self.lr2 * reward * self.eligibility_trace2
        self.w_kc2_mbon2 = np.clip(self.w_kc2_mbon2 + delta_w2, 0.0, self.w_max2)

        # 2. อัปเดต Synapse ใน Layer 1 (Perceptual Abstraction)
        delta_w1 = self.lr1 * reward * self.eligibility_trace1
        self.w_kc1_mbon1 = np.clip(self.w_kc1_mbon1 + delta_w1, 0.0, self.w_max1)

        # ปรับอุณหภูมิสำรวจ (Exploration Temperature Decay)
        self.temperature = max(self.temp_min, self.temperature * self.temp_decay)

    def reset_traces(self):
        """รีเซ็ต Eligibility Traces ของทั้งสองชั้นเมื่อจบเกม"""
        self.eligibility_trace1.fill(0.0)
        self.eligibility_trace2.fill(0.0)

    def get_diagnostics(self) -> Dict[str, Any]:
        """ดึงข้อมูลสถิติของสมองทั้งสองชั้นสำหรับการวิเคราะห์และการแสดงผล"""
        active1 = int(np.sum(self.last_kc1 > 0))
        active2 = int(np.sum(self.last_kc2 > 0))
        return {
            "active_kc1": active1,
            "active_kc2": active2,
            "sparsity1_pct": (active1 / self.num_kc1) * 100.0,
            "sparsity2_pct": (active2 / self.num_kc2) * 100.0,
            "mean_weight_layer1": float(np.mean(self.w_kc1_mbon1)),
            "mean_weight_layer2": float(np.mean(self.w_kc2_mbon2)),
            "concepts": list(self.last_concepts),
            "mbon_probs": list(self.last_mbon2_probs),
            "temperature": self.temperature
        }
