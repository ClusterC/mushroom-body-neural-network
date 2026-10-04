import numpy as np
from typing import Dict, Any, Tuple, Optional

# ทิศทางการเคลื่อนที่หลัก
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class HDCVisualMushroomBody:
    """
    Hyperdimensional Computing / Vector Symbolic Architecture (HDC-VSA) Mushroom Body
    
    จำลองวงจรประสาทชีวภาพผ่านพีชคณิตเวกเตอร์มิติสูง (High-Dimensional Vector Algebra):
    1. Item Memory (Basis Hypervectors D=2,048): เก็บเวกเตอร์ที่ตั้งฉากกันโดยธรรมชาติ
    2. Role-Filler Binding (⊗): ผูกทิศทางสัมพัทธ์ (Roles) กับสิ่งที่ตรวจพบ (Fillers)
    3. Scene Bundling (+): รวมสถานการณ์ของสภาพแวดล้อมเป็น Hypervector ก้อนเดียว (Translation Invariant)
    4. APL Lateral Inhibition (k-WTA): คัดเลือก 5% Highest Activated Components (k=100) เลียนแบบ Kenyon Cells
    5. MBON Associative Memory: Clean-up Memory เทียบ Cosine Similarity สั่งการทิศทางเดิน
    6. Three-Factor Plasticity: ปรับปรุง Action Prototypes ตามรางวัลโดปามีน (Dopamine Bundling)
    """
    def __init__(
        self,
        dim: int = 2048,
        k_active: int = 100,
        num_mbon: int = 4,
        learning_rate: float = 0.08,
        gamma: float = 0.90,
        lambda_trace: float = 0.75,
        temperature: float = 0.05,
        seed: Optional[int] = 42
    ):
        self.dim = dim
        self.k_active = k_active  # ~5.0% Sparsity (100 จาก 2,048)
        self.num_mbon = num_mbon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_trace = lambda_trace
        self.temperature = temperature
        
        self.rng = np.random.default_rng(seed)

        # 1. Item Memory: สุ่มสร้าง Basis Hypervectors (Bipolar: -1 หรือ +1)
        self.item_memory = {}
        self._init_item_memory()

        # 2. Associative Clean-up Memory (MBON Action Prototypes: 4 x D)
        self.mbon_prototypes = np.zeros((self.num_mbon, self.dim), dtype=np.float32)
        self.eligibility_trace = np.zeros((self.num_mbon, self.dim), dtype=np.float32)
        self._init_innate_prototypes()

        # สถานะล่าสุดสำหรับ UI Visualizer และการวิเคราะห์
        self.last_scene_hypervector = np.zeros(self.dim, dtype=np.float32)
        self.last_sparse_kc = np.zeros(self.dim, dtype=np.float32)
        self.last_similarities = np.zeros(self.num_mbon, dtype=np.float32)
        self.last_action = 0
        self.last_dopamine = 0.0

    def _random_bipolar(self) -> np.ndarray:
        """สร้าง Random Bipolar Hypervector (-1 หรือ +1) ที่ตั้งฉากกันโดยธรรมชาติในมิติสูง"""
        return self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)

    def _init_item_memory(self):
        """
        สร้างคลังเวกเตอร์มิติสูงพื้นฐาน (Basis Hypervectors):
        - Roles (ทิศทางสัมพัทธ์รอบตัวงู): AHEAD, LEFT, RIGHT, BEHIND
        - Fillers (สิ่งที่ตรวจพบ): FOOD, WALL, BODY, EMPTY
        - Spatial Qualifiers: NEAR, FAR, TRAPPED, OPEN
        """
        keys = [
            # Roles
            "DIR_AHEAD", "DIR_LEFT", "DIR_RIGHT", "DIR_BEHIND",
            # Fillers
            "ENT_FOOD", "ENT_WALL", "ENT_BODY", "ENT_EMPTY",
            # Qualifiers
            "QUAL_NEAR", "QUAL_FAR", "STATE_TRAPPED", "STATE_OPEN"
        ]
        for key in keys:
            self.item_memory[key] = self._random_bipolar()

    def _init_innate_prototypes(self):
        """
        Innate Grounding: กำหนดความรู้สัญชาตญาณเริ่มต้นให้ Action Prototypes
        เช่น ทิศทางที่จะเลี้ยวสอดคล้องกับการเข้าหาอาหารและการหลบหลีกกำแพง
        """
        # ทิศทางการเลี้ยวสัมพัทธ์
        food = self.item_memory["ENT_FOOD"]
        ahead = self.item_memory["DIR_AHEAD"]
        left = self.item_memory["DIR_LEFT"]
        right = self.item_memory["DIR_RIGHT"]
        
        # ผูกคู่สัญลักษณ์ตั้งต้น
        # Forward Motion Prototype
        self.mbon_prototypes[0] += (ahead * food) * 1.5
        # Right Turn Prototype
        self.mbon_prototypes[1] += (right * food) * 1.5
        # Backward / Down Prototype
        self.mbon_prototypes[2] += (self.item_memory["DIR_BEHIND"] * food) * 1.0
        # Left Turn Prototype
        self.mbon_prototypes[3] += (left * food) * 1.5

        # Normalization
        for a in range(self.num_mbon):
            norm = np.linalg.norm(self.mbon_prototypes[a])
            if norm > 0:
                self.mbon_prototypes[a] /= norm

    def bind(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        """
        VSA Binding Operation (⊗): Element-wise multiplication (Hadamard product)
        คุณสมบัติ: ผลลัพธ์จะตั้งฉากกับทั้ง vec_a และ vec_b
        """
        return vec_a * vec_b

    def encode_scene(
        self,
        ego_obs: np.ndarray,
        visual_obs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        แปลงสภาพแวดล้อมเป็น Hypervector เชิงสัญลักษณ์ผ่าน Binding (⊗) และ Bundling (+)
        
        ego_obs: เวกเตอร์เรดาร์ 12 มิติ
        - 0..2: Wall Distances (Forward, Left, Right)
        - 3..5: Body Danger (Forward, Left, Right)
        - 6..9: Food Relative Bearings (Ahead, Left, Right, Behind)
        - 10: Food Distance
        - 11: Trapped Pocket Factor
        """
        scene = np.zeros(self.dim, dtype=np.float32)

        # 1. Food Relation Binding & Bundling
        # ถ้ามีอาหารอยู่ด้านไหน ให้ผูกทิศนั้นเข้ากับเวกเตอร์อาหาร
        if ego_obs[6] > 0.5:  # Food Ahead
            scene += 2.0 * self.bind(self.item_memory["DIR_AHEAD"], self.item_memory["ENT_FOOD"])
        if ego_obs[7] > 0.5:  # Food Left
            scene += 2.0 * self.bind(self.item_memory["DIR_LEFT"], self.item_memory["ENT_FOOD"])
        if ego_obs[8] > 0.5:  # Food Right
            scene += 2.0 * self.bind(self.item_memory["DIR_RIGHT"], self.item_memory["ENT_FOOD"])
        if ego_obs[9] > 0.5:  # Food Behind
            scene += 1.0 * self.bind(self.item_memory["DIR_BEHIND"], self.item_memory["ENT_FOOD"])

        # 2. Obstacle / Danger Binding & Bundling (กำแพงและลำตัว)
        # ตรวจสอบทิศทางด้านหน้า
        wall_threat_ahead = 1.0 - ego_obs[0]
        if wall_threat_ahead > 0.6:
            scene += 2.5 * self.bind(self.item_memory["DIR_AHEAD"], self.item_memory["ENT_WALL"])
        if ego_obs[3] > 0.0:  # Body Ahead
            scene += 3.0 * self.bind(self.item_memory["DIR_AHEAD"], self.item_memory["ENT_BODY"])

        # ด้านซ้าย
        wall_threat_left = 1.0 - ego_obs[1]
        if wall_threat_left > 0.6:
            scene += 2.5 * self.bind(self.item_memory["DIR_LEFT"], self.item_memory["ENT_WALL"])
        if ego_obs[4] > 0.0:  # Body Left
            scene += 3.0 * self.bind(self.item_memory["DIR_LEFT"], self.item_memory["ENT_BODY"])

        # ด้านขวา
        wall_threat_right = 1.0 - ego_obs[2]
        if wall_threat_right > 0.6:
            scene += 2.5 * self.bind(self.item_memory["DIR_RIGHT"], self.item_memory["ENT_WALL"])
        if ego_obs[5] > 0.0:  # Body Right
            scene += 3.0 * self.bind(self.item_memory["DIR_RIGHT"], self.item_memory["ENT_BODY"])

        # 3. Pocket State (ภาวะจนตรอกหรือพื้นที่เปิด)
        if ego_obs[11] > 0.6:
            scene += 2.0 * self.item_memory["STATE_TRAPPED"]
        else:
            scene += 1.0 * self.item_memory["STATE_OPEN"]

        self.last_scene_hypervector = scene
        return scene

    def sparsify_kc(self, scene_hypervector: np.ndarray) -> np.ndarray:
        """
        APL Lateral Inhibition (k-WTA): คัดเลือก Top 5% (k=100 จาก 2,048)
        เลียนแบบคุณสมบัติ Sparse Firing ของ Kenyon Cells ใน Mushroom Body
        """
        sparse_kc = np.zeros(self.dim, dtype=np.float32)
        top_k_indices = np.argpartition(scene_hypervector, -self.k_active)[-self.k_active:]
        sparse_kc[top_k_indices] = scene_hypervector[top_k_indices]
        self.last_sparse_kc = sparse_kc
        return sparse_kc

    def forward(
        self,
        ego_obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        cpg_safe_mask: Optional[np.ndarray] = None,
        direction: int = RIGHT,
        temperature: Optional[float] = None,
        deterministic: bool = True
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward Pass:
        1. Encode Scene -> VSA Binding & Bundling
        2. k-WTA Sparsification -> Kenyon Cell Hypervector
        3. Clean-up Memory Matching (Cosine Similarity กับ Action Prototypes)
        4. CPG Safe Mask Filtering & Softmax / Argmax Selection
        """
        # 1. แปลงสภาพแวดล้อมเป็น Scene Hypervector
        scene = self.encode_scene(ego_obs)

        # 2. ทำ k-WTA เป็น Sparse KC
        kc_sparse = self.sparsify_kc(scene)

        # 3. Clean-up Memory: คำนวณ Cosine Similarity เทียบกับ Action Prototypes
        # แปลง Relative Action (Ahead, Right, Down, Left) สู่ Global Actions (UP, RIGHT, DOWN, LEFT)
        # ตามทิศทางปัจจุบันของหัวงู
        rel_to_global = {
            0: direction,            # Ahead
            1: (direction + 1) % 4,  # Right
            2: (direction + 2) % 4,  # Behind
            3: (direction - 1) % 4   # Left
        }

        similarities = np.zeros(self.num_mbon, dtype=np.float32)
        norm_kc = np.linalg.norm(kc_sparse) + 1e-8

        for rel_act in range(4):
            glob_act = rel_to_global[rel_act]
            proto = self.mbon_prototypes[rel_act]
            norm_proto = np.linalg.norm(proto) + 1e-8
            sim = np.dot(kc_sparse, proto) / (norm_kc * norm_proto)
            similarities[glob_act] = sim

        self.last_similarities = similarities

        # 4. Action Masking & CPG Reflex Filter
        effective_mask = cpg_safe_mask if cpg_safe_mask is not None else action_mask
        masked_scores = similarities.copy()
        if effective_mask is not None:
            masked_scores[effective_mask == False] = -1e9

        eff_temp = temperature if temperature is not None else self.temperature

        # Softmax over Cosine Similarities
        scaled_scores = (masked_scores - np.max(masked_scores)) / max(eff_temp, 0.01)
        exp_scores = np.exp(scaled_scores)
        action_probs = exp_scores / max(float(np.sum(exp_scores)), 1e-6)

        if deterministic or eff_temp <= 0.05:
            chosen_action = int(np.argmax(masked_scores))
        else:
            chosen_action = int(self.rng.choice(self.num_mbon, p=action_probs))

        self.last_action = chosen_action

        # 5. สะสม Eligibility Trace (จับคู่ Relative Action กับ Sparse KC)
        # หา relative action ที่สอดคล้องกับ chosen_action
        chosen_rel = 0
        for rel_idx, g_idx in rel_to_global.items():
            if g_idx == chosen_action:
                chosen_rel = rel_idx
                break

        act_onehot = np.zeros((self.num_mbon, 1), dtype=np.float32)
        act_onehot[chosen_rel] = 1.0

        self.eligibility_trace = (self.gamma * self.lambda_trace * self.eligibility_trace) + \
                                 (act_onehot * kc_sparse)

        return chosen_action, action_probs, kc_sparse, similarities

    def update_plasticity(self, reward: float):
        """
        Three-Factor VSA Plasticity (Dopamine Bundling):
        ปรับปรุง Action Prototypes ใน Associative Clean-up Memory ตามรางวัล
        M_action = Normalize(M_action + eta * Dopamine * E_trace)
        """
        self.last_dopamine = reward
        if abs(reward) < 1e-5:
            return

        # ปรับปรุง Action Prototypes
        delta_m = self.learning_rate * reward * self.eligibility_trace
        self.mbon_prototypes += delta_m

        # Normalize แต่ละ Prototype เพื่อคงเสถียรภาพของความจำ
        for a in range(self.num_mbon):
            norm = np.linalg.norm(self.mbon_prototypes[a])
            if norm > 0:
                self.mbon_prototypes[a] /= norm

    def reset_traces(self):
        """รีเซ็ต Eligibility Trace เมื่อจบเกม"""
        self.eligibility_trace.fill(0.0)

    def export_weights(self) -> Dict[str, Any]:
        """ส่งออก Prototypes และ Item Memory"""
        return {
            "mbon_prototypes": self.mbon_prototypes.copy(),
            "item_memory": {k: v.copy() for k, v in self.item_memory.items()},
            "dim": self.dim,
            "k_active": self.k_active
        }

    def load_weights(self, weights: Dict[str, Any]):
        """โหลด Prototypes เข้าสู่โมเดล"""
        if "mbon_prototypes" in weights:
            self.mbon_prototypes = weights["mbon_prototypes"].copy()
        if "item_memory" in weights:
            self.item_memory = {k: v.copy() for k, v in weights["item_memory"].items()}

    def get_diagnostics(self) -> Dict[str, Any]:
        """รายงานสถิติของ HDC-VSA Circuit"""
        active_kc = int(np.sum(self.last_sparse_kc > 0))
        return {
            "dim": self.dim,
            "active_kc": active_kc,
            "sparsity_pct": (active_kc / self.dim) * 100.0,
            "similarities": list(self.last_similarities),
            "last_action": self.last_action,
            "last_dopamine": self.last_dopamine
        }
