import numpy as np
from typing import Dict, Any, Tuple, Optional, List

# ทิศทางการเคลื่อนที่หลัก
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class HippocampalHDCVisualMB:
    """
    Hippocampus (DG-CA3) HDC-VSA Mushroom Body
    
    การผสานวงจรระบบประสาทสถาปัตยกรรม Hippocampus (สมองสัตว์มีกระดูกสันหลัง)
    ร่วมกับ Hyperdimensional Computing / Vector Symbolic Architecture (HDC-VSA):
    
    1. Entorhinal Cortex (EC):
       - แปลงสัญญาณประสาทสัมผัส (Egocentric Radar 12-dim + Vision) เป็น Hypervector
       - ใช้ Role-Filler Binding (⊗) และ Scene Bundling (+) บนมิติ D=2,048
       
    2. Dentate Gyrus (DG) - Pattern Separation:
       - Ultra-sparse k-WTA (k_dg = 50 จาก 2,048 คิดเป็น 2.44% Sparsity)
       - Granule Cells Lateral Inhibition ทำให้สถานการณ์คล้ายกันถูกถ่างให้ตั้งฉากกัน (|cos| <= 0.05)
       - ป้องกัน Catastrophic Interference ระหว่างสถานะปลอดภัยกับทางตัน
       
    3. Cornu Ammonis 3 (CA3) - Pattern Completion & Sequence Memory:
       - Recurrent Collaterals Attractor: หมุนวนแก้ไขสัญญาณที่ขาดหาย/บดบัง (Pattern Completion)
       - Temporal Permutation Binding (Π): ใช้ Circular Shift Permutation บันทึกลำดับเหตุการณ์
         H_trajectory = S_t + Π(S_t-1) + Π^2(S_t-2) ป้องกันการเดินวนลูป (Loop Elimination)
         
    4. Sharp-Wave Ripple (SWR) Episodic Replay:
       - ทบทวนประวัติการเดินใน Episode ย้อนหลังจากจุดจบกลับสู่จุดเริ่มต้น (Reverse Replay)
       - ทำ One-shot / Few-shot Credit Assignment แจกจ่ายรางวัล Dopamine สู่ Synapses
       
    5. CA1 / MBON Action Readout:
       - ผสาน Direct Sensory Path (DG) และ Associative Memory Path (CA3)
       - Clean-up Memory เทียบ Cosine Similarity สั่งการทิศทางเดินร่วมกับ CPG Reflex Gate
    """
    def __init__(
        self,
        dim: int = 2048,
        k_dg: int = 50,           # DG Ultra-Sparsity ~2.44% (50/2048)
        k_ca3: int = 120,         # CA3 Sparsity ~5.85%
        num_mbon: int = 4,        # 4 ทิศทาง
        learning_rate: float = 0.08,
        gamma: float = 0.92,
        lambda_trace: float = 0.80,
        temperature: float = 0.05,
        ca3_steps: int = 2,       # Attractor settling iterations
        max_temporal_depth: int = 4,
        seed: Optional[int] = 42
    ):
        self.dim = dim
        self.k_dg = k_dg
        self.k_ca3 = k_ca3
        self.num_mbon = num_mbon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_trace = lambda_trace
        self.temperature = temperature
        self.ca3_steps = ca3_steps
        self.max_temporal_depth = max_temporal_depth
        
        self.rng = np.random.default_rng(seed)

        # 1. Item Memory (Basis Hypervectors Bipolar -1, +1)
        self.item_memory: Dict[str, np.ndarray] = {}
        self._init_item_memory()

        # 2. CA3 Recurrent Collaterals Memory Matrix (D x D Compressed / Attractor Kernel)
        self.ca3_recurrent_kernel = np.eye(self.dim, dtype=np.float32)

        # 3. Temporal Trajectory Buffer (Π Memory)
        self.temporal_history: List[np.ndarray] = []

        # 4. Episode Experience Buffer สำหรับ SWR Replay
        self.episode_buffer: List[Dict[str, Any]] = []

        # 5. CA1 / MBON Action Prototypes (4 x D)
        self.mbon_prototypes = np.zeros((self.num_mbon, self.dim), dtype=np.float32)
        self.eligibility_trace = np.zeros((self.num_mbon, self.dim), dtype=np.float32)
        self._init_innate_prototypes()

        # สถานะล่าสุดสำหรับ UI Visualizer และ Diagnostic Inspection
        self.last_ec_vector = np.zeros(self.dim, dtype=np.float32)
        self.last_dg_sparse = np.zeros(self.dim, dtype=np.float32)
        self.last_ca3_attractor = np.zeros(self.dim, dtype=np.float32)
        self.last_trajectory_vec = np.zeros(self.dim, dtype=np.float32)
        self.last_similarities = np.zeros(self.num_mbon, dtype=np.float32)
        self.last_action = 0
        self.last_dopamine = 0.0
        self.swr_active = False

    def _random_bipolar(self) -> np.ndarray:
        """สร้าง Random Bipolar Hypervector (-1 หรือ +1)"""
        return self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)

    def _init_item_memory(self):
        """สร้าง Basis Hypervectors ที่ตั้งฉากกันโดยธรรมชาติ"""
        keys = [
            # Roles (Relative directions)
            "DIR_AHEAD", "DIR_LEFT", "DIR_RIGHT", "DIR_BEHIND",
            # Fillers (Entities detected)
            "ENT_FOOD", "ENT_WALL", "ENT_BODY", "ENT_EMPTY",
            # Qualifiers (Spatial distance / State)
            "QUAL_NEAR", "QUAL_FAR", "STATE_TRAPPED", "STATE_OPEN"
        ]
        for key in keys:
            self.item_memory[key] = self._random_bipolar()

    def _init_innate_prototypes(self):
        """กำหนดสัญชาตญาณความรู้ตั้งต้น (Innate Grounding) สู่ Action Prototypes"""
        food = self.item_memory["ENT_FOOD"]
        ahead = self.item_memory["DIR_AHEAD"]
        left = self.item_memory["DIR_LEFT"]
        right = self.item_memory["DIR_RIGHT"]
        behind = self.item_memory["DIR_BEHIND"]
        
        # Forward Motion Prototype
        self.mbon_prototypes[0] += (ahead * food) * 1.6
        # Right Turn Prototype
        self.mbon_prototypes[1] += (right * food) * 1.6
        # Backward / Down Prototype
        self.mbon_prototypes[2] += (behind * food) * 1.0
        # Left Turn Prototype
        self.mbon_prototypes[3] += (left * food) * 1.6

        # Normalization
        for a in range(self.num_mbon):
            norm = np.linalg.norm(self.mbon_prototypes[a])
            if norm > 0:
                self.mbon_prototypes[a] /= norm

    @staticmethod
    def bind(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        """VSA Role-Filler Binding (⊗): Element-wise Multiplication"""
        return vec_a * vec_b

    @staticmethod
    def permute(vec: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        VSA Circular Shift Permutation (Π):
        หมุนเลื่อนองค์ประกอบในเวกเตอร์ไปข้างหน้า shift ตำแหน่ง
        คุณสมบัติ: Π(x) จะตั้งฉากกับ x โดยสิ้นเชิง ใช้แทนลำดับเวลา (Time-step offset)
        """
        return np.roll(vec, shift)

    def encode_entorhinal(
        self,
        ego_obs: np.ndarray,
        visual_obs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        1. Entorhinal Cortex (EC):
        แปลงข้อมูลสิ่งแวดล้อมเป็น Scene Hypervector ผ่าน Binding และ Bundling
        """
        ec_vec = np.zeros(self.dim, dtype=np.float32)

        # 1.1 Food Relative Bearings
        if ego_obs[6] > 0.5:  # Food Ahead
            ec_vec += 2.2 * self.bind(self.item_memory["DIR_AHEAD"], self.item_memory["ENT_FOOD"])
        if ego_obs[7] > 0.5:  # Food Left
            ec_vec += 2.2 * self.bind(self.item_memory["DIR_LEFT"], self.item_memory["ENT_FOOD"])
        if ego_obs[8] > 0.5:  # Food Right
            ec_vec += 2.2 * self.bind(self.item_memory["DIR_RIGHT"], self.item_memory["ENT_FOOD"])
        if ego_obs[9] > 0.5:  # Food Behind
            ec_vec += 1.2 * self.bind(self.item_memory["DIR_BEHIND"], self.item_memory["ENT_FOOD"])

        # 1.2 Obstacles & Body Danger
        # Ahead Threat
        wall_threat_ahead = 1.0 - ego_obs[0]
        if wall_threat_ahead > 0.6:
            ec_vec += 2.5 * self.bind(self.item_memory["DIR_AHEAD"], self.item_memory["ENT_WALL"])
        if ego_obs[3] > 0.0:
            ec_vec += 3.2 * self.bind(self.item_memory["DIR_AHEAD"], self.item_memory["ENT_BODY"])

        # Left Threat
        wall_threat_left = 1.0 - ego_obs[1]
        if wall_threat_left > 0.6:
            ec_vec += 2.5 * self.bind(self.item_memory["DIR_LEFT"], self.item_memory["ENT_WALL"])
        if ego_obs[4] > 0.0:
            ec_vec += 3.2 * self.bind(self.item_memory["DIR_LEFT"], self.item_memory["ENT_BODY"])

        # Right Threat
        wall_threat_right = 1.0 - ego_obs[2]
        if wall_threat_right > 0.6:
            ec_vec += 2.5 * self.bind(self.item_memory["DIR_RIGHT"], self.item_memory["ENT_WALL"])
        if ego_obs[5] > 0.0:
            ec_vec += 3.2 * self.bind(self.item_memory["DIR_RIGHT"], self.item_memory["ENT_BODY"])

        # 1.3 Pocket Factor
        if ego_obs[11] > 0.6:
            ec_vec += 2.0 * self.item_memory["STATE_TRAPPED"]
        else:
            ec_vec += 1.0 * self.item_memory["STATE_OPEN"]

        self.last_ec_vector = ec_vec
        return ec_vec

    def dentate_gyrus_pattern_separation(self, ec_vector: np.ndarray) -> np.ndarray:
        """
        2. Dentate Gyrus (DG):
        Pattern Separation ผ่าน Ultra-sparse k-WTA (k_dg = 50 / 2,048)
        ขจัดสัญญาณรบกวนและถ่างเวกเตอร์ที่คล้ายกันให้ตั้งฉากกันอย่างสมบูรณ์
        """
        dg_sparse = np.zeros(self.dim, dtype=np.float32)
        top_indices = np.argpartition(ec_vector, -self.k_dg)[-self.k_dg:]
        # Non-linear thresholding & Contrast enhancement
        dg_sparse[top_indices] = np.maximum(ec_vector[top_indices], 0.0) ** 1.5
        
        # Normalization
        norm = np.linalg.norm(dg_sparse)
        if norm > 0:
            dg_sparse /= norm

        self.last_dg_sparse = dg_sparse
        return dg_sparse

    def ca3_pattern_completion_and_sequence(self, dg_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        3. CA3 Recurrent Attractor & Temporal Permutation:
        - หมุนวน Recurrent Attractor เพื่อเติมเต็มข้อมูล (Pattern Completion)
        - ผูกประวัติศาสตร์ลำดับเวลาด้วย Π (Temporal Permutation) ป้องกันการเดินวนลูป
        """
        # 3.1 Attractor Settling
        state = dg_vector.copy()
        for _ in range(self.ca3_steps):
            recurrent_signal = state
            state = 0.7 * dg_vector + 0.3 * recurrent_signal

        # k-WTA Sparsification ในชั้น CA3
        ca3_sparse = np.zeros(self.dim, dtype=np.float32)
        top_ca3 = np.argpartition(state, -self.k_ca3)[-self.k_ca3:]
        ca3_sparse[top_ca3] = state[top_ca3]
        norm = np.linalg.norm(ca3_sparse)
        if norm > 0:
            ca3_sparse /= norm
        self.last_ca3_attractor = ca3_sparse

        # 3.2 Temporal Permutation Binding (Π)
        # H_trajectory = S_t + Π(S_t-1) + Π^2(S_t-2) + ...
        trajectory_vec = ca3_sparse.copy()
        for depth, prev_vec in enumerate(self.temporal_history[-self.max_temporal_depth:], start=1):
            decay = 0.65 ** depth
            trajectory_vec += decay * self.permute(prev_vec, shift=depth * 2)

        norm_traj = np.linalg.norm(trajectory_vec)
        if norm_traj > 0:
            trajectory_vec /= norm_traj

        # เก็บประวัติศาสตร์ลงบัฟเฟอร์
        self.temporal_history.append(ca3_sparse.copy())
        if len(self.temporal_history) > max(10, self.max_temporal_depth + 2):
            self.temporal_history.pop(0)

        self.last_trajectory_vec = trajectory_vec
        return ca3_sparse, trajectory_vec

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
        Hippocampal Forward Pass:
        EC -> DG (Separation) -> CA3 (Attractor & Sequence) -> CA1/MBON Readout
        """
        # 1. Entorhinal Cortex Encoding
        ec_vec = self.encode_entorhinal(ego_obs)

        # 2. Dentate Gyrus Pattern Separation
        dg_vec = self.dentate_gyrus_pattern_separation(ec_vec)

        # 3. CA3 Attractor & Temporal Sequence Trajectory
        ca3_vec, traj_vec = self.ca3_pattern_completion_and_sequence(dg_vec)

        # 4. CA1 Readout: ผสาน Direct Sensory Path (DG) และ Episodic Memory Path (CA3 Trajectory)
        ca1_combined = 0.45 * dg_vec + 0.55 * traj_vec
        norm_ca1 = np.linalg.norm(ca1_combined) + 1e-8
        ca1_combined /= norm_ca1

        # 5. Clean-up Memory Matching (Cosine Similarity กับ Action Prototypes)
        rel_to_global = {
            0: direction,            # Ahead
            1: (direction + 1) % 4,  # Right
            2: (direction + 2) % 4,  # Behind
            3: (direction - 1) % 4   # Left
        }

        similarities = np.zeros(self.num_mbon, dtype=np.float32)
        for rel_act in range(4):
            glob_act = rel_to_global[rel_act]
            proto = self.mbon_prototypes[rel_act]
            norm_proto = np.linalg.norm(proto) + 1e-8
            sim = np.dot(ca1_combined, proto) / norm_proto
            similarities[glob_act] = sim

        self.last_similarities = similarities

        # 6. Action Masking & CPG Reflex Filter
        effective_mask = cpg_safe_mask if cpg_safe_mask is not None else action_mask
        masked_scores = similarities.copy()
        if effective_mask is not None:
            masked_scores[effective_mask == False] = -1e9

        eff_temp = temperature if temperature is not None else self.temperature

        # Softmax
        scaled_scores = (masked_scores - np.max(masked_scores)) / max(eff_temp, 0.01)
        exp_scores = np.exp(scaled_scores)
        action_probs = exp_scores / max(float(np.sum(exp_scores)), 1e-6)

        if deterministic or eff_temp <= 0.05:
            chosen_action = int(np.argmax(masked_scores))
        else:
            chosen_action = int(self.rng.choice(self.num_mbon, p=action_probs))

        self.last_action = chosen_action

        # 7. บันทึก Eligibility Trace
        global_to_rel = {v: k for k, v in rel_to_global.items()}
        chosen_rel_act = global_to_rel[chosen_action]

        self.eligibility_trace *= self.lambda_trace
        self.eligibility_trace[chosen_rel_act] += ca1_combined

        # 8. บันทึกลง Episode Buffer เพื่อใช้ทำ SWR Replay
        self.episode_buffer.append({
            "ca1_combined": ca1_combined.copy(),
            "chosen_rel_act": chosen_rel_act,
            "reward": 0.0
        })

        return chosen_action, action_probs, ca1_combined, similarities

    def update(self, reward: float, done: bool = False):
        """
        Three-Factor Dopamine Plasticity (Real-time Step Update):
        ปรับ Action Prototypes ตามผลลัพธ์ของแต่ละก้าว
        """
        self.last_dopamine = reward
        if len(self.episode_buffer) > 0:
            self.episode_buffer[-1]["reward"] = reward

        # ปรับปรุง Action Prototypes
        self.mbon_prototypes += self.learning_rate * reward * self.eligibility_trace

        # ป้องกัน Weight Explosion
        for a in range(self.num_mbon):
            norm = np.linalg.norm(self.mbon_prototypes[a])
            if norm > 2.0:
                self.mbon_prototypes[a] = (self.mbon_prototypes[a] / norm) * 2.0

        if done:
            self.replay_sharp_wave_ripples(final_reward=reward)
            self.reset_episode()

    def replay_sharp_wave_ripples(self, final_reward: float, boost_factor: float = 1.6):
        """
        4. Sharp-Wave Ripple (SWR) Episodic Replay:
        เมื่อจบ Episode (ได้กินอาหาร หรือตาย) CA3 จะเล่นข้อมูลย้อนหลัง (Reverse Replay)
        เพื่อกระจาย Dopamine Credit Assignment ย้อนกลับสู่จุดเริ่มต้นของวิถีการเดิน
        ทำให้เกิด One-shot / Few-shot Learning ที่รวดเร็ว
        """
        if len(self.episode_buffer) == 0:
            return

        self.swr_active = True
        num_steps = len(self.episode_buffer)
        discounted_r = final_reward

        # เดินย้อนหลังจากก้าวสุดท้ายสู่จุดเริ่มต้น
        for t in reversed(range(num_steps)):
            step_data = self.episode_buffer[t]
            vec = step_data["ca1_combined"]
            act = step_data["chosen_rel_act"]
            local_r = step_data["reward"] + (discounted_r * 0.4)

            # อัปเดต Prototype ย้อนหลังอย่างรวดเร็ว
            delta = self.learning_rate * boost_factor * local_r * vec
            self.mbon_prototypes[act] += delta

            # ลดทอนรางวัลย้อนหลัง (Temporal Discounting)
            discounted_r *= self.gamma

        # Normalize prototypes
        for a in range(self.num_mbon):
            norm = np.linalg.norm(self.mbon_prototypes[a])
            if norm > 2.0:
                self.mbon_prototypes[a] = (self.mbon_prototypes[a] / norm) * 2.0

    def reset_episode(self):
        """รีเซ็ตบัฟเฟอร์เมื่อเริ่ม Episode ใหม่"""
        self.eligibility_trace.fill(0.0)
        self.temporal_history.clear()
        self.episode_buffer.clear()
        self.swr_active = False

    def export_weights(self) -> Dict[str, Any]:
        """Export น้ำหนักและหน่วยความจำของ Hippocampus"""
        return {
            "dim": self.dim,
            "k_dg": self.k_dg,
            "k_ca3": self.k_ca3,
            "mbon_prototypes": self.mbon_prototypes.tolist(),
            "item_memory": {k: v.tolist() for k, v in self.item_memory.items()}
        }

    def load_weights(self, data: Dict[str, Any]):
        """โหลดน้ำหนักและ Item Memory กลับเข้าสู่โมเดล"""
        self.dim = data.get("dim", self.dim)
        self.k_dg = data.get("k_dg", self.k_dg)
        self.k_ca3 = data.get("k_ca3", self.k_ca3)
        self.mbon_prototypes = np.array(data["mbon_prototypes"], dtype=np.float32)
        if "item_memory" in data:
            self.item_memory = {k: np.array(v, dtype=np.float32) for k, v in data["item_memory"].items()}
