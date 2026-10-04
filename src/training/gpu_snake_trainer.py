import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch

from src.envs.snake_env import SnakeEnv, UP, RIGHT, DOWN, LEFT
from src.models.stacked_visual_mb import StackedVisualMushroomBody

class GPUMushroomBodyTrainer:
    """
    ระบบเร่งการฝึกฝน Mushroom Body ด้วยการ์ดจอ NVIDIA GPU (PyTorch CUDA)
    จำลองเกมงูแบบขนานหลายร้อยเกม (Batched Parallel Training)
    และคำนวณ Sparse k-WTA กับ Three-Factor Plasticity บน CUDA Cores
    """
    def __init__(
        self,
        model: StackedVisualMushroomBody,
        batch_size: int = 256,
        device: Optional[str] = None
    ):
        self.model = model
        self.batch_size = batch_size
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"

        # ย้าย Weights และ Traces ขึ้น GPU
        self.w_pn1_kc1 = torch.from_numpy(model.w_pn1_kc1).to(self.device)
        self.w_kc1_mbon1 = torch.from_numpy(model.w_kc1_mbon1).to(self.device)
        self.w_pn2_kc2 = torch.from_numpy(model.w_pn2_kc2).to(self.device)
        self.w_kc2_mbon2 = torch.from_numpy(model.w_kc2_mbon2).to(self.device)

        # Traces ขนาด Batch บน GPU
        self.trace1 = torch.zeros(
            (batch_size, model.num_kc1, model.num_concepts),
            dtype=torch.float32,
            device=self.device
        )
        self.trace2 = torch.zeros(
            (batch_size, model.num_kc2, model.num_mbon),
            dtype=torch.float32,
            device=self.device
        )

        self.k_active1 = model.k_active1
        self.k_active2 = model.k_active2
        self.gamma = model.gamma
        self.lambda1 = model.lambda1
        self.lambda2 = model.lambda2
        self.lr1 = model.lr1
        self.lr2 = model.lr2
        self.w_max1 = model.w_max1
        self.w_max2 = model.w_max2

    def train(
        self,
        total_episodes: int = 5000,
        cpg_reflex: bool = True,
        use_whiskers: bool = True,
        temperature: float = 0.4
    ) -> Dict[str, Any]:
        """
        ฝึกฝนโมเดลแบบคู่ขนานบน GPU จนครบจำนวนรอบที่กำหนด
        """
        start_time = time.time()
        envs = [SnakeEnv(width=10, height=10, seed=1000 + i) for i in range(self.batch_size)]
        
        completed_episodes = 0
        total_apples = 0
        total_steps = 0
        
        obs_batch = np.zeros((self.batch_size, 300), dtype=np.float32)
        ego_batch = np.zeros((self.batch_size, 12), dtype=np.float32)
        safe_mask_batch = np.zeros((self.batch_size, 4), dtype=bool)
        prev_actions = np.zeros(self.batch_size, dtype=np.int64)

        for i, env in enumerate(envs):
            obs_batch[i] = env.get_visual_observation().flatten()
            if use_whiskers:
                ego_batch[i] = env.get_egocentric_observation()
            safe_mask_batch[i] = env.get_safe_action_mask() if cpg_reflex else env.get_action_mask()

        while completed_episodes < total_episodes:
            # 1. ย้ายข้อมูลเข้าสู่ CUDA Tensor
            t_obs = torch.from_numpy(obs_batch).to(self.device)
            t_masks = torch.from_numpy(safe_mask_batch).to(self.device)
            t_prev = torch.from_numpy(prev_actions).to(self.device)

            # 2. Forward Pass Layer 1 (300 PNs -> 1,200 KC1 -> 12 Concepts)
            raw_kc1 = torch.matmul(t_obs, self.w_pn1_kc1).clamp_min_(0.0)
            topk1 = torch.topk(raw_kc1, k=self.k_active1, dim=-1)
            kc1_sparse = torch.zeros_like(raw_kc1).scatter_(-1, topk1.indices, topk1.values)

            # MBON1: Concepts
            concept_logits = torch.matmul(kc1_sparse, self.w_kc1_mbon1)
            concepts = torch.sigmoid(concept_logits)

            if use_whiskers:
                t_ego = torch.from_numpy(ego_batch).to(self.device)
                concepts[:, 0] = torch.clamp(concepts[:, 0] + 0.4 * t_ego[:, 6], 0.0, 1.0)
                concepts[:, 1] = torch.clamp(concepts[:, 1] + 0.4 * t_ego[:, 8], 0.0, 1.0)
                concepts[:, 2] = torch.clamp(concepts[:, 2] + 0.4 * t_ego[:, 9], 0.0, 1.0)
                concepts[:, 3] = torch.clamp(concepts[:, 3] + 0.4 * t_ego[:, 7], 0.0, 1.0)
                concepts[:, 4] = torch.clamp(concepts[:, 4] + 0.5 * (1.0 - t_ego[:, 0]), 0.0, 1.0)
                concepts[:, 6] = torch.clamp(concepts[:, 6] + 0.5 * t_ego[:, 3], 0.0, 1.0)
                concepts[:, 8] = torch.clamp(concepts[:, 8] + 0.5 * t_ego[:, 11], 0.0, 1.0)

            # Eligibility Trace 1 Update: E1 = γλ E1 + (KC1 ⊗ Concepts)
            self.trace1 = (self.gamma * self.lambda1 * self.trace1) + \
                          torch.bmm(kc1_sparse.unsqueeze(2), concepts.unsqueeze(1))

            # 3. Forward Pass Layer 2 (16 PNs -> 800 KC2 -> 4 Motor Actions)
            context = torch.zeros((self.batch_size, 4), dtype=torch.float32, device=self.device)
            context.scatter_(1, t_prev.unsqueeze(1), 1.0)
            pn2_in = torch.cat([concepts, context], dim=-1)

            raw_kc2 = torch.matmul(pn2_in, self.w_pn2_kc2).clamp_min_(0.0)
            topk2 = torch.topk(raw_kc2, k=self.k_active2, dim=-1)
            kc2_sparse = torch.zeros_like(raw_kc2).scatter_(-1, topk2.indices, topk2.values)

            motor_logits = torch.matmul(kc2_sparse, self.w_kc2_mbon2)
            motor_logits.masked_fill_(~t_masks, -1e9)

            probs = torch.softmax(motor_logits / max(temperature, 0.02), dim=-1)
            # สำรวจหรือเลือกเดิน
            actions_t = torch.multinomial(probs, num_samples=1).squeeze(-1)
            actions = actions_t.cpu().numpy()

            # Eligibility Trace 2 Update: E2 = γλ E2 + (KC2 ⊗ Action)
            act_onehot = torch.zeros((self.batch_size, 4), dtype=torch.float32, device=self.device)
            act_onehot.scatter_(1, actions_t.unsqueeze(1), 1.0)
            self.trace2 = (self.gamma * self.lambda2 * self.trace2) + \
                          torch.bmm(kc2_sparse.unsqueeze(2), act_onehot.unsqueeze(1))

            # 4. ดำเนินการ 1 ก้าวในทุกสภาพแวดล้อมพร้อมกัน
            rewards_np = np.zeros(self.batch_size, dtype=np.float32)
            dones_np = np.zeros(self.batch_size, dtype=bool)

            for idx in range(self.batch_size):
                env = envs[idx]
                act = int(actions[idx])
                _, reward, done, info = env.step(act)
                rewards_np[idx] = reward
                dones_np[idx] = done
                total_steps += 1

                if reward >= 1.0:
                    total_apples += 1

                if done:
                    completed_episodes += 1
                    env.reset()
                    # รีเซ็ต Traces เฉพาะเกมที่จบ
                    self.trace1[idx].fill_(0.0)
                    self.trace2[idx].fill_(0.0)
                    prev_actions[idx] = 0

                obs_batch[idx] = env.get_visual_observation().flatten()
                if use_whiskers:
                    ego_batch[idx] = env.get_egocentric_observation()
                safe_mask_batch[idx] = env.get_safe_action_mask() if cpg_reflex else env.get_action_mask()
                prev_actions[idx] = act

            # 5. Parallel Three-Factor Plasticity Update บน GPU
            t_rewards = torch.from_numpy(rewards_np).to(self.device).view(-1, 1, 1)
            
            delta_w1 = self.lr1 * torch.mean(t_rewards * self.trace1, dim=0)
            delta_w2 = self.lr2 * torch.mean(t_rewards * self.trace2, dim=0)

            self.w_kc1_mbon1.add_(delta_w1).clamp_(0.0, self.w_max1)
            self.w_kc2_mbon2.add_(delta_w2).clamp_(0.0, self.w_max2)

        # 6. ซิงค์ Weights กลับเข้าสู่โมเดล CPU
        self.sync_to_model()

        duration = time.time() - start_time
        return {
            "device": self.device_name,
            "episodes": completed_episodes,
            "duration": duration,
            "fps": total_steps / max(duration, 1e-4),
            "total_apples": total_apples,
            "avg_apples": total_apples / max(completed_episodes, 1),
            "mean_w1": float(self.w_kc1_mbon1.mean().cpu()),
            "mean_w2": float(self.w_kc2_mbon2.mean().cpu())
        }

    def sync_to_model(self):
        """คัดลอก Synaptic Weights จาก GPU กลับสู่โมเดลหลัก"""
        self.model.w_kc1_mbon1 = self.w_kc1_mbon1.cpu().numpy()
        self.model.w_kc2_mbon2 = self.w_kc2_mbon2.cpu().numpy()
