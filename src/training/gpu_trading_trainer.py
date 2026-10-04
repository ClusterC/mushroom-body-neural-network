"""
GPU-Accelerated Parallel Trainer for Hippocampal Trading Mushroom Body (PyTorch CUDA).

Executes batched parallel trading environments (B=64-256) entirely through
CUDA-accelerated High-Dimensional VSA, Dentate Gyrus k-WTA, CA3 Attractors,
and Batched Three-Factor Hebbian Plasticity.
"""

import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from src.envs.stock_trading_env import StockTradingEnv, HOLD, BUY, SELL
from src.models.hippocampal_trading_mb import (
    HippocampalTradingMB,
    FEATURE_ROLES,
    REGIME_BULL_EXPANSION,
    REGIME_BEAR_DISTRIBUTION,
    REGIME_CHOPPY_SIDEWAYS,
    REGIME_VOLATILE_SHOCK,
)


class GPUTradingTrainer:
    """
    High-Performance GPU-Accelerated Trainer for HippocampalTradingMB.
    Leverages NVIDIA CUDA cores to execute:
      1. Batched VSA Role-Filler Hypervector Binding (D=2,048)
      2. Dentate Gyrus Ultra-sparse k-WTA (k=50) via torch.topk
      3. CA3 Recurrent Attractor Permutations (torch.roll)
      4. Cognitive Regime Classification & Permissive Gating
      5. Batched Three-Factor Hebbian Plasticity with Dopamine Modulation
    """

    def __init__(
        self,
        model: HippocampalTradingMB,
        batch_size: int = 128,
        device: Optional[str] = None
    ):
        self.model = model
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        self.dim = model.dim
        self.num_actions = model.num_actions
        self.num_regimes = model.num_regimes
        self.k_dg = model.k_dg
        self.perm_shift = model.perm_shift
        self.ca3_steps = model.ca3_steps
        self.gamma = model.gamma
        self.lambda_trace = model.lambda_trace
        self.learning_rate = model.learning_rate
        self.temperature = model.temperature

        # 1. Hypervector Memory Tensors on GPU
        # Role hypervectors: (18, dim)
        roles = [model.item_memory[role] for role in FEATURE_ROLES]
        self.role_vectors = torch.from_numpy(np.array(roles, dtype=np.float32)).to(self.device)

        # Level hypervectors: (25, dim)
        self.level_vectors = torch.from_numpy(np.array(model.level_hypervectors, dtype=np.float32)).to(self.device)
        self.num_levels = len(model.level_hypervectors)

        # Action Prototypes: (dim, 3)
        self.action_prototypes = torch.from_numpy(model.action_prototypes.copy()).to(self.device)

        # Cognitive Regime Prototypes: (dim, 4)
        self.regime_prototypes = torch.from_numpy(model.regime_prototypes.copy()).to(self.device)

        # 2. Batched Eligibility Traces on GPU: (B, dim, 3)
        self.traces = torch.zeros((batch_size, self.dim, self.num_actions), dtype=torch.float32, device=self.device)

        # 3. Batched Temporal History Buffer on GPU: (B, 5, dim)
        self.temporal_history = torch.zeros((batch_size, 5, self.dim), dtype=torch.float32, device=self.device)
        self.temporal_depth = 5

    def encode_vsa_batch(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """
        Vectorized Entorhinal Cortex VSA Role-Filler Binding on CUDA:
        obs_batch: (B, 18) float32 in range [-2.0, 2.0]
        Returns: (B, dim) bipolar hypervectors
        """
        # Quantize all 18 features simultaneously to level indices [0, num_levels - 1]
        norm_val = torch.clamp((obs_batch - (-2.0)) / 4.0, 0.0, 0.999)
        level_indices = (norm_val * self.num_levels).long()  # (B, 18)

        # Gather level vectors: (B, 18, dim)
        fillers = self.level_vectors[level_indices]

        # Role-Filler Binding via Hadamard product: (1, 18, dim) * (B, 18, dim)
        roles_expanded = self.role_vectors.unsqueeze(0)  # (1, 18, dim)
        bound = roles_expanded * fillers                 # (B, 18, dim)

        # Superposition across 18 features & bipolar thresholding
        scene_sum = torch.sum(bound, dim=1)              # (B, dim)
        ec_vectors = torch.sign(scene_sum)
        # Handle exact zero
        ec_vectors = torch.where(ec_vectors == 0.0, torch.ones_like(ec_vectors), ec_vectors)
        return ec_vectors

    def dentate_gyrus_batch(self, ec_vectors: torch.Tensor) -> torch.Tensor:
        """
        Dentate Gyrus (DG) Ultra-sparse k-WTA (k=50) across batch on CUDA:
        Returns: (B, dim) binary sparse firing tensor
        """
        topk = torch.topk(ec_vectors, k=self.k_dg, dim=-1)
        dg_sparse = torch.zeros_like(ec_vectors)
        dg_sparse.scatter_(-1, topk.indices, 1.0)
        return dg_sparse

    def ca3_recurrent_batch(self, dg_sparse: torch.Tensor) -> torch.Tensor:
        """
        CA3 Temporal Permutation & Recurrent Attractor Dynamics on CUDA.
        """
        # Circular shift permutation
        perm_current = torch.roll(dg_sparse, shifts=self.perm_shift, dims=-1)

        # Shift temporal history buffer
        self.temporal_history = torch.cat(
            [self.temporal_history[:, 1:, :], perm_current.unsqueeze(1)],
            dim=1
        )

        # Accumulate temporal trajectory sequence
        ca3_trajectory = dg_sparse.clone()
        decay = 0.80
        curr_decay = decay
        for step_idx in range(self.temporal_depth - 2, -1, -1):
            ca3_trajectory += self.temporal_history[:, step_idx, :] * curr_decay
            curr_decay *= decay

        # Recurrent Attractor Pattern Completion
        for _ in range(self.ca3_steps):
            norms = torch.norm(ca3_trajectory, dim=-1, keepdim=True).clamp_min(1e-6)
            ca3_trajectory = ca3_trajectory / norms
            ca3_trajectory = torch.tanh(ca3_trajectory * 1.5)

        return ca3_trajectory

    def classify_regimes_batch(self, dg_sparse: torch.Tensor) -> torch.Tensor:
        """
        Cognitive Regime Classification across batch on CUDA.
        Returns: (B,) regime indices (0: BULL, 1: BEAR, 2: SIDEWAYS, 3: SHOCK)
        """
        # Cosine similarity with regime prototypes: (B, dim) x (dim, 4) -> (B, 4)
        dg_norm = dg_sparse / torch.norm(dg_sparse, dim=-1, keepdim=True).clamp_min(1e-6)
        proto_norm = self.regime_prototypes / torch.norm(self.regime_prototypes, dim=0, keepdim=True).clamp_min(1e-6)
        sims = torch.matmul(dg_norm, proto_norm)
        return torch.argmax(sims, dim=-1)

    def select_actions_batch(
        self,
        ca3_vectors: torch.Tensor,
        regimes: torch.Tensor,
        action_masks: torch.Tensor,
        is_holding: torch.Tensor,
        obs_batch: Optional[torch.Tensor] = None,
        temperature: float = 0.20
    ) -> torch.Tensor:
        """
        Batched Action Selection with Regime-Driven Permissive Gating on CUDA.
        Integrates Golden Pullback Sniper confluence and CPG Risk Reflex on GPU.
        """
        # Action similarities: (B, dim) x (dim, 3) -> (B, 3)
        action_sims = torch.matmul(ca3_vectors, self.action_prototypes)

        # 1. Action Masking (prevent illegal actions)
        action_sims = torch.where(action_masks, action_sims, torch.full_like(action_sims, -1e9))

        # 2. Golden Pullback Sniper Confluence & CPG Risk Reflex (if obs_batch provided)
        if obs_batch is not None:
            rsi_val = (obs_batch[:, 3] + 1.0) / 2.0
            dist_sma = obs_batch[:, 5] / 10.0
            channel_val = (obs_batch[:, 17] + 1.0) / 2.0
            is_bull_trend = (obs_batch[:, 12] > 0.0) & (obs_batch[:, 4] > 0.015)
            is_momentum = (obs_batch[:, 13] > 0.0)
            is_pullback = (rsi_val >= 0.38) & (rsi_val <= 0.54)
            is_near_sma = (dist_sma >= 0.000) & (dist_sma <= 0.016)
            is_green = (obs_batch[:, 0] > 0.0005)
            has_support = (channel_val >= 0.25)
            is_safe_vol = (obs_batch[:, 6] <= 0.40)
            is_bull_regime = (regimes == REGIME_BULL_EXPANSION)

            can_sniper_buy = (
                is_bull_trend & is_momentum & is_pullback & is_near_sma &
                is_green & has_support & is_safe_vol & is_bull_regime
            )

            # Not holding: Strict Sniper Gating
            not_holding = (is_holding <= 0.5)
            unauthorized_buy = not_holding & (~can_sniper_buy)
            action_sims[:, BUY] = torch.where(unauthorized_buy, torch.full_like(action_sims[:, BUY], -1e9), action_sims[:, BUY])
            action_sims[:, HOLD] = torch.where(unauthorized_buy, action_sims[:, HOLD] + 2.0, action_sims[:, HOLD])

            authorized_buy = not_holding & can_sniper_buy
            action_sims[:, BUY] = torch.where(authorized_buy, action_sims[:, BUY] + 2.5, action_sims[:, BUY])

            # Holding: CPG Risk Reflex on GPU
            holding_mask = (is_holding > 0.5)
            unrealized_pnl = obs_batch[:, 9] / 10.0

            hit_tp = (unrealized_pnl >= 0.0080)
            hit_rsi_tp = (rsi_val >= 0.56) & (unrealized_pnl >= 0.0048)
            hit_chan_tp = (channel_val >= 0.72) & (unrealized_pnl >= 0.0048)
            hit_sl = (unrealized_pnl <= -0.035)

            force_exit = holding_mask & (hit_tp | hit_rsi_tp | hit_chan_tp | hit_sl)
            action_sims[:, SELL] = torch.where(force_exit, torch.full_like(action_sims[:, SELL], 1e9), action_sims[:, SELL])
            action_sims[:, HOLD] = torch.where(force_exit, torch.full_like(action_sims[:, HOLD], -1e9), action_sims[:, HOLD])
            action_sims[:, BUY] = torch.where(force_exit, torch.full_like(action_sims[:, BUY], -1e9), action_sims[:, BUY])

            # Default holding inertia when not force-exiting
            normal_holding = holding_mask & (~force_exit)
            action_sims[:, HOLD] = torch.where(normal_holding, action_sims[:, HOLD] + 1.5, action_sims[:, HOLD])
            action_sims[:, SELL] = torch.where(normal_holding, action_sims[:, SELL] - 0.5, action_sims[:, SELL])
        else:
            # Fallback legacy regime permissive gating
            bear_mask = (regimes == REGIME_BEAR_DISTRIBUTION)
            action_sims[:, BUY] = torch.where(bear_mask, torch.full_like(action_sims[:, BUY], -1e9), action_sims[:, BUY])

            bull_holding = (regimes == REGIME_BULL_EXPANSION) & (is_holding > 0.5)
            action_sims[:, HOLD] = torch.where(bull_holding, action_sims[:, HOLD] + 0.35, action_sims[:, HOLD])

            choppy_mask = (regimes == REGIME_CHOPPY_SIDEWAYS)
            action_sims[:, HOLD] = torch.where(choppy_mask, action_sims[:, HOLD] + 0.20, action_sims[:, HOLD])

        # 3. Softmax Action Selection or Argmax
        temp = max(temperature, 0.05)
        probs = torch.softmax(action_sims / temp, dim=-1)

        # Check for NaN / invalid values
        if torch.isnan(probs).any():
            probs = torch.nan_to_num(probs, nan=0.333)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return actions

    def train(
        self,
        total_episodes: int = 500,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Train HippocampalTradingMB across batched parallel environments on GPU.
        """
        start_time = time.time()
        profiles = ["TECH_MOMENTUM", "INDEX_ETF", "CRYPTO_VOLATILE", "DEFENSIVE_VALUE"]

        # Dynamically scale batch size if total_episodes is smaller than batch_size
        effective_batch = min(self.batch_size, max(1, total_episodes))

        # Initialize parallel environments
        envs = [
            StockTradingEnv(
                initial_cash=10000.0,
                max_steps=252,
                asset_profile=profiles[i % len(profiles)],
                seed=2000 + i
            )
            for i in range(effective_batch)
        ]

        obs_batch = np.zeros((effective_batch, 18), dtype=np.float32)
        masks_batch = np.zeros((effective_batch, 3), dtype=bool)
        holding_batch = np.zeros(effective_batch, dtype=np.float32)

        for i, env in enumerate(envs):
            obs_batch[i] = env.reset()
            masks_batch[i] = env.get_action_mask()
            holding_batch[i] = 1.0 if env.shares > 0 else 0.0

        completed_episodes = 0
        total_steps = 0
        total_wins = 0
        total_trades = 0
        total_pnl = 0.0

        # Reset traces and buffers for effective batch size
        batch_traces = torch.zeros((effective_batch, self.dim, self.num_actions), dtype=torch.float32, device=self.device)
        batch_temporal = torch.zeros((effective_batch, self.temporal_depth, self.dim), dtype=torch.float32, device=self.device)

        while completed_episodes < total_episodes:
            # 1. Transfer current batch sensory inputs to GPU
            t_obs = torch.from_numpy(obs_batch).to(self.device)
            t_masks = torch.from_numpy(masks_batch).to(self.device)
            t_holding = torch.from_numpy(holding_batch).to(self.device)

            # 2. CUDA Forward Pass: VSA -> DG -> CA3 -> Regime -> Action
            ec_vectors = self.encode_vsa_batch(t_obs)
            dg_sparse = self.dentate_gyrus_batch(ec_vectors)
            
            # CA3 Recurrent Attractor with local batch temporal buffer
            perm_current = torch.roll(dg_sparse, shifts=self.perm_shift, dims=-1)
            batch_temporal = torch.cat(
                [batch_temporal[:, 1:, :], perm_current.unsqueeze(1)],
                dim=1
            )
            ca3_vectors = dg_sparse.clone()
            decay = 0.80
            curr_decay = decay
            for step_idx in range(self.temporal_depth - 2, -1, -1):
                ca3_vectors += batch_temporal[:, step_idx, :] * curr_decay
                curr_decay *= decay

            for _ in range(self.ca3_steps):
                norms = torch.norm(ca3_vectors, dim=-1, keepdim=True).clamp_min(1e-6)
                ca3_vectors = ca3_vectors / norms
                ca3_vectors = torch.tanh(ca3_vectors * 1.5)

            regimes = self.classify_regimes_batch(dg_sparse)

            actions_tensor = self.select_actions_batch(
                ca3_vectors, regimes, t_masks, t_holding, obs_batch=t_obs, temperature=self.temperature
            )
            actions = actions_tensor.cpu().numpy()

            # 3. Update Eligibility Traces on GPU: E = γλ E + (CA3 ⊗ Action)
            actions_onehot = torch.zeros((effective_batch, self.num_actions), dtype=torch.float32, device=self.device)
            actions_onehot.scatter_(1, actions_tensor.unsqueeze(1), 1.0)
            batch_traces = (self.gamma * self.lambda_trace * batch_traces) + \
                           torch.bmm(ca3_vectors.unsqueeze(2), actions_onehot.unsqueeze(1))

            # 4. Step Environments in Parallel
            rewards_np = np.zeros(effective_batch, dtype=np.float32)
            dones_np = np.zeros(effective_batch, dtype=bool)

            for i in range(effective_batch):
                env = envs[i]
                act = int(actions[i])
                next_obs, rew, done, info = env.step(act)
                rewards_np[i] = rew
                dones_np[i] = done
                total_steps += 1

                if done:
                    completed_episodes += 1
                    total_pnl += (env.net_worth - env.initial_cash)
                    total_wins += env.winning_trades
                    total_trades += env.total_trades

                    if completed_episodes >= total_episodes:
                        break

                    # Reset env with new seed / profile
                    next_prof = profiles[completed_episodes % len(profiles)]
                    envs[i] = StockTradingEnv(
                        initial_cash=10000.0,
                        max_steps=252,
                        asset_profile=next_prof,
                        seed=3000 + completed_episodes
                    )
                    next_obs = envs[i].reset()
                    batch_traces[i].fill_(0.0)
                    batch_temporal[i].fill_(0.0)

                obs_batch[i] = next_obs
                masks_batch[i] = envs[i].get_action_mask()
                holding_batch[i] = 1.0 if envs[i].shares > 0 else 0.0

            # 5. Batched Three-Factor Hebbian Plasticity Update on CUDA
            t_rewards = torch.from_numpy(rewards_np).to(self.device).view(-1, 1, 1)
            delta_w = self.learning_rate * torch.mean(t_rewards * batch_traces, dim=0)
            self.action_prototypes.add_(delta_w).clamp_(-3.0, 3.0)

            # Progress tracking
            if progress_callback is not None:
                progress_callback(min(completed_episodes, total_episodes), total_episodes)

            if completed_episodes >= total_episodes:
                break

        # 6. Synchronize trained weights back to CPU model
        self.sync_to_model()

        duration = time.time() - start_time
        win_rate = (total_wins / max(1, total_trades)) * 100.0

        return {
            "device": self.device_name,
            "episodes": completed_episodes,
            "duration": duration,
            "fps": total_steps / max(duration, 1e-4),
            "win_rate": win_rate,
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "mean_weight": float(self.action_prototypes.mean().cpu())
        }

    def sync_to_model(self):
        """Synchronize synaptic weights and prototypes from GPU back to NumPy model."""
        self.model.action_prototypes = self.action_prototypes.cpu().numpy().astype(np.float32)
        self.model.regime_prototypes = self.regime_prototypes.cpu().numpy().astype(np.float32)
