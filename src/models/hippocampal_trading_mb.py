"""
Hippocampal Cognitive Map (DG-CA3) HDC-VSA Neural Architecture for Algorithmic Stock Trading.
Integrates continuous feature quantization, Dentate Gyrus ultra-sparse pattern separation (2.44%),
CA3 temporal trajectory permutation (Π), Sharp-Wave Ripple (SWR) episodic replay,
and a spinal Central Pattern Generator (CPG) Risk Reflex (Hard Stop-Loss).
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List

HOLD = 0
BUY = 1
SELL = 2

FEATURE_ROLES = [
    "ROLE_RET1", "ROLE_RET5", "ROLE_RET20", "ROLE_RSI",
    "ROLE_SMA_RATIO", "ROLE_PRICE_SMA", "ROLE_VOL", "ROLE_VOL_RATIO",
    "ROLE_HOLDING", "ROLE_UNREALIZED", "ROLE_DURATION", "ROLE_DRAWDOWN",
    "ROLE_TREND", "ROLE_MOMENTUM", "ROLE_RSI_SIG", "ROLE_CUM_RET",
    "ROLE_VWAP", "ROLE_CHANNEL"
]


class HippocampalTradingMB:
    """
    Bio-inspired Hippocampal Trading Agent:
    - Entorhinal Cortex (EC): Thermometer level encoding & Role-Filler Hypervector Binding (D=2,048)
    - Dentate Gyrus (DG): Ultra-sparse k-WTA (k=50, 2.44%) for market regime separation
    - CA3 Attractor & Trajectory Memory: Multi-bar temporal permutation (Π)
    - SWR Episodic Replay: Post-trade reverse replay for one-shot dopamine credit assignment
    - Spinal CPG Risk Reflex: Enforces hard stop-loss (-3.0%), trailing profit lock, and volatility filters
    """
    def __init__(
        self,
        dim: int = 2048,
        k_dg: int = 50,              # DG Ultra-sparsity ~2.44% (50/2048)
        k_ca3: int = 120,            # CA3 Sparsity ~5.85%
        num_actions: int = 3,        # HOLD (0), BUY (1), SELL (2)
        learning_rate: float = 0.06,
        gamma: float = 0.92,
        lambda_trace: float = 0.85,
        temperature: float = 0.20,
        temp_min: float = 0.02,
        temp_decay: float = 0.995,
        stop_loss_pct: float = -0.03,         # -3.0% hard stop-loss
        trailing_stop_pct: float = 0.02,     # 2.0% trailing profit lock from peak
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
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.ca3_steps = 2
        self.perm_shift = 17

        self.rng = np.random.default_rng(seed)

        # 1. Item Memory & Continuous Level Quantization
        self.item_memory: Dict[str, np.ndarray] = {}
        self.num_levels = 25
        self.level_hypervectors: List[np.ndarray] = []
        self._init_item_memory()

        # 2. Temporal Trajectory Memory (Π Buffer)
        self.temporal_history: List[np.ndarray] = []
        self.max_temporal_depth = 5

        # 3. Action Prototypes & Plastic Synaptic Weights (dim x num_actions)
        self.action_prototypes = self.rng.uniform(0.05, 0.15, size=(dim, num_actions)).astype(np.float32)
        self._init_innate_prototypes()

        # 4. Eligibility Traces for Three-Factor Plasticity
        self.eligibility_traces = np.zeros((dim, num_actions), dtype=np.float32)

        # 5. Episodic Memory Buffer for SWR Replay
        self.episode_experiences: List[Dict[str, Any]] = []

        # 6. Telemetry & Alerts
        self.last_swr_active = False
        self.last_cpg_triggered = False
        self.last_cpg_reason = ""
        self.last_selected_action = HOLD
        self.last_action_probs = np.ones(num_actions) / num_actions
        self.last_dg_indices = np.array([], dtype=int)
        self.last_ca3_depth = 0
        self.inaction_counter = 0
        self.peak_unrealized_pnl = 0.0

    def _init_innate_prototypes(self):
        """Initialize innate biological grounding for BUY, SELL, and HOLD prototypes."""
        high_level = self.level_hypervectors[-1]
        low_level = self.level_hypervectors[0]
        mid_level = self.level_hypervectors[self.num_levels // 2]

        buy_innate = (
            self.item_memory["ROLE_TREND"] * high_level * 2.5 +
            self.item_memory["ROLE_MOMENTUM"] * high_level * 2.5 +
            self.item_memory["ROLE_SMA_RATIO"] * high_level * 2.0 +
            self.item_memory["ROLE_HOLDING"] * low_level * 2.0 +
            self.item_memory["ROLE_CHANNEL"] * high_level * 2.0 +
            self.item_memory["ROLE_VWAP"] * high_level * 1.5
        )

        sell_innate = (
            self.item_memory["ROLE_TREND"] * low_level * 2.5 +
            self.item_memory["ROLE_MOMENTUM"] * low_level * 2.0 +
            self.item_memory["ROLE_RSI_SIG"] * high_level * 2.5 +
            self.item_memory["ROLE_HOLDING"] * high_level * 2.0 +
            self.item_memory["ROLE_CHANNEL"] * low_level * 2.0
        )

        hold_innate = (
            self.item_memory["ROLE_VOL"] * mid_level * 0.5 +
            self.item_memory["ROLE_TREND"] * mid_level * 0.5
        )

        self.action_prototypes[:, HOLD] += np.maximum(0.01, hold_innate * 0.05)
        self.action_prototypes[:, BUY] += np.maximum(0.01, buy_innate * 0.25)
        self.action_prototypes[:, SELL] += np.maximum(0.01, sell_innate * 0.25)

    def _init_item_memory(self):
        """Initialize orthogonal bipolar hypervectors (-1, +1) for feature roles and level bins."""
        for role in FEATURE_ROLES:
            vec = self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)
            self.item_memory[role] = vec

        # Level hypervectors: Gradual flipping from Level 0 to Level N (Thermometer coding)
        base = self.rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)
        flips_per_level = self.dim // self.num_levels
        curr = base.copy()
        self.level_hypervectors.append(curr.copy())

        for _ in range(1, self.num_levels):
            flip_idx = self.rng.choice(self.dim, size=flips_per_level, replace=False)
            curr[flip_idx] *= -1.0
            self.level_hypervectors.append(curr.copy())

    def _quantize_feature(self, val: float, val_min: float = -2.0, val_max: float = 2.0) -> np.ndarray:
        """Map a scalar feature into a continuous level hypervector."""
        norm_val = np.clip((val - val_min) / max(1e-5, val_max - val_min), 0.0, 0.999)
        idx = int(norm_val * self.num_levels)
        return self.level_hypervectors[idx]

    def encode_entorhinal_cortex(self, obs: np.ndarray) -> np.ndarray:
        """
        Entorhinal Cortex (EC):
        Binds 16 technical features into high-dimensional scene hypervector via Role-Filler binding.
        """
        scene_vec = np.zeros(self.dim, dtype=np.float32)
        for i, role in enumerate(FEATURE_ROLES):
            val = float(obs[i]) if i < len(obs) else 0.0
            filler_vec = self._quantize_feature(val)
            role_vec = self.item_memory[role]
            # Role-Filler Binding via Hadamard product
            bound_vec = role_vec * filler_vec
            scene_vec += bound_vec

        # Bipolar normalization
        scene_vec = np.where(scene_vec >= 0, 1.0, -1.0).astype(np.float32)
        return scene_vec

    def dentate_gyrus_separation(self, ec_vector: np.ndarray) -> np.ndarray:
        """
        Dentate Gyrus (DG) Layer:
        Ultra-sparse k-WTA (k=50, 2.44% sparsity) enforcing orthogonal representation across market regimes.
        """
        # Linear projection with non-linear thresholding
        activations = ec_vector.copy()
        k = min(self.k_dg, self.dim)
        top_k_indices = np.argpartition(activations, -k)[-k:]

        dg_sparse = np.zeros(self.dim, dtype=np.float32)
        # Rectified positive firing rate for top-k granule cells
        dg_sparse[top_k_indices] = 1.0
        self.last_dg_indices = top_k_indices
        return dg_sparse

    def ca3_recurrent_sequence(self, dg_sparse: np.ndarray) -> np.ndarray:
        """
        Cornu Ammonis 3 (CA3) Layer:
        Recurrent Attractor Network + Multi-bar Temporal Permutation (Π) encoding market sequence history.
        """
        # Circular shift permutation
        perm_current = np.roll(dg_sparse, self.perm_shift)
        self.temporal_history.append(perm_current)
        if len(self.temporal_history) > self.max_temporal_depth:
            self.temporal_history.pop(0)

        # Accumulate temporal trajectory sequence
        ca3_trajectory = dg_sparse.copy()
        decay = 0.80
        curr_decay = decay
        for hist_vec in reversed(self.temporal_history[:-1]):
            ca3_trajectory += hist_vec * curr_decay
            curr_decay *= decay

        # Recurrent Attractor Pattern Completion (2 iterations)
        for _ in range(self.ca3_steps):
            norm = np.linalg.norm(ca3_trajectory)
            if norm > 1e-6:
                ca3_trajectory /= norm
            ca3_trajectory = np.maximum(0.0, ca3_trajectory - 0.05)

        # CA3 k-WTA Sparsification
        k = min(self.k_ca3, self.dim)
        top_indices = np.argpartition(ca3_trajectory, -k)[-k:]
        ca3_sparse = np.zeros(self.dim, dtype=np.float32)
        ca3_sparse[top_indices] = ca3_trajectory[top_indices]

        self.last_ca3_depth = len(self.temporal_history)
        return ca3_sparse

    def check_cpg_risk_reflex(self, obs: np.ndarray, intended_action: int) -> Tuple[int, bool, str]:
        """
        Central Pattern Generator (CPG) Risk Reflex:
        Spinal emergency reflex overriding decisions to enforce capital preservation & profit protection:
        1. Hard Stop-Loss: If holding position and unrealized loss exceeds threshold, force SELL.
        2. Trailing Profit Lock: If peak unrealized gain >= +3.0% and retraces by >= trailing_stop_pct, force SELL.
        3. High Volatility Anomaly: If volatility spike is extreme, veto BUY into HOLD.
        """
        is_holding = (obs[8] > 0.5)
        unrealized_pnl_pct = (obs[9] / 10.0)  # De-normalize
        vol_norm = (obs[6] + 1.0) / 2.0

        if is_holding:
            # Track highest unrealized profit achieved during this position
            self.peak_unrealized_pnl = max(self.peak_unrealized_pnl, unrealized_pnl_pct)

            # Reflex 1: Hard Stop-Loss Protection
            if unrealized_pnl_pct <= self.stop_loss_pct:
                return SELL, True, f"HARD STOP-LOSS ({unrealized_pnl_pct * 100:.1f}%)"

            # Reflex 2: Trailing Profit Lock (locking profits before they turn into losses)
            if self.peak_unrealized_pnl >= 0.03:
                giveback = self.peak_unrealized_pnl - unrealized_pnl_pct
                if giveback >= self.trailing_stop_pct:
                    return SELL, True, f"TRAILING PROFIT LOCK (+{unrealized_pnl_pct * 100:.1f}%)"
        else:
            self.peak_unrealized_pnl = 0.0

            # Reflex 3: Volatility Spike Veto
            if intended_action == BUY and vol_norm > 0.88:
                return HOLD, True, "EXTREME VOLATILITY SPIKE VETO"

        return intended_action, False, ""

    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        training: bool = True
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass & action readout:
        Combines Sensory Path (DG 45%) and Sequence Path (CA3 55%), applies Softmax policy with masking,
        and evaluates CPG Risk Reflex filter.
        """
        ec_vec = self.encode_entorhinal_cortex(obs)
        dg_sparse = self.dentate_gyrus_separation(ec_vec)
        ca3_sparse = self.ca3_recurrent_sequence(dg_sparse)

        # Dual-path representation
        combined_repr = (0.45 * dg_sparse) + (0.55 * ca3_sparse)
        norm = np.linalg.norm(combined_repr)
        if norm > 1e-6:
            combined_repr /= norm

        # Action Prototype Readout via Cosine Similarity
        scores = np.dot(combined_repr, self.action_prototypes).astype(np.float32)

        # Dynamic Trend Sensitivity & Exploration Drive to prevent Inaction Trap
        is_holding = (obs[8] > 0.5)
        if not is_holding:
            self.inaction_counter += 1
            # Trend Confluence: reward BUY readiness when market is bullish
            trend_confluence = (obs[12] * 0.08) + (obs[13] * 0.08)
            if trend_confluence > 0:
                scores[BUY] += float(trend_confluence)

            # Exploration Drive: if idling in cash for too long, boost BUY exploration
            if self.inaction_counter > 15:
                scores[BUY] += 0.08
        else:
            self.inaction_counter = 0
            # Exit Confluence: boost SELL readiness if trend breaks down or overbought
            if obs[12] < 0 or obs[14] > 0.5:
                scores[SELL] += 0.08

        # Mask illegal actions
        masked_scores = np.where(action_mask, scores, -1e9)

        # Softmax Policy
        temp = max(self.temp_min, self.temperature if training else self.temp_min)
        exp_scores = np.exp((masked_scores - np.max(masked_scores)) / temp)
        probs = exp_scores / np.sum(exp_scores)
        self.last_action_probs = probs

        if training:
            selected_action = int(self.rng.choice(self.num_actions, p=probs))
        else:
            selected_action = int(np.argmax(probs))

        # CPG Risk Reflex Filter
        final_action, cpg_triggered, cpg_reason = self.check_cpg_risk_reflex(obs, selected_action)
        self.last_cpg_triggered = cpg_triggered
        self.last_cpg_reason = cpg_reason
        self.last_selected_action = final_action

        # Update Eligibility Traces: presynaptic (combined_repr) x postsynaptic (action)
        self.eligibility_traces *= (self.gamma * self.lambda_trace)
        self.eligibility_traces[:, final_action] += combined_repr

        # Record experience for SWR replay
        self.episode_experiences.append({
            "repr": combined_repr.copy(),
            "action": final_action,
            "mask": action_mask.copy()
        })

        return final_action, probs, dg_sparse, ca3_sparse

    def update_plasticity(self, reward: float):
        """Three-Factor Hebbian update modulated by dopamine reward signal."""
        dopamine = float(reward)
        delta_w = self.learning_rate * dopamine * self.eligibility_traces
        self.action_prototypes += delta_w
        # Dale's Bound: enforce non-negative weights
        self.action_prototypes = np.maximum(0.01, self.action_prototypes)

        # Temperature decay
        self.temperature = max(self.temp_min, self.temperature * self.temp_decay)

    def trigger_swr_episodic_replay(self, final_reward: float, trade_return: float = 0.0):
        """
        Sharp-Wave Ripple (SWR) Episodic Replay:
        Executes prioritized reverse sequence replay from trade exit back to entry,
        distributing dopamine credit assignment to earlier decisions in one shot.
        Amplifies dopamine for Big Wins (> +2.0%) with prioritized multi-pass consolidation.
        """
        if not self.episode_experiences:
            return

        self.last_swr_active = True

        # Prioritize Big Wins: amplify replay dopamine for high R-multiple trades
        amp = 1.0
        if trade_return > 0.02:
            amp = 1.0 + min(3.0, trade_return * 25.0)
        elif trade_return < -0.02:
            amp = 1.4  # Strong negative memory reinforcement to prevent repeating large losses

        replay_dopamine = float(final_reward) * amp
        passes = 2 if abs(trade_return) >= 0.025 else 1

        for _ in range(passes):
            curr_signal = replay_dopamine
            discount = 0.85
            for exp in reversed(self.episode_experiences):
                repr_vec = exp["repr"]
                action = exp["action"]
                # Direct retroactive update on the action taken
                self.action_prototypes[:, action] += (self.learning_rate * 0.4 * curr_signal * repr_vec)
                self.action_prototypes[:, action] = np.maximum(0.01, self.action_prototypes[:, action])
                curr_signal *= discount

        # Clear buffer after replay
        self.episode_experiences.clear()

    def reset_traces(self):
        """Reset eligibility traces, temporal sequence history, and peak position metrics."""
        self.eligibility_traces.fill(0.0)
        self.temporal_history.clear()
        self.episode_experiences.clear()
        self.last_swr_active = False
        self.last_cpg_triggered = False
        self.last_cpg_reason = ""
        self.inaction_counter = 0
        self.peak_unrealized_pnl = 0.0
