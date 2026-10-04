"""
Scientific Benchmark: CPU Sequential vs. PyTorch CUDA Batched Training for Hippocampal Trading MB.
Measures wall-clock training time, speedup multiplier, steps-per-second throughput, and trade performance.
"""

import os
import sys
import time
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.hippocampal_trading_mb import HippocampalTradingMB
from src.envs.stock_trading_env import StockTradingEnv
from src.training.gpu_trading_trainer import GPUTradingTrainer


def run_cpu_sequential_training(episodes: int = 500) -> dict:
    """Benchmark traditional single-threaded CPU sequential training loop."""
    mb = HippocampalTradingMB(seed=42)
    profiles = ["TECH_MOMENTUM", "INDEX_ETF", "CRYPTO_VOLATILE", "DEFENSIVE_VALUE"]

    t0 = time.time()
    total_steps = 0
    total_trades = 0
    wins = 0

    for ep in range(episodes):
        prof = profiles[ep % len(profiles)]
        env = StockTradingEnv(initial_cash=10000.0, max_steps=252, asset_profile=prof, seed=1000 + ep)
        obs = env.reset()
        mb.reset_traces()

        while not env.done:
            mask = env.get_action_mask()
            action, _, _, _ = mb.select_action(obs, mask, training=True)
            next_obs, rew, done, info = env.step(action)
            mb.update_plasticity(rew)

            if info.get("trade_event") == "SELL":
                mb.trigger_swr_episodic_replay(rew, trade_return=env.last_trade_return)

            obs = next_obs
            total_steps += 1

        wins += env.winning_trades
        total_trades += env.total_trades

    dur = time.time() - t0
    win_rate = (wins / max(1, total_trades)) * 100.0

    return {
        "mode": "CPU Sequential",
        "device": "CPU",
        "episodes": episodes,
        "duration": dur,
        "fps": total_steps / max(dur, 1e-4),
        "total_steps": total_steps,
        "win_rate": win_rate,
        "trades": total_trades
    }


def run_gpu_batched_training(episodes: int = 500, batch_size: int = 128) -> dict:
    """Benchmark GPU CUDA batched parallel training loop."""
    mb = HippocampalTradingMB(seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = GPUTradingTrainer(mb, batch_size=batch_size, device=device)

    t0 = time.time()
    res = trainer.train(total_episodes=episodes)
    dur = time.time() - t0

    return {
        "mode": "CUDA Batched Parallel",
        "device": trainer.device_name,
        "episodes": episodes,
        "duration": dur,
        "fps": res["fps"],
        "win_rate": res["win_rate"],
        "trades": res["total_trades"],
        "mean_weight": res["mean_weight"]
    }


def main():
    print("=" * 80, flush=True)
    print("EXPERIMENT 12: GPU ACCELERATED TRADING BENCHMARK (CPU vs. CUDA BATCHED)", flush=True)
    print("=" * 80, flush=True)

    episodes = 64
    print(f"\n1. Benchmarking GPU Batched Training ({episodes} episodes, Batch=64)...", flush=True)
    gpu_res = run_gpu_batched_training(episodes=episodes, batch_size=64)
    print(f"   GPU Duration: {gpu_res['duration']:.2f}s | FPS: {gpu_res['fps']:.1f} steps/s | Win Rate: {gpu_res['win_rate']:.1f}% | Device: {gpu_res['device']}", flush=True)

    print(f"\n2. Benchmarking CPU Sequential Training ({episodes} episodes)...", flush=True)
    cpu_res = run_cpu_sequential_training(episodes=episodes)
    print(f"   CPU Duration: {cpu_res['duration']:.2f}s | FPS: {cpu_res['fps']:.1f} steps/s | Win Rate: {cpu_res['win_rate']:.1f}%", flush=True)

    speedup = cpu_res["duration"] / max(1e-4, gpu_res["duration"])
    fps_boost = gpu_res["fps"] / max(1e-4, cpu_res["fps"])

    print("\n" + "=" * 80, flush=True)
    print("BENCHMARK RESULTS SUMMARY:", flush=True)
    print("=" * 80, flush=True)
    print(f"• CPU Time:       {cpu_res['duration']:.2f} seconds ({cpu_res['fps']:.1f} steps/sec)", flush=True)
    print(f"• GPU CUDA Time:  {gpu_res['duration']:.2f} seconds ({gpu_res['fps']:.1f} steps/sec)", flush=True)
    print(f"• Speedup Factor: {speedup:.2f}x FASTER ({fps_boost:.2f}x throughput increase)", flush=True)
    print(f"• Hardware:       {gpu_res['device']}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
