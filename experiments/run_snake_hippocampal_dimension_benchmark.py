import os
import sys
import time
import numpy as np

# บังคับใช้ UTF-8 บน Windows Console
if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.snake_env import SnakeEnv
from src.models.hippocampal_hdc_mb import HippocampalHDCVisualMB

def benchmark_hippocampal_dimension(
    dim: int,
    k_dg: int,
    k_ca3: int,
    max_temporal_depth: int = 4,
    n_train: int = 150,
    n_eval: int = 50,
    seed: int = 42
):
    env = SnakeEnv(width=10, height=10, max_steps_without_food=100, seed=seed)
    model = HippocampalHDCVisualMB(
        dim=dim,
        k_dg=k_dg,
        k_ca3=k_ca3,
        max_temporal_depth=max_temporal_depth,
        seed=seed
    )

    # 1. Training with SWR Replay
    start_train_t = time.time()
    for _ in range(n_train):
        env.reset()
        done = False
        while not done:
            mask = env.get_action_mask()
            ego = env.get_egocentric_observation()
            cpg_mask = env.get_safe_action_mask()
            action, _, _, _ = model.forward(
                ego_obs=ego,
                action_mask=mask,
                cpg_safe_mask=cpg_mask,
                direction=env.direction,
                temperature=0.08,
                deterministic=False
            )
            _, reward, done, _ = env.step(action)
            model.update(reward=reward, done=done)
    train_time = time.time() - start_train_t

    # 2. Evaluation
    eval_apples = []
    eval_steps = []
    self_collisions = 0
    wall_collisions = 0
    timeout_games = 0

    for _ in range(n_eval):
        env.reset()
        done = False
        steps = 0
        apples = 0
        while not done:
            steps += 1
            mask = env.get_action_mask()
            ego = env.get_egocentric_observation()
            cpg_mask = env.get_safe_action_mask()
            action, _, _, _ = model.forward(
                ego_obs=ego,
                action_mask=mask,
                cpg_safe_mask=cpg_mask,
                direction=env.direction,
                temperature=0.02,
                deterministic=True
            )
            _, reward, done, info = env.step(action)
            if reward >= 1.0:
                apples += 1
            if done:
                cause = info.get("cause", "")
                if cause == "self_collision":
                    self_collisions += 1
                elif cause == "wall":
                    wall_collisions += 1
                elif cause == "starvation" or steps >= 300:
                    timeout_games += 1

        eval_apples.append(apples)
        eval_steps.append(steps)

    # 3. Micro-evaluation of DG Pattern Separation Quality
    ego1 = np.zeros(12, dtype=np.float32); ego1[6] = 1.0; ego1[0] = 0.5  # Food Ahead, wall mid
    ego2 = np.zeros(12, dtype=np.float32); ego2[7] = 1.0; ego2[0] = 0.5  # Food Left, wall mid
    ec1 = model.encode_entorhinal(ego1)
    ec2 = model.encode_entorhinal(ego2)
    dg1 = model.dentate_gyrus_pattern_separation(ec1)
    dg2 = model.dentate_gyrus_pattern_separation(ec2)
    sim_ec = float(np.dot(ec1, ec2) / (np.linalg.norm(ec1) * np.linalg.norm(ec2) + 1e-9))
    sim_dg = float(np.dot(dg1, dg2) / (np.linalg.norm(dg1) * np.linalg.norm(dg2) + 1e-9))

    return {
        "dim": dim,
        "k_dg": k_dg,
        "k_ca3": k_ca3,
        "depth": max_temporal_depth,
        "train_time": train_time,
        "eval_avg_apples": float(np.mean(eval_apples)),
        "eval_max_apples": int(np.max(eval_apples)),
        "eval_avg_steps": float(np.mean(eval_steps)),
        "self_collisions": self_collisions,
        "wall_collisions": wall_collisions,
        "sim_ec": sim_ec,
        "sim_dg": sim_dg,
        "separation_gain": (sim_ec - sim_dg) / (sim_ec + 1e-9) * 100.0
    }

if __name__ == "__main__":
    print("=" * 88)
    print("HIPPOCAMPUS (DG-CA3) HDC-VSA: DIMENSIONAL SCALING BENCHMARK (SNAKE GAME)")
    print("=" * 88)
    print("Configurations: 150 Train Episodes (SWR Replay) + 50 Eval Episodes per Dimension\n")

    configs = [
        {"dim": 1024, "k_dg": 25,  "k_ca3": 60,  "depth": 4},
        {"dim": 2048, "k_dg": 50,  "k_ca3": 120, "depth": 4},
        {"dim": 4096, "k_dg": 100, "k_ca3": 240, "depth": 4},
        {"dim": 8192, "k_dg": 200, "k_ca3": 480, "depth": 4},
    ]

    results = []
    for cfg in configs:
        d = cfg["dim"]
        print(f"--> Running Benchmark for D={d:,} (k_dg={cfg['k_dg']}, k_ca3={cfg['k_ca3']}, depth={cfg['depth']})...")
        res = benchmark_hippocampal_dimension(
            dim=cfg["dim"],
            k_dg=cfg["k_dg"],
            k_ca3=cfg["k_ca3"],
            max_temporal_depth=cfg["depth"],
            n_train=150,
            n_eval=50,
            seed=42
        )
        results.append(res)
        print(f"    Train Time: {res['train_time']:.2f}s | Avg Apples: {res['eval_avg_apples']:.2f} | Max: {res['eval_max_apples']} | Avg Steps: {res['eval_avg_steps']:.1f}")
        print(f"    Collisions: Self={res['self_collisions']}/50, Wall={res['wall_collisions']}/50 | DG Separation Gain: +{res['separation_gain']:.1f}%\n")

    print("=" * 88)
    print(f"{'Dimension (D)':<14} | {'Train Time':<11} | {'Avg Apples':<11} | {'Max Apples':<11} | {'Avg Steps':<11} | {'Self Crash':<10}")
    print("-" * 88)
    for r in results:
        print(f"D = {r['dim']:<10,d} | {r['train_time']:<8.2f}s  | {r['eval_avg_apples']:<11.2f} | {r['eval_max_apples']:<11} | {r['eval_avg_steps']:<11.1f} | {r['self_collisions']:<2}/50")
    print("=" * 88)
