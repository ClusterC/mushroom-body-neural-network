import os
import sys
import time
import numpy as np

# บังคับใช้ UTF-8 บน Windows Console
if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.snake_env import SnakeEnv
from src.models.visual_mushroom_body import VisualMushroomBody
from src.models.stacked_visual_mb import StackedVisualMushroomBody
from src.models.hdc_visual_mb import HDCVisualMushroomBody
from src.models.hippocampal_hdc_mb import HippocampalHDCVisualMB

def eval_single_mb(n_eval: int = 100, seed: int = 42):
    env = SnakeEnv(width=10, height=10, max_steps_without_food=100, seed=seed)
    model = VisualMushroomBody(channels=3, grid_h=10, grid_w=10, num_kc=2000, k_active=100, num_mbon=4, seed=seed)
    eval_apples = []
    eval_steps = []
    for _ in range(n_eval):
        obs = env.reset()
        done = False
        steps = 0
        apples = 0
        while not done:
            steps += 1
            action = model.select_action(env, training=False)
            _, r, done, _ = env.step(action)
            if r >= 1.0:
                apples += 1
        eval_apples.append(apples)
        eval_steps.append(steps)
    return {
        "agent": "Single Visual MB (Baseline)",
        "eval_avg_apples": float(np.mean(eval_apples)),
        "eval_max_apples": int(np.max(eval_apples)),
        "eval_avg_steps": float(np.mean(eval_steps))
    }

def eval_hdc_vsa(n_train: int = 200, n_eval: int = 100, seed: int = 42):
    env = SnakeEnv(width=10, height=10, max_steps_without_food=100, seed=seed)
    model = HDCVisualMushroomBody(dim=2048, k_active=100, seed=seed)
    start_t = time.time()
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
            model.update_plasticity(reward)
        model.reset_traces()
    train_time = time.time() - start_t

    eval_apples = []
    eval_steps = []
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
            _, reward, done, _ = env.step(action)
            if reward >= 1.0:
                apples += 1
        eval_apples.append(apples)
        eval_steps.append(steps)
    return {
        "agent": "HDC-VSA Mushroom Body",
        "train_time": train_time,
        "eval_avg_apples": float(np.mean(eval_apples)),
        "eval_max_apples": int(np.max(eval_apples)),
        "eval_avg_steps": float(np.mean(eval_steps))
    }

def eval_hippocampal_mb(n_train: int = 200, n_eval: int = 100, seed: int = 42):
    env = SnakeEnv(width=10, height=10, max_steps_without_food=100, seed=seed)
    model = HippocampalHDCVisualMB(dim=2048, k_dg=50, k_ca3=120, seed=seed)
    start_t = time.time()
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
    train_time = time.time() - start_t

    eval_apples = []
    eval_steps = []
    self_collisions = 0
    wall_collisions = 0

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
        eval_apples.append(apples)
        eval_steps.append(steps)

    return {
        "agent": "Hippocampal (DG-CA3) HDC-VSA MB",
        "train_time": train_time,
        "eval_avg_apples": float(np.mean(eval_apples)),
        "eval_max_apples": int(np.max(eval_apples)),
        "eval_avg_steps": float(np.mean(eval_steps)),
        "self_collisions": self_collisions,
        "wall_collisions": wall_collisions
    }

if __name__ == "__main__":
    print("=" * 70)
    print("HIPPOCAMPUS (DG-CA3) VS HDC-VSA VS BASELINE BENCHMARK")
    print("=" * 70)

    print("\n[1/3] Evaluating Baseline Single Visual MB (100 episodes)...")
    res_base = eval_single_mb(n_eval=100)
    print(f"  -> Avg Apples: {res_base['eval_avg_apples']:.2f} | Max: {res_base['eval_max_apples']} | Avg Steps: {res_base['eval_avg_steps']:.1f}")

    print("\n[2/3] Evaluating HDC-VSA Mushroom Body (200 train + 100 eval)...")
    res_hdc = eval_hdc_vsa(n_train=200, n_eval=100)
    print(f"  -> Train Time: {res_hdc['train_time']:.2f}s | Avg Apples: {res_hdc['eval_avg_apples']:.2f} | Max: {res_hdc['eval_max_apples']} | Avg Steps: {res_hdc['eval_avg_steps']:.1f}")

    print("\n[3/3] Evaluating Hippocampal (DG-CA3) HDC-VSA MB (200 train + 100 eval)...")
    res_hippo = eval_hippocampal_mb(n_train=200, n_eval=100)
    print(f"  -> Train Time: {res_hippo['train_time']:.2f}s | Avg Apples: {res_hippo['eval_avg_apples']:.2f} | Max: {res_hippo['eval_max_apples']} | Avg Steps: {res_hippo['eval_avg_steps']:.1f}")
    print(f"  -> Self Collisions: {res_hippo['self_collisions']} / 100 | Wall: {res_hippo['wall_collisions']} / 100")

    print("\n" + "=" * 70)
    print(f"{'Architecture':<35} | {'Avg Apples':<10} | {'Max Apples':<10} | {'Avg Steps':<10}")
    print("-" * 70)
    for res in [res_base, res_hdc, res_hippo]:
        print(f"{res['agent']:<35} | {res['eval_avg_apples']:<10.2f} | {res['eval_max_apples']:<10} | {res['eval_avg_steps']:<10.1f}")
    print("=" * 70)
