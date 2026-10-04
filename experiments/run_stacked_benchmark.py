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

def train_and_eval(agent_type: str, n_train: int = 250, n_eval: int = 100, seed: int = 42):
    env = SnakeEnv(width=10, height=10, max_steps_without_food=100, seed=seed)
    
    if agent_type == "single":
        model = VisualMushroomBody(channels=3, grid_h=10, grid_w=10, num_kc=2000, k_active=100, seed=seed)
    else:
        model = StackedVisualMushroomBody(
            channels=3, grid_h=10, grid_w=10,
            num_kc1=1200, k_active1=60, num_concepts=12,
            num_kc2=800, k_active2=40, seed=seed
        )

    # 1. การฝึกฝน (Training Phase)
    start_t = time.time()
    train_rewards = []
    train_apples = []
    
    for ep in range(n_train):
        obs = env.reset()
        ep_reward = 0.0
        apples = 0
        done = False
        prev_act = None
        
        while not done:
            mask = env.get_action_mask()
            if agent_type == "single":
                action = model.select_action(env, training=True)
            else:
                action, probs, _, _, _ = model.forward(obs, action_mask=mask, prev_action=prev_act)
                
            prev_act = action
            next_obs, reward, done, info = env.step(action)
            if agent_type == "single":
                model.update_synapses(reward)
            else:
                model.update_plasticity(reward)
            ep_reward += reward
            if reward >= 1.0:
                apples += 1
            obs = next_obs

        model.reset_traces()
        train_rewards.append(ep_reward)
        train_apples.append(apples)

    train_duration = time.time() - start_t

    # 2. การประเมินผลหลังฝึก (Evaluation Phase - Low Temperature)
    eval_apples = []
    eval_steps = []
    self_collisions = 0
    wall_collisions = 0
    
    for ep in range(n_eval):
        obs = env.reset()
        ep_steps = 0
        apples = 0
        done = False
        prev_act = None
        
        while not done:
            ep_steps += 1
            mask = env.get_action_mask()
            if agent_type == "single":
                action = model.select_action(env, training=False)
            else:
                action, probs, _, _, _ = model.forward(obs, action_mask=mask, prev_action=prev_act, temperature=0.1)
                
            prev_act = action
            next_obs, reward, done, info = env.step(action)
            if reward >= 1.0:
                apples += 1
            if done:
                cause = info.get("cause", "")
                if cause == "self_collision":
                    self_collisions += 1
                elif cause == "wall":
                    wall_collisions += 1
            obs = next_obs

        eval_apples.append(apples)
        eval_steps.append(ep_steps)

    return {
        "agent": agent_type,
        "train_time": train_duration,
        "avg_train_apples": float(np.mean(train_apples)),
        "eval_avg_apples": float(np.mean(eval_apples)),
        "eval_max_apples": int(np.max(eval_apples)),
        "eval_avg_steps": float(np.mean(eval_steps)),
        "self_collision_rate": (self_collisions / n_eval) * 100.0,
        "wall_collision_rate": (wall_collisions / n_eval) * 100.0
    }

def eval_supercharged_gpu(n_train: int = 2000, n_eval: int = 100, seed: int = 42):
    """ทดสอบโมเดล Supercharged ที่ฝึกฝนบน GPU พร้อม Egocentric Whiskers และ CPG Reflex"""
    from src.training.gpu_snake_trainer import GPUMushroomBodyTrainer
    env = SnakeEnv(width=10, height=10, max_steps_without_food=100, seed=seed)
    model = StackedVisualMushroomBody(
        channels=3, grid_h=10, grid_w=10,
        num_kc1=1200, k_active1=60, num_concepts=12,
        num_kc2=800, k_active2=40, seed=seed
    )

    # ฝึกฝนบน GPU
    start_t = time.time()
    trainer = GPUMushroomBodyTrainer(model, batch_size=256)
    gpu_res = trainer.train(total_episodes=n_train, cpg_reflex=True, use_whiskers=True)
    train_duration = time.time() - start_t

    # ประเมินผล 100 รอบ
    eval_apples = []
    eval_steps = []
    self_collisions = 0
    wall_collisions = 0

    for ep in range(n_eval):
        obs = env.reset()
        ep_steps = 0
        apples = 0
        done = False
        prev_act = None

        while not done:
            ep_steps += 1
            mask = env.get_action_mask()
            ego = env.get_egocentric_observation()
            cpg_mask = env.get_safe_action_mask()

            action, probs, _, _, _ = model.forward(
                obs,
                action_mask=mask,
                prev_action=prev_act,
                temperature=0.05,
                ego_obs=ego,
                cpg_safe_mask=cpg_mask
            )
            prev_act = action
            next_obs, reward, done, info = env.step(action)
            if reward >= 1.0:
                apples += 1
            if done:
                cause = info.get("cause", "")
                if cause == "self_collision":
                    self_collisions += 1
                elif cause == "wall":
                    wall_collisions += 1
            obs = next_obs

        eval_apples.append(apples)
        eval_steps.append(ep_steps)

    return {
        "agent": "supercharged_gpu",
        "device": gpu_res["device"],
        "train_time": train_duration,
        "eval_avg_apples": float(np.mean(eval_apples)),
        "eval_max_apples": int(np.max(eval_apples)),
        "eval_avg_steps": float(np.mean(eval_steps)),
        "self_collision_rate": (self_collisions / n_eval) * 100.0,
        "wall_collision_rate": (wall_collisions / n_eval) * 100.0,
        "gpu_fps": gpu_res["fps"]
    }

def main():
    print("=" * 75)
    print("MUSHROOM BODY BENCHMARK ON VISUAL SNAKE: CPU vs STACKED vs GPU")
    print("=" * 75)

    print("\n[1/3] กำลังทดสอบ Single-Layer Mushroom Body (2,000 KCs บน CPU)...")
    res_single = train_and_eval("single", n_train=250, n_eval=100, seed=42)
    print(f"   - เฉลี่ยแอปเปิล: {res_single['eval_avg_apples']:.2f} ลูก (สูงสุด {res_single['eval_max_apples']} ลูก) | อายุขัยเฉลี่ย: {res_single['eval_avg_steps']:.1f} ก้าว")

    print("\n[2/3] กำลังทดสอบ Stacked Deep MB (1,200 KC1 + 800 KC2 บน CPU)...")
    res_stacked = train_and_eval("stacked", n_train=250, n_eval=100, seed=42)
    print(f"   - เฉลี่ยแอปเปิล: {res_stacked['eval_avg_apples']:.2f} ลูก (สูงสุด {res_stacked['eval_max_apples']} ลูก) | อายุขัยเฉลี่ย: {res_stacked['eval_avg_steps']:.1f} ก้าว")

    print("\n[3/3] กำลังทดสอบ Supercharged GPU Deep MB (CUDA Acceleration + Whiskers + CPG)...")
    res_gpu = eval_supercharged_gpu(n_train=2000, n_eval=100, seed=42)
    print(f"   - อุปกรณ์: {res_gpu['device']} | ความเร็ว: {res_gpu['gpu_fps']:.0f} FPS")
    print(f"   - เฉลี่ยแอปเปิล: {res_gpu['eval_avg_apples']:.2f} ลูก (สูงสุด {res_gpu['eval_max_apples']} ลูก) | อายุขัยเฉลี่ย: {res_gpu['eval_avg_steps']:.1f} ก้าว")

    print("\n" + "=" * 75)
    print("ตารางสรุปผลการเปรียบเทียบเชิงประจักษ์ (Empirical Benchmark 100 Episodes)")
    print("=" * 75)
    print(f"{'ตัวชี้วัด (Metric)':<26} | {'Single MB (CPU)':<15} | {'Stacked MB (CPU)':<16} | {'Supercharged GPU'}")
    print("-" * 75)
    print(f"{'Avg Apples / Game':<26} | {res_single['eval_avg_apples']:<15.2f} | {res_stacked['eval_avg_apples']:<16.2f} | {res_gpu['eval_avg_apples']:.2f} ลูก")
    print(f"{'Max Apples in Game':<26} | {res_single['eval_max_apples']:<15} | {res_stacked['eval_max_apples']:<16} | {res_gpu['eval_max_apples']} ลูก")
    print(f"{'Avg Survival Steps':<26} | {res_single['eval_avg_steps']:<15.1f} | {res_stacked['eval_avg_steps']:<16.1f} | {res_gpu['eval_avg_steps']:.1f} ก้าว")
    print(f"{'Self-Trap Collision':<26} | {res_single['self_collision_rate']:<14.1f}% | {res_stacked['self_collision_rate']:<15.1f}% | {res_gpu['self_collision_rate']:.1f}%")
    print(f"{'Training Duration':<26} | {res_single['train_time']:<13.2f}s | {res_stacked['train_time']:<14.2f}s | {res_gpu['train_time']:.2f}s (2,000 EP)")
    print("=" * 75)

    gain_single = ((res_gpu['eval_avg_apples'] - res_single['eval_avg_apples']) / max(res_single['eval_avg_apples'], 0.01)) * 100.0
    print(f"\nผลสรุป: Supercharged GPU Deep MB มีประสิทธิภาพสูงขึ้น {gain_single:+.1f}% เทียบกับโมเดลตั้งต้น")
    print(f"ความเร็ว GPU ({res_gpu['device']}): ฝึก 2,000 รอบใน {res_gpu['train_time']:.2f} วินาที ({res_gpu['gpu_fps']:.0f} FPS)")

if __name__ == "__main__":
    main()
