import os
import sys
import time
import json
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# เพิ่ม Path ให้เรียกใช้โมดูลใน src ได้
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.tic_tac_toe import TicTacToeEnv, PLAYER_X, PLAYER_O
from src.models.mushroom_body import MushroomBodyNet
from src.baselines.q_learning import TabularQLearningAgent
from src.opponents.random_agent import RandomAgent
from src.opponents.heuristic_agent import HeuristicAgent
from src.opponents.minimax_agent import MinimaxAgent
from src.utils.metrics import evaluate_agent, measure_representation_orthogonality

def train_mushroom_body(mb_agent, opponent, episodes=1000, agent_as_x=True):
    env = TicTacToeEnv()
    agent_player = PLAYER_X if agent_as_x else PLAYER_O
    opponent_player = PLAYER_O if agent_as_x else PLAYER_X

    win_history = []

    for ep in range(episodes):
        mb_agent.reset_traces()
        env.reset(starting_player=PLAYER_X)

        while not env.done:
            if env.current_player == agent_player:
                action = mb_agent.select_action(env, training=True, player=agent_player)
            else:
                action = opponent.select_action(env, player=opponent_player)
            env.step(action)

        # Dopamine modulation จากผลลัพธ์สุดท้าย
        if env.winner == agent_player:
            reward = 1.0
            win_history.append(1)
        elif env.winner == "DRAW":
            reward = 0.0
            win_history.append(0)
        else:
            reward = -1.0
            win_history.append(-1)

        mb_agent.update_synapses(dopamine_signal=reward)
        mb_agent.decay_temperature()

    return win_history

def train_q_learning(q_agent, opponent, episodes=1000, agent_as_x=True):
    env = TicTacToeEnv()
    agent_player = PLAYER_X if agent_as_x else PLAYER_O
    opponent_player = PLAYER_O if agent_as_x else PLAYER_X

    win_history = []

    for ep in range(episodes):
        q_agent.reset_episode()
        env.reset(starting_player=PLAYER_X)

        while not env.done:
            if env.current_player == agent_player:
                action = q_agent.select_action(env, training=True, player=agent_player)
            else:
                action = opponent.select_action(env, player=opponent_player)
            env.step(action)

        if env.winner == agent_player:
            reward = 1.0
            win_history.append(1)
        elif env.winner == "DRAW":
            reward = 0.0
            win_history.append(0)
        else:
            reward = -1.0
            win_history.append(-1)

        q_agent.update_q_values(final_reward=reward)
        q_agent.decay_epsilon()

    return win_history

def run_experiment_1():
    print("\n" + "=" * 80)
    print("🚀 [EXPERIMENT 1] Learning Efficiency & Convergence vs. Tabular Q-Learning")
    print("=" * 80)
    print("หลักสูตร: Stage 1 (Random: 1,000 ep) -> Stage 2 (Heuristic: 1,000 ep) -> Benchmark (Minimax: 100 games)")

    seed = 42
    mb = MushroomBodyNet(num_kc=1000, k_active=75, learning_rate=0.08, seed=seed)
    ql = TabularQLearningAgent(learning_rate=0.15, epsilon=1.0, seed=seed)

    random_opp = RandomAgent(seed=101)
    heuristic_opp = HeuristicAgent(seed=202)
    minimax_opp = MinimaxAgent()

    # Stage 1: Train on Random
    print("\n--- กำลังฝึก Stage 1: Random Opponent (1,000 Episodes) ---")
    t0 = time.time()
    train_mushroom_body(mb, random_opp, episodes=1000, agent_as_x=True)
    t_mb1 = time.time() - t0

    t0 = time.time()
    train_q_learning(ql, random_opp, episodes=1000, agent_as_x=True)
    t_ql1 = time.time() - t0

    eval_mb_s1 = evaluate_agent(mb, random_opp, num_games=100)
    eval_ql_s1 = evaluate_agent(ql, random_opp, num_games=100)

    print(f"ผลประเมิน Stage 1 เทียบกับ Random Opponent (100 Games):")
    print(f"  Mushroom Body : Win={eval_mb_s1['win_rate']*100:5.1f}%, Draw={eval_mb_s1['draw_rate']*100:4.1f}%, Loss={eval_mb_s1['loss_rate']*100:4.1f}% (เวลาเทรน: {t_mb1:.2f}s)")
    print(f"  Q-Learning    : Win={eval_ql_s1['win_rate']*100:5.1f}%, Draw={eval_ql_s1['draw_rate']*100:4.1f}%, Loss={eval_ql_s1['loss_rate']*100:4.1f}% (เวลาเทรน: {t_ql1:.2f}s)")

    # Stage 2: Train on Heuristic
    print("\n--- กำลังฝึก Stage 2: Heuristic Opponent (1,000 Episodes) ---")
    t0 = time.time()
    train_mushroom_body(mb, heuristic_opp, episodes=1000, agent_as_x=True)
    t_mb2 = time.time() - t0

    t0 = time.time()
    train_q_learning(ql, heuristic_opp, episodes=1000, agent_as_x=True)
    t_ql2 = time.time() - t0

    eval_mb_s2 = evaluate_agent(mb, heuristic_opp, num_games=100)
    eval_ql_s2 = evaluate_agent(ql, heuristic_opp, num_games=100)

    print(f"ผลประเมิน Stage 2 เทียบกับ Heuristic Opponent (100 Games):")
    print(f"  Mushroom Body : Win={eval_mb_s2['win_rate']*100:5.1f}%, Draw={eval_mb_s2['draw_rate']*100:4.1f}%, Loss={eval_mb_s2['loss_rate']*100:4.1f}% (เวลาเทรน: {t_mb2:.2f}s)")
    print(f"  Q-Learning    : Win={eval_ql_s2['win_rate']*100:5.1f}%, Draw={eval_ql_s2['draw_rate']*100:4.1f}%, Loss={eval_ql_s2['loss_rate']*100:4.1f}% (เวลาเทรน: {t_ql2:.2f}s)")

    # Final Benchmark: Minimax
    print("\n--- ทดสอบ Benchmark สุดท้ายกับ Optimal Minimax Opponent (100 Games) ---")
    eval_mb_mm = evaluate_agent(mb, minimax_opp, num_games=100)
    eval_ql_mm = evaluate_agent(ql, minimax_opp, num_games=100)

    print(f"ผลประเมินเทียบกับ Optimal Minimax Opponent:")
    print(f"  Mushroom Body : Win={eval_mb_mm['win_rate']*100:5.1f}%, Draw={eval_mb_mm['draw_rate']*100:4.1f}%, Loss={eval_mb_mm['loss_rate']*100:4.1f}% | Non-loss={eval_mb_mm['non_loss_rate']*100:5.1f}%")
    print(f"  Q-Learning    : Win={eval_ql_mm['win_rate']*100:5.1f}%, Draw={eval_ql_mm['draw_rate']*100:4.1f}%, Loss={eval_ql_mm['loss_rate']*100:4.1f}% | Non-loss={eval_ql_mm['non_loss_rate']*100:5.1f}%")

    return {
        "mb": {"s1": eval_mb_s1, "s2": eval_mb_s2, "minimax": eval_mb_mm},
        "ql": {"s1": eval_ql_s1, "s2": eval_ql_s2, "minimax": eval_ql_mm}
    }

def run_experiment_2():
    print("\n" + "=" * 80)
    print("🔄 [EXPERIMENT 2] Dynamic Role Reversal & Adaptation Latency")
    print("=" * 80)
    print("สถานการณ์: โมเดลที่เชี่ยวชาญการเดินก่อน (Player X) ถูกสลับให้เดินทีหลัง (Player O)")
    print("วัด Adaptation Latency และการคงอยู่ของทักษะเดิม (Catastrophic Forgetting Resistance)")

    seed = 88
    mb = MushroomBodyNet(num_kc=1000, k_active=75, learning_rate=0.1, seed=seed)
    ql = TabularQLearningAgent(learning_rate=0.2, seed=seed)
    heuristic_opp = HeuristicAgent(seed=555)

    # 1. เทรนเบื้องต้นเป็น Player X 800 episodes
    print("\n1. เทรนโมเดลเป็น Player X (เดินก่อน) 800 Episodes...")
    train_mushroom_body(mb, heuristic_opp, episodes=800, agent_as_x=True)
    train_q_learning(ql, heuristic_opp, episodes=800, agent_as_x=True)

    baseline_x_mb = evaluate_agent(mb, heuristic_opp, num_games=100, agent_as_x=True)
    baseline_x_ql = evaluate_agent(ql, heuristic_opp, num_games=100, agent_as_x=True)
    print(f"  ความเชี่ยวชาญเดิมในฐานะ Player X:")
    print(f"    Mushroom Body Non-loss rate : {baseline_x_mb['non_loss_rate']*100:.1f}%")
    print(f"    Q-Learning Non-loss rate    : {baseline_x_ql['non_loss_rate']*100:.1f}%")

    # 2. ทำการสลับบทบาทเป็น Player O ทันที และบันทึกผลการปรับตัวทุกๆ 100 episodes
    print("\n2. สลับบทบาทเป็น Player O (เดินทีหลัง) ทันที ติดตามการปรับตัว...")
    adaptation_steps = [100, 200, 400, 600, 800]
    mb_adapt_scores = []
    ql_adapt_scores = []

    # Reset temperature / exploration เล็กน้อยสำหรับการเรียนรู้ใหม่
    mb.temperature = 0.5
    ql.epsilon = 0.3

    last_ep = 0
    for target_ep in adaptation_steps:
        ep_diff = target_ep - last_ep
        train_mushroom_body(mb, heuristic_opp, episodes=ep_diff, agent_as_x=False)
        train_q_learning(ql, heuristic_opp, episodes=ep_diff, agent_as_x=False)
        last_ep = target_ep

        res_mb = evaluate_agent(mb, heuristic_opp, num_games=50, agent_as_x=False)
        res_ql = evaluate_agent(ql, heuristic_opp, num_games=50, agent_as_x=False)

        mb_adapt_scores.append(res_mb['non_loss_rate'])
        ql_adapt_scores.append(res_ql['non_loss_rate'])

        print(f"  หลังสลับบทบาท {target_ep:3d} Episodes:")
        print(f"    Mushroom Body (Player O) Non-loss rate : {res_mb['non_loss_rate']*100:5.1f}% (Win={res_mb['win_rate']*100:4.1f}%, Draw={res_mb['draw_rate']*100:4.1f}%)")
        print(f"    Q-Learning    (Player O) Non-loss rate : {res_ql['non_loss_rate']*100:5.1f}% (Win={res_ql['win_rate']*100:4.1f}%, Draw={res_ql['draw_rate']*100:4.1f}%)")

    # 3. ทดสอบการลืมข้อมูลเก่า (Catastrophic Forgetting Test): นำกลับไปทดสอบเป็น Player X โดยไม่เทรนซ้ำ
    print("\n3. ตรวจสอบการลืมข้อมูลเก่า (Catastrophic Forgetting Test):")
    post_x_mb = evaluate_agent(mb, heuristic_opp, num_games=100, agent_as_x=True)
    post_x_ql = evaluate_agent(ql, heuristic_opp, num_games=100, agent_as_x=True)

    retention_mb = (post_x_mb['non_loss_rate'] / max(baseline_x_mb['non_loss_rate'], 1e-4)) * 100.0
    retention_ql = (post_x_ql['non_loss_rate'] / max(baseline_x_ql['non_loss_rate'], 1e-4)) * 100.0

    print(f"  ผลการประเมินทักษะ Player X ดั้งเดิมหลังถูกเทรนเป็น Player O:")
    print(f"    Mushroom Body : Non-loss rate = {post_x_mb['non_loss_rate']*100:5.1f}% | Retention = {retention_mb:5.1f}%")
    print(f"    Q-Learning    : Non-loss rate = {post_x_ql['non_loss_rate']*100:5.1f}% | Retention = {retention_ql:5.1f}%")

    return {
        "baseline_x": {"mb": baseline_x_mb, "ql": baseline_x_ql},
        "post_x": {"mb": post_x_mb, "ql": post_x_ql},
        "retention": {"mb": retention_mb, "ql": retention_ql}
    }

def run_experiment_3():
    print("\n" + "=" * 80)
    print("🧬 [EXPERIMENT 3] Biological Sparsity & Representation Orthogonality")
    print("=" * 80)
    print("ตรวจสอบคุณสมบัติการขยายมิติแบบเบาบาง (Sparse Expansion) และ Pattern Separation ในชั้น Kenyon Cells")

    mb = MushroomBodyNet(num_pn=27, num_kc=1000, k_active=75, seed=123)
    metrics = measure_representation_orthogonality(mb, num_samples=150, seed=999)

    # วัดเทียบกับ Raw PN input
    rng = np.random.default_rng(999)
    env = TicTacToeEnv()
    pn_patterns = []
    for _ in range(150):
        env.reset()
        steps = rng.integers(1, 8)
        for _ in range(steps):
            if env.done:
                break
            acts = env.get_legal_actions()
            if not acts:
                break
            env.step(rng.choice(acts))
        pn_patterns.append(env.get_observation(PLAYER_X))

    pn_matrix = np.array(pn_patterns)
    pn_norms = np.linalg.norm(pn_matrix, axis=1, keepdims=True)
    pn_norms = np.where(pn_norms == 0, 1.0, pn_norms)
    pn_norm_mat = pn_matrix / pn_norms
    pn_sim_mat = np.dot(pn_norm_mat, pn_norm_mat.T)
    pn_indices = np.triu_indices(len(pn_patterns), k=1)
    pn_mean_sim = float(np.mean(pn_sim_mat[pn_indices]))

    kc_mean_sim = metrics['mean_cosine_similarity']
    pattern_separation_gain = (pn_mean_sim - kc_mean_sim) / max(pn_mean_sim, 1e-4) * 100.0

    print(f"ผลการวิเคราะห์ชั้น Kenyon Cells (1,000 เซลล์, k-WTA=75):")
    print(f"  • Active Neuron Density (ความเบาบาง) : {metrics['active_density']*100:.2f}% (เป้าหมายชีวภาพ: 5% - 10%)")
    print(f"  • ค่า Cosine Similarity เฉลี่ยในชั้น Input (PN) : {pn_mean_sim:.4f}")
    print(f"  • ค่า Cosine Similarity เฉลี่ยในชั้น KC (Sparse) : {kc_mean_sim:.4f}")
    print(f"  • อัตราการแยก Pattern (Pattern Separation Gain)  : +{pattern_separation_gain:.1f}% (ลดความทับซ้อนของสัญญาณ)")
    print(f"  • Orthogonality Score (ความเป็นอิสระเชิงมิติ)    : {metrics['orthogonality_score']:.4f} / 1.0000")

    return metrics

def main():
    print("=" * 80)
    print("  เริ่มการทดสอบระบบ BIO-INSPIRED MUSHROOM BODY NEURAL NETWORK บนเกม XO")
    print("=" * 80)

    exp1_results = run_experiment_1()
    exp2_results = run_experiment_2()
    exp3_results = run_experiment_3()

    print("\n" + "=" * 80)
    print("🏁 สรุปผลการทดลองทั้งหมดเสร็จสมบูรณ์")
    print("=" * 80)

if __name__ == "__main__":
    main()
