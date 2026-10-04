import numpy as np
from src.envs.tic_tac_toe import TicTacToeEnv, PLAYER_X, PLAYER_O

def compute_moving_average(data, window=50):
    """
    คำนวณ Moving Average ของข้อมูลด้วยขนาดหน้าต่าง window
    """
    if len(data) < window:
        return np.array(data)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def evaluate_agent(agent, opponent, num_games=100, agent_as_x=True):
    """
    ประเมินผลตัวแทนแข่งขันกับคู่ต่อสู้จำนวน num_games เกม
    ส่งคืน Dict สรุป Win Rate, Draw Rate, Loss Rate
    """
    env = TicTacToeEnv()
    agent_player = PLAYER_X if agent_as_x else PLAYER_O
    opponent_player = PLAYER_O if agent_as_x else PLAYER_X

    wins = 0
    draws = 0
    losses = 0

    for _ in range(num_games):
        env.reset(starting_player=PLAYER_X)
        while not env.done:
            if env.current_player == agent_player:
                action = agent.select_action(env, training=False, player=agent_player)
            else:
                action = opponent.select_action(env, player=opponent_player)
            env.step(action)

        if env.winner == agent_player:
            wins += 1
        elif env.winner == "DRAW":
            draws += 1
        else:
            losses += 1

    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / num_games,
        "draw_rate": draws / num_games,
        "loss_rate": losses / num_games,
        "non_loss_rate": (wins + draws) / num_games
    }

def measure_representation_orthogonality(mb_model, num_samples=100, seed=42):
    """
    วัดความเบาบาง (Sparsity) และความเป็นอิสระเชิงมุมมอง (Cosine Orthogonality)
    ของ Activation ในชั้น Kenyon Cells (KC) จากหลากหลายสถานะกระดาน
    """
    rng = np.random.default_rng(seed)
    env = TicTacToeEnv()
    kc_patterns = []

    # สุ่มจำลองสถานะกระดานที่ถูกต้อง
    for _ in range(num_samples):
        env.reset()
        steps = rng.integers(1, 8)
        for _ in range(steps):
            if env.done:
                break
            actions = env.get_legal_actions()
            if not actions:
                break
            env.step(rng.choice(actions))

        obs = env.get_observation(PLAYER_X)
        kc_act = mb_model.encode_kc(obs)
        kc_patterns.append(kc_act)

    kc_matrix = np.array(kc_patterns)  # shape (num_samples, num_kc)

    # 1. Sparsity Density: อัตราส่วนของ Active Neurons เฉลี่ยต่อ Sample
    active_density = np.mean(kc_matrix > 0)

    # 2. Pairwise Cosine Similarity:
    # cos_sim = (u . v) / (||u|| * ||v||)
    norms = np.linalg.norm(kc_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    norm_matrix = kc_matrix / norms
    sim_matrix = np.dot(norm_matrix, norm_matrix.T)

    # ตัดเส้นทแยงมุมออก (ตัวเองเทียบตัวเอง = 1.0)
    indices = np.triu_indices(len(kc_patterns), k=1)
    pairwise_sims = sim_matrix[indices]

    mean_similarity = float(np.mean(pairwise_sims))
    max_similarity = float(np.max(pairwise_sims))
    min_similarity = float(np.min(pairwise_sims))

    return {
        "active_density": float(active_density),
        "mean_cosine_similarity": mean_similarity,
        "max_cosine_similarity": max_similarity,
        "min_cosine_similarity": min_similarity,
        "orthogonality_score": 1.0 - mean_similarity  # ยิ่งใกล้ 1 แสดงว่าแยก Pattern ได้ขาดมาก
    }
