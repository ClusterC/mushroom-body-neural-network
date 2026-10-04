import os
import sys
import time
import numpy as np

# บังคับใช้ UTF-8 บน Windows Console
if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.tic_tac_toe import TicTacToeEnv, PLAYER_X, PLAYER_O
from src.models.mushroom_body import MushroomBodyNet
from src.models.hippocampal_xo_mb import HippocampalXOMB
from src.opponents.heuristic_agent import HeuristicAgent
from src.opponents.random_agent import RandomAgent
from src.opponents.minimax_agent import MinimaxAgent

def evaluate_agent(agent, opponent, n_games=100, agent_player=PLAYER_X, is_hippo=False):
    env = TicTacToeEnv()
    wins = 0
    draws = 0
    losses = 0

    for _ in range(n_games):
        env.reset()
        if hasattr(agent, 'reset_traces'):
            agent.reset_traces()
        while not env.done:
            if env.current_player == agent_player:
                act = agent.select_action(env, training=False, player=agent_player)
            else:
                opp_player = PLAYER_O if agent_player == PLAYER_X else PLAYER_X
                act = opponent.select_action(env, player=opp_player)
            env.step(act)

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
        "win_rate": (wins / n_games) * 100.0,
        "non_loss_rate": ((wins + draws) / n_games) * 100.0
    }

def train_agent(agent, opponent, n_episodes=300, is_hippo=False, agent_player=PLAYER_X):
    env = TicTacToeEnv()
    start_t = time.time()
    for _ in range(n_episodes):
        env.reset()
        if hasattr(agent, 'reset_traces'):
            agent.reset_traces()
        while not env.done:
            if env.current_player == agent_player:
                act = agent.select_action(env, training=True, player=agent_player)
            else:
                opp_player = PLAYER_O if agent_player == PLAYER_X else PLAYER_X
                act = opponent.select_action(env, player=opp_player)
            env.step(act)

        rew = 1.0 if env.winner == agent_player else (0.0 if env.winner == "DRAW" else -1.0)
        if is_hippo:
            agent.update_synapses(rew, done=True)
        else:
            agent.update_synapses(rew)
        if hasattr(agent, 'decay_temperature'):
            agent.decay_temperature()

    duration = time.time() - start_t
    return duration

if __name__ == "__main__":
    print("=" * 80)
    print("TIC-TAC-TOE (XO): HIPPOCAMPAL (DG-CA3) VS STANDARD MUSHROOM BODY VS MINIMAX")
    print("=" * 80)

    heuristic = HeuristicAgent(seed=999)
    random_agent = RandomAgent(seed=123)
    minimax_agent = MinimaxAgent()

    # 1. Standard Mushroom Body
    print("\n[1/2] Training Standard Mushroom Body (300 episodes vs Heuristic)...")
    mb_agent = MushroomBodyNet(seed=42)
    mb_time = train_agent(mb_agent, heuristic, n_episodes=300, is_hippo=False)
    print(f"  -> Training Time: {mb_time:.2f}s")
    res_mb_rnd = evaluate_agent(mb_agent, random_agent, n_games=100)
    res_mb_heu = evaluate_agent(mb_agent, heuristic, n_games=100)
    res_mb_mm_x = evaluate_agent(mb_agent, minimax_agent, n_games=100, agent_player=PLAYER_X)
    res_mb_mm_o = evaluate_agent(mb_agent, minimax_agent, n_games=100, agent_player=PLAYER_O)
    print(f"  -> vs Random: Win {res_mb_rnd['win_rate']:.1f}% | Non-loss {res_mb_rnd['non_loss_rate']:.1f}%")
    print(f"  -> vs Heuristic: Win {res_mb_heu['win_rate']:.1f}% | Non-loss {res_mb_heu['non_loss_rate']:.1f}%")
    print(f"  -> vs Minimax (as X): Non-loss {res_mb_mm_x['non_loss_rate']:.1f}% (Draw: {res_mb_mm_x['draws']}, Loss: {res_mb_mm_x['losses']})")
    print(f"  -> vs Minimax (as O): Non-loss {res_mb_mm_o['non_loss_rate']:.1f}% (Draw: {res_mb_mm_o['draws']}, Loss: {res_mb_mm_o['losses']})")

    # 2. Hippocampal (DG-CA3) HDC-VSA MB
    print("\n[2/2] Training Hippocampal (DG-CA3) HDC-VSA MB (300 episodes with SWR Replay)...")
    hippo_agent = HippocampalXOMB(dim=2048, k_dg=50, k_ca3=120, seed=42)
    hippo_time = train_agent(hippo_agent, heuristic, n_episodes=300, is_hippo=True)
    print(f"  -> Training Time: {hippo_time:.2f}s")
    res_hippo_rnd = evaluate_agent(hippo_agent, random_agent, n_games=100, is_hippo=True)
    res_hippo_heu = evaluate_agent(hippo_agent, heuristic, n_games=100, is_hippo=True)
    res_hippo_mm_x = evaluate_agent(hippo_agent, minimax_agent, n_games=100, agent_player=PLAYER_X, is_hippo=True)
    res_hippo_mm_o = evaluate_agent(hippo_agent, minimax_agent, n_games=100, agent_player=PLAYER_O, is_hippo=True)
    # print debug
    # print(f"DEBUG: {res_hippo_mm_o}")
    print(f"  -> vs Random: Win {res_hippo_rnd['win_rate']:.1f}% | Non-loss {res_hippo_rnd['non_loss_rate']:.1f}%")
    print(f"  -> vs Heuristic: Win {res_hippo_heu['win_rate']:.1f}% | Non-loss {res_hippo_heu['non_loss_rate']:.1f}%")
    print(f"  -> vs Minimax (as X): Non-loss {res_hippo_mm_x['non_loss_rate']:.1f}% (Draw: {res_hippo_mm_x['draws']}, Loss: {res_hippo_mm_x['losses']})")
    print(f"  -> vs Minimax (as O): Non-loss {res_hippo_mm_o['non_loss_rate']:.1f}% (Draw: {res_hippo_mm_o['draws']}, Loss: {res_hippo_mm_o['losses']})")

    print("\n" + "=" * 80)
    print(f"{'Architecture':<28} | {'vs Random':<10} | {'vs Heuristic':<12} | {'vs MM (as X)':<12} | {'vs MM (as O)':<12}")
    print(f"{'':<28} | {'Win%':<10} | {'Non-Loss%':<12} | {'Non-Loss%':<12} | {'Non-Loss%':<12}")
    print("-" * 80)
    print(f"{'Standard Mushroom Body':<28} | {res_mb_rnd['win_rate']:<10.1f} | {res_mb_heu['non_loss_rate']:<12.1f} | {res_mb_mm_x['non_loss_rate']:<12.1f} | {res_mb_mm_o['non_loss_rate']:<12.1f}")
    print(f"{'Hippocampus (DG-CA3)':<28} | {res_hippo_rnd['win_rate']:<10.1f} | {res_hippo_heu['non_loss_rate']:<12.1f} | {res_hippo_mm_x['non_loss_rate']:<12.1f} | {res_hippo_mm_o['non_loss_rate']:<12.1f}")
    print("=" * 80)
