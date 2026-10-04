"""
Comparative Benchmark Experiment for Algorithmic Stock Trading.
Compares:
1. Buy & Hold Benchmark
2. Random Trading Agent
3. Rule-Based Heuristic (SMA Crossover + RSI)
4. Canonical Mushroom Body (1,000 KC)
5. Hippocampal Trading MB (DG-CA3 + HDC + SWR + CPG Risk Reflex)

Evaluates:
- Cumulative Return %
- Annualized Sharpe Ratio
- Maximum Drawdown %
- Trade Win Rate %
- Total Completed Trades
"""

import os
import sys
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.stock_trading_env import StockTradingEnv, HOLD, BUY, SELL
from src.models.hippocampal_trading_mb import HippocampalTradingMB
from src.models.mushroom_body import MushroomBodyNet


def evaluate_buy_and_hold(env: StockTradingEnv) -> dict:
    """Evaluate Buy & Hold: Buys immediately on step 0 and holds until completion."""
    env.reset()
    env.step(BUY)
    while not env.done:
        env.step(HOLD)

    ret = ((env.net_worth - env.initial_cash) / env.initial_cash) * 100.0
    p_hist = np.array(env.portfolio_history)
    peak = np.maximum.accumulate(p_hist)
    dd = (peak - p_hist) / np.maximum(1e-5, peak)
    max_dd = float(np.max(dd)) * 100.0

    daily_rets = np.diff(p_hist) / p_hist[:-1]
    sharpe = float(np.mean(daily_rets) / max(1e-5, np.std(daily_rets)) * np.sqrt(252))

    return {
        "strategy": "Buy & Hold Benchmark",
        "return_pct": ret,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "win_rate": 100.0 if ret > 0 else 0.0,
        "trades": 1
    }


def evaluate_random_agent(env: StockTradingEnv, seed: int = 42) -> dict:
    """Evaluate Random Agent: Chooses randomly from legal actions."""
    rng = np.random.default_rng(seed)
    env.reset()
    while not env.done:
        mask = env.get_action_mask()
        valid_actions = np.where(mask)[0]
        act = int(rng.choice(valid_actions))
        env.step(act)

    ret = ((env.net_worth - env.initial_cash) / env.initial_cash) * 100.0
    p_hist = np.array(env.portfolio_history)
    peak = np.maximum.accumulate(p_hist)
    dd = (peak - p_hist) / np.maximum(1e-5, peak)
    max_dd = float(np.max(dd)) * 100.0

    daily_rets = np.diff(p_hist) / p_hist[:-1]
    sharpe = float(np.mean(daily_rets) / max(1e-5, np.std(daily_rets)) * np.sqrt(252))
    win_rate = (env.winning_trades / max(1, env.total_trades)) * 100.0

    return {
        "strategy": "Random Agent",
        "return_pct": ret,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trades": env.total_trades
    }


def evaluate_heuristic_agent(env: StockTradingEnv) -> dict:
    """Evaluate Heuristic Agent (SMA 5/20 Crossover + RSI filter)."""
    obs = env.reset()
    while not env.done:
        mask = env.get_action_mask()
        # obs[3] is normalized RSI ([-1, 1]) -> map to [0, 100]
        rsi = (obs[3] + 1.0) / 2.0 * 100.0
        # obs[4] is normalized SMA ratio (SMA5 / SMA20 - 1)
        sma_ratio = obs[4]

        action = HOLD
        if mask[BUY] and (rsi < 40 or sma_ratio > 0.05):
            action = BUY
        elif mask[SELL] and (rsi > 70 or sma_ratio < -0.05):
            action = SELL

        obs, _, _, _ = env.step(action)

    ret = ((env.net_worth - env.initial_cash) / env.initial_cash) * 100.0
    p_hist = np.array(env.portfolio_history)
    peak = np.maximum.accumulate(p_hist)
    dd = (peak - p_hist) / np.maximum(1e-5, peak)
    max_dd = float(np.max(dd)) * 100.0

    daily_rets = np.diff(p_hist) / p_hist[:-1]
    sharpe = float(np.mean(daily_rets) / max(1e-5, np.std(daily_rets)) * np.sqrt(252))
    win_rate = (env.winning_trades / max(1, env.total_trades)) * 100.0

    return {
        "strategy": "Rule-Based Heuristic (SMA+RSI)",
        "return_pct": ret,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trades": env.total_trades
    }


def evaluate_canonical_mb(env: StockTradingEnv, episodes_train: int = 100) -> dict:
    """Evaluate Canonical Mushroom Body (1,000 KC)."""
    # 16 sensory inputs -> 1000 KCs -> 3 MBON actions
    mb = MushroomBodyNet(num_pn=16, num_kc=1000, num_mbon=3, k_active=75, seed=42)

    # Pre-train
    for ep in range(episodes_train):
        env_train = StockTradingEnv(initial_cash=10000.0, max_steps=150, seed=500 + ep)
        obs = env_train.reset()
        mb.reset_traces()
        while not env_train.done:
            mask = env_train.get_action_mask()
            # Canonical MB softmax
            logits = np.dot(mb.w_pn_kc.T, obs)
            # k-WTA
            k = 75
            top_k = np.argpartition(logits, -k)[-k:]
            kc_act = np.zeros(1000, dtype=np.float32)
            kc_act[top_k] = 1.0
            mbon_out = np.dot(mb.w_kc_mbon.T, kc_act)
            mbon_out = np.where(mask, mbon_out, -1e9)
            exp_o = np.exp(mbon_out - np.max(mbon_out))
            p = exp_o / np.sum(exp_o)
            act = int(np.random.choice(3, p=p))
            obs, rew, done, info = env_train.step(act)
            # Plasticity
            mb.eligibility_trace *= 0.8
            mb.eligibility_trace[:, act] += kc_act
            mb.w_kc_mbon += 0.05 * rew * mb.eligibility_trace
            mb.w_kc_mbon = np.maximum(0.01, mb.w_kc_mbon)

    # Test
    obs = env.reset()
    mb.reset_traces()
    while not env.done:
        mask = env.get_action_mask()
        logits = np.dot(mb.w_pn_kc.T, obs)
        top_k = np.argpartition(logits, -75)[-75:]
        kc_act = np.zeros(1000, dtype=np.float32)
        kc_act[top_k] = 1.0
        mbon_out = np.dot(mb.w_kc_mbon.T, kc_act)
        mbon_out = np.where(mask, mbon_out, -1e9)
        act = int(np.argmax(mbon_out))
        obs, _, _, _ = env.step(act)

    ret = ((env.net_worth - env.initial_cash) / env.initial_cash) * 100.0
    p_hist = np.array(env.portfolio_history)
    peak = np.maximum.accumulate(p_hist)
    dd = (peak - p_hist) / np.maximum(1e-5, peak)
    max_dd = float(np.max(dd)) * 100.0

    daily_rets = np.diff(p_hist) / p_hist[:-1]
    sharpe = float(np.mean(daily_rets) / max(1e-5, np.std(daily_rets)) * np.sqrt(252))
    win_rate = (env.winning_trades / max(1, env.total_trades)) * 100.0

    return {
        "strategy": "Canonical MB (1,000 KC)",
        "return_pct": ret,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trades": env.total_trades
    }


def evaluate_hippocampal_mb(env: StockTradingEnv, episodes_train: int = 100) -> dict:
    """Evaluate Hippocampal Trading MB (DG-CA3 + HDC + SWR + CPG)."""
    agent = HippocampalTradingMB(dim=2048, k_dg=50, k_ca3=120, stop_loss_pct=-0.03, seed=42)

    # Pre-train
    for ep in range(episodes_train):
        env_train = StockTradingEnv(initial_cash=10000.0, max_steps=150, seed=500 + ep)
        obs = env_train.reset()
        agent.reset_traces()
        while not env_train.done:
            mask = env_train.get_action_mask()
            act, _, _, _ = agent.select_action(obs, mask, training=True)
            obs, rew, done, info = env_train.step(act)
            agent.update_plasticity(rew)
            if info.get("trade_event") == "SELL":
                agent.trigger_swr_episodic_replay(rew)

    # Test
    obs = env.reset()
    agent.reset_traces()
    while not env.done:
        mask = env.get_action_mask()
        act, _, _, _ = agent.select_action(obs, mask, training=False)
        obs, rew, done, info = env.step(act)
        if info.get("trade_event") == "SELL":
            agent.trigger_swr_episodic_replay(rew)

    ret = ((env.net_worth - env.initial_cash) / env.initial_cash) * 100.0
    p_hist = np.array(env.portfolio_history)
    peak = np.maximum.accumulate(p_hist)
    dd = (peak - p_hist) / np.maximum(1e-5, peak)
    max_dd = float(np.max(dd)) * 100.0

    daily_rets = np.diff(p_hist) / p_hist[:-1]
    sharpe = float(np.mean(daily_rets) / max(1e-5, np.std(daily_rets)) * np.sqrt(252))
    win_rate = (env.winning_trades / max(1, env.total_trades)) * 100.0

    return {
        "strategy": "Hippocampus DG-CA3 Trading MB",
        "return_pct": ret,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trades": env.total_trades
    }


def run_multi_market_benchmark():
    print("=" * 86)
    print("BIOMIMETIC STOCK TRADING BENCHMARK: HIPPOCAMPUS (DG-CA3) VS BASELINES")
    print("Evaluated across 5 Diverse Market Regimes (Bull, Bear, Chop, Rally, Recovery)")
    print("=" * 86)

    test_seeds = [101, 202, 303, 404, 505]
    regime_names = ["Steady Bull", "Bear Correction", "Sideways Chop", "Momentum Rally", "Volatile Recovery"]

    # Pre-train Canonical MB and Hippocampal MB
    print("\nPre-training Biomimetic Agents on 60 training episodes...")
    env_sample = StockTradingEnv(initial_cash=10000.0, max_steps=252, seed=42)

    hippocampal_agent = HippocampalTradingMB(dim=2048, k_dg=50, k_ca3=120, stop_loss_pct=-0.03, seed=42)
    for ep in range(60):
        env_tr = StockTradingEnv(initial_cash=10000.0, max_steps=150, seed=500 + ep)
        obs = env_tr.reset()
        hippocampal_agent.reset_traces()
        while not env_tr.done:
            mask = env_tr.get_action_mask()
            act, _, _, _ = hippocampal_agent.select_action(obs, mask, training=True)
            obs, rew, done, info = env_tr.step(act)
            hippocampal_agent.update_plasticity(rew)
            if info.get("trade_event") == "SELL":
                hippocampal_agent.trigger_swr_episodic_replay(rew)

    strategy_stats = {
        "Buy & Hold Benchmark": {"returns": [], "max_dds": [], "sharpes": [], "wins": [], "trades": []},
        "Random Agent": {"returns": [], "max_dds": [], "sharpes": [], "wins": [], "trades": []},
        "Rule-Based Heuristic (SMA+RSI)": {"returns": [], "max_dds": [], "sharpes": [], "wins": [], "trades": []},
        "Canonical MB (1,000 KC)": {"returns": [], "max_dds": [], "sharpes": [], "wins": [], "trades": []},
        "Hippocampus DG-CA3 Trading MB": {"returns": [], "max_dds": [], "sharpes": [], "wins": [], "trades": []}
    }

    t0 = time.time()
    for s_idx, (seed, r_name) in enumerate(zip(test_seeds, regime_names)):
        env_eval = StockTradingEnv(initial_cash=10000.0, max_steps=252, seed=seed)

        # 1. Buy & Hold
        r_bh = evaluate_buy_and_hold(env_eval)
        strategy_stats["Buy & Hold Benchmark"]["returns"].append(r_bh["return_pct"])
        strategy_stats["Buy & Hold Benchmark"]["max_dds"].append(r_bh["max_dd"])
        strategy_stats["Buy & Hold Benchmark"]["sharpes"].append(r_bh["sharpe"])
        strategy_stats["Buy & Hold Benchmark"]["wins"].append(r_bh["win_rate"])
        strategy_stats["Buy & Hold Benchmark"]["trades"].append(r_bh["trades"])

        # 2. Random
        r_rnd = evaluate_random_agent(env_eval, seed=seed)
        strategy_stats["Random Agent"]["returns"].append(r_rnd["return_pct"])
        strategy_stats["Random Agent"]["max_dds"].append(r_rnd["max_dd"])
        strategy_stats["Random Agent"]["sharpes"].append(r_rnd["sharpe"])
        strategy_stats["Random Agent"]["wins"].append(r_rnd["win_rate"])
        strategy_stats["Random Agent"]["trades"].append(r_rnd["trades"])

        # 3. Heuristic
        r_heu = evaluate_heuristic_agent(env_eval)
        strategy_stats["Rule-Based Heuristic (SMA+RSI)"]["returns"].append(r_heu["return_pct"])
        strategy_stats["Rule-Based Heuristic (SMA+RSI)"]["max_dds"].append(r_heu["max_dd"])
        strategy_stats["Rule-Based Heuristic (SMA+RSI)"]["sharpes"].append(r_heu["sharpe"])
        strategy_stats["Rule-Based Heuristic (SMA+RSI)"]["wins"].append(r_heu["win_rate"])
        strategy_stats["Rule-Based Heuristic (SMA+RSI)"]["trades"].append(r_heu["trades"])

        # 4. Canonical MB
        r_cmb = evaluate_canonical_mb(env_eval, episodes_train=15)
        strategy_stats["Canonical MB (1,000 KC)"]["returns"].append(r_cmb["return_pct"])
        strategy_stats["Canonical MB (1,000 KC)"]["max_dds"].append(r_cmb["max_dd"])
        strategy_stats["Canonical MB (1,000 KC)"]["sharpes"].append(r_cmb["sharpe"])
        strategy_stats["Canonical MB (1,000 KC)"]["wins"].append(r_cmb["win_rate"])
        strategy_stats["Canonical MB (1,000 KC)"]["trades"].append(r_cmb["trades"])

        # 5. Hippocampus DG-CA3 MB
        obs = env_eval.reset()
        hippocampal_agent.reset_traces()
        while not env_eval.done:
            mask = env_eval.get_action_mask()
            act, _, _, _ = hippocampal_agent.select_action(obs, mask, training=False)
            obs, rew, done, info = env_eval.step(act)
            if info.get("trade_event") == "SELL":
                hippocampal_agent.trigger_swr_episodic_replay(rew)

        ret = ((env_eval.net_worth - env_eval.initial_cash) / env_eval.initial_cash) * 100.0
        p_hist = np.array(env_eval.portfolio_history)
        peak = np.maximum.accumulate(p_hist)
        dd = (peak - p_hist) / np.maximum(1e-5, peak)
        max_dd = float(np.max(dd)) * 100.0
        daily_rets = np.diff(p_hist) / p_hist[:-1]
        sharpe = float(np.mean(daily_rets) / max(1e-5, np.std(daily_rets)) * np.sqrt(252))
        win_rate = (env_eval.winning_trades / max(1, env_eval.total_trades)) * 100.0

        strategy_stats["Hippocampus DG-CA3 Trading MB"]["returns"].append(ret)
        strategy_stats["Hippocampus DG-CA3 Trading MB"]["max_dds"].append(max_dd)
        strategy_stats["Hippocampus DG-CA3 Trading MB"]["sharpes"].append(sharpe)
        strategy_stats["Hippocampus DG-CA3 Trading MB"]["wins"].append(win_rate)
        strategy_stats["Hippocampus DG-CA3 Trading MB"]["trades"].append(env_eval.total_trades)

    total_time = time.time() - t0

    print(f"\nCompleted multi-market evaluation in {total_time:.2f} seconds.\n")
    print(f"{'Strategy / Agent':<35} | {'Avg Return':<12} | {'Avg Sharpe':<11} | {'Avg Max DD':<12} | {'Avg Win Rate'}")
    print("-" * 88)

    for strat, data in strategy_stats.items():
        avg_ret = float(np.mean(data["returns"]))
        avg_sharpe = float(np.mean(data["sharpes"]))
        avg_dd = float(np.mean(data["max_dds"]))
        avg_win = float(np.mean(data["wins"]))
        print(f"{strat:<35} | {avg_ret:>+9.2f}% | {avg_sharpe:>9.2f} | {avg_dd:>10.2f}% | {avg_win:>10.1f}%")

    print("=" * 88)


if __name__ == "__main__":
    run_multi_market_benchmark()
