"""
Multi-Year (5-Year) Algorithmic Trading Benchmark.
Evaluates Buy & Hold vs. Original MB vs. Regime-Governed Hippocampus MB
across SPY, AAPL, QQQ, and BTC datasets spanning 2019 - 2024.
Computes Cumulative Return, Annualized Sharpe, Maximum Drawdown, and Trade Churn.
"""

import os
import sys
import time
import numpy as np

# Ensure root dir is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.stock_trading_env import StockTradingEnv, HOLD, BUY, SELL
from src.models.hippocampal_trading_mb import (
    HippocampalTradingMB,
    REGIME_BULL_EXPANSION,
    REGIME_BEAR_DISTRIBUTION,
    REGIME_CHOPPY_SIDEWAYS,
    REGIME_VOLATILE_SHOCK,
    REGIME_NAMES
)

BENCHMARK_DATASETS = [
    ("SPY 5-Year (S&P 500 ETF)", "data/assets/SPY_5Y_historical.csv"),
    ("AAPL 5-Year (Apple Tech)", "data/assets/AAPL_5Y_historical.csv"),
    ("QQQ 5-Year (Nasdaq 100 ETF)", "data/assets/QQQ_5Y_historical.csv"),
    ("BTC 5-Year (Bitcoin/USD)", "data/assets/BTC_5Y_historical.csv"),
]


def run_buy_and_hold(csv_path: str) -> dict:
    """Run baseline Buy & Hold on entire multi-year series."""
    env = StockTradingEnv(initial_cash=10000.0, csv_path=csv_path, seed=42)
    obs = env.reset()

    # Buy on step 0
    mask = env.get_action_mask()
    if mask[BUY]:
        env.step(BUY)

    while not env.done:
        env.step(HOLD)

    rets = np.diff(env.portfolio_history) / np.array(env.portfolio_history[:-1])
    sharpe = (np.mean(rets) / max(1e-6, np.std(rets))) * np.sqrt(252) if len(rets) > 0 else 0.0
    cum_ret = ((env.net_worth - env.initial_cash) / env.initial_cash) * 100.0

    peaks = np.maximum.accumulate(env.portfolio_history)
    dds = (peaks - env.portfolio_history) / np.maximum(1e-5, peaks)
    max_dd = float(np.max(dds)) * 100.0

    return {
        "net_worth": env.net_worth,
        "cum_ret": cum_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades": 1,
        "win_rate": 100.0 if cum_ret > 0 else 0.0
    }


def pretrain_agent(mb: HippocampalTradingMB, n_episodes: int = 30):
    """Fast pre-training curriculum on diverse macroeconomic profiles to ground synaptic weights."""
    profiles = ["TECH_MOMENTUM", "INDEX_ETF", "CRYPTO_VOLATILE", "DEFENSIVE_VALUE"]
    for ep in range(n_episodes):
        prof = profiles[ep % len(profiles)]
        env_train = StockTradingEnv(initial_cash=10000.0, max_steps=200, asset_profile=prof, seed=2000 + ep)
        obs = env_train.reset()
        mb.reset_traces()
        while not env_train.done:
            mask = env_train.get_action_mask()
            action, _, _, _ = mb.select_action(obs, mask, training=True)
            next_obs, rew, done, info = env_train.step(action)
            mb.update_plasticity(rew)
            if info.get("trade_event") == "SELL":
                mb.trigger_swr_episodic_replay(rew, trade_return=env_train.last_trade_return)
            obs = next_obs
    mb.reset_traces()


def run_regime_governed_mb(csv_path: str, use_regime_gating: bool = True) -> dict:
    """Run Hippocampal MB agent with or without Macro Cognitive Regime Gating."""
    env = StockTradingEnv(initial_cash=10000.0, csv_path=csv_path, seed=42)
    mb = HippocampalTradingMB(dim=2048, k_dg=50, k_ca3=120, stop_loss_pct=-0.04, seed=42)
    
    # Pre-train weights so the model possesses learned synaptic associations
    pretrain_agent(mb, n_episodes=24)

    obs = env.reset()

    # Walk-forward simulation over 5-year data
    while not env.done:
        mask = env.get_action_mask()
        
        if not use_regime_gating:
            # Bypass regime gating (Original MB baseline)
            ec_vec = mb.encode_entorhinal_cortex(obs)
            dg_sparse = mb.dentate_gyrus_separation(ec_vec)
            ca3_sparse = mb.ca3_recurrent_sequence(dg_sparse)
            combined_repr = (0.45 * dg_sparse) + (0.55 * ca3_sparse)
            norm = np.linalg.norm(combined_repr)
            if norm > 1e-6:
                combined_repr /= norm
            scores = np.dot(combined_repr, mb.action_prototypes).astype(np.float32)
            masked_scores = np.where(mask, scores, -1e9)
            action = int(np.argmax(masked_scores))
            final_action, _, _ = mb.check_cpg_risk_reflex(obs, action)
        else:
            final_action, probs, _, _ = mb.select_action(obs, mask, training=False)

        next_obs, rew, done, info = env.step(final_action)
        mb.update_plasticity(rew)

        if info.get("trade_event") == "SELL":
            mb.trigger_swr_episodic_replay(rew, trade_return=env.last_trade_return)

        obs = next_obs

    rets = np.diff(env.portfolio_history) / np.array(env.portfolio_history[:-1])
    sharpe = (np.mean(rets) / max(1e-6, np.std(rets))) * np.sqrt(252) if len(rets) > 0 else 0.0
    cum_ret = ((env.net_worth - env.initial_cash) / env.initial_cash) * 100.0

    peaks = np.maximum.accumulate(env.portfolio_history)
    dds = (peaks - env.portfolio_history) / np.maximum(1e-5, peaks)
    max_dd = float(np.max(dds)) * 100.0
    win_rate = (env.winning_trades / max(1, env.total_trades)) * 100.0

    return {
        "net_worth": env.net_worth,
        "cum_ret": cum_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades": env.total_trades,
        "win_rate": win_rate
    }


def main():
    print("=" * 86)
    print("      5-YEAR MULTI-YEAR BENCHMARK: HIPPOCAMPAL REGIME GOVERNOR (2019 - 2024)")
    print("=" * 86)
    print(f"{'Asset Class':<26} | {'Strategy':<22} | {'Return %':<10} | {'Sharpe':<7} | {'Max DD%':<8} | {'Trades':<6} | {'Win%':<6}")
    print("-" * 86)

    summary_stats = {"bh": [], "orig": [], "regime": []}

    for label, csv_file in BENCHMARK_DATASETS:
        if not os.path.exists(csv_file):
            print(f"Skipping {label}: {csv_file} not found.")
            continue

        res_bh = run_buy_and_hold(csv_file)
        res_orig = run_regime_governed_mb(csv_file, use_regime_gating=False)
        res_regime = run_regime_governed_mb(csv_file, use_regime_gating=True)

        summary_stats["bh"].append(res_bh)
        summary_stats["orig"].append(res_orig)
        summary_stats["regime"].append(res_regime)

        print(f"{label:<26} | {'Buy & Hold Baseline':<22} | {res_bh['cum_ret']:+8.2f}% | {res_bh['sharpe']:6.2f} | {res_bh['max_dd']:6.2f}% | {res_bh['trades']:6d} | {res_bh['win_rate']:5.1f}%")
        print(f"{'':<26} | {'Original MB (Direct)':<22} | {res_orig['cum_ret']:+8.2f}% | {res_orig['sharpe']:6.2f} | {res_orig['max_dd']:6.2f}% | {res_orig['trades']:6d} | {res_orig['win_rate']:5.1f}%")
        print(f"{'':<26} | {'Regime Governor MB':<22} | {res_regime['cum_ret']:+8.2f}% | {res_regime['sharpe']:6.2f} | {res_regime['max_dd']:6.2f}% | {res_regime['trades']:6d} | {res_regime['win_rate']:5.1f}%")
        print("-" * 86)

    # Average Metrics
    avg_bh_ret = np.mean([s['cum_ret'] for s in summary_stats["bh"]])
    avg_bh_dd = np.mean([s['max_dd'] for s in summary_stats["bh"]])
    avg_orig_ret = np.mean([s['cum_ret'] for s in summary_stats["orig"]])
    avg_orig_dd = np.mean([s['max_dd'] for s in summary_stats["orig"]])
    avg_reg_ret = np.mean([s['cum_ret'] for s in summary_stats["regime"]])
    avg_reg_dd = np.mean([s['max_dd'] for s in summary_stats["regime"]])
    avg_reg_win = np.mean([s['win_rate'] for s in summary_stats["regime"]])

    print("\n" + "=" * 86)
    print("                       5-YEAR AGGREGATE SUMMARY FINDINGS")
    print("=" * 86)
    print(f"1. Buy & Hold Benchmark : Avg Return = {avg_bh_ret:+.2f}%, Avg Max DD = {avg_bh_dd:.2f}%")
    print(f"2. Original Direct MB   : Avg Return = {avg_orig_ret:+.2f}%, Avg Max DD = {avg_orig_dd:.2f}% (Over-trading friction drag)")
    print(f"3. Regime Governor MB   : Avg Return = {avg_reg_ret:+.2f}%, Avg Max DD = {avg_reg_dd:.2f}%, Avg Win Rate = {avg_reg_win:.1f}%")
    print("=" * 86)


if __name__ == "__main__":
    main()
