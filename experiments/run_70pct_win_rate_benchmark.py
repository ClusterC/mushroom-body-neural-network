"""
Benchmark Evaluation Suite: 70%+ Win Rate Golden Pullback Sniper Architecture.

Tests the Hippocampal Trading Mushroom Body across 100 multi-asset, multi-regime
market episodes (TECH_MOMENTUM, INDEX_ETF, CRYPTO_VOLATILE, DEFENSIVE_VALUE).
Validates that the empirical Win Rate decisively surpasses the 70.0% target.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.hippocampal_trading_mb import HippocampalTradingMB
from src.envs.stock_trading_env import StockTradingEnv


def run_win_rate_benchmark(episodes_per_profile: int = 25):
    print("=" * 80)
    print("   HIPPOCAMPAL TRADING MUSHROOM BODY: 70%+ WIN RATE BENCHMARK SUITE   ")
    print("=" * 80)

    profiles = ["TECH_MOMENTUM", "INDEX_ETF", "CRYPTO_VOLATILE", "DEFENSIVE_VALUE"]
    mb = HippocampalTradingMB(seed=42)

    overall_trades = 0
    overall_wins = 0
    overall_losses = 0
    overall_pnl = 0.0
    overall_pnl_wins = 0.0
    overall_pnl_losses = 0.0

    profile_stats = {}

    for prof in profiles:
        prof_trades = 0
        prof_wins = 0
        prof_losses = 0
        prof_pnl = 0.0

        for ep in range(episodes_per_profile):
            seed = 7000 + (hash(prof) % 1000) + ep
            env = StockTradingEnv(
                initial_cash=10000.0,
                max_steps=252,
                asset_profile=prof,
                seed=seed
            )
            obs = env.reset()
            mb.reset_traces()

            while not env.done:
                mask = env.get_action_mask()
                action, _, _, _ = mb.select_action(obs, mask, training=False)
                next_obs, rew, done, info = env.step(action)
                obs = next_obs

            prof_trades += env.total_trades
            prof_wins += env.winning_trades
            prof_losses += env.losing_trades
            ep_pnl = env.net_worth - env.initial_cash
            prof_pnl += ep_pnl

            for trade in env.trade_history:
                pnl = trade.get("pnl", 0.0)
                if pnl > 0:
                    overall_pnl_wins += pnl
                else:
                    overall_pnl_losses += abs(pnl)

        prof_win_rate = (prof_wins / max(1, prof_trades)) * 100.0
        profile_stats[prof] = {
            "trades": prof_trades,
            "wins": prof_wins,
            "losses": prof_losses,
            "win_rate": prof_win_rate,
            "pnl": prof_pnl
        }

        overall_trades += prof_trades
        overall_wins += prof_wins
        overall_losses += prof_losses
        overall_pnl += prof_pnl

    aggregate_win_rate = (overall_wins / max(1, overall_trades)) * 100.0
    profit_factor = overall_pnl_wins / max(1e-4, overall_pnl_losses)

    print("\n--- RESULTS BY ASSET PROFILE ---")
    print(f"{'Asset Profile':<20} | {'Trades':<8} | {'Wins':<6} | {'Losses':<6} | {'Win Rate':<10} | {'Net PnL ($)':<12}")
    print("-" * 75)
    for prof, stat in profile_stats.items():
        print(f"{prof:<20} | {stat['trades']:<8} | {stat['wins']:<6} | {stat['losses']:<6} | {stat['win_rate']:>8.2f}% | ${stat['pnl']:>10.2f}")

    print("-" * 75)
    print("\n--- AGGREGATE PERFORMANCE SUMMARY ---")
    print(f"Total Market Episodes Evaluated: {episodes_per_profile * len(profiles)} (25,200 trading days)")
    print(f"Total Executed Trades:           {overall_trades}")
    print(f"Winning Trades:                  {overall_wins}")
    print(f"Losing Trades:                   {overall_losses}")
    print(f"Empirical Win Rate:              {aggregate_win_rate:.2f}%")
    print(f"Profit Factor:                   {profit_factor:.2f}")
    print(f"Cumulative Net PnL:              ${overall_pnl:.2f}")
    print("=" * 80)

    target_achieved = aggregate_win_rate >= 70.0
    status_str = "SUCCESS (>= 70.0%)" if target_achieved else "FAILED (< 70.0%)"
    print(f"Target Status: {status_str}")
    print("=" * 80)

    return {
        "aggregate_win_rate": aggregate_win_rate,
        "overall_trades": overall_trades,
        "overall_wins": overall_wins,
        "target_achieved": target_achieved,
        "profile_stats": profile_stats
    }


if __name__ == "__main__":
    run_win_rate_benchmark(episodes_per_profile=25)
