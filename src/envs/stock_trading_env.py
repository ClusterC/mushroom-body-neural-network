"""
Stock Trading Environment for Biomimetic Neuromorphic AI.
Simulates a realistic financial market with OHLCV data, portfolio accounting,
slippage, commission fees, technical indicators, and action masking.
"""

import os
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

HOLD = 0
BUY = 1
SELL = 2

ACTION_NAMES = {HOLD: "HOLD", BUY: "BUY", SELL: "SELL"}


class StockTradingEnv:
    """
    Realistic financial trading environment with portfolio management,
    technical indicator generation, transaction frictions, and action masking.
    """
    def __init__(
        self,
        initial_cash: float = 10000.0,
        max_steps: int = 252,          # 1 trading year of daily bars
        slippage_pct: float = 0.0005,  # 0.05% slippage
        fee_pct: float = 0.001,        # 0.10% commission fee
        seed: Optional[int] = None,
        csv_path: Optional[str] = None
    ):
        self.initial_cash = float(initial_cash)
        self.max_steps = int(max_steps)
        self.slippage_pct = float(slippage_pct)
        self.fee_pct = float(fee_pct)
        self.seed = seed
        self.csv_path = csv_path
        self.rng = np.random.default_rng(seed)

        # Market data arrays
        self.prices: np.ndarray = np.array([])
        self.highs: np.ndarray = np.array([])
        self.lows: np.ndarray = np.array([])
        self.volumes: np.ndarray = np.array([])
        self.current_step = 0
        self.done = False

        # Portfolio state
        self.cash = self.initial_cash
        self.shares = 0
        self.entry_price = 0.0
        self.position_bars = 0
        self.peak_net_worth = self.initial_cash
        self.net_worth = self.initial_cash
        self.portfolio_history: List[float] = []
        self.benchmark_history: List[float] = []
        self.trade_history: List[Dict[str, Any]] = []

        # Trade metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_realized_pnl = 0.0
        self.last_trade_return = 0.0

        self._load_or_generate_market()
        self.reset()

    def _load_or_generate_market(self):
        """Generate stochastic market series (GBM + Jump Diffusion + Regime Shifts) or load CSV."""
        if self.csv_path and os.path.exists(self.csv_path):
            import csv
            closes, highs, lows, vols = [], [], [], []
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    closes.append(float(row.get('Close', row.get('close', 100.0))))
                    highs.append(float(row.get('High', row.get('high', closes[-1] * 1.01))))
                    lows.append(float(row.get('Low', row.get('low', closes[-1] * 0.99))))
                    vols.append(float(row.get('Volume', row.get('volume', 1000000.0))))
            self.prices = np.array(closes, dtype=np.float64)
            self.highs = np.array(highs, dtype=np.float64)
            self.lows = np.array(lows, dtype=np.float64)
            self.volumes = np.array(vols, dtype=np.float64)
            self.warmup_steps = 30
            return

        # Generate realistic multi-regime market (Warmup 30 + max_steps + 10 buffer)
        total_len = self.max_steps + 40
        dt = 1.0 / 252.0
        base_price = 100.0

        prices = [base_price]
        highs = [base_price * 1.005]
        lows = [base_price * 0.995]
        vols = [1_000_000.0]

        # Multi-regime parameters: (drift, volatility, duration)
        regimes = [
            (0.12, 0.16, 60),    # Steady Bullish
            (-0.18, 0.28, 45),   # Bearish correction
            (0.02, 0.12, 50),    # Low-volatility Sideways chop
            (0.25, 0.22, 60),    # Strong Momentum Rally
            (-0.05, 0.18, 50),   # Range-bound choppy
            (0.10, 0.15, 60)     # Moderate Uptrend
        ]

        curr_p = base_price
        for drift, vol, duration in regimes:
            for _ in range(duration):
                if len(prices) >= total_len:
                    break
                # Geometric Brownian Motion with occasional jumps
                shock = self.rng.standard_normal()
                jump = 0.0
                if self.rng.random() < 0.03:  # 3% chance of jump
                    jump = self.rng.choice([-1, 1]) * self.rng.uniform(0.015, 0.035)

                ret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shock + jump
                curr_p = max(5.0, curr_p * np.exp(ret))

                intraday_vol = curr_p * vol * np.sqrt(dt) * self.rng.uniform(0.8, 1.8)
                h = curr_p + abs(intraday_vol * self.rng.uniform(0.3, 1.0))
                l = max(1.0, curr_p - abs(intraday_vol * self.rng.uniform(0.3, 1.0)))
                v = max(100_000.0, 1_000_000.0 * (1.0 + 2.0 * abs(ret)) * self.rng.uniform(0.7, 1.4))

                prices.append(curr_p)
                highs.append(h)
                lows.append(l)
                vols.append(v)

        self.prices = np.array(prices, dtype=np.float64)
        self.highs = np.array(highs, dtype=np.float64)
        self.lows = np.array(lows, dtype=np.float64)
        self.volumes = np.array(vols, dtype=np.float64)
        self.warmup_steps = 30

    def reset(self) -> np.ndarray:
        """Reset the environment to the initial state."""
        self.current_step = self.warmup_steps
        self.done = False
        self.cash = self.initial_cash
        self.shares = 0
        self.entry_price = 0.0
        self.position_bars = 0
        self.peak_net_worth = self.initial_cash
        self.net_worth = self.initial_cash
        self.last_trade_return = 0.0

        self.portfolio_history = [self.initial_cash]
        self.benchmark_history = [self.initial_cash]
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_realized_pnl = 0.0

        return self.get_observation()

    def get_action_mask(self) -> np.ndarray:
        """
        Mask legal actions:
        - HOLD (0): Always legal
        - BUY  (1): Legal only if currently 100% in cash and cash can afford at least 1 share
        - SELL (2): Legal only if holding stock shares
        """
        curr_price = self.prices[self.current_step]
        can_hold = True
        can_buy = (self.shares == 0) and (self.cash >= curr_price * (1.0 + self.fee_pct))
        can_sell = (self.shares > 0)
        return np.array([can_hold, can_buy, can_sell], dtype=bool)

    def get_observation(self) -> np.ndarray:
        """
        Extract 16-dimensional normalized market & portfolio sensory vector (Projection Neurons).
        """
        idx = self.current_step
        close = self.prices[idx]

        # 1. Normalized returns
        ret_1 = (close - self.prices[idx - 1]) / self.prices[idx - 1]
        ret_5 = (close - self.prices[idx - 5]) / self.prices[idx - 5]
        ret_20 = (close - self.prices[idx - 20]) / self.prices[idx - 20]

        # 2. Moving averages
        sma_5 = np.mean(self.prices[idx - 4: idx + 1])
        sma_20 = np.mean(self.prices[idx - 19: idx + 1])
        sma_ratio = (sma_5 / max(1e-5, sma_20)) - 1.0
        price_to_sma20 = (close / max(1e-5, sma_20)) - 1.0

        # 3. RSI 14-period
        diffs = np.diff(self.prices[idx - 14: idx + 1])
        gains = np.maximum(diffs, 0.0)
        losses = np.abs(np.minimum(diffs, 0.0))
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0.0:
            rsi = 1.0
        else:
            rs = avg_gain / avg_loss
            rsi = 1.0 - (1.0 / (1.0 + rs))

        # 4. Volatility / Normalized Range
        tr = max(self.highs[idx] - self.lows[idx], abs(self.highs[idx] - self.prices[idx - 1]))
        vol_norm = min(1.0, tr / max(1e-5, close * 0.05))

        # 5. Volume ratio
        vol_mean = np.mean(self.volumes[idx - 19: idx + 1])
        vol_ratio = np.clip((self.volumes[idx] / max(1.0, vol_mean)) - 1.0, -1.0, 1.0)

        # 6. Portfolio features
        is_holding = 1.0 if self.shares > 0 else 0.0
        if self.shares > 0 and self.entry_price > 0.0:
            unrealized_pnl_pct = (close - self.entry_price) / self.entry_price
        else:
            unrealized_pnl_pct = 0.0

        pos_duration = min(1.0, self.position_bars / 50.0)
        drawdown = (self.peak_net_worth - self.net_worth) / max(1e-5, self.peak_net_worth)
        drawdown = np.clip(drawdown, 0.0, 1.0)

        cum_return = (self.net_worth - self.initial_cash) / self.initial_cash

        # 7. Regimes & Signals
        trend_regime = 1.0 if close >= sma_20 else -1.0
        momentum_regime = 1.0 if ret_5 >= 0 else -1.0
        rsi_signal = 1.0 if rsi > 0.70 else (-1.0 if rsi < 0.30 else 0.0)

        obs = np.array([
            np.clip(ret_1 * 20.0, -2.0, 2.0),
            np.clip(ret_5 * 10.0, -2.0, 2.0),
            np.clip(ret_20 * 5.0, -2.0, 2.0),
            rsi * 2.0 - 1.0,                      # -1.0 (oversold) to +1.0 (overbought)
            np.clip(sma_ratio * 15.0, -2.0, 2.0),
            np.clip(price_to_sma20 * 10.0, -2.0, 2.0),
            vol_norm * 2.0 - 1.0,
            vol_ratio,
            is_holding,
            np.clip(unrealized_pnl_pct * 10.0, -2.0, 2.0),
            pos_duration,
            drawdown * 2.0 - 1.0,
            trend_regime,
            momentum_regime,
            rsi_signal,
            np.clip(cum_return, -1.0, 2.0)
        ], dtype=np.float32)

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one trading action (HOLD, BUY, SELL).
        Returns: (next_observation, reward, done, info)
        """
        if self.done:
            return self.get_observation(), 0.0, True, {}

        curr_price = self.prices[self.current_step]
        mask = self.get_action_mask()

        # If model selects an illegal action, force HOLD
        if not mask[action]:
            action = HOLD

        reward = 0.0
        executed_action = action
        trade_event = None

        if action == BUY and mask[BUY]:
            # Execute Long Buy with slippage and commission fee
            exec_price = curr_price * (1.0 + self.slippage_pct)
            fee_factor = 1.0 + self.fee_pct
            shares_to_buy = int(self.cash / (exec_price * fee_factor))
            if shares_to_buy > 0:
                cost = shares_to_buy * exec_price
                fee = cost * self.fee_pct
                self.cash -= (cost + fee)
                self.shares = shares_to_buy
                self.entry_price = exec_price
                self.position_bars = 0
                trade_event = "BUY"
                # Small execution penalty to prevent churn
                reward -= (self.fee_pct * 10.0)

        elif action == SELL and mask[SELL]:
            # Execute Sell with slippage and commission fee
            exec_price = curr_price * (1.0 - self.slippage_pct)
            gross_proceeds = self.shares * exec_price
            fee = gross_proceeds * self.fee_pct
            net_proceeds = gross_proceeds - fee
            cost_basis = self.shares * self.entry_price
            trade_pnl = net_proceeds - cost_basis
            trade_return = trade_pnl / max(1e-5, cost_basis)

            self.cash += net_proceeds
            self.total_trades += 1
            self.total_realized_pnl += trade_pnl
            self.last_trade_return = trade_return

            if trade_pnl > 0:
                self.winning_trades += 1
                # Scaled reward for profitable trades
                reward += min(2.5, trade_return * 15.0 + 0.5)
            else:
                self.losing_trades += 1
                # Scaled penalty for loss
                reward += max(-3.0, trade_return * 15.0 - 0.5)

            self.trade_history.append({
                "step": self.current_step,
                "type": "SELL",
                "entry_price": self.entry_price,
                "exit_price": exec_price,
                "pnl": trade_pnl,
                "return_pct": trade_return,
                "bars_held": self.position_bars
            })

            self.shares = 0
            self.entry_price = 0.0
            self.position_bars = 0
            trade_event = "SELL"

        else:
            # HOLD action
            if self.shares > 0:
                self.position_bars += 1
                # Unrealized mark-to-market incremental reward
                step_ret = (curr_price - self.prices[self.current_step - 1]) / self.prices[self.current_step - 1]
                reward += np.clip(step_ret * 3.0, -0.2, 0.2)
            else:
                # Holding cash during downtrend is rewarded (Patience bonus)
                sma_20 = np.mean(self.prices[self.current_step - 19: self.current_step + 1])
                if curr_price < sma_20:
                    reward += 0.02

        # Advance market step
        self.current_step += 1
        new_close = self.prices[self.current_step]

        # Update Net Worth
        self.net_worth = self.cash + (self.shares * new_close)
        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth

        drawdown = (self.peak_net_worth - self.net_worth) / max(1e-5, self.peak_net_worth)
        if drawdown > 0.05:
            # Quadratic drawdown penalty
            reward -= float(drawdown ** 2) * 2.0

        self.portfolio_history.append(self.net_worth)

        # Benchmark Buy & Hold tracking
        start_price = self.prices[self.warmup_steps]
        benchmark_val = self.initial_cash * (new_close / start_price)
        self.benchmark_history.append(benchmark_val)

        # Check termination
        if self.current_step >= len(self.prices) - 2 or (self.current_step - self.warmup_steps) >= self.max_steps:
            self.done = True

        info = {
            "step": self.current_step,
            "action": executed_action,
            "trade_event": trade_event,
            "net_worth": self.net_worth,
            "cash": self.cash,
            "shares": self.shares,
            "drawdown": drawdown,
            "total_trades": self.total_trades,
            "win_rate": (self.winning_trades / max(1, self.total_trades)),
            "benchmark_val": benchmark_val
        }

        return self.get_observation(), float(reward), self.done, info
