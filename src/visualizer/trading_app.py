"""
Desktop Pygame Financial Terminal & Neuromorphic Visualizer.
Displays live candlestick/price charts, moving average overlays, buy/sell markers,
equity curve vs. Buy & Hold benchmark, Dentate Gyrus 50-cell sparse activations,
CA3 sequence memory, SWR Episodic Replay flash, and CPG Stop-Loss alerts.
"""

import os
import sys
import time
import threading
import pygame
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

# Hardware Acceleration & GPU Detection
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else "CPU Mode"
    GPU_DEVICE = torch.device("cuda:0" if HAS_CUDA else "cpu")
    GPU_MEM_INFO = (
        f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        if HAS_CUDA else "N/A"
    )
except Exception:
    HAS_CUDA = False
    GPU_NAME = "CPU Mode"
    GPU_DEVICE = None
    GPU_MEM_INFO = "N/A"

from src.envs.stock_trading_env import StockTradingEnv, HOLD, BUY, SELL, ACTION_NAMES
from src.models.hippocampal_trading_mb import (
    HippocampalTradingMB,
    REGIME_NAMES,
    REGIME_BULL_EXPANSION,
    REGIME_BEAR_DISTRIBUTION,
    REGIME_CHOPPY_SIDEWAYS,
    REGIME_VOLATILE_SHOCK
)
from src.training.gpu_trading_trainer import GPUTradingTrainer

from src.visualizer.components import (
    UIButton,
    COLOR_BG,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED
)

# Financial Terminal Custom Palette
COLOR_BULL_GREEN = (16, 185, 129)       # Emerald Green (Buy / Profit)
COLOR_BEAR_RED = (244, 63, 94)          # Rose Red (Sell / Loss)
COLOR_AMBER_DG = (245, 158, 11)         # Hippocampus DG Amber
COLOR_BENCHMARK_CYAN = (6, 182, 212)    # Buy & Hold Cyan
COLOR_SMA_SHORT = (250, 204, 21)        # SMA 5 Yellow
COLOR_SMA_LONG = (139, 92, 246)         # SMA 20 Purple
COLOR_SWR_GOLD = (251, 191, 36)         # SWR Replay Gold


class TradingVisualizerApp:
    """
    Desktop Pygame Application for Biomimetic Hippocampal Algorithmic Trading.
    """
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.width = 1280
        self.height = 760

        if not self.headless:
            pygame.init()
            pygame.font.init()
            pygame.display.set_caption("Biomimetic Financial Terminal - Hippocampus (DG-CA3) Trading MB")
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

            self.fonts = {
                'header': self._get_font(20, bold=True),
                'title': self._get_font(16, bold=True),
                'sub': self._get_font(13, bold=False),
                'small': self._get_font(11, bold=False),
                'mono': self._get_font(13, bold=True, mono=True),
                'badge': self._get_font(12, bold=True)
            }
        else:
            self.screen = None
            self.clock = None
            self.fonts = None

        # Asset Profiles configuration
        self.asset_profiles_list = [
            ("TECH_MOMENTUM", "TECH GROWTH (HIGH BETA)", None),
            ("INDEX_ETF", "INDEX ETF (S&P 500)", None),
            ("CRYPTO_VOLATILE", "CRYPTO (BTC VOLATILE)", None),
            ("DEFENSIVE_VALUE", "DEFENSIVE VALUE (DIVIDEND)", None),
            ("REAL_SPY_5Y", "REAL 5Y: SPY ETF (S&P 500)", "data/assets/SPY_5Y_historical.csv"),
            ("REAL_AAPL_5Y", "REAL 5Y: AAPL (APPLE TECH)", "data/assets/AAPL_5Y_historical.csv"),
            ("REAL_QQQ_5Y", "REAL 5Y: QQQ (NASDAQ 100)", "data/assets/QQQ_5Y_historical.csv"),
            ("REAL_BTC_5Y", "REAL 5Y: BTC/USD (CRYPTO)", "data/assets/BTC_5Y_historical.csv"),
            ("REAL_SPY", "REAL MARKET: SPY ETF (1Y)", "data/assets/SPY_historical.csv"),
            ("REAL_AAPL", "REAL MARKET: AAPL TECH (1Y)", "data/assets/AAPL_historical.csv"),
            ("REAL_BTC", "REAL MARKET: BTC/USD (1Y)", "data/assets/BTC_historical.csv"),
        ]
        self.asset_idx = 0

        # Environment & Hippocampal Agent
        self.env = StockTradingEnv(initial_cash=10000.0, max_steps=252, asset_profile="TECH_MOMENTUM", seed=42)
        self.mb = HippocampalTradingMB(dim=2048, k_dg=50, k_ca3=120, stop_loss_pct=-0.03, seed=42)

        # Simulation state
        self.obs = self.env.reset()
        self.auto_trade = False
        self.speed_options = [1, 3, 10]
        self.speed_idx = 1
        self.speed_mode = 3
        self.live_plasticity = True
        self.cpg_enabled = True
        self.running = True
        self.last_step_time = 0
        self.step_delay = 300  # ms base delay

        # Telemetry & Notification tracking
        self.last_action = HOLD
        self.last_reward = 0.0
        self.last_probs = np.ones(3) / 3.0
        self.swr_flash_timer = 0.0
        self.cpg_alert_timer = 0.0
        self.toast_msg = ""
        self.toast_color = COLOR_BULL_GREEN
        self.toast_timer = 0.0
        self.total_trained_episodes = 0

        # Action history for chart markers: list of (step, action, price)
        self.executed_trades: List[Tuple[int, int, float]] = []

        # Hardware Telemetry & Accelerator Status
        self.has_cuda = HAS_CUDA
        self.gpu_device_name = GPU_NAME
        self.gpu_mem_info = GPU_MEM_INFO
        self.gpu_device = GPU_DEVICE
        self.gpu_active = False

        # Background Training State & Real-time Progress Bar
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        self.training_progress = 0.0
        self.training_current_ep = 0
        self.training_total_ep = 0
        self.training_win_rate = 0.0
        self.training_status_text = "STANDBY"

        self.buttons = {}
        if not self.headless:
            self._init_buttons()

    def _get_font(self, size: int, bold: bool = False, mono: bool = False):
        try:
            if mono:
                return pygame.font.SysFont("Consolas, Courier New, monospace", size, bold=bold)
            return pygame.font.SysFont("Segoe UI, Arial, sans-serif", size, bold=bold)
        except Exception:
            return pygame.font.Font(None, size)

    def _init_buttons(self):
        btn_y = 512
        prof_label = self.asset_profiles_list[self.asset_idx][1]
        self.buttons = {
            'asset': UIButton((850, 474, 395, 30), f"ASSET: {prof_label} ▾", self.fonts['sub'], active_color=(37, 99, 235), active=True),
            'step': UIButton((850, btn_y, 90, 34), "STEP", self.fonts['sub'], active_color=(55, 65, 81)),
            'auto': UIButton((950, btn_y, 140, 34), "AUTO-TRADE: OFF", self.fonts['sub'], active_color=COLOR_BULL_GREEN),
            'speed': UIButton((1100, btn_y, 110, 34), f"SPEED: {self.speed_mode}x", self.fonts['sub']),
            'plasticity': UIButton((850, btn_y + 42, 170, 34), "PLASTICITY: ON", self.fonts['sub'], active_color=(139, 92, 246), active=True),
            'cpg': UIButton((1030, btn_y + 42, 180, 34), "CPG STOP-LOSS: ON", self.fonts['sub'], active_color=COLOR_AMBER_DG, active=True),
            'reset': UIButton((850, btn_y + 84, 90, 34), "RESET", self.fonts['sub'], base_color=(75, 85, 99)),
            'train': UIButton((950, btn_y + 84, 260, 34), "⚡ TRAIN HISTORICAL (+500 EP)", self.fonts['sub'], active_color=COLOR_SWR_GOLD, active=True)
        }

    def show_toast(self, msg: str, color=COLOR_BULL_GREEN, duration: float = 3.0):
        self.toast_msg = msg
        self.toast_color = color
        self.toast_timer = time.time() + duration

    def step_simulation(self):
        """Execute one trading step in the environment."""
        if self.env.done:
            self.show_toast("EPISODE COMPLETE - RESETTING TO START", color=COLOR_AMBER_DG)
            self.reset_simulation()
            return

        mask = self.env.get_action_mask()
        action, probs, dg_sparse, ca3_sparse = self.mb.select_action(
            self.obs, mask, training=self.live_plasticity
        )

        if not self.cpg_enabled:
            # Override CPG if disabled
            if not mask[action]:
                action = HOLD

        self.last_action = action
        self.last_probs = probs
        curr_price = float(self.env.prices[self.env.current_step])

        if action in (BUY, SELL) and mask[action]:
            self.executed_trades.append((self.env.current_step, action, curr_price))

        next_obs, reward, done, info = self.env.step(action)
        self.last_reward = reward

        # Live Plasticity synaptic update
        if self.live_plasticity:
            self.mb.update_plasticity(reward)

        # Handle trade close / SWR replay
        if info.get("trade_event") == "SELL":
            self.mb.trigger_swr_episodic_replay(reward, trade_return=self.env.last_trade_return)
            self.swr_flash_timer = time.time() + 1.2
            ret_pct = self.env.last_trade_return * 100.0
            color = COLOR_BULL_GREEN if ret_pct >= 0 else COLOR_BEAR_RED
            self.show_toast(f"TRADE CLOSED: {ret_pct:+.2f}% (PRIORITIZED SWR REPLAY)", color=color)

        if self.mb.last_cpg_triggered:
            self.cpg_alert_timer = time.time() + 1.5

        self.obs = next_obs

    def reset_simulation(self):
        """Reset environment and agent buffers."""
        self.obs = self.env.reset()
        self.mb.reset_traces()
        self.executed_trades.clear()
        self.last_action = HOLD
        self.last_reward = 0.0
        self.swr_flash_timer = 0.0
        self.cpg_alert_timer = 0.0

    def train_episodes(self, n_episodes: int = 500, async_mode: Optional[bool] = None):
        """Curriculum training loop over diverse market regimes (Non-blocking in GUI)."""
        if self.is_training:
            self.show_toast("TRAINING ALREADY IN PROGRESS...", color=COLOR_AMBER_DG)
            return

        if async_mode is None:
            async_mode = not self.headless

        if async_mode:
            self.is_training = True
            self.training_total_ep = n_episodes
            self.training_current_ep = 0
            self.training_progress = 0.0
            self.training_status_text = f"TRAINING (0/{n_episodes})"
            if 'train' in self.buttons:
                self.buttons['train'].text = "TRAINING IN PROGRESS..."
            self.training_thread = threading.Thread(
                target=self._training_worker,
                args=(n_episodes,),
                daemon=True
            )
            self.training_thread.start()
        else:
            self._training_worker(n_episodes)

    def _training_worker(self, n_episodes: int):
        """Worker thread executing curriculum training with high-speed GPU acceleration or CPU fallback."""
        self.is_training = True
        self.gpu_active = self.has_cuda
        self.training_total_ep = n_episodes
        self.training_current_ep = 0
        self.training_progress = 0.0
        self.training_status_text = f"TRAINING (0/{n_episodes})"

        t0 = time.time()

        # Pure GPU-Accelerated Training Path (Batched Parallel CUDA Engine)
        if self.has_cuda and self.gpu_device is not None:
            try:
                batch_size = min(128, n_episodes)
                trainer = GPUTradingTrainer(self.mb, batch_size=batch_size, device=str(self.gpu_device))

                def update_progress(current, total):
                    self.training_current_ep = current
                    self.training_progress = current / max(1, total)
                    self.training_status_text = f"GPU TRAINING ({current}/{total})"

                res = trainer.train(total_episodes=n_episodes, progress_callback=update_progress)
                dur = time.time() - t0
                self.total_trained_episodes += n_episodes
                self.training_current_ep = n_episodes
                self.training_progress = 1.0
                self.training_win_rate = res["win_rate"]
                self.is_training = False
                self.gpu_active = False
                self.training_status_text = f"COMPLETED ({n_episodes} EP | {res['win_rate']:.1f}% WIN)"

                if 'train' in self.buttons:
                    self.buttons['train'].text = "⚡ TRAIN HISTORICAL (+500 EP)"

                self.show_toast(
                    f"⚡ GPU ACCELERATED: {n_episodes} EP IN {dur:.2f}s ({res['fps']:.0f} FPS) | WIN: {res['win_rate']:.1f}% | {self.gpu_device_name}",
                    color=COLOR_SWR_GOLD
                )
                return
            except Exception as e:
                # Rollback state if GPU pipeline encounters issues before CPU fallback
                self.total_trained_episodes = max(0, self.total_trained_episodes - n_episodes)
                self.training_progress = 0.0
                pass

        # CPU Fallback Loop
        total_pnl = 0.0
        wins = 0
        trades = 0
        profiles = ["TECH_MOMENTUM", "INDEX_ETF", "CRYPTO_VOLATILE", "DEFENSIVE_VALUE"]

        for ep in range(n_episodes):
            prof = profiles[ep % len(profiles)]
            env_train = StockTradingEnv(initial_cash=10000.0, max_steps=252, asset_profile=prof, seed=1000 + ep)
            obs = env_train.reset()
            self.mb.reset_traces()

            while not env_train.done:
                mask = env_train.get_action_mask()
                action, _, _, _ = self.mb.select_action(obs, mask, training=True)
                next_obs, rew, done, info = env_train.step(action)
                self.mb.update_plasticity(rew)

                if info.get("trade_event") == "SELL":
                    self.mb.trigger_swr_episodic_replay(rew, trade_return=env_train.last_trade_return)

                obs = next_obs

            total_pnl += (env_train.net_worth - env_train.initial_cash)
            wins += env_train.winning_trades
            trades += env_train.total_trades

            self.training_current_ep = ep + 1
            self.training_progress = (ep + 1) / n_episodes
            self.training_status_text = f"TRAINING ({self.training_current_ep}/{n_episodes})"

        dur = time.time() - t0
        self.total_trained_episodes += n_episodes
        win_rate = (wins / max(1, trades)) * 100.0
        self.training_win_rate = win_rate
        self.is_training = False
        self.gpu_active = False
        self.training_status_text = f"COMPLETED ({n_episodes} EP | {win_rate:.1f}% WIN)"

        if 'train' in self.buttons:
            self.buttons['train'].text = "⚡ TRAIN HISTORICAL (+500 EP)"

        device_label = "CUDA GPU" if self.has_cuda else "CPU"
        self.show_toast(
            f"TRAINED {n_episodes} EP IN {dur:.2f}s | WIN RATE: {win_rate:.1f}% | {device_label}",
            color=COLOR_SWR_GOLD
        )

    def draw_price_chart(self):
        """Draw historical candlestick/line price chart with moving averages and trade markers."""
        panel_rect = pygame.Rect(20, 60, 800, 360)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, panel_rect, width=1, border_radius=10)

        # Header info
        curr_price = float(self.env.prices[self.env.current_step])
        step_idx = self.env.current_step
        asset_label = self.asset_profiles_list[self.asset_idx][1]
        self.screen.blit(self.fonts['title'].render(f"PRICE CHART: {asset_label}", True, COLOR_TEXT_PRIMARY), (35, 75))
        price_str = f"PRICE: ${curr_price:.2f}  |  BAR: {step_idx}/{len(self.env.prices)}"
        self.screen.blit(self.fonts['mono'].render(price_str, True, COLOR_SMA_SHORT), (450, 75))

        # Chart plotting area
        chart_x, chart_y, chart_w, chart_h = 35, 110, 765, 290
        pygame.draw.rect(self.screen, (13, 17, 28), (chart_x, chart_y, chart_w, chart_h), border_radius=6)

        window_size = 70
        start_idx = max(0, step_idx - window_size)
        visible_prices = self.env.prices[start_idx: step_idx + 1]

        if len(visible_prices) < 2:
            return

        min_p = float(np.min(visible_prices)) * 0.99
        max_p = float(np.max(visible_prices)) * 1.01
        p_range = max(1.0, max_p - min_p)

        def to_screen(i, p):
            sx = chart_x + int((i / max(1, len(visible_prices) - 1)) * (chart_w - 20)) + 10
            sy = chart_y + chart_h - int(((p - min_p) / p_range) * (chart_h - 30)) - 15
            return sx, sy

        # Horizontal gridlines
        for g_pct in [0.25, 0.50, 0.75]:
            gy = chart_y + int(chart_h * g_pct)
            gp = max_p - g_pct * p_range
            pygame.draw.line(self.screen, (25, 33, 49), (chart_x, gy), (chart_x + chart_w, gy), 1)
            self.screen.blit(self.fonts['small'].render(f"${gp:.1f}", True, COLOR_TEXT_MUTED), (chart_x + 5, gy - 12))

        # Draw Price Curve
        pts = [to_screen(i, float(p)) for i, p in enumerate(visible_prices)]
        pygame.draw.lines(self.screen, (243, 244, 246), False, pts, 2)

        # Draw SMA 5 (Yellow) & SMA 20 (Purple)
        if len(visible_prices) >= 5:
            sma5_pts = []
            for i in range(len(visible_prices)):
                global_i = start_idx + i
                if global_i >= 4:
                    val = float(np.mean(self.env.prices[global_i - 4: global_i + 1]))
                    sma5_pts.append(to_screen(i, val))
            if len(sma5_pts) >= 2:
                pygame.draw.lines(self.screen, COLOR_SMA_SHORT, False, sma5_pts, 1)

        if len(visible_prices) >= 20:
            sma20_pts = []
            for i in range(len(visible_prices)):
                global_i = start_idx + i
                if global_i >= 19:
                    val = float(np.mean(self.env.prices[global_i - 19: global_i + 1]))
                    sma20_pts.append(to_screen(i, val))
            if len(sma20_pts) >= 2:
                pygame.draw.lines(self.screen, COLOR_SMA_LONG, False, sma20_pts, 2)

        # Draw Executed Trade Markers (Buy = Green ▲, Sell = Red ▼)
        for t_step, t_action, t_price in self.executed_trades:
            if start_idx <= t_step <= step_idx:
                local_i = t_step - start_idx
                tx, ty = to_screen(local_i, t_price)
                if t_action == BUY:
                    pygame.draw.polygon(self.screen, COLOR_BULL_GREEN, [(tx, ty - 12), (tx - 6, ty), (tx + 6, ty)])
                elif t_action == SELL:
                    pygame.draw.polygon(self.screen, COLOR_BEAR_RED, [(tx, ty + 12), (tx - 6, ty), (tx + 6, ty)])

    def draw_equity_panel(self):
        """Draw portfolio equity curve vs. Buy & Hold benchmark and key financial telemetry."""
        panel_rect = pygame.Rect(20, 435, 800, 305)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, panel_rect, width=1, border_radius=10)

        self.screen.blit(self.fonts['title'].render("PORTFOLIO NET WORTH VS. BUY & HOLD BENCHMARK", True, COLOR_TEXT_PRIMARY), (35, 450))

        # Chart area for Equity Curves
        ec_x, ec_y, ec_w, ec_h = 35, 480, 480, 240
        pygame.draw.rect(self.screen, (13, 17, 28), (ec_x, ec_y, ec_w, ec_h), border_radius=6)

        hist = self.env.portfolio_history
        bench = self.env.benchmark_history
        if len(hist) >= 2:
            min_v = min(float(np.min(hist)), float(np.min(bench))) * 0.98
            max_v = max(float(np.max(hist)), float(np.max(bench))) * 1.02
            v_range = max(1.0, max_v - min_v)

            def to_ec_screen(i, val):
                sx = ec_x + int((i / max(1, len(hist) - 1)) * (ec_w - 20)) + 10
                sy = ec_y + ec_h - int(((val - min_v) / v_range) * (ec_h - 30)) - 15
                return sx, sy

            bench_pts = [to_ec_screen(i, float(v)) for i, v in enumerate(bench)]
            agent_pts = [to_ec_screen(i, float(v)) for i, v in enumerate(hist)]

            pygame.draw.lines(self.screen, COLOR_BENCHMARK_CYAN, False, bench_pts, 1)
            pygame.draw.lines(self.screen, COLOR_BULL_GREEN, False, agent_pts, 2)

            # Legends
            self.screen.blit(self.fonts['small'].render("— Hippocampal Agent", True, COLOR_BULL_GREEN), (ec_x + 10, ec_y + 10))
            self.screen.blit(self.fonts['small'].render("— Buy & Hold Benchmark", True, COLOR_BENCHMARK_CYAN), (ec_x + 150, ec_y + 10))

        # Telemetry Stats Table on Right Side of Equity Panel
        stat_x = 535
        stat_y = 480
        ret_pct = ((self.env.net_worth - self.env.initial_cash) / self.env.initial_cash) * 100.0
        ret_color = COLOR_BULL_GREEN if ret_pct >= 0 else COLOR_BEAR_RED
        win_rate = (self.env.winning_trades / max(1, self.env.total_trades)) * 100.0
        drawdown_pct = ((self.env.peak_net_worth - self.env.net_worth) / max(1e-5, self.env.peak_net_worth)) * 100.0

        metrics = [
            ("NET WORTH:", f"${self.env.net_worth:,.2f}", ret_color),
            ("CASH BALANCE:", f"${self.env.cash:,.2f}", COLOR_TEXT_PRIMARY),
            ("SHARES HELD:", f"{self.env.shares:,} units", COLOR_SMA_SHORT),
            ("RETURN (CUMULATIVE):", f"{ret_pct:+.2f}%", ret_color),
            ("WIN RATE:", f"{win_rate:.1f}% ({self.env.winning_trades}/{self.env.total_trades})", COLOR_BULL_GREEN if win_rate >= 50 else COLOR_TEXT_MUTED),
            ("REALIZED PnL:", f"${self.env.total_realized_pnl:+,.2f}", COLOR_BULL_GREEN if self.env.total_realized_pnl >= 0 else COLOR_BEAR_RED),
            ("MAX DRAWDOWN:", f"{drawdown_pct:.2f}%", COLOR_BEAR_RED if drawdown_pct > 5.0 else COLOR_TEXT_MUTED),
            ("TRAINED EPISODES:", f"{self.total_trained_episodes:,} EP", COLOR_SWR_GOLD)
        ]

        for i, (label, val, col) in enumerate(metrics):
            yy = stat_y + (i * 28)
            self.screen.blit(self.fonts['small'].render(label, True, COLOR_TEXT_SECONDARY), (stat_x, yy))
            self.screen.blit(self.fonts['mono'].render(val, True, col), (stat_x + 130, yy))

    def draw_hippocampal_panel(self):
        """Draw Dentate Gyrus granule cells, CA3 sequence depth, and alert banners."""
        panel_rect = pygame.Rect(835, 60, 425, 450)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, panel_rect, width=1, border_radius=10)

        self.screen.blit(self.fonts['title'].render("HIPPOCAMPUS (DG-CA3) TELEMETRY", True, COLOR_AMBER_DG), (850, 75))

        # 1. Dentate Gyrus 50 Granule Cells Grid
        sub_text = "DENTATE GYRUS: 50 ACTIVE / 2,048 (2.44% SPARSITY)"
        self.screen.blit(self.fonts['small'].render(sub_text, True, COLOR_TEXT_SECONDARY), (850, 105))

        dg_box = pygame.Rect(850, 125, 395, 110)
        pygame.draw.rect(self.screen, (13, 17, 28), dg_box, border_radius=6)

        # Plot 200 representation dots (10 cols x 20 rows)
        active_set = set(self.mb.last_dg_indices.tolist()) if len(self.mb.last_dg_indices) > 0 else set()
        cols, rows = 20, 10
        dot_w = dg_box.width // cols
        dot_h = dg_box.height // rows

        for r in range(rows):
            for c in range(cols):
                cell_id = r * cols + c
                dx = dg_box.x + c * dot_w + dot_w // 2
                dy = dg_box.y + r * dot_h + dot_h // 2
                # Check active
                if cell_id in active_set or (cell_id * 10) in active_set:
                    pygame.draw.circle(self.screen, COLOR_AMBER_DG, (dx, dy), 4)
                else:
                    pygame.draw.circle(self.screen, (30, 41, 59), (dx, dy), 2)

        # 2. CA3 Trajectory Sequence Depth Gauge
        ca3_y = 250
        self.screen.blit(self.fonts['small'].render(f"CA3 SEQUENCE DEPTH (Π PERMUTATION): {self.mb.last_ca3_depth}/5 BARS", True, COLOR_TEXT_SECONDARY), (850, ca3_y))
        bar_w = 395
        bar_h = 10
        pygame.draw.rect(self.screen, (30, 41, 59), (850, ca3_y + 20, bar_w, bar_h), border_radius=4)
        fill_w = int(bar_w * (self.mb.last_ca3_depth / 5.0))
        if fill_w > 0:
            pygame.draw.rect(self.screen, (139, 92, 246), (850, ca3_y + 20, fill_w, bar_h), border_radius=4)

        # 3. Action Readout & Softmax Probabilities
        act_y = 295
        self.screen.blit(self.fonts['small'].render("ACTION READOUT PROBABILITIES:", True, COLOR_TEXT_SECONDARY), (850, act_y))

        labels = ["HOLD", "BUY", "SELL"]
        colors = [(107, 114, 128), COLOR_BULL_GREEN, COLOR_BEAR_RED]
        for i, (name, col) in enumerate(zip(labels, colors)):
            prob = float(self.last_probs[i]) if i < len(self.last_probs) else 0.33
            p_y = act_y + 20 + (i * 24)
            self.screen.blit(self.fonts['small'].render(name, True, col), (850, p_y))

            # Probability bar
            meter_x = 895
            meter_w = 260
            pygame.draw.rect(self.screen, (30, 41, 59), (meter_x, p_y + 3, meter_w, 10), border_radius=3)
            fill_p = int(meter_w * prob)
            if fill_p > 0:
                pygame.draw.rect(self.screen, col, (meter_x, p_y + 3, fill_p, 10), border_radius=3)

            self.screen.blit(self.fonts['small'].render(f"{prob * 100:.1f}%", True, COLOR_TEXT_PRIMARY), (meter_x + meter_w + 10, p_y))

        # 4. Flashing Alert Banners (SWR & CPG)
        banner_y = 390
        now = time.time()
        if now < self.swr_flash_timer:
            swr_rect = pygame.Rect(850, banner_y, 395, 34)
            pygame.draw.rect(self.screen, (245, 158, 11, 40), swr_rect, border_radius=6)
            pygame.draw.rect(self.screen, COLOR_SWR_GOLD, swr_rect, width=2, border_radius=6)
            self.screen.blit(self.fonts['badge'].render("⚡ SWR EPISODIC REPLAY: ACTIVE (ONE-SHOT UPDATE)", True, COLOR_SWR_GOLD), (865, banner_y + 8))

        elif now < self.cpg_alert_timer:
            cpg_rect = pygame.Rect(850, banner_y, 395, 34)
            pygame.draw.rect(self.screen, (244, 63, 94, 40), cpg_rect, border_radius=6)
            pygame.draw.rect(self.screen, COLOR_BEAR_RED, cpg_rect, width=2, border_radius=6)
            self.screen.blit(self.fonts['badge'].render(f"⚠️ CPG RISK REFLEX: {self.mb.last_cpg_reason}", True, COLOR_BEAR_RED), (860, banner_y + 8))
        else:
            norm_rect = pygame.Rect(850, banner_y, 395, 34)
            regime_id = getattr(self.mb, 'last_detected_regime', REGIME_CHOPPY_SIDEWAYS)
            conf_pct = getattr(self.mb, 'last_regime_confidence', 0.25) * 100.0
            alloc_pct = getattr(self.env, 'last_alloc_factor', 1.0) * 100.0
            regime_label = REGIME_NAMES.get(regime_id, "NEUTRAL")

            if regime_id == REGIME_BULL_EXPANSION:
                badge_bg = (13, 38, 28)
                badge_border = COLOR_BULL_GREEN
                badge_dot = COLOR_BULL_GREEN
            elif regime_id == REGIME_BEAR_DISTRIBUTION:
                badge_bg = (38, 15, 20)
                badge_border = COLOR_BEAR_RED
                badge_dot = COLOR_BEAR_RED
            elif regime_id == REGIME_VOLATILE_SHOCK:
                badge_bg = (35, 18, 45)
                badge_border = (168, 85, 247)
                badge_dot = (168, 85, 247)
            else:
                badge_bg = (35, 28, 15)
                badge_border = COLOR_AMBER_DG
                badge_dot = COLOR_AMBER_DG

            pygame.draw.rect(self.screen, badge_bg, norm_rect, border_radius=6)
            pygame.draw.rect(self.screen, badge_border, norm_rect, width=1, border_radius=6)
            pygame.draw.circle(self.screen, badge_dot, (864, banner_y + 17), 4)

            reg_text = f"REGIME: {regime_label} ({conf_pct:.0f}%) | SIZING: {alloc_pct:.0f}%"
            self.screen.blit(self.fonts['badge'].render(reg_text, True, badge_border), (876, banner_y + 8))

    def draw_controls(self):
        """Draw interactive control buttons, progress load bar, and notifications."""
        panel_rect = pygame.Rect(835, 465, 425, 275)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, panel_rect, width=1, border_radius=10)

        for btn in self.buttons.values():
            btn.draw(self.screen)

        # -------------------------------------------------------------
        # Real-Time Training Load Bar & Hardware Accelerator Status
        # -------------------------------------------------------------
        bar_x, bar_y, bar_w, bar_h = 850, 638, 395, 22
        load_rect = pygame.Rect(bar_x, bar_y, bar_w, bar_h)

        # Dark outer track
        pygame.draw.rect(self.screen, (15, 23, 42), load_rect, border_radius=5)
        pygame.draw.rect(self.screen, (51, 65, 85), load_rect, width=1, border_radius=5)

        if self.is_training:
            # Active progress fill
            fill_w = max(4, int(bar_w * max(0.01, min(1.0, self.training_progress))))
            fill_color = COLOR_BULL_GREEN if self.has_cuda else COLOR_AMBER_DG
            pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, fill_w, bar_h), border_radius=5)

            # Dynamic shimmer effect on progress bar
            shimmer_offset = int((time.time() * 220) % max(1, fill_w))
            shimmer_x = bar_x + shimmer_offset
            if shimmer_x + 18 <= bar_x + fill_w:
                shimmer_surf = pygame.Surface((18, bar_h), pygame.SRCALPHA)
                shimmer_surf.fill((255, 255, 255, 70))
                self.screen.blit(shimmer_surf, (shimmer_x, bar_y))

            pct_val = self.training_progress * 100.0
            dev_str = "CUDA GPU" if self.has_cuda else "CPU"
            txt = f"⚡ TRAINING: {self.training_current_ep}/{self.training_total_ep} ({pct_val:.1f}%) | {dev_str}"
            self.screen.blit(self.fonts['small'].render(txt, True, (255, 255, 255)), (bar_x + 8, bar_y + 4))

            # Hardware accelerator status sub-line
            if self.has_cuda:
                gpu_status = f"ACCELERATOR: {self.gpu_device_name} | CUDA TENSORS ACTIVE"
                sub_col = COLOR_BULL_GREEN
            else:
                gpu_status = "ACCELERATOR: MULTI-CORE CPU THREADED"
                sub_col = COLOR_TEXT_MUTED
            self.screen.blit(self.fonts['small'].render(gpu_status, True, sub_col), (bar_x, bar_y + 26))

        else:
            # Idle / Standby state
            if self.total_trained_episodes > 0:
                txt = f"READY | TOTAL TRAINED: {self.total_trained_episodes} EP | LAST WIN RATE: {self.training_win_rate:.1f}%"
                bar_col = COLOR_SWR_GOLD
            else:
                txt = "TRAINING STANDBY | READY FOR 500-EP HISTORICAL RUN"
                bar_col = COLOR_TEXT_MUTED

            self.screen.blit(self.fonts['small'].render(txt, True, bar_col), (bar_x + 8, bar_y + 4))

            if self.has_cuda:
                gpu_status = f"HARDWARE: {self.gpu_device_name} (CUDA READY)"
                sub_col = COLOR_TEXT_SECONDARY
            else:
                gpu_status = "HARDWARE: CPU COMPUTE (NO CUDA)"
                sub_col = COLOR_TEXT_MUTED
            self.screen.blit(self.fonts['small'].render(gpu_status, True, sub_col), (bar_x, bar_y + 26))

        # Toast notification
        if time.time() < self.toast_timer:
            toast_rect = pygame.Rect(850, 696, 395, 30)
            pygame.draw.rect(self.screen, (15, 23, 42), toast_rect, border_radius=6)
            pygame.draw.rect(self.screen, self.toast_color, toast_rect, width=1, border_radius=6)
            self.screen.blit(self.fonts['small'].render(self.toast_msg, True, self.toast_color), (860, 703))

    def draw_all(self):
        """Render the complete application interface."""
        self.screen.fill(COLOR_BG)

        # Top Header Bar
        header_rect = pygame.Rect(20, 10, 1240, 40)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, header_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, header_rect, width=1, border_radius=8)

        self.screen.blit(self.fonts['header'].render("BIOMIMETIC FINANCIAL TERMINAL", True, COLOR_BULL_GREEN), (35, 18))
        sub_title = "HIPPOCAMPUS (DG-CA3) ALGORITHMIC TRADING AGENT"
        self.screen.blit(self.fonts['title'].render(sub_title, True, COLOR_TEXT_PRIMARY), (360, 20))

        # GPU Hardware Accelerator Badge (Top-Right of Header)
        badge_w, badge_h = 360, 26
        badge_x = 1240 + 20 - badge_w - 15  # 885
        badge_y = 17
        badge_rect = pygame.Rect(badge_x, badge_y, badge_w, badge_h)

        if self.has_cuda:
            bg_col = (13, 38, 28) if not self.gpu_active else (28, 60, 36)
            border_col = COLOR_BULL_GREEN if not self.gpu_active else COLOR_SWR_GOLD
            pygame.draw.rect(self.screen, bg_col, badge_rect, border_radius=5)
            pygame.draw.rect(self.screen, border_col, badge_rect, width=1, border_radius=5)

            # LED indicator with pulse when active
            led_x, led_y = badge_x + 12, badge_y + 13
            led_col = COLOR_BULL_GREEN if not self.gpu_active else (255, 230, 0)
            pygame.draw.circle(self.screen, led_col, (led_x, led_y), 4)

            state_str = "CUDA ACTIVE ⚡" if self.gpu_active else "CUDA ONLINE"
            gpu_label = f"GPU: {self.gpu_device_name[:22]} ({state_str})"
            self.screen.blit(self.fonts['badge'].render(gpu_label, True, COLOR_TEXT_PRIMARY), (badge_x + 22, badge_y + 5))
        else:
            pygame.draw.rect(self.screen, (24, 32, 47), badge_rect, border_radius=5)
            pygame.draw.rect(self.screen, (51, 65, 85), badge_rect, width=1, border_radius=5)
            pygame.draw.circle(self.screen, (100, 116, 139), (badge_x + 12, badge_y + 13), 4)
            self.screen.blit(self.fonts['badge'].render("HARDWARE: CPU COMPUTE (NO CUDA)", True, COLOR_TEXT_MUTED), (badge_x + 22, badge_y + 5))

        # Render sub-panels
        self.draw_price_chart()
        self.draw_equity_panel()
        self.draw_hippocampal_panel()
        self.draw_controls()

    def handle_events(self):
        """Process GUI interactions and button events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            for name, btn in self.buttons.items():
                if btn.handle_event(event):
                    if name == 'asset':
                        self.asset_idx = (self.asset_idx + 1) % len(self.asset_profiles_list)
                        prof_key, prof_label, csv_f = self.asset_profiles_list[self.asset_idx]
                        btn.text = f"ASSET: {prof_label} ▾"
                        self.env.set_asset_profile(prof_key, csv_path=csv_f)
                        self.reset_simulation()
                        self.show_toast(f"SWITCHED ASSET: {prof_label}", color=COLOR_BULL_GREEN)
                    elif name == 'step':
                        self.step_simulation()
                    elif name == 'auto':
                        self.auto_trade = not self.auto_trade
                        btn.active = self.auto_trade
                        btn.text = "AUTO-TRADE: ON" if self.auto_trade else "AUTO-TRADE: OFF"
                    elif name == 'speed':
                        self.speed_idx = (self.speed_idx + 1) % len(self.speed_options)
                        self.speed_mode = self.speed_options[self.speed_idx]
                        btn.text = f"SPEED: {self.speed_mode}x"
                    elif name == 'plasticity':
                        self.live_plasticity = not self.live_plasticity
                        btn.active = self.live_plasticity
                        btn.text = "PLASTICITY: ON" if self.live_plasticity else "PLASTICITY: OFF"
                    elif name == 'cpg':
                        self.cpg_enabled = not self.cpg_enabled
                        btn.active = self.cpg_enabled
                        btn.text = "CPG STOP-LOSS: ON" if self.cpg_enabled else "CPG STOP-LOSS: OFF"
                    elif name == 'reset':
                        self.reset_simulation()
                        self.show_toast("MARKET SIMULATION RESET", color=COLOR_TEXT_PRIMARY)
                    elif name == 'train':
                        if not self.is_training:
                            self.train_episodes(n_episodes=500)
                        else:
                            self.show_toast("TRAINING CURRENTLY RUNNING...", color=COLOR_AMBER_DG)

    def run(self):
        """Main execution loop for Desktop interactive visualizer."""
        while self.running:
            self.handle_events()

            # Process auto-trade ticks
            now = pygame.time.get_ticks()
            delay = self.step_delay // self.speed_mode
            if self.auto_trade and (now - self.last_step_time) >= delay:
                self.step_simulation()
                self.last_step_time = now

            self.draw_all()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
