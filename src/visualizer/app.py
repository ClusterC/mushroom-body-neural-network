import os
import sys
import time
import pygame
import numpy as np

from src.envs.tic_tac_toe import TicTacToeEnv, EMPTY, PLAYER_X, PLAYER_O, WINNING_COMBINATIONS
from src.models.mushroom_body import MushroomBodyNet
from src.models.hippocampal_xo_mb import HippocampalXOMB
from src.baselines.q_learning import TabularQLearningAgent
from src.opponents.random_agent import RandomAgent
from src.opponents.heuristic_agent import HeuristicAgent
from src.opponents.minimax_agent import MinimaxAgent
from src.training.self_play import train_self_play
from src.visualizer.components import (
    BoardRenderer,
    NeuralRenderer,
    UIButton,
    COLOR_BG,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED,
    COLOR_X_PRIMARY,
    COLOR_O_PRIMARY,
    COLOR_ACCENT_AMBER
)

AVAILABLE_AGENTS = ["Hippocampal MB", "Mushroom Body", "Q-Learning", "Heuristic", "Minimax", "Random", "Human"]

class VisualizerApp:
    """
    Desktop Pygame Application สำหรับแสดงผลการแข่งขันของ AI และวงจรประสาท Mushroom Body / Hippocampus DG-CA3
    """
    def __init__(self, headless=False):
        self.headless = headless
        self.width = 1280
        self.height = 720

        if not headless:
            pygame.init()
            pygame.display.set_caption("Bio-Inspired Mushroom Body & Hippocampus DG-CA3 Matchup Visualizer")
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

            # Fonts setup
            self.fonts = {
                'title': self._get_font(18, bold=True),
                'sub': self._get_font(14, bold=False),
                'small': self._get_font(12, bold=False),
                'badge': self._get_font(13, bold=True),
                'header': self._get_font(20, bold=True)
            }
        else:
            self.screen = None
            self.clock = None
            self.fonts = None

        # สภาพแวดล้อมเกม
        self.env = TicTacToeEnv()

        # โมเดล AI ทั้งหมด
        self.hippo_agent = HippocampalXOMB(seed=42)
        self.mb_agent = MushroomBodyNet(seed=42)
        self.ql_agent = TabularQLearningAgent(seed=42)
        self.heuristic_agent = HeuristicAgent(seed=101)
        self.minimax_agent = MinimaxAgent()
        self.random_agent = RandomAgent(seed=202)

        # ฝึกฝนโมเดลเบื้องต้นอย่างรวดเร็ว (Pre-train 500 episodes) เพื่อให้มีทักษะพร้อมโชว์ทันที
        self._pretrain_models(episodes=500)

        # การตั้งค่าผู้เล่น
        self.agent_x_idx = 0  # Hippocampal MB (ค่าเริ่มต้นใหม่)
        self.agent_o_idx = 3  # Heuristic

        # สถานะการแข่งขัน
        self.last_move = None
        self.winning_combination = None
        self.live_plasticity = True
        self.auto_play = False
        self.speed_mode = "NORMAL"  # SLOW, NORMAL, FAST
        self.last_step_time = 0.0
        self.game_over_timer = 0.0

        # ข้อมูลสำหรับ Neural Circuit
        self.mb_neural_data = {
            "sparse_kc": None,
            "mbon_probs": None,
            "legal_mask": None,
            "selected_action": None
        }

        # สถิติคะแนน
        self.stats = {"wins_x": 0, "wins_o": 0, "draws": 0, "total": 0}

        # การติดตามผลการฝึกฝนและชีวประสาท (Training & Synapse Tracking)
        self.total_trained_episodes = 500
        self.mean_synapse_weight = float(np.mean(self.mb_agent.w_kc_mbon))
        self.last_weight_delta = 0.0
        self.training_toast = None
        self.toast_timer = 0.0

        self.buttons = {}
        if not headless:
            # Component Renderers
            self.board_renderer = BoardRenderer(origin_x=45, origin_y=160, size=330)
            self.neural_renderer = NeuralRenderer(origin_x=405, origin_y=30, width=460, height=660)
            self._init_buttons()

    def _get_font(self, size, bold=False):
        try:
            return pygame.font.SysFont('segoeui', size, bold=bold)
        except Exception:
            return pygame.font.Font(None, size)

    def _pretrain_models(self, episodes=500, is_user_request=False):
        """
        ฝึกฝนโมเดล MB และ QL พร้อมติดตามผลการเปลี่ยนแปลงของค่าน้ำหนัก Synapse
        """
        w_before = float(np.mean(self.mb_agent.w_kc_mbon))
        env = TicTacToeEnv()
        heuristic = HeuristicAgent(seed=999)
        for _ in range(episodes):
            env.reset()
            self.mb_agent.reset_traces()
            self.ql_agent.reset_episode()
            self.hippo_agent.reset_traces()
            while not env.done:
                if env.current_player == PLAYER_X:
                    act_mb = self.mb_agent.select_action(env, training=True)
                    env.step(act_mb)
                else:
                    act_opp = heuristic.select_action(env)
                    env.step(act_opp)
            rew = 1.0 if env.winner == PLAYER_X else (0.0 if env.winner == "DRAW" else -1.0)
            self.mb_agent.update_synapses(rew)
            self.mb_agent.decay_temperature()

        # ฝึกฝน Hippocampal MB ต่อ Minimax และ Heuristic (ทั้งเดินก่อน X และเดินทีหลัง O)
        minimax = MinimaxAgent()
        for p in [PLAYER_X, PLAYER_O]:
            for _ in range(100):
                env.reset()
                self.hippo_agent.reset_traces()
                while not env.done:
                    if env.current_player == p:
                        act_hippo = self.hippo_agent.select_action(env, training=True, player=p)
                        env.step(act_hippo)
                    else:
                        opp_p = PLAYER_O if p == PLAYER_X else PLAYER_X
                        act_opp = minimax.select_action(env, player=opp_p)
                        env.step(act_opp)
                rew_h = 1.0 if env.winner == p else (0.0 if env.winner == "DRAW" else -1.0)
                self.hippo_agent.update_synapses(rew_h, done=True)
                self.hippo_agent.decay_temperature()

        w_after = float(np.mean(self.mb_agent.w_kc_mbon))
        self.mean_synapse_weight = w_after
        self.last_weight_delta = w_after - w_before

        if is_user_request:
            self.total_trained_episodes += episodes

            # ประเมินวัดผลอย่างรวดเร็ว 20 เกมเพื่อดูอัตรา Non-loss ล่าสุด
            eval_wins = 0
            eval_draws = 0
            for _ in range(20):
                env.reset()
                while not env.done:
                    if env.current_player == PLAYER_X:
                        act = self.mb_agent.select_action(env, training=False)
                    else:
                        act = heuristic.select_action(env)
                    env.step(act)
                if env.winner == PLAYER_X:
                    eval_wins += 1
                elif env.winner == "DRAW":
                    eval_draws += 1
            non_loss_pct = ((eval_wins + eval_draws) / 20) * 100.0

            self.training_toast = f"+{episodes} EPISODES APPLIED! (Total: {self.total_trained_episodes:,} Ep) | ΔW: {self.last_weight_delta:+.4f} | Non-loss: {non_loss_pct:.0f}%"
            self.toast_timer = time.time() + 4.0

            # อัปเดตการแสดงผลในแผงสมองสำหรับสถานะกระดานปัจจุบันทันที
            if not self.env.done:
                obs = self.env.get_observation(self.env.current_player)
                mask = self.env.get_action_mask()
                probs, sparse_kc, _ = self.mb_agent.forward(obs, mask)
                self.mb_neural_data["sparse_kc"] = sparse_kc
                self.mb_neural_data["mbon_probs"] = probs
                self.mb_neural_data["legal_mask"] = mask

    def _init_buttons(self):
        btn_font = self.fonts['badge']
        rx = 890
        self.buttons = {
            # Player Selection Buttons
            'prev_x': UIButton((rx, 75, 30, 28), "<", btn_font),
            'next_x': UIButton((rx + 310, 75, 30, 28), ">", btn_font),
            'prev_o': UIButton((rx, 140, 30, 28), "<", btn_font),
            'next_o': UIButton((rx + 310, 140, 30, 28), ">", btn_font),

            # Playback Controls
            'step': UIButton((rx, 220, 100, 36), "STEP", btn_font, active_color=(59, 130, 246)),
            'auto': UIButton((rx + 115, 220, 110, 36), "AUTO: OFF", btn_font, active_color=(16, 185, 129)),
            'reset_board': UIButton((rx + 240, 220, 100, 36), "RESET", btn_font),

            # Speed Presets
            'speed_slow': UIButton((rx, 275, 105, 28), "SLOW", self.fonts['small']),
            'speed_normal': UIButton((rx + 118, 275, 105, 28), "NORMAL", self.fonts['small'], active=True),
            'speed_fast': UIButton((rx + 235, 275, 105, 28), "FAST", self.fonts['small']),

            # Learning & Training Controls
            'toggle_plasticity': UIButton((rx, 340, 340, 32), "LIVE PLASTICITY: ACTIVE", btn_font, active=True, active_color=(16, 185, 129)),
            'train_500': UIButton((rx, 380, 165, 34), "TRAIN HEU (+500)", self.fonts['small'], base_color=(55, 65, 81), hover_color=(75, 85, 99)),
            'train_self_play': UIButton((rx + 175, 380, 165, 34), "SELF-PLAY (+500)", self.fonts['small'], base_color=(99, 102, 241), hover_color=(129, 140, 248), active_color=(16, 185, 129)),

            # Clear Stats
            'clear_stats': UIButton((rx, 650, 340, 32), "RESET SCOREBOARD", self.fonts['small'], base_color=(31, 41, 55))
        }

    def get_agent_name(self, player):
        idx = self.agent_x_idx if player == PLAYER_X else self.agent_o_idx
        return AVAILABLE_AGENTS[idx]

    def reset_game(self):
        self.env.reset()
        self.last_move = None
        self.winning_combination = None
        self.game_over_timer = 0.0
        self.mb_agent.reset_traces()
        self.hippo_agent.reset_traces()
        self.ql_agent.reset_episode()
        self.mb_neural_data = {
            "sparse_kc": None,
            "mbon_probs": None,
            "legal_mask": None,
            "selected_action": None
        }

    def check_winning_combination(self):
        b = self.env.board
        for a, b_idx, c in WINNING_COMBINATIONS:
            if b[a] != EMPTY and b[a] == b[b_idx] == b[c]:
                return (a, b_idx, c)
        return None

    def execute_step(self, human_action=None):
        """
        ดำเนินการตาเดิน 1 ก้าวของฝ่ายปัจจุบัน
        """
        if self.env.done:
            return

        current_player = self.env.current_player
        agent_type = self.get_agent_name(current_player)

        # 1. การตัดสินใจเลือกก้าวเดิน
        if agent_type == "Human":
            if human_action is None or human_action not in self.env.get_legal_actions():
                return
            action = human_action
        elif agent_type == "Hippocampal MB":
            obs = self.env.get_observation(current_player)
            mask = self.env.get_action_mask()
            probs, sparse_kc, _ = self.hippo_agent.forward(obs, mask=mask, board=self.env.board, player=current_player)
            action = self.hippo_agent.select_action(self.env, training=self.live_plasticity, player=current_player)

            # บันทึกข้อมูลเพื่อแสดงผลในแผงสมอง
            self.mb_neural_data["sparse_kc"] = sparse_kc
            self.mb_neural_data["mbon_probs"] = probs
            self.mb_neural_data["legal_mask"] = mask
            self.mb_neural_data["selected_action"] = action
        elif agent_type == "Mushroom Body":
            obs = self.env.get_observation(current_player)
            mask = self.env.get_action_mask()
            probs, sparse_kc, _ = self.mb_agent.forward(obs, mask)
            action = self.mb_agent.select_action(self.env, training=self.live_plasticity, player=current_player)

            # บันทึกข้อมูลเพื่อแสดงผลในแผงสมอง
            self.mb_neural_data["sparse_kc"] = sparse_kc
            self.mb_neural_data["mbon_probs"] = probs
            self.mb_neural_data["legal_mask"] = mask
            self.mb_neural_data["selected_action"] = action
        elif agent_type == "Q-Learning":
            action = self.ql_agent.select_action(self.env, training=self.live_plasticity, player=current_player)
        elif agent_type == "Heuristic":
            action = self.heuristic_agent.select_action(self.env, player=current_player)
        elif agent_type == "Minimax":
            action = self.minimax_agent.select_action(self.env, player=current_player)
        else:  # Random
            action = self.random_agent.select_action(self.env, player=current_player)

        # 2. ทำการเดินในสภาพแวดล้อม
        self.last_move = action
        self.env.step(action)

        # 3. หากเกมจบ ตรวจสอบผู้ชนะ อัปเดต Dopamine และสถิติ
        if self.env.done:
            self.winning_combination = self.check_winning_combination()
            self.stats["total"] += 1

            if self.env.winner == PLAYER_X:
                self.stats["wins_x"] += 1
            elif self.env.winner == PLAYER_O:
                self.stats["wins_o"] += 1
            else:
                self.stats["draws"] += 1

            # ประเมิน Dopamine สำหรับ Hippocampal MB
            if "Hippocampal MB" in [self.get_agent_name(PLAYER_X), self.get_agent_name(PLAYER_O)]:
                hippo_player = PLAYER_X if self.get_agent_name(PLAYER_X) == "Hippocampal MB" else PLAYER_O
                if self.env.winner == hippo_player:
                    reward = 1.0
                elif self.env.winner == "DRAW":
                    reward = 0.0
                else:
                    reward = -1.0

                if self.live_plasticity:
                    self.hippo_agent.update_synapses(reward, done=True)

                if not self.headless and self.neural_renderer:
                    self.neural_renderer.trigger_dopamine_flash(reward)

            # ประเมิน Dopamine สำหรับ Mushroom Body
            if "Mushroom Body" in [self.get_agent_name(PLAYER_X), self.get_agent_name(PLAYER_O)]:
                mb_player = PLAYER_X if self.get_agent_name(PLAYER_X) == "Mushroom Body" else PLAYER_O
                if self.env.winner == mb_player:
                    reward = 1.0
                elif self.env.winner == "DRAW":
                    reward = 0.0
                else:
                    reward = -1.0

                if self.live_plasticity:
                    self.mb_agent.update_synapses(reward)

                if not self.headless and self.neural_renderer:
                    self.neural_renderer.trigger_dopamine_flash(reward)

            # อัปเดต Q-learning ถ้ามี
            if self.live_plasticity and "Q-Learning" in [self.get_agent_name(PLAYER_X), self.get_agent_name(PLAYER_O)]:
                ql_player = PLAYER_X if self.get_agent_name(PLAYER_X) == "Q-Learning" else PLAYER_O
                ql_rew = 1.0 if self.env.winner == ql_player else (0.0 if self.env.winner == "DRAW" else -1.0)
                self.ql_agent.update_q_values(ql_rew)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            # คลิกปุ่มต่างๆ
            for name, btn in self.buttons.items():
                if btn.handle_event(event):
                    self._on_button_click(name)

            # การคลิกกระดานของ Human Player
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                current_type = self.get_agent_name(self.env.current_player)
                if current_type == "Human" and not self.env.done:
                    cell = self.board_renderer.get_cell_at_pos(event.pos)
                    if cell is not None and cell in self.env.get_legal_actions():
                        self.execute_step(human_action=cell)

        return True

    def _on_button_click(self, name):
        if name == 'prev_x':
            self.agent_x_idx = (self.agent_x_idx - 1) % len(AVAILABLE_AGENTS)
            self.reset_game()
        elif name == 'next_x':
            self.agent_x_idx = (self.agent_x_idx + 1) % len(AVAILABLE_AGENTS)
            self.reset_game()
        elif name == 'prev_o':
            self.agent_o_idx = (self.agent_o_idx - 1) % len(AVAILABLE_AGENTS)
            self.reset_game()
        elif name == 'next_o':
            self.agent_o_idx = (self.agent_o_idx + 1) % len(AVAILABLE_AGENTS)
            self.reset_game()
        elif name == 'step':
            if not self.env.done:
                self.execute_step()
        elif name == 'auto':
            self.auto_play = not self.auto_play
            self.buttons['auto'].active = self.auto_play
            self.buttons['auto'].text = "AUTO: ON" if self.auto_play else "AUTO: OFF"
        elif name == 'reset_board':
            self.reset_game()
        elif name == 'speed_slow':
            self.speed_mode = "SLOW"
            self.buttons['speed_slow'].active = True
            self.buttons['speed_normal'].active = False
            self.buttons['speed_fast'].active = False
        elif name == 'speed_normal':
            self.speed_mode = "NORMAL"
            self.buttons['speed_slow'].active = False
            self.buttons['speed_normal'].active = True
            self.buttons['speed_fast'].active = False
        elif name == 'speed_fast':
            self.speed_mode = "FAST"
            self.buttons['speed_slow'].active = False
            self.buttons['speed_normal'].active = False
            self.buttons['speed_fast'].active = True
        elif name == 'toggle_plasticity':
            self.live_plasticity = not self.live_plasticity
            self.buttons['toggle_plasticity'].active = self.live_plasticity
            self.buttons['toggle_plasticity'].text = "LIVE PLASTICITY: ACTIVE" if self.live_plasticity else "LIVE PLASTICITY: FROZEN"
        elif name == 'train_500':
            self._pretrain_models(episodes=500, is_user_request=True)
            if hasattr(self, 'buttons') and 'train_500' in self.buttons:
                self.buttons['train_500'].text = "✓ HEU DONE!"
                self.buttons['train_500'].active = True
        elif name == 'train_self_play':
            res = train_self_play(self.mb_agent, episodes=500, minimax_mix_ratio=0.30)
            self.total_trained_episodes += 500
            self.mean_synapse_weight = res['weight_after']
            self.last_weight_delta = res['weight_delta']
            self.training_toast = f"✓ SELF-PLAY +500 EP! (Total: {self.total_trained_episodes:,}) | Self-Draw: {res['draw_rate']*100:.0f}% | ΔW: {res['weight_delta']:+.4f}"
            self.toast_timer = time.time() + 4.0
            if hasattr(self, 'buttons') and 'train_self_play' in self.buttons:
                self.buttons['train_self_play'].text = "✓ SELF-PLAY DONE!"
                self.buttons['train_self_play'].active = True
            if not self.env.done:
                obs = self.env.get_observation(self.env.current_player)
                mask = self.env.get_action_mask()
                probs, sparse_kc, _ = self.mb_agent.forward(obs, mask)
                self.mb_neural_data["sparse_kc"] = sparse_kc
                self.mb_neural_data["mbon_probs"] = probs
                self.mb_neural_data["legal_mask"] = mask
        elif name == 'clear_stats':
            self.stats = {"wins_x": 0, "wins_o": 0, "draws": 0, "total": 0}

    def update_auto_play(self):
        """
        จัดการการเล่นอัตโนมัติ (Auto-Play) ตามความเร็วที่กำหนด
        """
        now = time.time()
        delays = {"SLOW": 0.7, "NORMAL": 0.3, "FAST": 0.06}
        delay = delays[self.speed_mode]

        if not self.env.done:
            if now - self.last_step_time >= delay:
                current_type = self.get_agent_name(self.env.current_player)
                if current_type != "Human":
                    self.execute_step()
                    self.last_step_time = now
        else:
            # จบเกมแล้ว รอสักครู่แล้วเริ่มเกมใหม่
            if self.game_over_timer == 0.0:
                self.game_over_timer = now
            elif now - self.game_over_timer >= (1.4 if self.speed_mode != "FAST" else 0.4):
                self.reset_game()

    def draw(self):
        self.screen.fill(COLOR_BG)

        # 1. แผงซ้าย: XO Board & Match Status
        left_panel_rect = pygame.Rect(20, 30, 365, 660)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, left_panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, left_panel_rect, width=1, border_radius=12)

        # หัวข้อแผงซ้าย
        title_surf = self.fonts['header'].render("TIC-TAC-TOE ARENA", True, (243, 244, 246))
        self.screen.blit(title_surf, (35, 45))

        match_label = f"{self.get_agent_name(PLAYER_X)} (X)  vs  {self.get_agent_name(PLAYER_O)} (O)"
        match_surf = self.fonts['sub'].render(match_label, True, COLOR_TEXT_SECONDARY)
        self.screen.blit(match_surf, (35, 75))

        # ป้ายสถานะตาเดิน (Turn Badge)
        if not self.env.done:
            cur_p = self.env.current_player
            p_name = self.get_agent_name(cur_p)
            turn_text = f"CURRENT TURN: {p_name} ({'X' if cur_p == PLAYER_X else 'O'})"
            badge_color = COLOR_X_PRIMARY if cur_p == PLAYER_X else COLOR_O_PRIMARY
        else:
            if self.env.winner == PLAYER_X:
                turn_text = f"GAME OVER: {self.get_agent_name(PLAYER_X)} (X) WINS!"
                badge_color = COLOR_X_PRIMARY
            elif self.env.winner == PLAYER_O:
                turn_text = f"GAME OVER: {self.get_agent_name(PLAYER_O)} (O) WINS!"
                badge_color = COLOR_O_PRIMARY
            else:
                turn_text = "GAME OVER: DRAW GAME!"
                badge_color = COLOR_ACCENT_AMBER

        turn_surf = self.fonts['badge'].render(turn_text, True, badge_color)
        self.screen.blit(turn_surf, (35, 115))

        # เรนเดอร์กระดาน
        mouse_pos = pygame.mouse.get_pos()
        hover_cell = self.board_renderer.get_cell_at_pos(mouse_pos)
        self.board_renderer.draw(
            self.screen,
            self.env.board,
            last_move=self.last_move,
            winning_combination=self.winning_combination,
            hover_cell=hover_cell
        )

        # คำแนะนำสำหรับ Human
        if self.get_agent_name(self.env.current_player) == "Human" and not self.env.done:
            guide_surf = self.fonts['sub'].render("-> Your turn: Click on any empty cell", True, (52, 211, 153))
            self.screen.blit(guide_surf, (35, 520))

        # 2. แผงกลาง: Neural Circuit
        self.neural_renderer.update()
        cur_agent = self.get_agent_name(self.env.current_player)
        is_hippo = (cur_agent == "Hippocampal MB") or (
            "Hippocampal MB" in [self.get_agent_name(PLAYER_X), self.get_agent_name(PLAYER_O)]
            and "Mushroom Body" not in [self.get_agent_name(PLAYER_X), self.get_agent_name(PLAYER_O)]
        )
        circuit_mode = "hippocampal" if is_hippo else "standard"
        self.neural_renderer.draw(
            self.screen,
            self.fonts,
            sparse_kc=self.mb_neural_data["sparse_kc"],
            mbon_probs=self.mb_neural_data["mbon_probs"],
            legal_mask=self.mb_neural_data["legal_mask"],
            selected_action=self.mb_neural_data["selected_action"],
            circuit_mode=circuit_mode,
            hippo_agent=self.hippo_agent
        )

        # 3. แผงขวา: Controls & Scoreboard
        right_panel_rect = pygame.Rect(880, 30, 380, 660)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, right_panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, right_panel_rect, width=1, border_radius=12)

        # หัวข้อแผงขวา
        ctrl_title = self.fonts['header'].render("CONTROLS & METRICS", True, (243, 244, 246))
        self.screen.blit(ctrl_title, (895, 45))

        # Label: Player X
        x_lbl = self.fonts['sub'].render("PLAYER X (FIRST MOVER):", True, COLOR_X_PRIMARY)
        self.screen.blit(x_lbl, (895, 55))
        agent_x_surf = self.fonts['title'].render(self.get_agent_name(PLAYER_X), True, (255, 255, 255))
        self.screen.blit(agent_x_surf, (940, 78))

        # Label: Player O
        o_lbl = self.fonts['sub'].render("PLAYER O (SECOND MOVER):", True, COLOR_O_PRIMARY)
        self.screen.blit(o_lbl, (895, 120))
        agent_o_surf = self.fonts['title'].render(self.get_agent_name(PLAYER_O), True, (255, 255, 255))
        self.screen.blit(agent_o_surf, (940, 143))

        # Label: Playback Controls
        pb_lbl = self.fonts['sub'].render("PLAYBACK CONTROLS:", True, COLOR_TEXT_SECONDARY)
        self.screen.blit(pb_lbl, (895, 195))

        # Label: Learning & Plasticity
        lp_lbl = self.fonts['sub'].render("BIO-PLASTICITY MANAGEMENT:", True, COLOR_TEXT_SECONDARY)
        self.screen.blit(lp_lbl, (895, 318))

        # Live Synapse & Training Status Monitor Box
        status_box = pygame.Rect(895, 425, 340, 48)
        pygame.draw.rect(self.screen, (15, 23, 42), status_box, border_radius=6)
        pygame.draw.rect(self.screen, (30, 41, 59), status_box, width=1, border_radius=6)

        now = time.time()
        if self.training_toast and now < self.toast_timer:
            # วาด Toast Banner สว่างวาบสีเขียวเมื่อเพิ่งฝึกฝนเสร็จ
            toast_surf = pygame.Surface((340, 48), pygame.SRCALPHA)
            toast_surf.fill((16, 185, 129, 230))
            pygame.draw.rect(toast_surf, (52, 211, 153), (0, 0, 340, 48), width=2, border_radius=6)
            self.screen.blit(toast_surf, (895, 425))

            toast_text1 = f"✓ TRAINED +500 EPISODES! (TOTAL: {self.total_trained_episodes:,})"
            toast_text2 = f"ΔW: {self.last_weight_delta:+.4f} | Softmax Temp: {self.mb_agent.temperature:.3f}"
            self.screen.blit(self.fonts['small'].render(toast_text1, True, (255, 255, 255)), (905, 430))
            self.screen.blit(self.fonts['small'].render(toast_text2, True, (220, 252, 231)), (905, 450))
        else:
            # แสดงค่าสถิติสถานะสมองสด
            if 'train_500' in self.buttons and self.buttons['train_500'].active:
                self.buttons['train_500'].text = "TRAIN HEU (+500)"
                self.buttons['train_500'].active = False
            if 'train_self_play' in self.buttons and self.buttons['train_self_play'].active:
                self.buttons['train_self_play'].text = "SELF-PLAY (+500)"
                self.buttons['train_self_play'].active = False

            line_ep = f"Total Trained: {self.total_trained_episodes:,} Ep | Temp: {self.mb_agent.temperature:.3f}"
            delta_color = (52, 211, 153) if self.last_weight_delta >= 0 else (248, 113, 113)
            line_w = f"Mean Synapse W: {self.mean_synapse_weight:.4f} (Last ΔW: {self.last_weight_delta:+.4f})"

            self.screen.blit(self.fonts['small'].render(line_ep, True, COLOR_TEXT_PRIMARY), (905, 430))
            self.screen.blit(self.fonts['small'].render(line_w, True, delta_color), (905, 450))

        # Scoreboard Section
        sc_lbl = self.fonts['sub'].render("MATCHUP SCOREBOARD:", True, COLOR_TEXT_SECONDARY)
        self.screen.blit(sc_lbl, (895, 485))

        sc_box = pygame.Rect(895, 505, 340, 135)
        pygame.draw.rect(self.screen, (15, 23, 42), sc_box, border_radius=8)

        tot = max(self.stats["total"], 1)
        win_x_pct = (self.stats["wins_x"] / tot) * 100.0
        win_o_pct = (self.stats["wins_o"] / tot) * 100.0
        draw_pct = (self.stats["draws"] / tot) * 100.0

        line1 = f"Total Matches Played : {self.stats['total']}"
        line2 = f"Player X Wins : {self.stats['wins_x']:3d}  ({win_x_pct:4.1f}%)"
        line3 = f"Player O Wins : {self.stats['wins_o']:3d}  ({win_o_pct:4.1f}%)"
        line4 = f"Draw Games    : {self.stats['draws']:3d}  ({draw_pct:4.1f}%)"

        self.screen.blit(self.fonts['sub'].render(line1, True, COLOR_TEXT_PRIMARY), (915, 515))
        self.screen.blit(self.fonts['sub'].render(line2, True, COLOR_X_PRIMARY), (915, 542))
        self.screen.blit(self.fonts['sub'].render(line3, True, COLOR_O_PRIMARY), (915, 569))
        self.screen.blit(self.fonts['sub'].render(line4, True, COLOR_ACCENT_AMBER), (915, 596))


        # วาดปุ่มทั้งหมด
        for btn in self.buttons.values():
            btn.draw(self.screen)

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            if self.auto_play:
                self.update_auto_play()
            self.draw()
            self.clock.tick(60)

        pygame.quit()
