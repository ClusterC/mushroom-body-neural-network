import os
import sys
import time
import pygame
import numpy as np
import torch

from src.envs.snake_env import SnakeEnv, UP, RIGHT, DOWN, LEFT
from src.models.visual_mushroom_body import VisualMushroomBody
from src.models.stacked_visual_mb import StackedVisualMushroomBody
from src.models.hdc_visual_mb import HDCVisualMushroomBody
from src.models.hippocampal_hdc_mb import HippocampalHDCVisualMB
from src.training.gpu_snake_trainer import GPUMushroomBodyTrainer
from src.visualizer.components import (
    UIButton,
    COLOR_BG,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED
)

DIRECTION_NAMES = {UP: "UP", RIGHT: "RIGHT", DOWN: "DOWN", LEFT: "LEFT"}

class SnakeVisualizerApp:
    """
    Desktop Pygame Application สำหรับเกมงู (Visual Snake)
    แสดงผลกระดานเกมงู, ภาพพิกเซล 3 แชนแนล (Optic Lobe), และสมองชีวภาพ KC 2,000 เซลล์
    พร้อมระบบ Hyperdimensional Computing (HDC-VSA) และเร่งการฝึกบน GPU (CUDA)
    """
    def __init__(self, headless=False):
        self.headless = headless
        self.width = 1280
        self.height = 720

        if not headless:
            pygame.init()
            pygame.display.set_caption("Bio-Inspired Visual Snake - HDC-VSA Mushroom Body")
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

            self.fonts = {
                'header': self._get_font(20, bold=True),
                'title': self._get_font(16, bold=True),
                'sub': self._get_font(13, bold=False),
                'small': self._get_font(11, bold=False),
                'badge': self._get_font(13, bold=True)
            }
        else:
            self.screen = None
            self.clock = None
            self.fonts = None

        # สภาพแวดล้อมเกมงู 10x10
        self.env = SnakeEnv(width=10, height=10, seed=42)

        # โมเดล Visual Mushroom Body ทั้ง 4 รูปแบบ
        self.mb_hippo = HippocampalHDCVisualMB(
            dim=2048, k_dg=50, k_ca3=120, num_mbon=4,
            seed=42
        )
        self.mb_hdc = HDCVisualMushroomBody(
            dim=2048, k_active=100, num_mbon=4,
            seed=42
        )
        self.mb_stacked = StackedVisualMushroomBody(
            channels=3, grid_h=10, grid_w=10,
            num_kc1=1200, k_active1=60, num_concepts=12,
            num_kc2=800, k_active2=40, num_mbon=4,
            seed=42
        )
        self.mb_single = VisualMushroomBody(
            channels=3, grid_h=10, grid_w=10,
            num_kc=2000, k_active=100, num_mbon=4,
            seed=42
        )
        self.brain_mode = "HIPPOCAMPUS"  # "HIPPOCAMPUS" (ค่าเริ่มต้น), "HDC_VSA", "STACKED", หรือ "SINGLE"
        self.mb = self.mb_hippo

        # สถานะ Supercharged Features
        self.cpg_enabled = True
        self.ego_obs = None
        self.prev_action = None
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        # ฝึกฝนเบื้องต้นด้วย GPU 1,000 เกมอย่างรวดเร็ว (ใช้เวลาไม่ถึง 1 วินาที)
        if not headless:
            try:
                trainer = GPUMushroomBodyTrainer(self.mb_stacked, batch_size=256)
                trainer.train(total_episodes=1000, cpg_reflex=True, use_whiskers=True)
                self.total_trained_episodes = 1000
            except Exception:
                self._train_episodes(episodes=300)
                self.total_trained_episodes = 300
        else:
            self.total_trained_episodes = 0

        # สถานะการควบคุม
        self.auto_play = False
        self.live_plasticity = True
        self.speed_mode = "NORMAL"  # SLOW, NORMAL, FAST
        self.last_step_time = 0.0
        self.game_over_timer = 0.0

        # ข้อมูลสำหรับ Neural Circuit
        self.sparse_kc = None
        self.kc1_sparse = None
        self.concepts = None
        self.kc2_sparse = None
        self.mbon_probs = None
        self.legal_mask = None
        self.selected_action = None
        self.last_dopamine = 0.0
        self.dopamine_alpha = 0

        # ข้อมูลสถิติ
        self.total_trained_episodes = 300
        self.high_score = 0
        self.total_games = 0
        self.total_apples = 0
        self.training_toast = None
        self.toast_timer = 0.0

        # 1. พิกัดสำหรับ Single Mode (KC 2,000 จุด: 50 cols x 40 rows)
        self.kc_points = []
        grid_w, grid_h = 50, 40
        start_x, start_y = 660, 95
        spacing_x = 280 / grid_w
        spacing_y = 200 / grid_h
        for r in range(grid_h):
            for c in range(grid_w):
                px = int(start_x + c * spacing_x + spacing_x // 2)
                py = int(start_y + r * spacing_y + spacing_y // 2)
                self.kc_points.append((px, py))

        # 2. พิกัดสำหรับ Stacked Mode:
        # Layer 1: KC1 (1,200 จุด: 40 cols x 30 rows)
        self.kc1_points = []
        l1_w, l1_h = 40, 30
        l1_sx, l1_sy = 655, 92
        l1_dx = 290 / l1_w
        l1_dy = 95 / l1_h
        for r in range(l1_h):
            for c in range(l1_w):
                self.kc1_points.append((int(l1_sx + c * l1_dx + l1_dx // 2), int(l1_sy + r * l1_dy + l1_dy // 2)))

        # Layer 2: KC2 (800 จุด: 40 cols x 20 rows)
        self.kc2_points = []
        l2_w, l2_h = 40, 20
        l2_sx, l2_sy = 655, 275
        l2_dx = 290 / l2_w
        l2_dy = 75 / l2_h
        for r in range(l2_h):
            for c in range(l2_w):
                self.kc2_points.append((int(l2_sx + c * l2_dx + l2_dx // 2), int(l2_sy + r * l2_dy + l2_dy // 2)))

        self.buttons = {}
        if not headless:
            self._init_buttons()

    def _get_font(self, size, bold=False):
        try:
            return pygame.font.SysFont('segoeui', size, bold=bold)
        except Exception:
            return pygame.font.Font(None, size)

    def _init_buttons(self):
        btn_font = self.fonts['badge']
        rx = 980
        self.buttons = {
            'step': UIButton((rx, 80, 125, 34), "STEP", btn_font, active_color=(59, 130, 246)),
            'auto': UIButton((rx + 140, 80, 125, 34), "AUTO: OFF", btn_font, active_color=(16, 185, 129)),
            'reset_game': UIButton((rx, 120, 265, 30), "RESET SNAKE", btn_font),

            'toggle_brain': UIButton((rx, 156, 265, 34), "BRAIN: HIPPOCAMPUS (DG-CA3)", btn_font, active=True, active_color=(245, 158, 11)),

            'speed_slow': UIButton((rx, 196, 85, 26), "SLOW", self.fonts['small']),
            'speed_normal': UIButton((rx + 90, 196, 85, 26), "NORMAL", self.fonts['small'], active=True),
            'speed_fast': UIButton((rx + 180, 196, 85, 26), "FAST", self.fonts['small']),

            'toggle_plasticity': UIButton((rx, 228, 265, 30), "LIVE PLASTICITY: ON", btn_font, active=True, active_color=(16, 185, 129)),
            'toggle_cpg': UIButton((rx, 264, 265, 30), "CPG REFLEX: ACTIVE", btn_font, active=True, active_color=(16, 185, 129)),
            'train_gpu': UIButton((rx, 300, 265, 38), "TRAIN ON GPU (+2,000 EP)", btn_font, base_color=(16, 185, 129), hover_color=(52, 211, 153), active_color=(5, 150, 105))
        }

    def _train_episodes(self, episodes=500):
        """
        ฝึกฝน Visual Mushroom Body (ทั้ง Single หรือ Stacked ตาม brain_mode)
        """
        env = SnakeEnv(width=10, height=10)
        total_apples_eaten = 0
        w_before = float(np.mean(self.mb_stacked.w_kc2_mbon2)) if self.brain_mode == "STACKED" else float(np.mean(self.mb_single.w_kc_mbon))

        for _ in range(episodes):
            env.reset()
            active_mb = self.mb_stacked if self.brain_mode == "STACKED" else self.mb_single
            active_mb.reset_traces()
            prev_act = None
            while not env.done:
                obs = env.get_visual_observation()
                mask = env.get_action_mask()
                if self.brain_mode == "STACKED":
                    action, _, _, _, _ = self.mb_stacked.forward(obs, mask, prev_action=prev_act)
                    prev_act = action
                    _, reward, done, _ = env.step(action)
                    self.mb_stacked.update_plasticity(reward)
                else:
                    action = self.mb_single.select_action(env, training=True)
                    _, reward, done, _ = env.step(action)
                    self.mb_single.update_synapses(reward)
            if self.brain_mode == "SINGLE":
                self.mb_single.decay_temperature()
            total_apples_eaten += env.score

        w_after = float(np.mean(self.mb_stacked.w_kc2_mbon2)) if self.brain_mode == "STACKED" else float(np.mean(self.mb_single.w_kc_mbon))
        return {
            "episodes": episodes,
            "weight_delta": w_after - w_before,
            "mean_weight": w_after,
            "avg_apples": total_apples_eaten / max(episodes, 1)
        }

    def execute_step(self):
        """
        ดำเนินการ 1 ก้าวของงู
        """
        if self.env.done:
            return

        obs = self.env.get_visual_observation()
        legal_mask = self.env.get_action_mask()
        ego_obs = self.env.get_egocentric_observation()
        self.ego_obs = ego_obs
        cpg_mask = self.env.get_safe_action_mask() if self.cpg_enabled else legal_mask

        if self.brain_mode == "HIPPOCAMPUS":
            temp = 0.03 if not self.live_plasticity else 0.20
            action, probs, combined, sims = self.mb_hippo.forward(
                ego_obs=ego_obs,
                action_mask=legal_mask,
                cpg_safe_mask=cpg_mask,
                direction=self.env.direction,
                temperature=temp,
                deterministic=not self.live_plasticity
            )
            self.sparse_kc = self.mb_hippo.last_ca3_attractor
            self.mbon_probs = probs
            self.legal_mask = cpg_mask
            self.selected_action = action

            _, reward, done, info = self.env.step(action)
            if self.live_plasticity:
                self.mb_hippo.update(reward=reward, done=done)
            if done:
                self.mb_hippo.reset_episode()
        elif self.brain_mode == "HDC_VSA":
            temp = 0.04 if not self.live_plasticity else 0.25
            action, probs, kc, sims = self.mb_hdc.forward(
                ego_obs=ego_obs,
                action_mask=legal_mask,
                cpg_safe_mask=cpg_mask,
                direction=self.env.direction,
                temperature=temp,
                deterministic=not self.live_plasticity
            )
            self.sparse_kc = kc
            self.mbon_probs = probs
            self.legal_mask = cpg_mask
            self.selected_action = action

            _, reward, done, info = self.env.step(action)
            if self.live_plasticity:
                self.mb_hdc.update_plasticity(reward)
            if done:
                self.mb_hdc.reset_traces()
        elif self.brain_mode == "STACKED":
            temp = 0.05 if not self.live_plasticity else 0.35
            action, probs, kc1, concepts, kc2 = self.mb_stacked.forward(
                obs,
                action_mask=legal_mask,
                prev_action=self.prev_action,
                temperature=temp,
                ego_obs=ego_obs,
                cpg_safe_mask=cpg_mask
            )
            self.prev_action = action
            self.sparse_kc = kc1
            self.kc1_sparse = kc1
            self.concepts = concepts
            self.kc2_sparse = kc2
            self.mbon_probs = probs
            self.legal_mask = cpg_mask
            self.selected_action = action

            _, reward, done, info = self.env.step(action)
            if self.live_plasticity:
                self.mb_stacked.update_plasticity(reward)
            if done:
                self.mb_stacked.reset_traces()
                self.prev_action = None
        else:
            probs, sparse_kc, _ = self.mb_single.forward(obs, cpg_mask)
            action = self.mb_single.select_action(self.env, training=self.live_plasticity)
            self.sparse_kc = sparse_kc
            self.mbon_probs = probs
            self.legal_mask = cpg_mask
            self.selected_action = action

            _, reward, done, info = self.env.step(action)
            if self.live_plasticity:
                self.mb_single.update_synapses(reward)
            if done:
                self.mb_single.reset_traces()

        self.last_dopamine = reward
        self.dopamine_alpha = 255

        if done:
            self.total_games += 1
            self.total_apples += self.env.score
            if self.env.score > self.high_score:
                self.high_score = self.env.score

    def reset_game(self):
        self.env.reset()
        self.sparse_kc = None
        self.kc1_sparse = None
        self.concepts = None
        self.kc2_sparse = None
        self.mbon_probs = None
        self.legal_mask = None
        self.selected_action = None
        self.game_over_timer = 0.0
        self.mb_single.reset_traces()
        self.mb_stacked.reset_traces()
        self.mb_hdc.reset_traces()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            for name, btn in self.buttons.items():
                if btn.handle_event(event):
                    self._on_button_click(name)

            # รองรับการบังคับด้วยปุ่มลูกศร (Human control)
            if event.type == pygame.KEYDOWN and not self.env.done:
                key_map = {
                    pygame.K_UP: UP,
                    pygame.K_RIGHT: RIGHT,
                    pygame.K_DOWN: DOWN,
                    pygame.K_LEFT: LEFT
                }
                if event.key in key_map:
                    manual_action = key_map[event.key]
                    if self.env.get_action_mask()[manual_action]:
                        # ดำเนินการเดิน
                        _, rew, _, _ = self.env.step(manual_action)
                        self.last_dopamine = rew
                        self.dopamine_alpha = 255
                        if self.live_plasticity:
                            if self.brain_mode == "STACKED":
                                self.mb_stacked.update_plasticity(rew)
                            else:
                                self.mb_single.update_synapses(rew)

        return True

    def _on_button_click(self, name):
        if name == 'step':
            if not self.env.done:
                self.execute_step()
        elif name == 'auto':
            self.auto_play = not self.auto_play
            self.buttons['auto'].active = self.auto_play
            self.buttons['auto'].text = "AUTO: ON" if self.auto_play else "AUTO: OFF"
        elif name == 'reset_game':
            self.reset_game()
        elif name == 'toggle_brain':
            if self.brain_mode == "HIPPOCAMPUS":
                self.brain_mode = "HDC_VSA"
                self.mb = self.mb_hdc
                self.buttons['toggle_brain'].text = "BRAIN: HDC-VSA MB"
                self.buttons['toggle_brain'].active = True
                self.buttons['toggle_brain'].active_color = (139, 92, 246)
            elif self.brain_mode == "HDC_VSA":
                self.brain_mode = "STACKED"
                self.mb = self.mb_stacked
                self.buttons['toggle_brain'].text = "BRAIN: STACKED DEEP MB"
                self.buttons['toggle_brain'].active = True
                self.buttons['toggle_brain'].active_color = (6, 182, 212)
            elif self.brain_mode == "STACKED":
                self.brain_mode = "SINGLE"
                self.mb = self.mb_single
                self.buttons['toggle_brain'].text = "BRAIN: SINGLE MB"
                self.buttons['toggle_brain'].active = False
                self.buttons['toggle_brain'].active_color = (59, 130, 246)
            else:
                self.brain_mode = "HIPPOCAMPUS"
                self.mb = self.mb_hippo
                self.buttons['toggle_brain'].text = "BRAIN: HIPPOCAMPUS (DG-CA3)"
                self.buttons['toggle_brain'].active = True
                self.buttons['toggle_brain'].active_color = (245, 158, 11)
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
            self.buttons['toggle_plasticity'].text = "LIVE PLASTICITY: ON" if self.live_plasticity else "LIVE PLASTICITY: FROZEN"
        elif name == 'toggle_cpg':
            self.cpg_enabled = not self.cpg_enabled
            self.buttons['toggle_cpg'].active = self.cpg_enabled
            self.buttons['toggle_cpg'].text = "CPG REFLEX: ACTIVE" if self.cpg_enabled else "CPG REFLEX: OFF"
        elif name == 'train_gpu':
            try:
                trainer = GPUMushroomBodyTrainer(self.mb_stacked, batch_size=256)
                res = trainer.train(total_episodes=2000, cpg_reflex=self.cpg_enabled, use_whiskers=True)
                self.total_trained_episodes += 2000
                dev_short = "GTX 1060" if "1060" in res['device'] else res['device'][:12]
                self.training_toast = f"⚡ GPU TRAINED 2,000 EP ({dev_short}) | Avg: {res['avg_apples']:.2f} | {res['fps']:.0f} FPS"
                self.toast_timer = time.time() + 4.5
                if 'train_gpu' in self.buttons:
                    self.buttons['train_gpu'].text = "✓ GPU TRAINED +2K"
                    self.buttons['train_gpu'].active = True
            except Exception as e:
                self.training_toast = f"GPU Error: {str(e)[:30]}"
                self.toast_timer = time.time() + 3.0

    def update_auto_play(self):
        now = time.time()
        delays = {"SLOW": 0.4, "NORMAL": 0.15, "FAST": 0.03}
        delay = delays[self.speed_mode]

        if not self.env.done:
            if now - self.last_step_time >= delay:
                self.execute_step()
                self.last_step_time = now
        else:
            if self.game_over_timer == 0.0:
                self.game_over_timer = now
            elif now - self.game_over_timer >= (1.2 if self.speed_mode != "FAST" else 0.3):
                self.reset_game()

    def draw(self):
        self.screen.fill(COLOR_BG)

        # 1. แผงซ้าย: Snake Arena (10x10 Grid)
        arena_panel = pygame.Rect(20, 20, 360, 680)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, arena_panel, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, arena_panel, width=1, border_radius=12)

        self.screen.blit(self.fonts['header'].render("VISUAL SNAKE ARENA", True, (243, 244, 246)), (35, 35))
        status_sub = f"Apples Eaten: {self.env.score}   |   Steps: {self.env.steps}"
        self.screen.blit(self.fonts['sub'].render(status_sub, True, (52, 211, 153)), (35, 65))

        # ตารางเกมงู (320x320 px, แต่ละช่อง 32x32 px)
        origin_x, origin_y = 40, 100
        cell_size = 32
        board_rect = pygame.Rect(origin_x, origin_y, 320, 320)
        pygame.draw.rect(self.screen, (15, 23, 42), board_rect, border_radius=8)
        pygame.draw.rect(self.screen, (30, 41, 59), board_rect, width=2, border_radius=8)

        # เส้นตาราง
        for i in range(1, 10):
            x = origin_x + i * cell_size
            y = origin_y + i * cell_size
            pygame.draw.line(self.screen, (24, 33, 47), (x, origin_y), (x, origin_y + 320), 1)
            pygame.draw.line(self.screen, (24, 33, 47), (origin_x, y), (origin_x + 320, y), 1)

        # วาดอาหาร (Apple)
        if self.env.food and self.env.food[0] >= 0:
            fr, fc = self.env.food
            fx = origin_x + fc * cell_size + cell_size // 2
            fy = origin_y + fr * cell_size + cell_size // 2
            # Glow effect
            glow_surf = pygame.Surface((cell_size * 2, cell_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (244, 63, 94, 60), (cell_size, cell_size), 14)
            self.screen.blit(glow_surf, (fx - cell_size, fy - cell_size))
            pygame.draw.circle(self.screen, (244, 63, 94), (fx, fy), 10)

        # วาดลำตัวงู
        for idx, (br, bc) in enumerate(self.env.snake[1:]):
            bx = origin_x + bc * cell_size + 3
            by = origin_y + br * cell_size + 3
            # ลำตัวไล่ระดับสีเขียว
            green_val = int(max(140, 210 - idx * 5))
            pygame.draw.rect(self.screen, (16, green_val, 129), (bx, by, cell_size - 6, cell_size - 6), border_radius=6)

        # วาดหัวงู พร้อม Egocentric Whiskers Radar
        if self.env.snake:
            hr, hc = self.env.snake[0]
            hx = origin_x + hc * cell_size + 2
            hy = origin_y + hr * cell_size + 2
            head_color = (52, 211, 153) if not self.env.done else (239, 68, 68)
            pygame.draw.rect(self.screen, head_color, (hx, hy, cell_size - 4, cell_size - 4), border_radius=8)

            # วาดลำแสง Egocentric Whiskers (3 ทิศทาง: Forward, Left, Right)
            if self.ego_obs is not None and not self.env.done:
                center_hx = hx + (cell_size - 4) // 2
                center_hy = hy + (cell_size - 4) // 2
                d = self.env.direction
                rel_deltas = [
                    ((-1, 0) if d == UP else (0, 1) if d == RIGHT else (1, 0) if d == DOWN else (0, -1)),  # Forward
                    ((0, -1) if d == UP else (-1, 0) if d == RIGHT else (0, 1) if d == DOWN else (1, 0)),  # Left
                    ((0, 1) if d == UP else (1, 0) if d == RIGHT else (0, -1) if d == DOWN else (-1, 0)),  # Right
                ]
                for wi in range(3):
                    dr, dc = rel_deltas[wi]
                    # ความยาวตามระยะห่างกำแพง
                    ray_len = int(14 + self.ego_obs[wi] * 22)
                    target_x = center_hx + dc * ray_len
                    target_y = center_hy + dr * ray_len
                    # สีแดงหากมีลำตัวขวาง สีฟ้าหากโล่ง
                    has_danger = self.ego_obs[3 + wi] > 0
                    ray_color = (239, 68, 68) if has_danger else (56, 189, 248)
                    pygame.draw.line(self.screen, ray_color, (center_hx, center_hy), (target_x, target_y), 2)
                    pygame.draw.circle(self.screen, ray_color, (target_x, target_y), 3)

        # แถบสถานะใต้กระดาน
        if self.env.done:
            msg = f"GAME OVER! Collided with Wall/Self"
            color = (239, 68, 68)
        else:
            cpg_tag = "[CPG SAFE]" if self.cpg_enabled else "[NO REFLEX]"
            msg = f"HEADING: {DIRECTION_NAMES.get(self.env.direction, 'NONE')} {cpg_tag}"
            color = (52, 211, 153)
        self.screen.blit(self.fonts['badge'].render(msg, True, color), (40, 435))

        # 2. แผงกลางที่ 1: Visual Pixel Channels (Optic Lobe)
        optic_panel = pygame.Rect(395, 20, 235, 680)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, optic_panel, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, optic_panel, width=1, border_radius=12)

        self.screen.blit(self.fonts['header'].render("OPTIC LOBE", True, (243, 244, 246)), (410, 35))
        self.screen.blit(self.fonts['sub'].render("3x10x10 Visual Channels", True, COLOR_TEXT_SECONDARY), (410, 65))

        obs = self.env.get_visual_observation()
        channel_names = ["Ch 0: Snake Head", "Ch 1: Snake Body", "Ch 2: Food Target"]
        channel_colors = [(52, 211, 153), (6, 182, 212), (244, 63, 94)]

        for ch in range(3):
            cy = 100 + ch * 190
            self.screen.blit(self.fonts['sub'].render(channel_names[ch], True, channel_colors[ch]), (410, cy))
            mini_origin_x = 410
            mini_origin_y = cy + 24
            mini_cell = 14
            mini_rect = pygame.Rect(mini_origin_x, mini_origin_y, 140, 140)
            pygame.draw.rect(self.screen, (15, 23, 42), mini_rect, border_radius=6)

            for r in range(10):
                for c in range(10):
                    val = obs[ch, r, c]
                    if val > 0:
                        mx = mini_origin_x + c * mini_cell + 1
                        my = mini_origin_y + r * mini_cell + 1
                        c_rgb = tuple(int(val * cv) for cv in channel_colors[ch])
                        pygame.draw.rect(self.screen, c_rgb, (mx, my, mini_cell - 2, mini_cell - 2), border_radius=2)

        # 3. แผงกลางที่ 2: Mushroom Body Biological Circuit
        mb_panel = pygame.Rect(645, 20, 310, 680)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, mb_panel, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, mb_panel, width=1, border_radius=12)

        if self.brain_mode == "HIPPOCAMPUS":
            self.screen.blit(self.fonts['header'].render("HIPPOCAMPUS (DG-CA3)", True, (245, 158, 11)), (660, 32))
            dg_act = self.mb_hippo.k_dg
            ca3_act = self.mb_hippo.k_ca3
            sub_title = f"EC D=2,048 -> DG ({dg_act} k-WTA 2.4%) -> CA3 ({ca3_act}) -> CA1"
            self.screen.blit(self.fonts['small'].render(sub_title, True, (253, 230, 138)), (660, 58))

            # 1. Dentate Gyrus (DG) Ultra-Sparse Pattern Separation (50 cells)
            dg_box = pygame.Rect(655, 76, 290, 80)
            pygame.draw.rect(self.screen, (15, 23, 42), dg_box, border_radius=6)
            self.screen.blit(self.fonts['small'].render("DENTATE GYRUS (DG): PATTERN SEPARATION", True, (245, 158, 11)), (660, 78))
            self.screen.blit(self.fonts['badge'].render(f"Ultra-Sparse: {dg_act} Active Granule Cells (2.44%)", True, (251, 191, 36)), (660, 96))
            # วาดจุดแทน Granule Cells
            dg_pts_x = 660
            for i in range(50):
                px = dg_pts_x + (i % 25) * 11
                py = 125 + (i // 25) * 14
                pygame.draw.circle(self.screen, (245, 158, 11), (px, py), 3)

            # 2. CA3 Recurrent Attractor & Temporal Permutation (Π)
            ca3_box = pygame.Rect(655, 162, 290, 88)
            pygame.draw.rect(self.screen, (15, 23, 42), ca3_box, border_radius=6)
            self.screen.blit(self.fonts['small'].render("CA3 ATTRACTOR & TEMPORAL SEQUENCE (Π)", True, (6, 182, 212)), (660, 164))
            seq_len = len(self.mb_hippo.temporal_history)
            self.screen.blit(self.fonts['small'].render(f"Attractor Settling: 2 Steps | Π Depth: {seq_len}/4 steps", True, COLOR_TEXT_SECONDARY), (660, 184))
            
            # SWR Status Indicator
            swr_status = "⚡ SWR EPISODIC REPLAY: ACTIVE" if self.mb_hippo.swr_active else "SWR REPLAY: STANDBY (EPISODIC BUFFER)"
            swr_c = (250, 204, 21) if self.mb_hippo.swr_active else COLOR_TEXT_MUTED
            self.screen.blit(self.fonts['small'].render(swr_status, True, swr_c), (660, 206))
            buf_info = f"Buffered Trajectory Steps: {len(self.mb_hippo.episode_buffer)}"
            self.screen.blit(self.fonts['small'].render(buf_info, True, (148, 163, 184)), (660, 226))

            # 3. CA1 / MBON Action Readout (Clean-up Memory Matching)
            mbon_y = 258
            self.screen.blit(self.fonts['title'].render("CA1 / MBON ACTION READOUT", True, (243, 244, 246)), (660, mbon_y))
            mbon_box = pygame.Rect(655, mbon_y + 22, 290, 120)
            pygame.draw.rect(self.screen, (15, 23, 42), mbon_box, border_radius=8)

            bar_w = 48
            gap = (mbon_box.width - (bar_w * 4)) // 5
            chart_base = mbon_box.bottom - 22

            for a_idx in range(4):
                bx = mbon_box.left + gap + a_idx * (bar_w + gap)
                prob = self.mbon_probs[a_idx] if self.mbon_probs is not None else 0.0
                is_legal = self.legal_mask[a_idx] if self.legal_mask is not None else True
                is_chosen = (self.selected_action == a_idx)

                bar_h = int(prob * 75)
                if not is_legal:
                    bar_color = (75, 85, 99)
                    bar_h = 3
                elif is_chosen:
                    bar_color = (245, 158, 11)
                else:
                    bar_color = (6, 182, 212)

                if bar_h > 0:
                    pygame.draw.rect(self.screen, bar_color, (bx, chart_base - bar_h, bar_w, bar_h), border_radius=4)

                lbl = DIRECTION_NAMES[a_idx][:2]
                lbl_surf = self.fonts['badge'].render(lbl, True, bar_color)
                self.screen.blit(lbl_surf, (bx + bar_w // 2 - lbl_surf.get_width() // 2, chart_base + 3))

                if is_legal and prob > 0.02:
                    pct_surf = self.fonts['small'].render(f"{int(prob*100)}%", True, COLOR_TEXT_SECONDARY)
                    self.screen.blit(pct_surf, (bx + bar_w // 2 - pct_surf.get_width() // 2, chart_base - bar_h - 15))

        elif self.brain_mode == "HDC_VSA":
            self.screen.blit(self.fonts['header'].render("HDC-VSA MUSHROOM BODY", True, (139, 92, 246)), (660, 32))
            sub_title = "VSA Role-Filler Binding (⊗) & Bundling (+) | D=2,048"
            self.screen.blit(self.fonts['small'].render(sub_title, True, (196, 181, 253)), (660, 58))

            # 1. Hyperdimensional Basis & Scene Hypervector
            scene_box = pygame.Rect(655, 76, 290, 86)
            pygame.draw.rect(self.screen, (15, 23, 42), scene_box, border_radius=6)
            self.screen.blit(self.fonts['small'].render("SCENE HYPERVECTOR BUNDLING (D=2,048)", True, (139, 92, 246)), (660, 78))
            self.screen.blit(self.fonts['badge'].render("Quasi-Orthogonal Basis Vectors (|cos| < 0.12)", True, (167, 139, 250)), (660, 98))
            self.screen.blit(self.fonts['small'].render("Binding: Direction ⊗ Entity (Translation Invariant)", True, COLOR_TEXT_SECONDARY), (660, 120))
            self.screen.blit(self.fonts['small'].render("Bundling: Food + Obstacles + Pocket Qualifiers", True, COLOR_TEXT_MUTED), (660, 138))

            # 2. APL Lateral Inhibition k-WTA (100 active)
            apl_box = pygame.Rect(655, 168, 290, 80)
            pygame.draw.rect(self.screen, (15, 23, 42), apl_box, border_radius=6)
            self.screen.blit(self.fonts['small'].render("APL LATERAL INHIBITION (k-WTA SPARSITY)", True, (52, 211, 153)), (660, 170))
            self.screen.blit(self.fonts['badge'].render("k = 100 Active Kenyon Cells (4.88% Sparsity)", True, (52, 211, 153)), (660, 192))
            self.screen.blit(self.fonts['small'].render("Replaces dense PN-KC Synapses with Vector Algebra", True, COLOR_TEXT_MUTED), (660, 216))

            # 3. MBON Clean-up Memory Prototypes
            mbon_y = 258
            self.screen.blit(self.fonts['title'].render("MBON ASSOCIATIVE CLEAN-UP", True, (243, 244, 246)), (660, mbon_y))
            mbon_box = pygame.Rect(655, mbon_y + 22, 290, 120)
            pygame.draw.rect(self.screen, (15, 23, 42), mbon_box, border_radius=8)

            bar_w = 48
            gap = (mbon_box.width - (bar_w * 4)) // 5
            chart_base = mbon_box.bottom - 22

            for a_idx in range(4):
                bx = mbon_box.left + gap + a_idx * (bar_w + gap)
                prob = self.mbon_probs[a_idx] if self.mbon_probs is not None else 0.0
                is_legal = self.legal_mask[a_idx] if self.legal_mask is not None else True
                is_chosen = (self.selected_action == a_idx)

                bar_h = int(prob * 75)
                if not is_legal:
                    bar_color = (75, 85, 99)
                    bar_h = 3
                elif is_chosen:
                    bar_color = (139, 92, 246)
                else:
                    bar_color = (6, 182, 212)

                if bar_h > 0:
                    pygame.draw.rect(self.screen, bar_color, (bx, chart_base - bar_h, bar_w, bar_h), border_radius=4)

                lbl = DIRECTION_NAMES[a_idx][:2]
                lbl_surf = self.fonts['badge'].render(lbl, True, bar_color)
                self.screen.blit(lbl_surf, (bx + bar_w // 2 - lbl_surf.get_width() // 2, chart_base + 3))

                if is_legal and prob > 0.02:
                    pct_surf = self.fonts['small'].render(f"{int(prob*100)}%", True, COLOR_TEXT_SECONDARY)
                    self.screen.blit(pct_surf, (bx + bar_w // 2 - pct_surf.get_width() // 2, chart_base - bar_h - 15))

        elif self.brain_mode == "STACKED":
            self.screen.blit(self.fonts['header'].render("STACKED DEEP MB", True, (243, 244, 246)), (660, 32))
            act1 = int(np.sum(self.kc1_sparse > 0)) if self.kc1_sparse is not None else 0
            act2 = int(np.sum(self.kc2_sparse > 0)) if self.kc2_sparse is not None else 0
            sub_title = f"L1: 1,200 KC1 ({act1}) -> 12 Concepts -> L2: 800 KC2 ({act2})"
            self.screen.blit(self.fonts['small'].render(sub_title, True, (147, 197, 253)), (660, 58))

            # 1. Layer 1 KC Matrix (1,200 dots)
            kc1_box = pygame.Rect(655, 76, 290, 96)
            pygame.draw.rect(self.screen, (15, 23, 42), kc1_box, border_radius=6)
            self.screen.blit(self.fonts['small'].render("LAYER 1: PERCEPTUAL KCs (1,200)", True, (52, 211, 153)), (660, 78))
            if self.kc1_sparse is not None:
                for i, pt in enumerate(self.kc1_points):
                    if self.kc1_sparse[i] > 0:
                        pygame.draw.circle(self.screen, (52, 211, 153), pt, 2)
                    else:
                        pygame.draw.circle(self.screen, (25, 35, 48), pt, 1)
            else:
                for pt in self.kc1_points:
                    pygame.draw.circle(self.screen, (25, 35, 48), pt, 1)

            # 2. Intermediate Situational Concepts (12 Concepts)
            conc_box = pygame.Rect(655, 178, 290, 80)
            pygame.draw.rect(self.screen, (15, 23, 42), conc_box, border_radius=6)
            self.screen.blit(self.fonts['small'].render("INTERMEDIATE SITUATIONAL CONCEPTS (12)", True, (250, 204, 21)), (660, 180))
            
            c_names = StackedVisualMushroomBody.CONCEPT_NAMES
            c_vals = self.concepts if self.concepts is not None else np.zeros(12)
            for c_i in range(12):
                col = c_i // 6
                row = c_i % 6
                cx = 660 + col * 142
                cy = 195 + row * 10
                m_val = float(np.clip(c_vals[c_i], 0.0, 1.0))
                pygame.draw.rect(self.screen, (30, 41, 59), (cx + 65, cy + 2, 65, 6), border_radius=2)
                if m_val > 0.05:
                    bar_c = (250, 204, 21) if m_val > 0.6 else (147, 197, 253)
                    pygame.draw.rect(self.screen, bar_c, (cx + 65, cy + 2, int(65 * m_val), 6), border_radius=2)
                name_lbl = self.fonts['small'].render(c_names[c_i][:8], True, COLOR_TEXT_MUTED)
                self.screen.blit(name_lbl, (cx, cy))

            # 3. Layer 2 KC Matrix (800 dots)
            kc2_box = pygame.Rect(655, 264, 290, 78)
            pygame.draw.rect(self.screen, (15, 23, 42), kc2_box, border_radius=6)
            self.screen.blit(self.fonts['small'].render("LAYER 2: STRATEGIC KCs (800)", True, (6, 182, 212)), (660, 266))
            if self.kc2_sparse is not None:
                for i, pt in enumerate(self.kc2_points):
                    if self.kc2_sparse[i] > 0:
                        pygame.draw.circle(self.screen, (6, 182, 212), pt, 2)
                    else:
                        pygame.draw.circle(self.screen, (25, 35, 48), pt, 1)
            else:
                for pt in self.kc2_points:
                    pygame.draw.circle(self.screen, (25, 35, 48), pt, 1)

            # 4. MBON Actions (4 Direction bars)
            mbon_y = 350
            self.screen.blit(self.fonts['title'].render("EXECUTIVE MOTOR ACTIONS", True, (243, 244, 246)), (660, mbon_y))
            mbon_box = pygame.Rect(655, mbon_y + 22, 290, 136)
            pygame.draw.rect(self.screen, (15, 23, 42), mbon_box, border_radius=8)

            bar_w = 48
            gap = (mbon_box.width - (bar_w * 4)) // 5
            chart_base = mbon_box.bottom - 22

            for a_idx in range(4):
                bx = mbon_box.left + gap + a_idx * (bar_w + gap)
                prob = self.mbon_probs[a_idx] if self.mbon_probs is not None else 0.0
                is_legal = self.legal_mask[a_idx] if self.legal_mask is not None else True
                is_chosen = (self.selected_action == a_idx)

                bar_h = int(prob * 85)
                if not is_legal:
                    bar_color = (75, 85, 99)
                    bar_h = 3
                elif is_chosen:
                    bar_color = (250, 204, 21)
                else:
                    bar_color = (139, 92, 246)

                if bar_h > 0:
                    pygame.draw.rect(self.screen, bar_color, (bx, chart_base - bar_h, bar_w, bar_h), border_radius=4)

                lbl = DIRECTION_NAMES[a_idx][:2]
                lbl_surf = self.fonts['badge'].render(lbl, True, bar_color)
                self.screen.blit(lbl_surf, (bx + bar_w // 2 - lbl_surf.get_width() // 2, chart_base + 3))

                if is_legal and prob > 0.02:
                    pct_surf = self.fonts['small'].render(f"{int(prob*100)}%", True, COLOR_TEXT_SECONDARY)
                    self.screen.blit(pct_surf, (bx + bar_w // 2 - pct_surf.get_width() // 2, chart_base - bar_h - 15))
        else:
            # Single-Layer Mode (Original Display)
            self.screen.blit(self.fonts['header'].render("MUSHROOM BODY (SINGLE)", True, (243, 244, 246)), (660, 35))
            active_kc_cnt = int(np.sum(self.sparse_kc == 1.0)) if self.sparse_kc is not None else 0
            kc_sub = f"KC: 2,000 | Active: {active_kc_cnt} (5.0% Sparsity)"
            self.screen.blit(self.fonts['sub'].render(kc_sub, True, (52, 211, 153)), (660, 65))

            # วาดเมทริกซ์ KC 2,000 จุด
            kc_box = pygame.Rect(655, 90, 290, 210)
            pygame.draw.rect(self.screen, (15, 23, 42), kc_box, border_radius=8)

            if self.sparse_kc is not None:
                for i, pt in enumerate(self.kc_points):
                    if self.sparse_kc[i] > 0:
                        pygame.draw.circle(self.screen, (52, 211, 153), pt, 2)
                    else:
                        pygame.draw.circle(self.screen, (30, 41, 59), pt, 1)
            else:
                for pt in self.kc_points:
                    pygame.draw.circle(self.screen, (30, 41, 59), pt, 1)

            # MBON Direction Action Distribution (4 bars: UP, RIGHT, DOWN, LEFT)
            mbon_y = 315
            self.screen.blit(self.fonts['title'].render("MBON DIRECTION SELECTION", True, (243, 244, 246)), (660, mbon_y))
            mbon_box = pygame.Rect(655, mbon_y + 25, 290, 160)
            pygame.draw.rect(self.screen, (15, 23, 42), mbon_box, border_radius=8)

            bar_w = 48
            gap = (mbon_box.width - (bar_w * 4)) // 5
            chart_base = mbon_box.bottom - 25

            for a_idx in range(4):
                bx = mbon_box.left + gap + a_idx * (bar_w + gap)
                prob = self.mbon_probs[a_idx] if self.mbon_probs is not None else 0.0
                is_legal = self.legal_mask[a_idx] if self.legal_mask is not None else True
                is_chosen = (self.selected_action == a_idx)

                bar_h = int(prob * 100)
                if not is_legal:
                    bar_color = (75, 85, 99)
                    bar_h = 3
                elif is_chosen:
                    bar_color = (250, 204, 21)
                else:
                    bar_color = (6, 182, 212)

                if bar_h > 0:
                    pygame.draw.rect(self.screen, bar_color, (bx, chart_base - bar_h, bar_w, bar_h), border_radius=4)

                lbl = DIRECTION_NAMES[a_idx][:2]
                lbl_surf = self.fonts['badge'].render(lbl, True, bar_color)
                self.screen.blit(lbl_surf, (bx + bar_w // 2 - lbl_surf.get_width() // 2, chart_base + 4))

                if is_legal and prob > 0.02:
                    pct_surf = self.fonts['small'].render(f"{int(prob*100)}%", True, COLOR_TEXT_SECONDARY)
                    self.screen.blit(pct_surf, (bx + bar_w // 2 - pct_surf.get_width() // 2, chart_base - bar_h - 16))

        # Dopamine Indicator Box
        dopa_box = pygame.Rect(655, 515, 290, 170)
        pygame.draw.rect(self.screen, (15, 23, 42), dopa_box, border_radius=8)
        self.screen.blit(self.fonts['sub'].render("DOPAMINE PLASTICITY SIGNAL", True, COLOR_TEXT_SECONDARY), (670, 525))

        if self.dopamine_alpha > 0:
            self.dopamine_alpha = max(0, self.dopamine_alpha - 10)

        if self.last_dopamine > 0.5:
            dopa_text = f"DOPAMINE BURST: +{self.last_dopamine:.1f} (ATE FOOD!)"
            dopa_color = (52, 211, 153)
        elif self.last_dopamine < -0.5:
            dopa_text = f"DOPAMINE DIP: {self.last_dopamine:.1f} (COLLISION!)"
            dopa_color = (239, 68, 68)
        elif self.last_dopamine > 0.0:
            dopa_text = f"SHAPING: +{self.last_dopamine:.2f} (APPROACH)"
            dopa_color = (6, 182, 212)
        else:
            dopa_text = "STANDBY / STEPPING"
            dopa_color = COLOR_TEXT_MUTED

        self.screen.blit(self.fonts['title'].render(dopa_text, True, dopa_color), (670, 555))
        rule_desc = "ΔW = η · E_t · Dopamine (Live Three-Factor Update)"
        self.screen.blit(self.fonts['small'].render(rule_desc, True, COLOR_TEXT_MUTED), (670, 590))
        if self.brain_mode == "HIPPOCAMPUS":
            mean_w = float(np.mean(self.mb_hippo.mbon_prototypes))
            cur_temp = self.mb_hippo.temperature
        elif self.brain_mode == "HDC_VSA":
            mean_w = float(np.mean(self.mb_hdc.mbon_prototypes))
            cur_temp = self.mb_hdc.temperature
        elif self.brain_mode == "STACKED":
            mean_w = float(np.mean(self.mb_stacked.w_kc2_mbon2))
            cur_temp = self.mb_stacked.temperature
        else:
            mean_w = float(np.mean(self.mb_single.w_kc_mbon))
            cur_temp = self.mb_single.temperature
        self.screen.blit(self.fonts['small'].render(f"Mode: {self.brain_mode} | Mean W: {mean_w:.4f} | Temp: {cur_temp:.3f}", True, (226, 232, 240)), (670, 618))


        # 4. แผงขวา: Controls & Stats
        ctrl_panel = pygame.Rect(965, 20, 295, 680)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, ctrl_panel, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, ctrl_panel, width=1, border_radius=12)

        self.screen.blit(self.fonts['header'].render("CONTROLS & STATS", True, (243, 244, 246)), (980, 35))
        self.screen.blit(self.fonts['sub'].render("Visual MB Learning Loop", True, COLOR_TEXT_SECONDARY), (980, 65))

        # Scoreboard
        stats_box = pygame.Rect(980, 350, 265, 170)
        pygame.draw.rect(self.screen, (15, 23, 42), stats_box, border_radius=8)

        self.screen.blit(self.fonts['sub'].render("CAREER STATISTICS:", True, COLOR_TEXT_SECONDARY), (995, 365))
        self.screen.blit(self.fonts['badge'].render(f"High Score : {self.high_score} Apples", True, (250, 204, 21)), (995, 395))
        self.screen.blit(self.fonts['sub'].render(f"Games Played : {self.total_games}", True, COLOR_TEXT_PRIMARY), (995, 425))
        self.screen.blit(self.fonts['sub'].render(f"Total Apples : {self.total_apples}", True, (52, 211, 153)), (995, 455))
        self.screen.blit(self.fonts['sub'].render(f"Trained Episodes : {self.total_trained_episodes:,}", True, (6, 182, 212)), (995, 485))

        # Toast notification
        now = time.time()
        if self.training_toast and now < self.toast_timer:
            toast_rect = pygame.Rect(980, 535, 265, 45)
            pygame.draw.rect(self.screen, (16, 185, 129), toast_rect, border_radius=6)
            self.screen.blit(self.fonts['small'].render(self.training_toast, True, (255, 255, 255)), (990, 548))
        else:
            if 'train_gpu' in self.buttons and self.buttons['train_gpu'].active:
                self.buttons['train_gpu'].text = "TRAIN ON GPU (+2,000 EP)"
                self.buttons['train_gpu'].active = False

        # Hardware acceleration badge
        hw_label = f"ACCEL: {self.device_name}"
        self.screen.blit(self.fonts['small'].render(hw_label, True, (148, 163, 184)), (985, 590))

        # วาดปุ่ม
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
