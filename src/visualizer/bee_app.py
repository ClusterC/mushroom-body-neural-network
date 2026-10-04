import os
import sys
import math
import pygame
import numpy as np
from typing import Optional, List, Tuple

from src.envs.bee_foraging_env import BeeForagingEnv, Flower
from src.models.bee_mushroom_body import BeeMushroomBody
from src.visualizer.components import (
    UIButton,
    COLOR_BG,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED
)

# ชุดสีเฉพาะสำหรับระบบจำลองผึ้งน้ำหวาน
COLOR_MEADOW_BG = (13, 31, 20)           # Deep Turf Green
COLOR_MEADOW_BORDER = (30, 64, 45)
COLOR_HIVE_GOLD = (245, 158, 11)          # Amber Honey Gold
COLOR_HIVE_FILL = (120, 53, 15)
COLOR_BEE_YELLOW = (251, 191, 36)
COLOR_BEE_STRIPE = (24, 24, 27)
COLOR_WING_WHITE = (240, 249, 255, 180)

FLOWER_SPECIES_COLORS = [
    (168, 85, 247),   # Species 0: Lavender (Purple)
    (250, 204, 21),   # Species 1: Chamomile (Yellow)
    (244, 63, 94),    # Species 2: Wild Rose (Pink/Rose)
    (6, 182, 212)     # Species 3: Toxic Blue (Cyan)
]

FLOWER_SPECIES_NAMES = ["Lavender", "Chamomile", "Wild Rose", "Toxic Blue"]

class BeeVisualizerApp:
    """
    แอปพลิเคชัน Desktop แสดงผลกราฟิกจำลองพฤติกรรมผึ้งน้ำหวาน (Apis mellifera)
    ด้วยวงจรประสาท Mushroom Body แบบมัลติโมดัล (กลิ่น + สเปกตรัมแสงยูวี + สารสื่อประสาท Octopamine)
    """
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.width = 1280
        self.height = 800

        if not self.headless:
            pygame.init()
            pygame.font.init()
            pygame.display.set_caption("Bio-Inspired Honeybee Foraging Simulator (Apis mellifera Mushroom Body)")
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.fonts = {
                'title': pygame.font.SysFont("Segoe UI, Arial, sans-serif", 15, bold=True),
                'header': pygame.font.SysFont("Segoe UI, Arial, sans-serif", 18, bold=True),
                'body': pygame.font.SysFont("Segoe UI, Arial, sans-serif", 13),
                'small': pygame.font.SysFont("Segoe UI, Arial, sans-serif", 11),
                'badge': pygame.font.SysFont("Segoe UI, Arial, sans-serif", 12, bold=True)
            }
        else:
            self.screen = None
            self.clock = None
            self.fonts = {}

        # 1. สร้างสภาพแวดล้อมและโมเดล Mushroom Body
        self.env = BeeForagingEnv(meadow_size=20.0, max_steps=600, seed=42)
        self.mb = BeeMushroomBody(n_pn=36, n_kc=2500, n_mbon=5, sparsity_ratio=0.05, seed=42)
        
        # สถานะการจำลอง
        self.obs = self.env.reset()
        self.running = True
        self.auto_fly = True
        self.enable_plasticity = True
        self.speed_mode = 1  # 1x, 2x, 5x, 10x
        self.step_delay = 60  # ms
        self.last_step_time = 0
        self.speed_options = [1, 2, 5, 10]
        self.speed_idx = 0

        # ข้อมูลการตอบสนองล่าสุด
        self.last_action = 0
        self.last_reward = 0.0
        self.last_info = {"event": "INIT"}
        self.octopamine_alpha = 0
        self.dopamine_alpha = 0
        
        # ข้อความแจ้งเตือน Toast Feedback
        self.toast_msg = ""
        self.toast_timer = 0
        self.toast_color = (52, 211, 153)

        # 2. พิกัดจุด KC 2,500 จุด (50 cols x 50 rows)
        self.kc_points = []
        self._init_kc_layout()

        # 3. สร้างปุ่มควบคุม UI
        self._init_ui_buttons()

        # ฝึกฝนเบื้องต้นสั้นๆ เพื่อให้มีค่าน้ำหนักพื้นฐาน (ข้ามหากเป็น headless)
        if not self.headless:
            self._pretrain_quick(n_steps=200)

    def _init_kc_layout(self):
        """จัดวางพิกัดจุด KC 2,500 จุด บนผืนผ้าใบแผงวงจรประสาท"""
        kc_box_x = 688 + 15
        kc_box_y = 216 + 55
        kc_w = 568 - 30
        kc_h = 130
        cols = 50
        rows = 50
        dx = kc_w / cols
        dy = kc_h / rows

        for r in range(rows):
            for c in range(cols):
                px = int(kc_box_x + c * dx + dx / 2)
                py = int(kc_box_y + r * dy + dy / 2)
                self.kc_points.append((px, py))

    def _init_ui_buttons(self):
        """สร้างปุ่มอินเทอร์แอคทีฟด้านล่างขวา"""
        if self.headless:
            self.buttons = {}
            return

        btn_font = self.fonts['body']
        btn_y1 = 558 + 105
        btn_y2 = 558 + 155

        self.btn_step = UIButton((688 + 20, btn_y1, 85, 36), "STEP", btn_font)
        self.btn_autofly = UIButton((688 + 115, btn_y1, 120, 36), "AUTO-FLY: ON", btn_font, active=True)
        self.btn_speed = UIButton((688 + 245, btn_y1, 95, 36), "SPEED: 1x", btn_font)
        self.btn_plasticity = UIButton((688 + 350, btn_y1, 120, 36), "PLASTICITY: ON", btn_font, active=True)
        self.btn_reset = UIButton((688 + 480, btn_y1, 80, 36), "RESET", btn_font)

        # ปุ่มใหญ่สำหรับฝึกฝนแบบก้าวกระโดด
        self.btn_train = UIButton((688 + 20, btn_y2, 540, 42), "TRAIN FORAGING (+500 TRIPS)", self.fonts['header'],
                                  base_color=(30, 58, 138), hover_color=(37, 99, 235), active_color=(59, 130, 246))

        self.buttons = {
            'step': self.btn_step,
            'autofly': self.btn_autofly,
            'speed': self.btn_speed,
            'plasticity': self.btn_plasticity,
            'reset': self.btn_reset,
            'train': self.btn_train
        }

    def _pretrain_quick(self, n_steps: int = 200):
        """การฝึกเริ่มต้นสั้นๆ เพื่อสร้าง Synaptic Base"""
        for _ in range(n_steps):
            action, probs, _ = self.mb.forward(self.obs)
            next_obs, reward, done, info = self.env.step(action)
            self.mb.update_plasticity(reward)
            if done:
                self.obs = self.env.reset()
                self.mb.reset_traces()
            else:
                self.obs = next_obs

    def show_toast(self, message: str, color=(52, 211, 153), duration_frames: int = 150):
        """แสดงกล่องข้อความเรืองแสงแจ้งเตือน"""
        self.toast_msg = message
        self.toast_color = color
        self.toast_timer = duration_frames

    def step_simulation(self):
        """ก้าวเดินจำลอง 1 ก้าวของผึ้ง"""
        # Forward ผ่าน Mushroom Body
        action, probs, kc_act = self.mb.forward(self.obs)
        self.last_action = action

        # ก้าวในสิ่งแวดล้อม
        next_obs, reward, done, info = self.env.step(action)
        self.last_reward = reward
        self.last_info = info

        # อัปเดต Plasticity
        if self.enable_plasticity:
            self.mb.update_plasticity(reward)

        # กระตุ้นเอฟเฟกต์สารสื่อประสาท
        if reward > 0.3:
            self.octopamine_alpha = 255
            self.dopamine_alpha = 0
        elif reward < -0.4:
            self.dopamine_alpha = 255
            self.octopamine_alpha = 0

        if done:
            self.obs = self.env.reset()
            self.mb.reset_traces()
        else:
            self.obs = next_obs

    def train_episodes(self, n_trips: int = 500):
        """ฝึกฝนพฤติกรรมหาอาหารและจำแนกดอกไม้แบบรวดเร็ว"""
        trips = 0
        step_total = 0
        initial_honey = self.env.total_hive_nectar
        
        while trips < n_trips and step_total < 30000:
            step_total += 1
            action, probs, _ = self.mb.forward(self.obs)
            next_obs, reward, done, info = self.env.step(action)
            self.mb.update_plasticity(reward)

            if "UNLOADED_NECTAR" in info.get("event", ""):
                trips += 1

            if done:
                self.obs = self.env.reset()
                self.mb.reset_traces()
            else:
                self.obs = next_obs

        harvested = self.env.total_hive_nectar - initial_honey
        self.show_toast(f"TRAINED +{trips} TRIPS! HARVESTED +{harvested:.1f} NECTAR (SYNAPSE OPTIMIZED)",
                        color=(250, 204, 21), duration_frames=180)

    def handle_events(self):
        """จัดการเหตุการณ์คลิกและคีย์บอร์ด"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return
                elif event.key == pygame.K_SPACE:
                    self.step_simulation()
                elif event.key == pygame.K_a:
                    self.auto_fly = not self.auto_fly
                    self.btn_autofly.active = self.auto_fly
                    self.btn_autofly.text = "AUTO-FLY: ON" if self.auto_fly else "AUTO-FLY: OFF"

            # จัดการคลิกปุ่ม UI
            for name, btn in self.buttons.items():
                if btn.handle_event(event):
                    if name == 'step':
                        self.step_simulation()
                    elif name == 'autofly':
                        self.auto_fly = not self.auto_fly
                        btn.active = self.auto_fly
                        btn.text = "AUTO-FLY: ON" if self.auto_fly else "AUTO-FLY: OFF"
                    elif name == 'speed':
                        self.speed_idx = (self.speed_idx + 1) % len(self.speed_options)
                        self.speed_mode = self.speed_options[self.speed_idx]
                        btn.text = f"SPEED: {self.speed_mode}x"
                    elif name == 'plasticity':
                        self.enable_plasticity = not self.enable_plasticity
                        btn.active = self.enable_plasticity
                        btn.text = "PLASTICITY: ON" if self.enable_plasticity else "PLASTICITY: OFF"
                    elif name == 'reset':
                        self.obs = self.env.reset()
                        self.mb.reset_traces()
                        self.show_toast("FLIGHT SIMULATION RESET", color=(147, 197, 253))
                    elif name == 'train':
                        self.train_episodes(n_trips=500)

    def draw_meadow_arena(self):
        """วาดทุ่งหญ้า ดอกไม้ รังผึ้ง และตัวผึ้งพร้อมลำแสงสายตา"""
        arena_x = 24
        arena_y = 24
        arena_w = 640
        arena_h = 640
        arena_rect = pygame.Rect(arena_x, arena_y, arena_w, arena_h)

        # พื้นหลังทุ่งหญ้า
        pygame.draw.rect(self.screen, COLOR_MEADOW_BG, arena_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_MEADOW_BORDER, arena_rect, width=2, border_radius=12)

        # วาดตารางกริดจางๆ
        grid_step = arena_w / 20.0
        for i in range(1, 20):
            gx = int(arena_x + i * grid_step)
            gy = int(arena_y + i * grid_step)
            pygame.draw.line(self.screen, (20, 45, 30), (gx, arena_y), (gx, arena_y + arena_h), 1)
            pygame.draw.line(self.screen, (20, 45, 30), (arena_x, gy), (arena_x + arena_w, gy), 1)

        # ฟังก์ชันแปลงพิกัดทุ่งหญ้า (0..20) สู่หน้าจอ
        def to_screen(px, py):
            sx = int(arena_x + (px / 20.0) * arena_w)
            sy = int(arena_y + ((20.0 - py) / 20.0) * arena_h)  # กลับแกน Y ให้ 0 อยู่ล่าง
            return sx, sy

        # 1. วาดเส้นทางบินล่าสุด (Fading Flight Trail)
        if len(self.env.recent_positions) > 1:
            points = [to_screen(p[0], p[1]) for p in self.env.recent_positions]
            for i in range(len(points) - 1):
                alpha_factor = (i + 1) / len(points)
                c_val = int(120 * alpha_factor)
                pygame.draw.line(self.screen, (c_val, c_val + 30, c_val), points[i], points[i + 1], 2)

        # 2. วาดดอกไม้ทั้ง 24 ต้น
        for f in self.env.flowers:
            fx, fy = to_screen(f.x, f.y)
            
            if f.is_depleted:
                # ดอกไม้ที่แห้งเหือดน้ำหวานหมดแล้ว: กลีบเหี่ยวเฉาเป็นสีเทาหม่น ไม่มีรัศมีกลิ่น
                for petal_i in range(5):
                    ang = petal_i * (2 * math.pi / 5)
                    p_x = int(fx + 4 * math.cos(ang))
                    p_y = int(fy + 4 * math.sin(ang))
                    pygame.draw.circle(self.screen, (55, 65, 81), (p_x, p_y), 3)
                pygame.draw.circle(self.screen, (31, 41, 55), (fx, fy), 2)
            else:
                color = FLOWER_SPECIES_COLORS[f.species]
                
                # วาดรัศมีกลิ่นฟุ้งจางๆ (Odor Halo) เฉพาะดอกไม้ที่มีน้ำหวาน
                if f.current_nectar > 0.05 or f.species == 3:
                    halo_surf = pygame.Surface((36, 36), pygame.SRCALPHA)
                    halo_color = (*color[:3], 35)
                    pygame.draw.circle(halo_surf, halo_color, (18, 18), 16)
                    self.screen.blit(halo_surf, (fx - 18, fy - 18))

                # วาดกลีบดอกไม้ 5 กลีบ ขนาดแปรผันตามน้ำหวานที่เหลือ
                nectar_ratio = f.current_nectar / max(f.total_reserve, 0.1) if f.species != 3 else 1.0
                radius = int(5 + 3 * nectar_ratio)
                for petal_i in range(5):
                    ang = petal_i * (2 * math.pi / 5)
                    p_x = int(fx + 6 * math.cos(ang))
                    p_y = int(fy + 6 * math.sin(ang))
                    pygame.draw.circle(self.screen, color, (p_x, p_y), radius)

                # เกสรกลางดอกไม้
                center_color = (255, 255, 255) if f.species != 3 else (239, 68, 68)
                pygame.draw.circle(self.screen, center_color, (fx, fy), 3)


        # 3. วาดรังผึ้งกึ่งกลาง (Hive)
        hx, hy = to_screen(self.env.hive_pos[0], self.env.hive_pos[1])
        # วาดออร่าสีทองรอบรังผึ้ง
        hive_aura = pygame.Surface((60, 60), pygame.SRCALPHA)
        pygame.draw.circle(hive_aura, (245, 158, 11, 45), (30, 30), 28)
        self.screen.blit(hive_aura, (hx - 30, hy - 30))
        
        # รูปรังผึ้งหกเหลี่ยม
        hex_points = []
        for i in range(6):
            ang = i * (2 * math.pi / 6)
            hex_points.append((hx + int(18 * math.cos(ang)), hy + int(18 * math.sin(ang))))
        pygame.draw.polygon(self.screen, COLOR_HIVE_FILL, hex_points)
        pygame.draw.polygon(self.screen, COLOR_HIVE_GOLD, hex_points, 2)
        
        # ป้ายข้อความรังผึ้ง
        hive_txt = self.fonts['badge'].render("HIVE", True, COLOR_HIVE_GOLD)
        self.screen.blit(hive_txt, (hx - hive_txt.get_width() // 2, hy - 7))

        # 4. วาดตัวผึ้ง (The Honeybee)
        bx, by = to_screen(self.env.bee_pos[0], self.env.bee_pos[1])
        rad_head = math.radians(self.env.bee_heading)

        # วาดกรวยสายตาของผึ้ง (Visual Field Cone)
        cone_len = 45
        left_ang = rad_head + math.radians(35)
        right_ang = rad_head - math.radians(35)
        p_left = (bx + int(cone_len * math.cos(left_ang)), by - int(cone_len * math.sin(left_ang)))
        p_right = (bx + int(cone_len * math.cos(right_ang)), by - int(cone_len * math.sin(right_ang)))
        cone_surf = pygame.Surface((arena_w, arena_h), pygame.SRCALPHA)
        cone_poly = [(bx - arena_x, by - arena_y), (p_left[0] - arena_x, p_left[1] - arena_y), (p_right[0] - arena_x, p_right[1] - arena_y)]
        pygame.draw.polygon(cone_surf, (254, 240, 138, 25), cone_poly)
        self.screen.blit(cone_surf, (arena_x, arena_y))

        # วาดปีกผึ้งกระพือ (Flapping Wings)
        wing_flutter = math.sin(pygame.time.get_ticks() * 0.04) * 6
        wing_l = (bx - int(7 * math.sin(rad_head)) + int(wing_flutter), by - int(7 * math.cos(rad_head)))
        wing_r = (bx + int(7 * math.sin(rad_head)) - int(wing_flutter), by + int(7 * math.cos(rad_head)))
        pygame.draw.circle(self.screen, (224, 242, 254), wing_l, 6)
        pygame.draw.circle(self.screen, (224, 242, 254), wing_r, 6)

        # วาดลำตัวผึ้งลายทางเหลือง-ดำ
        pygame.draw.circle(self.screen, COLOR_BEE_YELLOW, (bx, by), 8)
        pygame.draw.circle(self.screen, COLOR_BEE_STRIPE, (bx, by), 5)
        head_x = bx + int(7 * math.cos(rad_head))
        head_y = by - int(7 * math.sin(rad_head))
        pygame.draw.circle(self.screen, (15, 23, 42), (head_x, head_y), 4)

        # หัวข้อแผงสนาม
        arena_title = self.fonts['header'].render("MEADOW ARENA (20x20 GRID | FORAGING GROUNDS)", True, (241, 245, 249))
        self.screen.blit(arena_title, (arena_x + 15, arena_y + 15))

    def draw_multisensory_panel(self):
        """วาดจอแสดงผลสัญญาณพหุสัมผัส (กลิ่น 4 ชนิด + ตาประกอบ 3 ทิศทาง + สถานะภายใน)"""
        panel_x = 688
        panel_y = 24
        panel_w = 568
        panel_h = 180
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, panel_rect, width=1, border_radius=12)

        # หัวข้อแผง
        title_surf = self.fonts['header'].render("MULTISENSORY OPTIC & OLFACTORY LOBES (PN: 36)", True, (226, 232, 240))
        self.screen.blit(title_surf, (panel_x + 15, panel_y + 12))

        # 1. Olfactory Spectrum (กลิ่น 4 ชนิด)
        sub_odor = self.fonts['title'].render("ANTENNAL LOBE (ODOR SPECTRUM)", True, COLOR_TEXT_SECONDARY)
        self.screen.blit(sub_odor, (panel_x + 15, panel_y + 36))

        odor_vals = self.obs[0:4]
        for i in range(4):
            bar_y = panel_y + 60 + i * 22
            # ชื่อดอกไม้
            name_txt = self.fonts['small'].render(FLOWER_SPECIES_NAMES[i][:8], True, FLOWER_SPECIES_COLORS[i])
            self.screen.blit(name_txt, (panel_x + 15, bar_y))
            
            # แถบเกจวัดกลิ่น
            gauge_bg = pygame.Rect(panel_x + 85, bar_y + 2, 130, 12)
            pygame.draw.rect(self.screen, (30, 41, 59), gauge_bg, border_radius=4)
            fill_w = int(130 * np.clip(odor_vals[i], 0.0, 1.0))
            if fill_w > 0:
                fill_rect = pygame.Rect(panel_x + 85, bar_y + 2, fill_w, 12)
                pygame.draw.rect(self.screen, FLOWER_SPECIES_COLORS[i], fill_rect, border_radius=4)

        # 2. Visual Ommatidia (ตาประกอบ 3 ทิศทาง)
        sub_vis = self.fonts['title'].render("COMPOUND EYES (OMMATIDIA)", True, COLOR_TEXT_SECONDARY)
        self.screen.blit(sub_vis, (panel_x + 245, panel_y + 36))

        ommatidia_labels = ["LEFT (+35°)", "CENTER (0°)", "RIGHT (-35°)"]
        for idx in range(3):
            ox = panel_x + 270 + idx * 75
            oy = panel_y + 80
            # ดึงสีจาก observation (UV, Blue, Green, Lum)
            vis_ch = self.obs[8 + idx * 4 : 8 + (idx + 1) * 4]
            # แปลงเป็น RGB
            r_val = int(np.clip((vis_ch[0] * 0.8 + vis_ch[3] * 0.2) * 255, 20, 255))
            g_val = int(np.clip((vis_ch[2] * 0.8 + vis_ch[3] * 0.2) * 255, 20, 255))
            b_val = int(np.clip((vis_ch[1] * 0.8 + vis_ch[0] * 0.4) * 255, 20, 255))
            
            # วาดลูกตาดาวเทียม
            pygame.draw.circle(self.screen, (r_val, g_val, b_val), (ox, oy), 18)
            pygame.draw.circle(self.screen, (75, 85, 99), (ox, oy), 18, 2)
            
            lbl_surf = self.fonts['small'].render(ommatidia_labels[idx], True, COLOR_TEXT_MUTED)
            self.screen.blit(lbl_surf, (ox - lbl_surf.get_width() // 2, oy + 22))

        # 3. Internal State Badges (ถุงน้ำหวาน + พลังงาน)
        sub_state = self.fonts['title'].render("INTERNAL STATUS", True, COLOR_TEXT_SECONDARY)
        self.screen.blit(sub_state, (panel_x + 475, panel_y + 36))

        # Crop Nectar Gauge
        crop_txt = self.fonts['small'].render(f"Crop: {self.env.crop_nectar:.1f}/5.0", True, COLOR_HIVE_GOLD)
        self.screen.blit(crop_txt, (panel_x + 475, panel_y + 60))
        crop_bg = pygame.Rect(panel_x + 475, panel_y + 78, 80, 10)
        pygame.draw.rect(self.screen, (30, 41, 59), crop_bg, border_radius=4)
        crop_fill = int(80 * (self.env.crop_nectar / 5.0))
        if crop_fill > 0:
            pygame.draw.rect(self.screen, COLOR_HIVE_GOLD, pygame.Rect(panel_x + 475, panel_y + 78, crop_fill, 10), border_radius=4)

        # Energy Gauge
        energy_txt = self.fonts['small'].render(f"Energy: {self.env.energy:.0f}%", True, (52, 211, 153))
        self.screen.blit(energy_txt, (panel_x + 475, panel_y + 100))
        energy_bg = pygame.Rect(panel_x + 475, panel_y + 118, 80, 10)
        pygame.draw.rect(self.screen, (30, 41, 59), energy_bg, border_radius=4)
        energy_fill = int(80 * (self.env.energy / 100.0))
        if energy_fill > 0:
            pygame.draw.rect(self.screen, (52, 211, 153), pygame.Rect(panel_x + 475, panel_y + 118, energy_fill, 10), border_radius=4)

        # Home Vector Distance
        dist_h = float(np.linalg.norm(self.env.bee_pos - self.env.hive_pos))
        home_lbl = self.fonts['small'].render(f"Home Dist: {dist_h:.1f}m", True, COLOR_TEXT_SECONDARY)
        self.screen.blit(home_lbl, (panel_x + 475, panel_y + 140))

    def draw_mushroom_body_circuit(self):
        """วาดโครงข่าย Kenyon Cells 2,500 จุด แบ่ง 3 โซน และ MBON 5 ทิศทาง"""
        panel_x = 688
        panel_y = 216
        panel_w = 568
        panel_h = 330
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, panel_rect, width=1, border_radius=12)

        # 1. หัวข้อและสถิติ Sparsity
        diag = self.mb.get_diagnostics()
        active_kc = diag["active_kc"]
        title_surf = self.fonts['header'].render(
            f"MUSHROOM BODY (CALYX: 2,500 KCs | ACTIVE: {active_kc} [5.0% SPARSITY])",
            True, (226, 232, 240)
        )
        self.screen.blit(title_surf, (panel_x + 15, panel_y + 12))

        # คำอธิบายโซน
        zone_info = self.fonts['small'].render(
            f"Lip (Odor): {diag['lip_active']}/800  |  Collar (Visual): {diag['collar_active']}/1000  |  Basal (Multi): {diag['basal_active']}/700",
            True, COLOR_TEXT_SECONDARY
        )
        self.screen.blit(zone_info, (panel_x + 15, panel_y + 34))

        # กล่องพื้นหลังของ KC
        kc_box = pygame.Rect(panel_x + 15, panel_y + 55, panel_w - 30, 130)
        pygame.draw.rect(self.screen, (15, 23, 42), kc_box, border_radius=8)

        # วาดจุด KC 2,500 จุด
        kc_act = self.mb.last_kc_activation
        for i, pt in enumerate(self.kc_points):
            if kc_act[i] > 0:
                # Active Cell เรืองแสงสว่างจ้า
                pygame.draw.circle(self.screen, (255, 255, 255), pt, 3)
                pygame.draw.circle(self.screen, (52, 211, 153), pt, 2)
            else:
                # Inactive Cell แยกสีตามโซนจางๆ
                if i < 800:
                    c = (45, 30, 20)     # Lip (Amber tint)
                elif i < 1800:
                    c = (20, 35, 45)     # Collar (Cyan tint)
                else:
                    c = (25, 40, 25)     # Basal (Lime tint)
                pygame.draw.circle(self.screen, c, pt, 1)

        # 2. MBON Action Probabilities Bar Chart (5 พฤติกรรม)
        mbon_title = self.fonts['title'].render("MBON MOTOR DECISION POLICY", True, (226, 232, 240))
        self.screen.blit(mbon_title, (panel_x + 15, panel_y + 195))

        action_names = ["FORWARD", "TURN_LEFT", "TURN_RIGHT", "LAND_AND_FEED", "RETURN_HOME"]
        probs = self.mb.last_mbon_probs
        bar_w = 95
        spacing = (panel_w - 30 - (bar_w * 5)) // 4

        for a_idx in range(5):
            bx = panel_x + 15 + a_idx * (bar_w + spacing)
            by = panel_y + 225
            
            # ป้ายชื่อการกระทำ
            is_chosen = (a_idx == self.last_action)
            txt_color = (52, 211, 153) if is_chosen else COLOR_TEXT_SECONDARY
            act_txt = self.fonts['small'].render(action_names[a_idx][:10], True, txt_color)
            self.screen.blit(act_txt, (bx, by))

            # แถบความน่าจะเป็น
            prob_bg = pygame.Rect(bx, by + 18, bar_w, 14)
            pygame.draw.rect(self.screen, (30, 41, 59), prob_bg, border_radius=4)
            fill_len = int(bar_w * np.clip(probs[a_idx], 0.0, 1.0))
            if fill_len > 0:
                bar_color = (52, 211, 153) if is_chosen else (59, 130, 246)
                pygame.draw.rect(self.screen, bar_color, pygame.Rect(bx, by + 18, fill_len, 14), border_radius=4)

            # เปอร์เซ็นต์
            pct_txt = self.fonts['small'].render(f"{probs[a_idx]*100:.0f}%", True, COLOR_TEXT_MUTED)
            self.screen.blit(pct_txt, (bx + bar_w - pct_txt.get_width(), by + 35))

        # 3. แถบแสดงสารสื่อประสาท Octopamine / Dopamine
        badge_y = panel_y + 285
        # Octopamine Indicator (รางวัลหวาน)
        oct_color = (250, 204, 21) if self.octopamine_alpha > 0 else (100, 80, 20)
        oct_txt = self.fonts['badge'].render(f"OCTOPAMINE BURST: +{self.mb.last_octopamine:.2f}", True, oct_color)
        self.screen.blit(oct_txt, (panel_x + 20, badge_y))

        # Dopamine Indicator (การลงโทษ)
        dop_color = (244, 63, 94) if self.dopamine_alpha > 0 else (100, 30, 40)
        dop_txt = self.fonts['badge'].render(f"DOPAMINE DEPRESSION: -{self.mb.last_dopamine:.2f}", True, dop_color)
        self.screen.blit(dop_txt, (panel_x + 300, badge_y))

    def draw_telemetry_and_controls(self):
        """วาดข้อมูลสถิติการบิน และปุ่มควบคุมการจำลอง"""
        panel_x = 688
        panel_y = 558
        panel_w = 568
        panel_h = 218
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, COLOR_PANEL_BORDER, panel_rect, width=1, border_radius=12)

        # 1. สรุปสถิติการบินและการเก็บเกี่ยว
        title_telemetry = self.fonts['header'].render("FORAGING TELEMETRY & FLORAL CONSTANCY", True, (226, 232, 240))
        self.screen.blit(title_telemetry, (panel_x + 15, panel_y + 12))

        stat_str = (
            f"Hive Honey: {self.env.total_hive_nectar:.1f} units  |  "
            f"Trips Completed: {self.env.trips_completed}  |  "
            f"Step: {self.env.step_count}/{self.env.max_steps}"
        )
        stat_surf = self.fonts['body'].render(stat_str, True, COLOR_HIVE_GOLD)
        self.screen.blit(stat_surf, (panel_x + 15, panel_y + 36))

        # สถิติดอกไม้ในทุ่งหญ้า (Blooming vs Depleted)
        f_stats = self.env.get_flower_stats()
        flora_str = f"Meadow Flora: {f_stats['active']}/{f_stats['total']} Blooming ({f_stats['depleted']} Depleted)"
        flora_color = (251, 191, 36) if f_stats['depleted'] > 0 else (52, 211, 153)
        flora_surf = self.fonts['small'].render(flora_str, True, flora_color)
        self.screen.blit(flora_surf, (panel_x + 350, panel_y + 38))

        # สถิติการเลือกชนิดดอกไม้ (Floral Preference)
        visits = self.env.species_visit_count
        visit_str = f"Visits -> Lav: {visits[0]} | Cham: {visits[1]} | Rose: {visits[2]} | Toxic: {visits[3]}"
        visit_surf = self.fonts['small'].render(visit_str, True, COLOR_TEXT_SECONDARY)
        self.screen.blit(visit_surf, (panel_x + 15, panel_y + 60))

        # เหตุการณ์ล่าสุด
        ev_str = f"Last Event: {self.last_info.get('event', 'FLYING')} (Reward: {self.last_reward:+.2f})"
        ev_surf = self.fonts['small'].render(ev_str, True, (147, 197, 253))
        self.screen.blit(ev_surf, (panel_x + 15, panel_y + 80))

        # 2. วาดปุ่มควบคุม
        for btn in self.buttons.values():
            btn.draw(self.screen)

    def draw_toast_notification(self):
        """วาดข้อความเรืองแสงแจ้งเตือนตรงกลางจอด้านบน"""
        if self.toast_timer > 0:
            self.toast_timer -= 1
            alpha = min(255, self.toast_timer * 8)
            toast_surf = self.fonts['header'].render(self.toast_msg, True, self.toast_color)
            tw = toast_surf.get_width() + 40
            th = toast_surf.get_height() + 16
            tx = (self.width - tw) // 2
            ty = 16

            box_surf = pygame.Surface((tw, th), pygame.SRCALPHA)
            pygame.draw.rect(box_surf, (15, 23, 42, int(alpha * 0.9)), (0, 0, tw, th), border_radius=10)
            pygame.draw.rect(box_surf, (*self.toast_color[:3], int(alpha * 0.8)), (0, 0, tw, th), width=2, border_radius=10)
            toast_surf.set_alpha(alpha)
            box_surf.blit(toast_surf, (20, 8))
            self.screen.blit(box_surf, (tx, ty))

    def run(self):
        """ลูปหลักของการจำลอง"""
        while self.running:
            self.handle_events()

            # อัปเดตการทำงานอัตโนมัติ
            now = pygame.time.get_ticks()
            delay = self.step_delay // self.speed_mode
            if self.auto_fly and (now - self.last_step_time) >= delay:
                self.step_simulation()
                self.last_step_time = now

            # ค่อยๆ ลดแสงเอฟเฟกต์สารสื่อประสาท
            if self.octopamine_alpha > 0:
                self.octopamine_alpha = max(0, self.octopamine_alpha - 10)
            if self.dopamine_alpha > 0:
                self.dopamine_alpha = max(0, self.dopamine_alpha - 10)

            # เรนเดอร์ส่วนประกอบทั้งหมด
            self.screen.fill(COLOR_BG)
            self.draw_meadow_arena()
            self.draw_multisensory_panel()
            self.draw_mushroom_body_circuit()
            self.draw_telemetry_and_controls()
            self.draw_toast_notification()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
