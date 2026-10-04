import pygame
import numpy as np

# Palette Constants
COLOR_BG = (11, 15, 25)
COLOR_PANEL_BG = (17, 24, 39)
COLOR_PANEL_BORDER = (31, 41, 55)
COLOR_TEXT_PRIMARY = (243, 244, 246)
COLOR_TEXT_SECONDARY = (156, 163, 175)
COLOR_TEXT_MUTED = (107, 114, 128)

COLOR_X_PRIMARY = (244, 63, 94)      # Neon Coral
COLOR_X_GLOW = (251, 113, 133, 80)
COLOR_O_PRIMARY = (6, 182, 212)       # Neon Cyan
COLOR_O_GLOW = (34, 211, 238, 80)

COLOR_KC_INACTIVE = (30, 41, 59)
COLOR_KC_ACTIVE = (52, 211, 153)      # Bright Emerald
COLOR_KC_GLOW = (16, 185, 129)

COLOR_ACCENT_PURPLE = (139, 92, 246)
COLOR_ACCENT_AMBER = (245, 158, 11)

class UIButton:
    """
    ปุ่มกดแบบอินเทอร์แอคทีฟ รองรับการ Hover, Click, และสถานะ Active / Toggle
    """
    def __init__(self, rect, text, font, base_color=(31, 41, 55), hover_color=(55, 65, 81), active_color=(139, 92, 246), text_color=COLOR_TEXT_PRIMARY, active=False):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.base_color = base_color
        self.hover_color = hover_color
        self.active_color = active_color
        self.text_color = text_color
        self.active = active
        self.is_hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, surface):
        if self.active:
            color = self.active_color
        elif self.is_hovered:
            color = self.hover_color
        else:
            color = self.base_color

        # วาดกล่องพื้นหลังโค้งมน
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        border_color = (139, 92, 246) if self.active else (75, 85, 99)
        pygame.draw.rect(surface, border_color, self.rect, width=1, border_radius=6)

        # เรนเดอร์ข้อความ
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)


class BoardRenderer:
    """
    เรนเดอร์กระดาน XO ขนาด 3x3 พร้อมเส้นตารางเรืองแสง, ไฮไลต์ช่องที่เพิ่งเดิน, และเส้นขีดชนะ
    """
    def __init__(self, origin_x, origin_y, size=330):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.size = size
        self.cell_size = size // 3

    def get_cell_at_pos(self, pos):
        x, y = pos
        if not (self.origin_x <= x <= self.origin_x + self.size and self.origin_y <= y <= self.origin_y + self.size):
            return None
        col = (x - self.origin_x) // self.cell_size
        row = (y - self.origin_y) // self.cell_size
        if 0 <= row < 3 and 0 <= col < 3:
            return int(row * 3 + col)
        return None

    def draw(self, surface, board, last_move=None, winning_combination=None, hover_cell=None):
        # 1. กรอบพื้นหลังกระดาน
        board_rect = pygame.Rect(self.origin_x, self.origin_y, self.size, self.size)
        pygame.draw.rect(surface, (15, 23, 42), board_rect, border_radius=12)
        pygame.draw.rect(surface, (30, 41, 59), board_rect, width=2, border_radius=12)

        # 2. ไฮไลต์ Hover cell
        if hover_cell is not None and 0 <= hover_cell < 9 and board[hover_cell] == 0:
            hr = hover_cell // 3
            hc = hover_cell % 3
            cell_rect = pygame.Rect(
                self.origin_x + hc * self.cell_size,
                self.origin_y + hr * self.cell_size,
                self.cell_size, self.cell_size
            )
            hover_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            hover_surf.fill((255, 255, 255, 20))
            surface.blit(hover_surf, cell_rect.topleft)

        # 3. ไฮไลต์ Last Move
        if last_move is not None and 0 <= last_move < 9:
            lr = last_move // 3
            lc = last_move % 3
            last_rect = pygame.Rect(
                self.origin_x + lc * self.cell_size + 4,
                self.origin_y + lr * self.cell_size + 4,
                self.cell_size - 8, self.cell_size - 8
            )
            pygame.draw.rect(surface, (59, 130, 246, 50), last_rect, width=2, border_radius=8)

        # 4. เส้นตารางกระดาน
        line_color = (51, 65, 85)
        for i in range(1, 3):
            # เส้นตั้ง
            x = self.origin_x + i * self.cell_size
            pygame.draw.line(surface, line_color, (x, self.origin_y + 10), (x, self.origin_y + self.size - 10), 3)
            # เส้นนอน
            y = self.origin_y + i * self.cell_size
            pygame.draw.line(surface, line_color, (self.origin_x + 10, y), (self.origin_x + self.size - 10, y), 3)

        # 5. วาดสัญลักษณ์ X และ O
        for r in range(3):
            for c in range(3):
                idx = r * 3 + c
                cx = self.origin_x + c * self.cell_size + self.cell_size // 2
                cy = self.origin_y + r * self.cell_size + self.cell_size // 2
                val = board[idx]

                if val == 1:  # Player X
                    offset = 28
                    thickness = 7
                    pygame.draw.line(surface, COLOR_X_PRIMARY, (cx - offset, cy - offset), (cx + offset, cy + offset), thickness)
                    pygame.draw.line(surface, COLOR_X_PRIMARY, (cx + offset, cy - offset), (cx - offset, cy + offset), thickness)
                elif val == 2:  # Player O
                    radius = 30
                    thickness = 7
                    pygame.draw.circle(surface, COLOR_O_PRIMARY, (cx, cy), radius, thickness)

        # 6. เส้นขีด Winning Strike
        if winning_combination is not None:
            a, b, c = winning_combination
            c1x = self.origin_x + (a % 3) * self.cell_size + self.cell_size // 2
            c1y = self.origin_y + (a // 3) * self.cell_size + self.cell_size // 2
            c3x = self.origin_x + (c % 3) * self.cell_size + self.cell_size // 2
            c3y = self.origin_y + (c // 3) * self.cell_size + self.cell_size // 2

            # ขยายปลายเส้นเล็กน้อย
            dx = c3x - c1x
            dy = c3y - c1y
            p1 = (c1x - int(dx * 0.15), c1y - int(dy * 0.15))
            p2 = (c3x + int(dx * 0.15), c3y + int(dy * 0.15))
            pygame.draw.line(surface, (250, 204, 21), p1, p2, 8)


class NeuralRenderer:
    """
    เรนเดอร์วงจรประสาทชีวภาพ Mushroom Body:
    - Kenyon Cells Grid (1,000 จุด จัดเรียง 40 x 25 พร้อม Glow บน Active 75 เซลล์)
    - MBON Action Probabilities Bar Chart (9 ช่อง)
    - Dopamine Burst Flash & Indicator
    """
    def __init__(self, origin_x, origin_y, width=460, height=660):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height

        # สร้างพิกัดคงที่สำหรับ KC 1,000 จุด (40 cols x 25 rows)
        self.kc_points = []
        grid_w = 40
        grid_h = 25
        start_x = self.origin_x + 20
        start_y = self.origin_y + 60
        spacing_x = (width - 40) / grid_w
        spacing_y = 170 / grid_h

        for r in range(grid_h):
            for c in range(grid_w):
                px = int(start_x + c * spacing_x + spacing_x // 2)
                py = int(start_y + r * spacing_y + spacing_y // 2)
                self.kc_points.append((px, py))

        # ค่าสำหรับ Dopamine Animation
        self.dopamine_val = 0.0
        self.dopamine_alpha = 0

    def trigger_dopamine_flash(self, reward):
        self.dopamine_val = reward
        self.dopamine_alpha = 255

    def update(self):
        if self.dopamine_alpha > 0:
            self.dopamine_alpha = max(0, self.dopamine_alpha - 5)

    def draw(self, surface, fonts, sparse_kc=None, mbon_probs=None, legal_mask=None, selected_action=None, circuit_mode="standard", hippo_agent=None):
        font_title = fonts['title']
        font_sub = fonts['sub']
        font_small = fonts['small']

        panel_rect = pygame.Rect(self.origin_x, self.origin_y, self.width, self.height)
        pygame.draw.rect(surface, COLOR_PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(surface, COLOR_PANEL_BORDER, panel_rect, width=1, border_radius=12)

        if circuit_mode == "hippocampal":
            # 1. หัวข้อแผง Hippocampus
            title_surf = font_title.render("HIPPOCAMPUS (DG-CA3) HDC-VSA", True, (245, 158, 11))
            surface.blit(title_surf, (self.origin_x + 20, self.origin_y + 15))

            sub_info = "EC D=2,048 -> DG (k=50 / 2.4%) -> CA3 (Attractor & Π) -> CA1"
            surface.blit(font_small.render(sub_info, True, (253, 230, 138)), (self.origin_x + 20, self.origin_y + 38))

            # 2. Dentate Gyrus & CA3 Section Box
            hippo_box = pygame.Rect(self.origin_x + 15, self.origin_y + 55, self.width - 30, 180)
            pygame.draw.rect(surface, (15, 23, 42), hippo_box, border_radius=8)

            surface.blit(font_sub.render("DENTATE GYRUS (DG): PATTERN SEPARATION", True, (245, 158, 11)), (hippo_box.left + 15, hippo_box.top + 10))
            surface.blit(font_small.render("Ultra-Sparse Granule Cells (50 Active / 2,048, 2.44% Sparsity)", True, COLOR_TEXT_SECONDARY), (hippo_box.left + 15, hippo_box.top + 28))

            # วาดจุด 50 เซลล์ของ Dentate Gyrus
            for i in range(50):
                px = hippo_box.left + 20 + (i % 25) * 16
                py = hippo_box.top + 50 + (i // 25) * 16
                pygame.draw.circle(surface, (245, 158, 11), (px, py), 4)
                pygame.draw.circle(surface, (251, 191, 36), (px, py), 2)

            # CA3 Attractor & SWR Status Line
            surface.blit(font_sub.render("CA3 RECURRENT ATTRACTOR & TEMPORAL Π", True, (6, 182, 212)), (hippo_box.left + 15, hippo_box.top + 95))
            swr_on = hippo_agent.swr_active if hippo_agent is not None else False
            depth_len = len(hippo_agent.temporal_history) if hippo_agent is not None else 0
            seq_info = f"Temporal Sequence Depth: {depth_len}/5 | Attractor: 2 Steps"
            surface.blit(font_small.render(seq_info, True, COLOR_TEXT_MUTED), (hippo_box.left + 15, hippo_box.top + 115))

            swr_status = "⚡ SWR EPISODIC REPLAY: ACTIVE" if swr_on else "SWR REPLAY: STANDBY (EPISODIC BUFFER)"
            swr_color = (250, 204, 21) if swr_on else COLOR_TEXT_MUTED
            surface.blit(font_small.render(swr_status, True, swr_color), (hippo_box.left + 15, hippo_box.top + 138))

            bar_section_y = self.origin_y + 250
            mbon_title = font_title.render("CA1 / MBON 9 ACTIONS (CLEAN-UP)", True, (226, 232, 240))
            surface.blit(mbon_title, (self.origin_x + 20, bar_section_y))
            surface.blit(font_sub.render("HDC Prototypes Matching with Legal Action Mask", True, COLOR_TEXT_SECONDARY), (self.origin_x + 20, bar_section_y + 22))

        else:
            # 1. หัวข้อแผง Standard Mushroom Body
            title_surf = font_title.render("MUSHROOM BODY BIOLOGICAL CIRCUIT", True, (226, 232, 240))
            surface.blit(title_surf, (self.origin_x + 20, self.origin_y + 15))

            # 2. KC Section
            active_count = int(np.sum(sparse_kc > 0)) if sparse_kc is not None else 0
            kc_label = f"Kenyon Cells (1,000 KCs)  |  Active: {active_count} ({active_count/10:.1f}% Sparsity)"
            kc_sub_surf = font_sub.render(kc_label, True, COLOR_TEXT_SECONDARY)
            surface.blit(kc_sub_surf, (self.origin_x + 20, self.origin_y + 38))

            # กล่องพื้นหลังของ KC
            kc_box = pygame.Rect(self.origin_x + 15, self.origin_y + 55, self.width - 30, 180)
            pygame.draw.rect(surface, (15, 23, 42), kc_box, border_radius=8)

            # วาดจุด KC
            if sparse_kc is None or len(sparse_kc) != 1000:
                for pt in self.kc_points:
                    pygame.draw.circle(surface, COLOR_KC_INACTIVE, pt, 1)
            else:
                for i, pt in enumerate(self.kc_points):
                    if sparse_kc[i] > 0:
                        pygame.draw.circle(surface, (16, 185, 129), pt, 3)
                        pygame.draw.circle(surface, (52, 211, 153), pt, 2)
                    else:
                        pygame.draw.circle(surface, COLOR_KC_INACTIVE, pt, 1)

            # 3. MBON Section
            bar_section_y = self.origin_y + 250
            mbon_title = font_title.render("MBON OUTPUT & ACTION SELECTION", True, (226, 232, 240))
            surface.blit(mbon_title, (self.origin_x + 20, bar_section_y))
            surface.blit(font_sub.render("Softmax Probabilities over Legal Action Masks", True, COLOR_TEXT_SECONDARY), (self.origin_x + 20, bar_section_y + 22))

        # วาด Bar Chart สำหรับ 9 ช่องเดิน
        chart_box = pygame.Rect(self.origin_x + 15, bar_section_y + 45, self.width - 30, 200)
        pygame.draw.rect(surface, (15, 23, 42), chart_box, border_radius=8)

        bar_width = 32
        gap = (chart_box.width - (bar_width * 9)) // 10
        chart_bottom = chart_box.bottom - 28

        for idx in range(9):
            bx = chart_box.left + gap + idx * (bar_width + gap)
            prob = mbon_probs[idx] if mbon_probs is not None else 0.0
            is_legal = legal_mask[idx] if legal_mask is not None else True
            is_chosen = (selected_action == idx)

            bar_h = int(prob * 130)

            # สีของแท่ง
            if not is_legal:
                bar_color = (75, 85, 99)
                bar_h = 4
            elif is_chosen:
                bar_color = (250, 204, 21)  # Golden Yellow for chosen
            else:
                bar_color = (6, 182, 212)    # Cyan for legal

            # วาดแท่ง
            if bar_h > 0:
                bar_rect = pygame.Rect(bx, chart_bottom - bar_h, bar_width, bar_h)
                pygame.draw.rect(surface, bar_color, bar_rect, border_radius=4)

            # ป้ายกำกับช่อง (0-8)
            lbl_color = (250, 204, 21) if is_chosen else (COLOR_TEXT_PRIMARY if is_legal else (107, 114, 128))
            idx_surf = font_small.render(f"#{idx}", True, lbl_color)
            surface.blit(idx_surf, (bx + bar_width // 2 - idx_surf.get_width() // 2, chart_bottom + 4))

            # เปอร์เซ็นต์ความน่าจะเป็น
            if is_legal and prob > 0.01:
                pct_surf = font_small.render(f"{int(prob*100)}%", True, COLOR_TEXT_SECONDARY)
                surface.blit(pct_surf, (bx + bar_width // 2 - pct_surf.get_width() // 2, chart_bottom - bar_h - 16))

        # 4. Dopamine Signal Section
        dopa_section_y = self.origin_y + 515
        dopa_box = pygame.Rect(self.origin_x + 15, dopa_section_y, self.width - 30, 120)
        pygame.draw.rect(surface, (15, 23, 42), dopa_box, border_radius=8)

        dopa_hdr = font_sub.render("DAN DOPAMINERGIC REWARD MODULATION", True, COLOR_TEXT_SECONDARY)
        surface.blit(dopa_hdr, (dopa_box.left + 15, dopa_box.top + 12))

        # ไฟสถานะ Dopamine
        indicator_center = (dopa_box.left + 45, dopa_box.top + 65)
        if self.dopamine_alpha > 0:
            if self.dopamine_val > 0:
                glow_color = (16, 185, 129, self.dopamine_alpha)
                solid_color = (52, 211, 153)
                status_text = f"DOPAMINE BURST: +{self.dopamine_val:.1f} (REWARD LTP)"
                sub_text = "Synaptic Potentiation applied to active KC-MBON pairs"
                status_color = (52, 211, 153)
            elif self.dopamine_val < 0:
                glow_color = (239, 68, 68, self.dopamine_alpha)
                solid_color = (248, 113, 113)
                status_text = f"DOPAMINE DIP: {self.dopamine_val:.1f} (PUNISHMENT LTD)"
                sub_text = "Synaptic Depression applied to active KC-MBON pairs"
                status_color = (248, 113, 113)
            else:
                glow_color = (245, 158, 11, self.dopamine_alpha)
                solid_color = (251, 191, 36)
                status_text = "DOPAMINE SIGNAL: 0.0 (NEUTRAL / DRAW)"
                sub_text = "No synaptic weight modulation applied"
                status_color = (251, 191, 36)

            # Glow surface
            glow_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, (30, 30), 26)
            surface.blit(glow_surf, (indicator_center[0] - 30, indicator_center[1] - 30))
            pygame.draw.circle(surface, solid_color, indicator_center, 12)
        else:
            pygame.draw.circle(surface, (55, 65, 81), indicator_center, 12)
            status_text = "DOPAMINE SIGNAL: STANDBY"
            sub_text = "Awaiting terminal episode reward from environment"
            status_color = COLOR_TEXT_MUTED

        stat_surf = font_title.render(status_text, True, status_color)
        surface.blit(stat_surf, (dopa_box.left + 75, dopa_box.top + 45))
        desc_surf = font_small.render(sub_text, True, COLOR_TEXT_MUTED)
        surface.blit(desc_surf, (dopa_box.left + 75, dopa_box.top + 75))
