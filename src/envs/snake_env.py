import numpy as np

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

OPPOSITE_ACTIONS = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT
}

class SnakeEnv:
    """
    สภาพแวดล้อมจำลองเกมงู (Snake Game) ขนาด 10x10 ตาราง
    พร้อมระบบส่งออกภาพพิกเซล 3 แชนแนล (Visual Pixel Input) สำหรับโครงข่ายประสาท
    - Channel 0: ตำแหน่งหัวงู (Snake Head)
    - Channel 1: ลำตัวงู (Snake Body)
    - Channel 2: ตำแหน่งอาหาร (Food)
    """
    def __init__(self, width=10, height=10, max_steps_without_food=100, seed=None):
        self.width = width
        self.height = height
        self.max_steps_without_food = max_steps_without_food
        self.rng = np.random.default_rng(seed)

        self.snake = []
        self.direction = RIGHT
        self.food = None
        self.done = False
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0

        self.reset()

    def reset(self):
        """
        รีเซ็ตสถานะเกมงูสู่จุดเริ่มต้น
        """
        # งูเริ่มต้นความยาว 3 ช่อง กลางกระดาน หันไปทางขวา
        center_r = self.height // 2
        center_c = self.width // 2
        self.snake = [
            (center_r, center_c),
            (center_r, center_c - 1),
            (center_r, center_c - 2)
        ]
        self.direction = RIGHT
        self.done = False
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0

        self._spawn_food()
        return self.get_visual_observation()

    def _spawn_food(self):
        """
        สุ่มตำแหน่งอาหารในช่องว่างที่ไม่มีลำตัวงูอยู่
        """
        occupied = set(self.snake)
        free_cells = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in occupied
        ]
        if not free_cells:
            # ชนะเกม งูเต็มกระดาน
            self.food = (-1, -1)
            self.done = True
            return

        self.food = free_cells[int(self.rng.integers(0, len(free_cells)))]

    def get_action_mask(self):
        """
        ส่งคืน Boolean mask ขนาด 4 ทิศทาง (True = เดินได้, False = เดินไม่ได้)
        ป้องกันการหักเลี้ยวกลับหลัง 180 องศาชนคอตนเอง
        """
        mask = np.ones(4, dtype=bool)
        if len(self.snake) > 1:
            opposite = OPPOSITE_ACTIONS[self.direction]
            mask[opposite] = False
        return mask

    def get_visual_observation(self):
        """
        แปลงสถานะปัจจุบันเป็น Visual Tensor ขนาด 3 x Height x Width (3 x 10 x 10 = 300 พิกเซล)
        Channel 0: Head
        Channel 1: Body (มีค่า Gradient จาก 1.0 ถึง 0.4 ไล่ระดับไปหาหาง)
        Channel 2: Food
        """
        obs = np.zeros((3, self.height, self.width), dtype=np.float32)

        # 1. Head Channel
        head_r, head_c = self.snake[0]
        if 0 <= head_r < self.height and 0 <= head_c < self.width:
            obs[0, head_r, head_c] = 1.0

        # 2. Body Channel
        body_len = max(len(self.snake) - 1, 1)
        for idx, (br, bc) in enumerate(self.snake[1:]):
            if 0 <= br < self.height and 0 <= bc < self.width:
                # Gradient intensity เพื่อให้โมเดลแยกแยะทิศทางของลำตัวได้
                decay = 1.0 - (idx / body_len) * 0.5
                obs[1, br, bc] = decay

        # 3. Food Channel
        if self.food is not None and self.food[0] >= 0:
            fr, fc = self.food
            obs[2, fr, fc] = 1.0

        return obs

    def get_safe_action_mask(self):
        """
        CPG Reflex Mask: ตรวจสอบสิ่งกีดขวาง 1 ก้าวล่วงหน้า
        ยับยั้ง Action ที่จะพุ่งชนกำแพงหรือลำตัวตนเอง หากยังมีทางเลือกที่ปลอดภัย
        """
        legal_mask = self.get_action_mask()
        safe_mask = legal_mask.copy()
        head_r, head_c = self.snake[0]

        deltas = {
            UP: (-1, 0),
            RIGHT: (0, 1),
            DOWN: (1, 0),
            LEFT: (0, -1)
        }

        body_set = set(self.snake[:-1])  # หางสุดจะขยับออก
        safe_count = 0

        for act in range(4):
            if not legal_mask[act]:
                safe_mask[act] = False
                continue

            dr, dc = deltas[act]
            nr, nc = head_r + dr, head_c + dc

            # ตรวจสอบการชนกำแพง
            if nr < 0 or nr >= self.height or nc < 0 or nc >= self.width:
                safe_mask[act] = False
                continue

            # ตรวจสอบการชนลำตัว
            if (nr, nc) in body_set:
                safe_mask[act] = False
                continue

            safe_count += 1

        # หากมีทางรอดอย่างน้อย 1 ทาง ให้ใช้ safe_mask มิฉะนั้นใช้ legal_mask เดิม
        return safe_mask if safe_count > 0 else legal_mask

    def get_egocentric_observation(self):
        """
        คำนวณเวกเตอร์เรดาร์ 12 มิติตามมุมมองของตัวงู (Egocentric Whiskers):
        - 0..2: Wall Distances (Forward, Left, Right) [0.0 - 1.0]
        - 3..5: Body Obstacle Danger (Forward, Left, Right) [0.0 - 1.0]
        - 6..9: Relative Food Bearing (Food Ahead, Left, Right, Behind) One-Hot/Magnitude
        - 10: Normalized Food Distance [0.0 - 1.0]
        - 11: Trapped Pocket Factor (0.0 = Open, 1.0 = Dead End)
        """
        head_r, head_c = self.snake[0]
        d = self.direction

        # กำหนดทิศทางสัมพัทธ์ (Relative Directions): Forward, Left, Right
        rel_dirs = {
            'forward': d,
            'left': (d - 1) % 4,
            'right': (d + 1) % 4,
            'behind': (d + 2) % 4
        }

        deltas = {
            UP: (-1, 0),
            RIGHT: (0, 1),
            DOWN: (1, 0),
            LEFT: (0, -1)
        }

        body_set = set(self.snake[:-1])
        ego = np.zeros(12, dtype=np.float32)

        # 1. Wall Distance & Body Obstacle (Forward, Left, Right)
        for i, key in enumerate(['forward', 'left', 'right']):
            act = rel_dirs[key]
            dr, dc = deltas[act]

            # วัดระยะกำแพง
            steps_wall = 0
            curr_r, curr_c = head_r, head_c
            while 0 <= curr_r + dr < self.height and 0 <= curr_c + dc < self.width:
                curr_r += dr
                curr_c += dc
                steps_wall += 1
            max_span = max(self.height, self.width)
            ego[i] = min(1.0, steps_wall / max(max_span, 1))

            # ตรวจสอบสิ่งกีดขวางลำตัวในระยะ 1, 2 ก้าว
            body_danger = 0.0
            for step_dist in [1, 2]:
                check_pos = (head_r + dr * step_dist, head_c + dc * step_dist)
                if check_pos in body_set:
                    body_danger = max(body_danger, 1.0 / step_dist)
            ego[3 + i] = body_danger

        # 2. Relative Food Bearing & Distance
        if self.food is not None and self.food[0] >= 0:
            fr, fc = self.food
            # เวกเตอร์สัมพัทธ์ระดับ Global
            diff_r = fr - head_r
            diff_c = fc - head_c

            # แปลงสู่ระบบแกนของตัวงู (Rotated to Head Frame)
            # UP: forward=-r, right=+c
            # RIGHT: forward=+c, right=+r
            # DOWN: forward=+r, right=-c
            # LEFT: forward=-c, right=-r
            if d == UP:
                fwd_val, rgt_val = -diff_r, diff_c
            elif d == RIGHT:
                fwd_val, rgt_val = diff_c, diff_r
            elif d == DOWN:
                fwd_val, rgt_val = diff_r, -diff_c
            else:  # LEFT
                fwd_val, rgt_val = -diff_c, -diff_r

            ego[6] = 1.0 if fwd_val > 0 else 0.0   # Food Ahead
            ego[7] = 1.0 if rgt_val < 0 else 0.0   # Food Left
            ego[8] = 1.0 if rgt_val > 0 else 0.0   # Food Right
            ego[9] = 1.0 if fwd_val < 0 else 0.0   # Food Behind

            total_dist = abs(diff_r) + abs(diff_c)
            max_dist = self.height + self.width
            ego[10] = min(1.0, total_dist / max_dist)

        # 3. Trapped Pocket Factor
        safe_mask = self.get_safe_action_mask()
        open_ways = np.sum(safe_mask)
        ego[11] = 1.0 - (open_ways / 3.0)  # 1.0 เมื่อไม่มีทางไป, 0.0 เมื่อโล่งทุกทาง

        return ego

    def step(self, action):
        """
        ดำเนินการเลี้ยวและเดิน 1 ก้าว
        ส่งคืน (visual_obs, reward, done, info)
        """
        if self.done:
            raise RuntimeError("เกมจบแล้ว ต้องเรียก reset() ก่อนเล่นรอบใหม่")

        # ตรวจสอบ Action Mask
        mask = self.get_action_mask()
        if not mask[action]:
            # ถ้าเลือกทิศทางห้ามเลี้ยว ให้เดินต่อไปในทิศทางเดิม
            action = self.direction

        self.direction = action
        self.steps += 1
        self.steps_since_food += 1

        head_r, head_c = self.snake[0]

        # คำนวณตำแหน่งหัวใหม่
        if action == UP:
            new_head = (head_r - 1, head_c)
        elif action == RIGHT:
            new_head = (head_r, head_c + 1)
        elif action == DOWN:
            new_head = (head_r + 1, head_c)
        elif action == LEFT:
            new_head = (head_r, head_c - 1)
        else:
            raise ValueError(f"Action {action} ไม่ถูกต้อง (ต้องเป็น 0-3)")

        new_r, new_c = new_head
        reward = 0.0

        # 1. ตรวจสอบการชนกำแพง
        if new_r < 0 or new_r >= self.height or new_c < 0 or new_c >= self.width:
            self.done = True
            reward = -1.0
            return self.get_visual_observation(), reward, self.done, {"cause": "wall", "score": self.score}

        # 2. ตรวจสอบการชนลำตัวตนเอง (ไม่นับช่องหางสุดที่จะขยับออก)
        if new_head in self.snake[:-1]:
            self.done = True
            reward = -1.0
            return self.get_visual_observation(), reward, self.done, {"cause": "self_collision", "score": self.score}

        # 3. คำนวณระยะห่างเดิมเทียบกับระยะห่างใหม่ (Shaping Reward)
        fr, fc = self.food
        old_dist = abs(head_r - fr) + abs(head_c - fc)
        new_dist = abs(new_r - fr) + abs(new_c - fc)

        # ขยับหัวงู
        self.snake.insert(0, new_head)

        # 4. ตรวจสอบการกินอาหาร
        if new_head == self.food:
            self.score += 1
            self.steps_since_food = 0
            reward = 1.0  # Dopamine Burst!
            self._spawn_food()
        else:
            # ไม่อาหาร ให้ตัดหางออก (ไม่เพิ่มความยาว)
            self.snake.pop()
            # Reward Shaping เล็กน้อยตามระยะเข้าหาอาหาร
            reward = 0.05 if new_dist < old_dist else -0.05

        # 5. ป้องกันกรณีเดินวนไม่กินอาหารนานเกินไป
        if self.steps_since_food >= self.max_steps_without_food:
            self.done = True
            reward = -0.5

        info = {
            "score": self.score,
            "steps": self.steps,
            "snake_len": len(self.snake),
            "cause": "starvation" if self.done and self.steps_since_food >= self.max_steps_without_food else None
        }

        return self.get_visual_observation(), reward, self.done, info
