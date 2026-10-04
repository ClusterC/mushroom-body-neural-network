import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class Flower:
    """จำลองดอกไม้แต่ละต้นในทุ่งหญ้า พร้อมลักษณะเฉพาะของสปีชีส์ และปริมาณน้ำหวานจำกัด (Finite Reserve)"""
    def __init__(self, flower_id: int, species: int, x: float, y: float, total_reserve: Optional[float] = None):
        self.id = flower_id
        self.species = species
        self.x = x
        self.y = y
        
        # คุณสมบัติตามสปีชีส์ (สี, กลิ่น, ความจุน้ำหวานจำกัด, ความคุ้มค่า)
        if species == 0:  # Lavender (ม่วง/น้ำเงิน) - น้ำหวานสูง แต่มีจำกัด (3.0 หน่วย)
            self.name = "Lavender"
            self.color = np.array([0.4, 0.2, 0.9, 0.8], dtype=np.float32)  # UV, Blue, Green, Lum
            self.odor = np.array([0.9, 0.1, 0.1, 0.1], dtype=np.float32)
            self.max_nectar = 3.0
            self.sweetness = 1.0  # High reward
        elif species == 1:  # Chamomile (ขาว/เหลือง) - น้ำหวานปานกลาง (1.5 หน่วย)
            self.name = "Chamomile"
            self.color = np.array([0.7, 0.9, 0.3, 0.7], dtype=np.float32)
            self.odor = np.array([0.1, 0.8, 0.2, 0.1], dtype=np.float32)
            self.max_nectar = 1.5
            self.sweetness = 0.6
        elif species == 2:  # Wild Rose (ชมพู/แดง) - น้ำหวานน้อยจำกัด (1.0 หน่วย)
            self.name = "Wild Rose"
            self.color = np.array([0.9, 0.2, 0.4, 0.6], dtype=np.float32)
            self.odor = np.array([0.1, 0.1, 0.9, 0.1], dtype=np.float32)
            self.max_nectar = 1.0
            self.sweetness = 0.35
        else:  # Toxic / Deceptive (ฟ้าสดใส/ไม่มีน้ำหวาน มีสารพิษ Quinine)
            self.name = "Toxic Blue"
            self.color = np.array([0.2, 0.9, 0.9, 0.9], dtype=np.float32)
            self.odor = np.array([0.2, 0.1, 0.2, 0.8], dtype=np.float32)
            self.max_nectar = 0.0
            self.sweetness = -0.8  # Punishment
            
        # ปริมาณน้ำหวานจำกัดต่อจุด
        self.total_reserve = total_reserve if total_reserve is not None else self.max_nectar
        self.current_nectar = float(self.total_reserve)
        self.is_depleted = (self.current_nectar <= 0.001 and self.species != 3)
        self.times_visited = 0

    def reset_reserve(self, multiplier: float = 1.0):
        """เติมน้ำหวานรอบใหม่เมื่อผึ้งบินนำอาหารกลับรังผึ้งสำเร็จ (New Bloom Cycle)"""
        self.current_nectar = float(self.total_reserve * multiplier)
        self.is_depleted = (self.current_nectar <= 0.001 and self.species != 3)
        self.times_visited = 0

    def extract_nectar(self, amount: float = 1.0) -> Tuple[float, float]:
        """ดูดน้ำหวานจากดอกไม้ คืนค่า (ปริมาณที่ได้, ความหวาน/รสชาติ)"""
        self.times_visited += 1
        if self.species == 3:
            return 0.0, self.sweetness

        # หากน้ำหวานหมดเกลี้ยงแล้ว (Depleted)
        if self.is_depleted or self.current_nectar <= 0.01:
            self.is_depleted = True
            return 0.0, -0.2  # ไม่ได้น้ำหวานและเสียเวลาตอมฟรี

        extracted = min(self.current_nectar, amount)
        self.current_nectar -= extracted
        if self.current_nectar <= 0.05:
            self.current_nectar = 0.0
            self.is_depleted = True
            
        return extracted, self.sweetness



class BeeForagingEnv:
    """
    สภาพแวดล้อมจำลองพฤติกรรมผึ้งน้ำหวาน (Apis mellifera) หาอาหารในทุ่งดอกไม้
    - แผนที่ 2D Meadow ขนาด 20 x 20 หน่วย
    - รังผึ้ง (Hive) อยู่กึ่งกลาง (10.0, 10.0)
    - Observation: 36 มิติ (8 Olfactory, 12 Visual, 8 Proximity, 8 Internal State)
    - Actions (5 ท่า):
      0: FORWARD (บินตรงไปข้างหน้า 1 หน่วย)
      1: TURN_LEFT (หันซ้าย 45 องศา)
      2: TURN_RIGHT (หันขวา 45 องศา)
      3: LAND_AND_FEED (ลงตอมดอกไม้และดูดน้ำหวาน)
      4: RETURN_TO_HIVE (หันหน้าและก้าวตรงกลับสู่รังผึ้ง)
    """
    ACTIONS = {
        0: "FORWARD",
        1: "TURN_LEFT",
        2: "TURN_RIGHT",
        3: "LAND_AND_FEED",
        4: "RETURN_TO_HIVE"
    }

    def __init__(self, meadow_size: float = 20.0, max_steps: int = 400, seed: Optional[int] = None):
        self.meadow_size = meadow_size
        self.hive_pos = np.array([meadow_size / 2.0, meadow_size / 2.0], dtype=np.float32)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        
        self.max_crop_capacity = 5.0
        self.max_energy = 100.0
        
        # สร้างตำแหน่งดอกไม้ในทุ่งหญ้า 24 ต้น (สปีชีส์ละ 6 ต้น)
        self.flowers: List[Flower] = []
        self._init_flowers()
        
        # ตัวแปรสถานะผึ้ง
        self.bee_pos = np.copy(self.hive_pos)
        self.bee_heading = 0.0  # องศา (0 = ทิศตะวันออก, 90 = ทิศเหนือ)
        self.crop_nectar = 0.0
        self.energy = self.max_energy
        self.step_count = 0
        self.total_hive_nectar = 0.0
        self.trips_completed = 0
        self.species_visit_count = [0, 0, 0, 0]
        
        self.recent_positions: List[Tuple[float, float]] = []
        self.last_reward = 0.0
        self.last_action = 0

    def _init_flowers(self):
        """กระจายกลุ่มดอกไม้รอบรังผึ้งในรัศมีที่เหมาะสม"""
        self.flowers.clear()
        flower_id = 0
        
        # จัดตำแหน่งเป็น 4 โซนทุ่งดอกไม้ (Patches)
        patch_centers = [
            (5.0, 5.0, 0),    # โซนตะวันตกเฉียงใต้: Lavender (ม่วง)
            (15.0, 5.0, 1),   # โซนตะวันออกเฉียงใต้: Chamomile (เหลือง)
            (5.0, 15.0, 2),   # โซนตะวันตกเฉียงเหนือ: Wild Rose (ชมพู)
            (15.0, 15.0, 3),  # โซนตะวันออกเฉียงเหนือ: Toxic Blue (ฟ้า)
        ]
        
        for cx, cy, species in patch_centers:
            for _ in range(6):
                # สุ่มตำแหน่งกระจายรอบจุดศูนย์กลางโซน (ระยะ 1.0 - 3.5 หน่วย)
                angle = self.rng.uniform(0, 2 * math.pi)
                radius = self.rng.uniform(0.8, 3.2)
                fx = np.clip(cx + radius * math.cos(angle), 1.0, self.meadow_size - 1.0)
                fy = np.clip(cy + radius * math.sin(angle), 1.0, self.meadow_size - 1.0)
                self.flowers.append(Flower(flower_id, species, fx, fy))
                flower_id += 1

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """รีเซ็ตสถานะการบินกลับสู่รังผึ้ง"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._init_flowers()
            
        self.bee_pos = np.copy(self.hive_pos)
        self.bee_heading = float(self.rng.uniform(0, 360))
        self.crop_nectar = 0.0
        self.energy = self.max_energy
        self.step_count = 0
        self.recent_positions = [(float(self.bee_pos[0]), float(self.bee_pos[1]))]
        self.last_reward = 0.0
        self.last_action = 0
        
        # ฟื้นฟูดอกไม้รอบใหม่
        for flower in self.flowers:
            flower.reset_reserve()
            
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        """
        สร้างเวกเตอร์สังเกตการณ์พหุสัมผัสขนาด 36 มิติ:
        - 0..7   : Olfactory Profile (กลิ่นดอกไม้ 4 ชนิด + เกรเดียนต์ความเข้มข้น)
        - 8..19  : Visual Ommatidia (ตาประกอบ 3 ทิศทาง ซ้าย/กลาง/ขวา x 4 ช่องแสง UV, Blue, Green, Lum)
        - 20..27 : Proximity & Spatial Sensors (ระยะขอบทุ่ง, ระยะรัง, ระยะดอกไม้ใกล้สุด)
        - 28..35 : Internal State & Navigation (Home Vector, Crop, Energy, Heading sin/cos, Speed)
        """
        obs = np.zeros(36, dtype=np.float32)
        
        # 1. Olfactory Profile (มิติ 0..7)
        # ดมกลิ่นดอกไม้ผ่านหนวด โดยความเข้มข้นลดลงตามระยะทางแบบ Exponential decay
        ambient_odor = np.zeros(4, dtype=np.float32)
        front_pos = self.bee_pos + 0.8 * np.array([
            math.cos(math.radians(self.bee_heading)),
            math.sin(math.radians(self.bee_heading))
        ], dtype=np.float32)
        front_odor = np.zeros(4, dtype=np.float32)
        
        for f in self.flowers:
            # คำนวณระยะทางถึงหนวดผึ้ง
            dist_curr = float(np.linalg.norm(self.bee_pos - np.array([f.x, f.y])))
            dist_front = float(np.linalg.norm(front_pos - np.array([f.x, f.y])))
            decay_curr = math.exp(-dist_curr / 3.0)
            decay_front = math.exp(-dist_front / 3.0)
            
            # ดอกไม้ที่มีน้ำหวานจะปล่อยกลิ่นเข้มข้นกว่า (ถ้าหมดแล้วกลิ่นจะจางลงเหลือ 5%)
            if f.is_depleted:
                nectar_factor = 0.05
            else:
                nectar_factor = 0.2 + 0.8 * (f.current_nectar / max(f.total_reserve, 0.1)) if f.species != 3 else 1.0
            ambient_odor += f.odor * (decay_curr * nectar_factor)
            front_odor += f.odor * (decay_front * nectar_factor)
            
        obs[0:4] = np.clip(ambient_odor, 0.0, 1.0)
        # เกรเดียนต์กลิ่นด้านหน้าลบด้วยกลิ่นรอบตัว (ชี้ทิศทางตามกลิ่น)
        obs[4:8] = np.clip((front_odor - ambient_odor) * 3.0 + 0.5, 0.0, 1.0)

        # 2. Visual Ommatidia (มิติ 8..19)
        # เซนเซอร์สายตาตรวจจับสี 3 ทิศทาง: ซ้าย (+35°), กลาง (0°), ขวา (-35°)
        angles = [self.bee_heading + 35.0, self.bee_heading, self.bee_heading - 35.0]
        for idx, ang in enumerate(angles):
            rad = math.radians(ang)
            dir_vec = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)
            detected_color = np.zeros(4, dtype=np.float32)
            
            for f in self.flowers:
                rel_vec = np.array([f.x, f.y], dtype=np.float32) - self.bee_pos
                dist = float(np.linalg.norm(rel_vec))
                if dist < 6.0 and dist > 0.05:
                    rel_norm = rel_vec / dist
                    cos_sim = float(np.dot(dir_vec, rel_norm))
                    if cos_sim > 0.80:  # อยู่ในกรวยสายตา ~36 องศา
                        intensity = (cos_sim ** 3) * (1.0 - dist / 6.0)
                        # ดอกไม้ที่หมดน้ำหวานแล้ว สีจะหม่นลง (Desaturated/Dimmed)
                        dim_factor = 0.25 if f.is_depleted else 1.0
                        detected_color += f.color * intensity * dim_factor
                        
            obs[8 + idx * 4 : 8 + (idx + 1) * 4] = np.clip(detected_color, 0.0, 1.0)

        # 3. Proximity & Spatial Sensors (มิติ 20..27)
        # ระยะห่างจากขอบทุ่ง 4 ทิศ (N, S, E, W)
        obs[20] = min(1.0, (self.meadow_size - self.bee_pos[1]) / self.meadow_size)  # North
        obs[21] = min(1.0, self.bee_pos[1] / self.meadow_size)                       # South
        obs[22] = min(1.0, (self.meadow_size - self.bee_pos[0]) / self.meadow_size)  # East
        obs[23] = min(1.0, self.bee_pos[0] / self.meadow_size)                       # West
        
        # ระยะห่างจากรังผึ้ง
        dist_hive = float(np.linalg.norm(self.bee_pos - self.hive_pos))
        obs[24] = min(1.0, dist_hive / (self.meadow_size * 0.7))
        
        # ระยะห่างดอกไม้ที่ใกล้ที่สุด
        min_dist_flower = min(float(np.linalg.norm(self.bee_pos - np.array([f.x, f.y]))) for f in self.flowers)
        obs[25] = min(1.0, min_dist_flower / 5.0)
        
        # ตัวชี้ว่าอยู่บนดอกไม้หรือไม่ (ระยะ < 0.9 หน่วย)
        obs[26] = 1.0 if min_dist_flower < 0.9 else 0.0
        obs[27] = 1.0 if dist_hive < 1.2 else 0.0  # อยู่ในรังหรือไม่

        # 4. Internal State & Compass (มิติ 28..35)
        # Home Vector (เวกเตอร์ชี้ตรงกลับรัง)
        home_vec = self.hive_pos - self.bee_pos
        home_dist = float(np.linalg.norm(home_vec))
        if home_dist > 0.001:
            home_norm = home_vec / home_dist
            obs[28] = (home_norm[0] + 1.0) * 0.5  # Normalized 0..1
            obs[29] = (home_norm[1] + 1.0) * 0.5
        else:
            obs[28] = 0.5
            obs[29] = 0.5
            
        obs[30] = self.crop_nectar / self.max_crop_capacity
        obs[31] = self.energy / self.max_energy
        rad_head = math.radians(self.bee_heading)
        obs[32] = (math.sin(rad_head) + 1.0) * 0.5
        obs[33] = (math.cos(rad_head) + 1.0) * 0.5
        obs[34] = 1.0 if self.crop_nectar >= (self.max_crop_capacity * 0.8) else 0.0  # สัญญาณถุงน้ำหวานใกล้เต็ม
        obs[35] = float(self.step_count) / float(self.max_steps)
        
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        ดำเนินการ 1 ก้าวตามการกระทำของผึ้ง:
        0: FORWARD
        1: TURN_LEFT (45°)
        2: TURN_RIGHT (45°)
        3: LAND_AND_FEED
        4: RETURN_TO_HIVE
        """
        self.step_count += 1
        self.last_action = action
        reward = -0.01  # ต้นทุนการบินขั้นต่ำในแต่ละก้าว
        self.energy = max(0.0, self.energy - 0.2)
        done = False
        info = {
            "action_name": self.ACTIONS.get(action, "UNKNOWN"),
            "event": "FLYING",
            "crop": self.crop_nectar,
            "hive_nectar": self.total_hive_nectar,
            "species_visited": list(self.species_visit_count)
        }

        # ดำเนินการตามแอ็กชัน
        if action == 0:  # FORWARD
            rad = math.radians(self.bee_heading)
            move_vec = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)
            self.bee_pos += move_vec * 0.8
        elif action == 1:  # TURN_LEFT
            self.bee_heading = (self.bee_heading + 45.0) % 360.0
            rad = math.radians(self.bee_heading)
            self.bee_pos += np.array([math.cos(rad), math.sin(rad)], dtype=np.float32) * 0.3
        elif action == 2:  # TURN_RIGHT
            self.bee_heading = (self.bee_heading - 45.0) % 360.0
            rad = math.radians(self.bee_heading)
            self.bee_pos += np.array([math.cos(rad), math.sin(rad)], dtype=np.float32) * 0.3
        elif action == 3:  # LAND_AND_FEED
            # ค้นหาดอกไม้ที่ใกล้ที่สุดในระยะ 1.1 หน่วย
            nearest_flower = None
            min_dist = 999.0
            for f in self.flowers:
                d = float(np.linalg.norm(self.bee_pos - np.array([f.x, f.y])))
                if d < min_dist:
                    min_dist = d
                    nearest_flower = f
                    
            if nearest_flower is not None and min_dist <= 1.2:
                # บินลงเกาะและดูดน้ำหวาน (ซึ่งมีจำกัดต่อจุด)
                free_space = self.max_crop_capacity - self.crop_nectar
                amount, sweetness = nearest_flower.extract_nectar(amount=min(1.5, free_space))
                self.species_visit_count[nearest_flower.species] += 1
                
                if nearest_flower.species == 3:  # Toxic Blue
                    reward -= 1.0  # การลงโทษรุนแรง
                    info["event"] = "POISONED"
                elif amount > 0.05:
                    self.crop_nectar += amount
                    reward += amount * sweetness * 1.5  # Octopamine reward!
                    if nearest_flower.is_depleted:
                        info["event"] = f"DEPLETED_{nearest_flower.name.upper()}"
                    else:
                        info["event"] = f"FED_{nearest_flower.name.upper()}"
                else:
                    reward -= 0.15  # ดอกไม้ไม่มีน้ำหวานเหลือแล้ว (หมดเกลี้ยง)
                    info["event"] = "EMPTY_FLOWER"
            else:
                reward -= 0.15  # ลงจอดบนพื้นหญ้าว่างเปล่า
                info["event"] = "LANDED_ON_GRASS"
        elif action == 4:  # RETURN_TO_HIVE
            # ปรับทิศทางมุ่งตรงเข้าหารังผึ้งและก้าวไปข้างหน้า
            to_hive = self.hive_pos - self.bee_pos
            dist_to_hive = float(np.linalg.norm(to_hive))
            if dist_to_hive > 0.05:
                angle_to_hive = math.degrees(math.atan2(to_hive[1], to_hive[0])) % 360.0
                self.bee_heading = angle_to_hive
                self.bee_pos += (to_hive / dist_to_hive) * 1.0
                info["event"] = "HEADING_HOME"

        # ป้องกันการบินทะลุกรอบทุ่งหญ้า
        prev_x, prev_y = self.bee_pos[0], self.bee_pos[1]
        self.bee_pos[0] = np.clip(self.bee_pos[0], 0.5, self.meadow_size - 0.5)
        self.bee_pos[1] = np.clip(self.bee_pos[1], 0.5, self.meadow_size - 0.5)
        if self.bee_pos[0] != prev_x or self.bee_pos[1] != prev_y:
            reward -= 0.2  # ชนขอบอาณาเขต

        # ตรวจสอบการกลับสู่รังผึ้ง
        dist_to_hive = float(np.linalg.norm(self.bee_pos - self.hive_pos))
        if dist_to_hive <= 1.2:
            if self.crop_nectar > 0.2:
                # ถ่ายทอดน้ำหวานเข้าสู่รัง (รางวัลใหญ่แก่รังผึ้ง)
                unloaded = self.crop_nectar
                self.total_hive_nectar += unloaded
                self.crop_nectar = 0.0
                self.energy = min(self.max_energy, self.energy + 40.0)
                reward += unloaded * 2.5  # รางวัลมหาศาลจากการนำอาหารกลับรัง
                self.trips_completed += 1
                info["event"] = f"UNLOADED_NECTAR (+{unloaded:.1f})"
                # ฤดูกาลใหม่: เมื่อนำน้ำหวานส่งรังสำเร็จ ดอกไม้ในทุ่งจะผลิดอกและมีน้ำหวานรอบใหม่
                for f in self.flowers:
                    f.reset_reserve()

        # บันทึกเส้นทางบิน
        self.recent_positions.append((float(self.bee_pos[0]), float(self.bee_pos[1])))
        if len(self.recent_positions) > 50:
            self.recent_positions.pop(0)

        # ตรวจสอบเงื่อนไขจบตอน (พลังงานหมด หรือครบขีดจำกัดก้าว)
        if self.energy <= 0.0:
            reward -= 1.0
            done = True
            info["event"] = "ENERGY_DEPLETED"
        elif self.step_count >= self.max_steps:
            done = True
            info["event"] = "TIME_LIMIT"

        self.last_reward = reward
        return self.get_observation(), float(reward), done, info

    def get_flower_stats(self) -> Dict[str, int]:
        """คืนค่าจำนวนดอกไม้ที่ยังบานและที่แห้งเหือดน้ำหวานหมดแล้ว"""
        total = len(self.flowers)
        depleted = sum(1 for f in self.flowers if f.is_depleted)
        active = total - depleted
        return {
            "total": total,
            "active": active,
            "depleted": depleted
        }

