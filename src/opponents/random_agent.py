import numpy as np

class RandomAgent:
    """
    คู่ต่อสู้แบบสุ่ม: สุ่มเลือกตาเดินจากช่องที่ยังว่างอย่างสม่ำเสมอ
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, env, player=None):
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            raise RuntimeError("ไม่มีช่องที่สามารถเดินได้")
        return int(self.rng.choice(legal_actions))
