import unittest
import numpy as np
from src.envs.tic_tac_toe import TicTacToeEnv, EMPTY, PLAYER_X, PLAYER_O
from src.models.hippocampal_xo_mb import HippocampalXOMB

class TestHippocampalXOMB(unittest.TestCase):
    def setUp(self):
        self.model = HippocampalXOMB(dim=2048, k_dg=50, k_ca3=120, seed=42)

    def test_initialization(self):
        """ตรวจสอบมิติและการเตรียม Item Memory สำหรับ XO"""
        self.assertEqual(self.model.dim, 2048)
        self.assertEqual(self.model.k_dg, 50)
        self.assertEqual(self.model.num_actions, 9)
        self.assertEqual(self.model.mbon_prototypes.shape, (9, 2048))
        self.assertIn("POS_0", self.model.item_memory)
        self.assertIn("POS_8", self.model.item_memory)
        self.assertIn("STATE_SELF", self.model.item_memory)
        self.assertIn("QUAL_WIN_THREAT", self.model.item_memory)

    def test_dentate_gyrus_pattern_separation(self):
        """
        ทดสอบ Pattern Separation บนกระดาน XO:
        กระดานที่มีตาเดินต่างกัน 1 ช่อง เมื่อผ่าน DG Ultra-sparse k-WTA
        จะต้องมี Cosine Similarity ลดลงอย่างชัดเจน
        """
        # กระดาน 1: ลงตรงกลาง (ช่อง 4)
        board_1 = np.zeros(9, dtype=int)
        board_1[4] = PLAYER_X

        # กระดาน 2: มีช่อง 4 เหมือนกัน แต่คู่ต่อสู้ลงช่อง 0 เพิ่ม
        board_2 = np.zeros(9, dtype=int)
        board_2[4] = PLAYER_X
        board_2[0] = PLAYER_O

        ec_1 = self.model.encode_entorhinal(board_1, current_player=PLAYER_X)
        ec_2 = self.model.encode_entorhinal(board_2, current_player=PLAYER_X)

        sim_ec = np.dot(ec_1, ec_2) / (np.linalg.norm(ec_1) * np.linalg.norm(ec_2))

        dg_1 = self.model.dentate_gyrus_pattern_separation(ec_1)
        dg_2 = self.model.dentate_gyrus_pattern_separation(ec_2)

        sim_dg = np.dot(dg_1, dg_2) / (np.linalg.norm(dg_1) * np.linalg.norm(dg_2))

        # ค่าความคล้ายคลึงหลังผ่าน DG ต้องลดลงอย่างมีนัยสำคัญ
        self.assertLess(sim_dg, sim_ec)
        # ตรวจสอบจำนวน Active Cells ใน DG ต้องเท่ากับ k_dg
        self.assertEqual(np.count_nonzero(dg_1), self.model.k_dg)

    def test_ca3_temporal_sequence(self):
        """ทดสอบการจำแนกลำดับตาเดินผ่าน Permutation (Π)"""
        board = np.zeros(9, dtype=int)
        board[4] = PLAYER_X

        ec = self.model.encode_entorhinal(board, current_player=PLAYER_X)
        dg = self.model.dentate_gyrus_pattern_separation(ec)

        ca3_1, traj_1 = self.model.ca3_pattern_completion_and_sequence(dg)
        ca3_2, traj_2 = self.model.ca3_pattern_completion_and_sequence(dg)

        sim = np.dot(traj_1, traj_2)
        self.assertLess(sim, 0.99)

    def test_swr_replay_learning(self):
        """
        ทดสอบ Sharp-Wave Ripple (SWR) Reverse Replay:
        เมื่อจบเกมและได้รางวัล +1.0 การ Replay ต้องปรับ Action Prototypes อย่างรวดเร็ว
        """
        env = TicTacToeEnv()
        initial_protos = self.model.mbon_prototypes.copy()

        # เล่น 3 ตา
        for _ in range(3):
            if not env.done:
                self.model.select_action(env, training=True, player=PLAYER_X)
                # จำลองการเดินของคู่ต่อสู้
                legal = env.get_legal_actions()
                if legal and not env.done:
                    env.step(legal[0])

        # สั่งอัปเดตเมื่อจบเกมพร้อมรางวัลชนะ (+1.0)
        self.model.update_synapses(reward=1.0, done=True)

        diff = np.linalg.norm(self.model.mbon_prototypes - initial_protos)
        self.assertGreater(diff, 0.05)
        # ตรวจสอบการรีเซ็ตบัฟเฟอร์
        self.assertEqual(len(self.model.episode_buffer), 0)
        self.assertEqual(len(self.model.temporal_history), 0)

    def test_legal_action_masking(self):
        """ทดสอบว่าจะไม่เลือกช่องที่ถูกลงไปแล้ว"""
        env = TicTacToeEnv()
        env.step(4)  # X ลงช่อง 4
        env.step(0)  # O ลงช่อง 0

        # ตอนนี้ช่อง 0 และ 4 ไม่ว่าง
        obs = env.get_observation(PLAYER_X)
        mask = env.get_action_mask()

        probs, _, _ = self.model.forward(obs, mask=mask, board=env.board, player=PLAYER_X)

        self.assertEqual(probs[0], 0.0)
        self.assertEqual(probs[4], 0.0)

        # สั่ง select_action ซ้ำๆ 20 ครั้ง ต้องไม่ลงช่อง 0 หรือ 4
        for _ in range(20):
            act = self.model.select_action(env, training=True, player=PLAYER_X)
            self.assertNotIn(act, [0, 4])

    def test_export_and_load_weights(self):
        """ทดสอบการบันทึกและโหลดน้ำหนัก"""
        data = self.model.export_weights()
        self.assertIn("mbon_prototypes", data)
        self.assertIn("item_memory", data)

        new_model = HippocampalXOMB(dim=2048, seed=999)
        new_model.load_weights(data)

        np.testing.assert_allclose(
            new_model.mbon_prototypes,
            self.model.mbon_prototypes,
            atol=1e-5
        )

    def test_tactical_defense_vs_minimax(self):
        """
        ทดสอบว่า HippocampalXOMB มีการป้องกันระดับ Master (Non-loss Rate 100%)
        เมื่อประลองกับ MinimaxAgent ทั้งเป็นฝ่ายเดินก่อน (X) และเดินทีหลัง (O)
        """
        from src.opponents.minimax_agent import MinimaxAgent
        minimax = MinimaxAgent()

        # ทดสอบ 10 เกมเป็น Player X
        for _ in range(10):
            env = TicTacToeEnv()
            self.model.reset_traces()
            while not env.done:
                if env.current_player == PLAYER_X:
                    act = self.model.select_action(env, training=False, player=PLAYER_X)
                else:
                    act = minimax.select_action(env, player=PLAYER_O)
                env.step(act)
            self.assertNotEqual(env.winner, PLAYER_O, "Hippocampus ต้องไม่แพ้ Minimax เมื่อเดินก่อน (X)")

        # ทดสอบ 10 เกมเป็น Player O (ทดสอบการแก้ Fork และแย่งกลาง)
        for _ in range(10):
            env = TicTacToeEnv()
            self.model.reset_traces()
            while not env.done:
                if env.current_player == PLAYER_O:
                    act = self.model.select_action(env, training=False, player=PLAYER_O)
                else:
                    act = minimax.select_action(env, player=PLAYER_X)
                env.step(act)
            self.assertNotEqual(env.winner, PLAYER_X, "Hippocampus ต้องไม่แพ้ Minimax เมื่อเดินทีหลัง (O)")

if __name__ == "__main__":
    unittest.main()
