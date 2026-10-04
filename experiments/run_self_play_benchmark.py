import os
import sys
import time
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.mushroom_body import MushroomBodyNet
from src.training.self_play import train_self_play
from src.opponents.heuristic_agent import HeuristicAgent
from src.opponents.minimax_agent import MinimaxAgent
from src.utils.metrics import evaluate_agent
from experiments.run_poc_experiments import train_mushroom_body

def run_benchmark():
    print("=" * 85)
    print("  🧠 [MUSHROOM BODY INTELLIGENCE UPGRADE] SELF-PLAY & CURRICULUM BENCHMARK")
    print("=" * 85)

    seed = 42
    mb = MushroomBodyNet(num_kc=1000, k_active=75, learning_rate=0.08, seed=seed)
    heuristic = HeuristicAgent(seed=101)
    minimax = MinimaxAgent()

    # 1. Phase 1: เทรนแบบเดิม (Heuristic Only 1,000 Episodes)
    print("\n1. กำลังสร้าง Baseline Model (เทรนกับ Heuristic อย่างเดียว 1,000 Episodes)...")
    t0 = time.time()
    train_mushroom_body(mb, heuristic, episodes=1000)
    t_baseline = time.time() - t0
    print(f"   ฝึกฝนเสร็จสิ้นภายใน {t_baseline:.2f} วินาที")

    eval_base_heu = evaluate_agent(mb, heuristic, num_games=100)
    eval_base_mm = evaluate_agent(mb, minimax, num_games=100)

    print("\n   [ผลประเมิน Baseline Model เดิม]")
    print(f"   • เทียบกับ Heuristic : Win={eval_base_heu['win_rate']*100:5.1f}%, Draw={eval_base_heu['draw_rate']*100:5.1f}%, Loss={eval_base_heu['loss_rate']*100:5.1f}%")
    print(f"   • เทียบกับ Minimax   : Win={eval_base_mm['win_rate']*100:5.1f}%, Draw={eval_base_mm['draw_rate']*100:5.1f}%, Loss={eval_base_mm['loss_rate']*100:5.1f}% | Non-loss={eval_base_mm['non_loss_rate']*100:5.1f}%")

    # 2. Phase 2: ยกระดับด้วย Self-Play Co-Evolution + Minimax Mix (1,500 Episodes)
    print("\n2. กำลังยกระดับความฉลาดด้วยระบบ Self-Play Co-Evolution (1,500 Episodes)...")
    t0 = time.time()
    sp_stats = train_self_play(mb, episodes=1500, snapshot_interval=150, minimax_mix_ratio=0.30)
    t_selfplay = time.time() - t0

    print(f"   การฝึก Self-Play สำเร็จภายใน {t_selfplay:.2f} วินาที")
    print(f"   • อัตราการเสมอระหว่างสู้กับตนเอง (Self-Play Draw Rate) : {sp_stats['draw_rate']*100:.1f}%")
    print(f"   • การเปลี่ยนแปลงของค่าน้ำหนัก Synapse (Mean ΔW)        : {sp_stats['weight_delta']:+.4f} (เฉลี่ยใหม่: {sp_stats['weight_after']:.4f})")

    # 3. Phase 3: ประเมินผลเทียบกับคู่ต่อสู้ทั้งสองระดับ
    print("\n3. ประเมินผลเปรียบเทียบหลังผ่าน Self-Play Co-Evolution...")
    eval_sp_heu = evaluate_agent(mb, heuristic, num_games=100)
    eval_sp_mm = evaluate_agent(mb, minimax, num_games=100)

    print("\n" + "=" * 85)
    print("  📊 ตารางเปรียบเทียบความเก่ง: BASELINE vs SELF-PLAY MUSHROOM BODY")
    print("=" * 85)
    print(f"  {'ตัวชี้วัด (Metric)':<35} | {'Baseline (เดิม)':<20} | {'Self-Play (ขั้นที่ 3)':<20}")
    print("  " + "-" * 81)
    print(f"  {'Vs. Heuristic Non-loss (เสมอ/ชนะ)':<35} | {eval_base_heu['non_loss_rate']*100:5.1f}%{'':<14} | {eval_sp_heu['non_loss_rate']*100:5.1f}%")
    print(f"  {'Vs. Minimax Non-loss (เสมอ)':<35} | {eval_base_mm['non_loss_rate']*100:5.1f}%{'':<14} | {eval_sp_mm['non_loss_rate']*100:5.1f}%")
    print(f"  {'Vs. Minimax Draw Count':<35} | {eval_base_mm['draws']:3d} / 100 Games{'':<5} | {eval_sp_mm['draws']:3d} / 100 Games")
    print("=" * 85)

    diff_mm = (eval_sp_mm['non_loss_rate'] - eval_base_mm['non_loss_rate']) * 100.0
    if diff_mm > 0:
        print(f"\n🎉 สำเร็จ: Self-Play ช่วยเพิ่มอัตราการเอาตัวรอดไม่แพ้ Minimax ขึ้น +{diff_mm:.1f}%!")
    else:
        print(f"\nผลลัพธ์ได้รับการบันทึกเรียบร้อย")

if __name__ == "__main__":
    run_benchmark()
