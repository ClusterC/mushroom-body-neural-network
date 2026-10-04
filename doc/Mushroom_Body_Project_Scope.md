# Project Scope: Bio-Inspired Mushroom Body Neural Network for Adaptive Game Playing

---

## 1. บทนำและภาพรวมของโครงการ (Project Overview)

โครงการนี้มีเป้าหมายเพื่อศึกษา ออกแบบ และพัฒนาโมเดลปัญญาประดิษฐ์ที่ได้รับแรงบันดาลใจจากวงจรประสาท **Mushroom Body (MB)** ของแมลง (อาทิ *Drosophila melanogaster*) เพื่อนำมาประยุกต์ใช้ในการควบคุม **Agent** เล่นเกมและแก้ปัญหาในสภาพแวดล้อมที่มีการเปลี่ยนแปลงแบบไดนามิก (Dynamic Environments)

การวิจัยนี้มุ่งเน้นการตอบคำถามหลักว่า: **"กลไก Sparse Expansion ร่วมกับ Three-Factor Local Plasticity ของ Mushroom Body มีประสิทธิภาพ ความยืดหยุ่นในการปรับตัว (Adaptability) และความสามารถในการแก้ปัญหาตามเป้าหมาย (Goal-Directed Problem Solving) ได้เหนือกว่าหรือทัดเทียมกับ Reinforcement Learning ยุคใหม่หรือไม่ โดยเฉพาะในสถานการณ์ที่มี Non-Stationary Dynamics หรือ Reward Sparsity"**

---

## 2. วัตถุประสงค์ของโครงการ (Objectives)

1. **การจำลองสถาปัตยกรรมชีวภาพ (Biological Circuit Modeling):**
   * จำลองโครงสร้างประสาท Projection Neurons (PN) $\rightarrow$ Kenyon Cells (KC) $\rightarrow$ Mushroom Body Output Neurons (MBON)
   * ศึกษาบทบาทของ Anterior Paired Lateral (APL) neuron ในการสร้าง Sparse Coding ผ่านกลไก Global Lateral Inhibition
2. **การพัฒนาระบบการเรียนรู้แบบ Three-Factor Plasticity:**
   * ออกแบบและพัฒนากฎการเรียนรู้แบบ Local Hebbian Learning ที่ถูกควบคุมด้วย Dopaminergic Neurons (DANs) เพื่อปรับค่าน้ำหนัก Synaptic Weights ตามสัญญาณผลลัพธ์ (Reward/Punishment Modulation)
3. **การประเมินความสามารถในการปรับตัวและการแก้ปัญหา (Adaptability & Generalization Evaluation):**
   * วัดประสิทธิภาพการเรียนรู้ใหม่แบบรวดเร็ว (Fast Adaptation / Few-shot Re-learning) เมื่อมีการเปลี่ยนกติกาหรือเป้าหมายกลางคัน (Dynamic Goal Shifting)
   * ประเมินความต้านทานต่อปัญหาการลืมข้อมูลเก่าอย่างรุนแรง (Catastrophic Forgetting) เมื่อต้องสลับไปมาระหว่างหลายภารกิจ (Continual Learning)
4. **การเปรียบเทียบเชิงลึกกับอัลกอริทึมมาตรฐาน (Benchmarking):**
   * ประเมินสมรรถนะเปรียบเทียบกับอัลกอริทึม Deep Reinforcement Learning มาตรฐาน เช่น DQN (Deep Q-Network) และ PPO (Proximal Policy Optimization)

---

## 3. ขอบเขตการดำเนินงาน (Scope of Work)

### 3.1 ขอบเขตด้านสถาปัตยกรรมโมเดล (Model Architecture)
* **Input Layer (Sensory / Projection Neurons - PN):**
  * รับค่า State Representation (Feature Vector) จากเกม ทำการ Normalize และ Encode เป็นสัญญาณประสาท
  * กรณีเป็น Visual Pixel State จะใช้ Feature Encoder ขนาดเล็ก (เช่น Shallow CNN หรือ Spatial Linear Projection) เพื่อแปลงเป็น Vector ก่อนส่งต่อ
* **Expansion Layer (Kenyon Cells - KC):**
  * ขยายมิติของ State Vector สู่ High-Dimensional Sparse Space โดยใช้ Random Fixed Projections
  * ใช้ฟังก์ชัน Sparsification เช่น $k$-Winner-Take-All ($k$-WTA) เพื่อรักษาระดับ Activation Density ให้อยู่ที่ 5% - 10% เลียนแบบการทำงานของสมองแมลง
* **Modulation & Output (DAN & MBON):**
  * ชั้น Output ทำหน้าที่เลือก Action (Discrete Action Space)
  * การอัปเดตน้ำหนัก Synapse ระหว่าง KC และ MBON จะใช้ Three-factor Hebbian Plasticity Rule:
    $$\Delta w_{ij} = \eta \cdot \text{EligibilityTrace}(x_i, y_j) \cdot \text{DopamineSignal}(R)$$
    โดยที่ $\eta$ คือ Learning Rate, $x_i$ คือ Activation ของ KC, $y_j$ คือ Activation ของ MBON, และ $R$ คือผลต่างของ Reward จากสภาพแวดล้อม

---

### 3.2 ขอบเขตด้านสภาพแวดล้อมการทดสอบ (Game Environments & Task Design)

การทดสอบจะดำเนินการบน Gymnasium Framework (OpenAI Gym) โดยแบ่งชุดการทดสอบออกเป็น 3 ระดับ:

| ระดับการทดสอบ | Environment ที่ใช้ | รายละเอียดการทดลอง (Experiment Scenarios) |
| :--- | :--- | :--- |
| **Stage 1: Core Navigation & Goal Seeking** | **MiniGrid (Empty / DoorKey / Dynamic Obstacles)** | ทดสอบการสำรวจพื้นที่และการเดินไปหาเป้าหมายในเขาวงกตที่มี State Sparsity สูง |
| **Stage 2: Dynamic Goal Adaptation** | **MiniGrid-Dynamic / Modified GridWorld** | **Task Switching:** ย้ายตำแหน่ง Goal กะทันหัน หรือสลับเงื่อนไขของ Key-Door เพื่อวัดความเร็วในการปรับตัว |
| **Stage 3: Physical Control & Dynamic Inversion** | **Gymnasium CartPole / MountainCar** | **Dynamic Inversion:** สลับทิศทางการควบคุม (Invert Action) หรือเปลี่ยนมวล/แรงโน้มถ่วงกลางคัน |

---

### 3.3 ขอบเขตด้านการวัดผลและประเมินประสิทธิภาพ (Evaluation Metrics)

1. **Adaptation Latency (Steps to Recovery):** จำนวน Steps/Episodes ที่ Agent ต้องใช้เพื่อกลับมาทำ Success Rate $\ge 90\%$ หลังสภาพแวดล้อมถูกเปลี่ยนกฎ
2. **Sample Efficiency:** ปริมาณ Interaction Step ทั้งหมดที่ Agent ต้องใช้ในการบรรลุเป้าหมายเปรียบเทียบกับ Baseline
3. **Catastrophic Forgetting Index (Retention Ratio):** ประสิทธิภาพในการเล่น Task A หลังจากถูกนำไปเทรนบน Task B โดยไม่มีการย้อนกลับไปเทรน Task A ซ้ำ
4. **Representation Orthogonality:** ความทับซ้อนของ Activation Pattern ในชั้น Kenyon Cells ระหว่างแต่ละ State เพื่อพิสูจน์คุณสมบัติ Sparse Representation

---

### 3.4 สิ่งที่อยู่นอกเหนือขอบเขต (Out of Scope)
* การบีบอัดโมเดลเพื่อลงอุปกรณ์ฮาร์ดแวร์ฝังตัวขนาดเล็ก (Lightweight Edge Deployment / MCU Optimization)
* การจำลอง Spike timing แบบ Real-time SNN ระดับ Sub-millisecond (Biophysical Hodgkin-Huxley Level) โดยจะใช้ Rate-coded Model หรือ Leaky Integrate-and-Fire (LIF) ระดับ Abstract
* สภาพแวดล้อมเกมแบบ 3D เชิงซ้อนสูงระดับ AAA (เช่น Minecraft หรือ FPS) เพื่อควบคุมตัวแปรในการวิจัย

---

## 4. แผนการดำเนินงานและส่งมอบ (Project Roadmap & Deliverables)

```
[Phase 1] ศึกษาทฤษฎี & ออกแบบ Architecture
    │
    ▼
[Phase 2] Implement Mushroom Body Model & Plasticity Rules (PyTorch)
    │
    ▼
[Phase 3] ทดสอบบน Stage 1 Baseline Environments (Gymnasium)
    │
    ▼
[Phase 4] การทดสอบ Dynamic Adaptation & Continual Learning (Stage 2 & 3)
    │
    ▼
[Phase 5] Benchmark เทียบกับ Deep RL (DQN, PPO) & วิเคราะห์ผลสรุปรายงาน
```

### รายการส่งมอบ (Deliverables):
1. **Source Code Repository:** โค้ดสำหรับ Model Architecture, Plasticity Optimizer, Custom Environment Wrappers, และ Benchmark Scripts
2. **Experimental Evaluation Report:** รายงานวิเคราะห์ผลการทดลอง กราฟ Learning Curve, Adaptation Latency และตารางวิเคราะห์ Representation Sparsity
3. **Research Documentation & Presentation:** เอกสารสรุปขอบเขตและผลการวิจัยเชิงลึก

---

## 5. แผนการรับมือความเสี่ยงทางเทคนิค (Technical Risk Assessment)

| ปัญหาความเสี่ยง (Risk) | ผลกระทบ (Impact) | แนวทางแก้ไข (Mitigation Plan) |
| :--- | :--- | :--- |
| Three-factor Plasticity ไม่ลู่เข้าหาคำตอบใน Action Space ขนาดใหญ่ | สูง | กำหนดวงจร Softmax Action Selection และเพิ่ม TD-Error สำหรับขับเคลื่อน Dopamine Signal |
| สัญญาณ Reward ในสภาพแวดล้อมเบาบางเกินไป (Sparse Reward) | ปานกลาง | เสริมกลไก Novelty / Intrinsic Curiosity Signal เข้าสู่ DANs เพื่อกระตุ้นการสำรวจ |
| Sparse Expansion ใช้ Memory สูงเมื่อ Dimension ของ KC ใหญ่เกินไป | ต่ำ | ใช้ Sparse Matrix Computation (`torch.sparse`) ในการคำนวณ Synapse Weight ระหว่าง KC-MBON |
