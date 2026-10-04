import os
import sys

# บังคับใช้การถอดรหัส UTF-8 บน Windows Console
if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# เพิ่ม Path ให้เรียกใช้โมดูลในโปรเจกต์ได้
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.visualizer.bee_app import BeeVisualizerApp

def main():
    print("กำลังเริ่มเปิดหน้าต่าง Bio-Inspired Honeybee Foraging Simulator (Apis mellifera Mushroom Body)...")
    app = BeeVisualizerApp()
    app.run()

if __name__ == "__main__":
    main()
