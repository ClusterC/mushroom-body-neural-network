import os
import sys

# เพิ่ม Path ให้เรียกใช้โมดูลในโปรเจกต์ได้
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.visualizer.snake_app import SnakeVisualizerApp

def main():
    print("กำลังเริ่มเปิดหน้าต่าง Bio-Inspired Visual Snake (Mushroom Body)...")
    app = SnakeVisualizerApp()
    app.run()

if __name__ == "__main__":
    main()
