"""
Automated script to record high-resolution, lightweight animated demonstration GIFs
from the three neuromorphic Pygame visualizers:
1. Visual Snake Arena (Hippocampus DG-CA3 + Egocentric Whiskers)
2. Honeybee Meadow Foraging Simulator (Multisensory Floral Calyx)
3. Tic-Tac-Toe (XO) Arena (Hippocampus vs Heuristic / Minimax)
"""

import os
import sys
import time

# Set headless SDL video driver before importing pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.visualizer.snake_app import SnakeVisualizerApp
from src.visualizer.bee_app import BeeVisualizerApp, COLOR_BG
from src.visualizer.app import VisualizerApp
from src.envs.snake_env import UP, RIGHT, DOWN, LEFT


def surface_to_pil(surf, target_size=(800, 450)):
    """Convert a Pygame Surface to a PIL Image, resized for web performance."""
    raw_str = pygame.image.tostring(surf, "RGB")
    img = Image.frombytes("RGB", surf.get_size(), raw_str)
    if target_size and target_size != surf.get_size():
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    return img


def save_optimized_gif(frames, output_path, duration=150, loop=0):
    """Save a list of PIL Images as an optimized looping GIF."""
    if not frames:
        print(f"Warning: No frames captured for {output_path}")
        return

    # Quantize frames to a shared palette for clean rendering & compact size
    quantized_frames = []
    # Use first frame to generate palette
    palette_img = frames[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    for frame in frames:
        quantized_frames.append(frame.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG))

    quantized_frames[0].save(
        output_path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved {output_path} ({len(frames)} frames, {size_kb:.1f} KB)")


def record_snake_demo(output_path, num_steps=24):
    """Record Visual Snake Arena navigating towards food with Hippocampus DG-CA3 brain."""
    print("Recording Visual Snake Demo...")
    app = SnakeVisualizerApp(headless=False)
    # Ensure Hippocampus brain is active
    app.brain_mode = "HIPPOCAMPUS"
    app.mb = app.mb_hippo
    app.cpg_enabled = True
    app.live_plasticity = True
    app.reset_game()

    frames = []
    # Initial frame
    app.draw()
    frames.append(surface_to_pil(app.screen, target_size=(800, 450)))

    for step in range(num_steps):
        if app.env.done:
            app.reset_game()
        app.execute_step()
        app.draw()
        frames.append(surface_to_pil(app.screen, target_size=(800, 450)))

    save_optimized_gif(frames, output_path, duration=180)


def record_bee_demo(output_path, num_steps=26):
    """Record Honeybee Meadow Simulator with multisensory foraging flight."""
    print("Recording Honeybee Foraging Demo...")
    app = BeeVisualizerApp(headless=False)
    app.auto_fly = True
    app.enable_plasticity = True

    frames = []

    def draw_current():
        app.screen.fill(COLOR_BG)
        app.draw_meadow_arena()
        app.draw_multisensory_panel()
        app.draw_mushroom_body_circuit()
        app.draw_telemetry_and_controls()
        app.draw_toast_notification()

    draw_current()
    frames.append(surface_to_pil(app.screen, target_size=(800, 500)))

    for step in range(num_steps):
        app.step_simulation()
        if app.octopamine_alpha > 0:
            app.octopamine_alpha = max(0, app.octopamine_alpha - 10)
        if app.dopamine_alpha > 0:
            app.dopamine_alpha = max(0, app.dopamine_alpha - 10)
        draw_current()
        frames.append(surface_to_pil(app.screen, target_size=(800, 500)))

    save_optimized_gif(frames, output_path, duration=170)


def record_xo_demo(output_path):
    """Record Tic-Tac-Toe Arena: Hippocampal MB vs Heuristic with SWR Replay."""
    print("Recording Tic-Tac-Toe Matchup Demo...")
    app = VisualizerApp(headless=False)
    app.agent_x_idx = 0  # Hippocampal MB
    app.agent_o_idx = 3  # Heuristic Agent
    app.reset_game()

    frames = []
    app.draw()
    frames.append(surface_to_pil(app.screen, target_size=(800, 450)))

    max_moves = 10
    move_count = 0
    while not app.env.done and move_count < max_moves:
        app.execute_step()
        app.draw()
        frames.append(surface_to_pil(app.screen, target_size=(800, 450)))
        move_count += 1

    # Repeat the final state with SWR alert for a short pause (4 frames)
    for _ in range(4):
        app.draw()
        frames.append(surface_to_pil(app.screen, target_size=(800, 450)))

    save_optimized_gif(frames, output_path, duration=400)


def main():
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
    os.makedirs(assets_dir, exist_ok=True)

    snake_gif = os.path.join(assets_dir, "snake_demo.gif")
    bee_gif = os.path.join(assets_dir, "bee_demo.gif")
    xo_gif = os.path.join(assets_dir, "xo_demo.gif")

    record_snake_demo(snake_gif, num_steps=24)
    record_bee_demo(bee_gif, num_steps=26)
    record_xo_demo(xo_gif)

    print("\nAll demonstration GIFs generated successfully in 'assets/' directory.")


if __name__ == "__main__":
    main()
