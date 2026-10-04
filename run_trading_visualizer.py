"""
Desktop Entrypoint for Biomimetic Financial Terminal Visualizer.
Runs the Hippocampus (DG-CA3) algorithmic stock trading simulation.
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.visualizer.trading_app import TradingVisualizerApp


def main():
    print("Launching Biomimetic Financial Terminal (Hippocampus DG-CA3 Trading MB)...")
    app = TradingVisualizerApp(headless=False)
    app.run()


if __name__ == "__main__":
    main()
