"""The separate JPEG figures the journal asks for, from the rendered PNGs (no legends in the images). Usage: python export_figures.py"""
import os
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), "results", "figures")
MAP = {1: "note_fig1_reproduction.png", 2: "note_fig3_haze.png", 3: "fig4_identifiability.png", 4: "fig5_muting.png", 5: "fig6_budget.png"}
for n, f in MAP.items():
    Image.open(os.path.join(SRC, f)).convert("RGB").save(os.path.join(HERE, "figures", f"Figure_{n}.jpg"), quality=95, dpi=(300, 300)); print("Figure", n, f)
