#!/usr/bin/env python3
"""Title and end cards for the showcase compilation: python3 scripts/cards.py <outdir>"""
import subprocess, sys
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
F = lambda n, s: ImageFont.truetype(f"/usr/share/fonts/TTF/{n}", s)
BIG, MID, SM = F("DejaVuSans-Bold.ttf", 78), F("DejaVuSans.ttf", 38), F("DejaVuSans.ttf", 30)
BG, TXT, DIM, ACC, GOOD = (16, 18, 28), (236, 240, 250), (140, 150, 175), (255, 196, 61), (96, 214, 140)


def card(path, draw_fn, secs):
    img = Image.new("RGB", (W, H), BG)
    draw_fn(ImageDraw.Draw(img))
    img.save(path + ".png")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-i", path + ".png", "-t", str(secs), "-r", "60",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path], check=True)


def title(d):
    d.text((W // 2, 360), "I taught a neural network", font=BIG, fill=TXT, anchor="mm")
    d.text((W // 2, 460), "to play Super Mario World", font=BIG, fill=ACC, anchor="mm")
    d.text((W // 2, 620), "No pixels. No human input while it plays.", font=MID, fill=DIM, anchor="mm")
    d.text((W // 2, 680), "It reads the game's memory and picks buttons.", font=MID, fill=DIM, anchor="mm")
    d.text((W // 2, 900), "Rust · Burn · PPO · 38 million steps of practice", font=SM, fill=DIM, anchor="mm")


def end(d):
    d.text((W // 2, 170), "6 stages cleared, from the start, every time it tries", font=F("DejaVuSans-Bold.ttf", 54), fill=TXT, anchor="mm")
    rows = [("Stage 1", 100), ("Stage 2", 95), ("Stage 3", 95), ("Stage 4", 98), ("Stage 5", 92), ("Stage 6", 85)]
    for i, (name, pct) in enumerate(rows):
        y = 320 + i * 90
        d.text((420, y), name, font=MID, fill=TXT, anchor="lm")
        d.rounded_rectangle([620, y - 22, 1520, y + 22], 22, fill=(44, 50, 72))
        d.rounded_rectangle([620, y - 22, 620 + int(900 * pct / 100), y + 22], 22, fill=GOOD)
        d.text((1560, y), f"{pct}%", font=MID, fill=TXT, anchor="lm")
    d.text((W // 2, 900), "% of 40 attempts that reach the finish", font=SM, fill=DIM, anchor="mm")
    d.text((W // 2, 960), "Stage 6 was only possible after one human demonstration", font=SM, fill=ACC, anchor="mm")


card(f"{sys.argv[1]}/card_title.mp4", title, 4)
card(f"{sys.argv[1]}/card_end.mp4", end, 7)
