#!/usr/bin/env python3
"""Render one cleared attempt (from examples/showcase.rs) with the AI's inner workings next to the game.

    python3 scripts/showcase.py runs/showcase/L7 out.mp4 --stage 1 --of 6 --rate "100% (40/40)" [--speed 2] [--note "text"]

Left: the game. Right: the tile grid the network sees, the action probabilities and buttons it picks, and the
critic's forecast. Bottom: progress through the level. `speed` is emulator frames per video frame (60 fps out).
"""
import argparse, json, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG, PANEL, TXT, DIM = (16, 18, 28), (28, 32, 48), (236, 240, 250), (140, 150, 175)
ACC, GOOD = (255, 196, 61), (96, 214, 140)
CELL = {"0": (34, 38, 56), "1": (96, 120, 190), "2": (255, 205, 60), "3": (170, 120, 70), "4": (240, 70, 70), "5": (80, 210, 120)}
F = lambda n, s: ImageFont.truetype(f"/usr/share/fonts/TTF/{n}", s)
BOLD, REG, MONO = F("DejaVuSans-Bold.ttf", 30), F("DejaVuSans.ttf", 22), F("DejaVuSansMNerdFont-Bold.ttf", 22)
HEAD, SMALL = F("DejaVuSans-Bold.ttf", 20), F("DejaVuSans.ttf", 18)
ACTION_LABELS = ["parado", "andar", "andar + pulo", "correr", "correr + pulo", "pulo parado", "voltar"]
RIGHT, LEFT, B, Y = 1 << 7, 1 << 6, 1 << 0, 1 << 1
PX, GX, GY, GC = 1120, 1120, 150, 22  # panel x, grid origin, grid cell size


def text(d, xy, s, font, fill=TXT, anchor="la"):
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def draw_grid(d, g):
    for r in range(14):
        for c in range(20):
            col = CELL[g[r * 20 + c]]
            d.rectangle([GX + c * GC, GY + r * GC, GX + (c + 1) * GC - 2, GY + (r + 1) * GC - 2], fill=col)
    # Mario's cell
    x0, y0 = GX + 5 * GC, GY + 8 * GC
    d.rectangle([x0 - 2, y0 - 2, x0 + GC, y0 + GC], outline=(255, 255, 255), width=3)
    text(d, (x0 + GC // 2 - 1, y0 - 10), "Mario", SMALL, TXT, "ms")
    lx = GX + 20 * GC + 28
    for i, (k, name) in enumerate([("1", "chão e parede"), ("3", "outro bloco"), ("2", "moeda"), ("4", "inimigo")]):
        y = GY + 8 + i * 36
        d.rectangle([lx, y, lx + 22, y + 22], fill=CELL[k])
        text(d, (lx + 34, y - 2), name, SMALL, DIM)


def draw_pad(d, mask, x0, y0):
    on, off = ACC, (52, 58, 82)
    cx, cy, s = x0 + 70, y0 + 80, 36
    for dx, dy, bit in [(1, 0, RIGHT), (-1, 0, LEFT)]:
        c = on if mask & bit else off
        d.rectangle([cx + dx * s - 17, cy + dy * s - 17, cx + dx * s + 17, cy + dy * s + 17], fill=c)
    d.rectangle([cx - 17, cy - 17, cx + 17, cy + 17], fill=(40, 44, 64))
    d.rectangle([cx - 17, cy - s - 17, cx + 17, cy - s + 17], fill=off)
    d.rectangle([cx - 17, cy + s - 17, cx + 17, cy + s + 17], fill=off)
    for name, bit, bx, by in [("B", B, x0 + 200, y0 + 100), ("Y", Y, x0 + 150, y0 + 60)]:
        d.ellipse([bx - 24, by - 24, bx + 24, by + 24], fill=on if mask & bit else off)
        text(d, (bx, by), name, BOLD, (20, 20, 30) if mask & bit else DIM, "mm")


def base(stage, of, note):
    px_note = 48
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    text(d, (48, 26), "Reinforcement learning no Super Mario World", BOLD, TXT)
    text(d, (860, 34), f"fase {stage} de {of}", REG, ACC)
    d.rounded_rectangle([PX - 20, 84, W - 28, 976], 14, fill=PANEL)
    text(d, (PX, 108), "O QUE A IA VÊ", HEAD, ACC)
    text(d, (PX, 480), "O QUE ELA PENSA", HEAD, ACC)
    text(d, (PX + 520, 520), "BOTÕES", HEAD, ACC)
    text(d, (PX, 806), "PREVISÃO DE RECOMPENSA", HEAD, ACC)
    if note:
        text(d, (px_note, 1046), note, SMALL, ACC)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("out")
    ap.add_argument("--stage", type=int, default=1); ap.add_argument("--of", type=int, default=6)
    ap.add_argument("--rate", default=""); ap.add_argument("--speed", type=float, default=1.5); ap.add_argument("--note", default="")
    a = ap.parse_args()
    steps = [json.loads(l) for l in open(f"{a.dir}/steps.jsonl")]
    frames = np.fromfile(f"{a.dir}/frames.rgb", dtype=np.uint8).reshape(-1, 224, 256, 3)
    nf = len(frames)
    step_of = np.zeros(nf, dtype=int)
    for s in steps:
        step_of[s["f"]: s["f"] + s["n"]] = s["t"]
    xs = [s["x"] for s in steps]; vs = [s["v"] for s in steps]
    x_end = max(xs) + 1
    v_lo, v_hi = min(vs), max(vs)
    bg = base(a.stage, a.of, a.note)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "60", "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", a.out], stdin=subprocess.PIPE)
    cache_t, cache_img = -1, None
    for k in range(int(nf / a.speed)):
        fi = min(nf - 1, int(k * a.speed))
        t = int(step_of[fi]); s = steps[t]
        if t != cache_t:  # right panel changes once per agent step
            img = bg.copy(); d = ImageDraw.Draw(img)
            draw_grid(d, s["g"])
            for i, p in enumerate(s["p"]):
                y = 520 + i * 40
                chosen = i == s["a"]
                text(d, (PX, y), ACTION_LABELS[i], REG, TXT if chosen else DIM)
                d.rectangle([PX + 230, y + 4, PX + 430, y + 28], fill=(44, 50, 72))
                d.rectangle([PX + 230, y + 4, PX + 230 + int(200 * p), y + 28], fill=ACC if chosen else (90, 100, 140))
                text(d, (PX + 440, y + 2), f"{p * 100:.0f}%", SMALL, TXT if chosen else DIM)
            draw_pad(d, s["mask"], PX + 520, 548)
            # critic forecast sparkline
            sx0, sy0, sw, sh = PX, 850, 740, 100
            d.rectangle([sx0, sy0, sx0 + sw, sy0 + sh], fill=(22, 25, 38))
            n = len(steps)
            pts = [(sx0 + sw * i / (n - 1), sy0 + sh - 6 - (sh - 12) * (vs[i] - v_lo) / max(1e-6, v_hi - v_lo)) for i in range(t + 1)]
            if len(pts) > 1:
                d.line(pts, fill=GOOD, width=3)
            d.ellipse([pts[-1][0] - 5, pts[-1][1] - 5, pts[-1][0] + 5, pts[-1][1] + 5], fill=GOOD)
            # progress
            px0, py0, pw = 48, 996, W - 96
            d.rounded_rectangle([px0, py0, px0 + pw, py0 + 14], 7, fill=(44, 50, 72))
            d.rounded_rectangle([px0, py0, px0 + max(18, int(pw * s["x"] / x_end)), py0 + 14], 7, fill=ACC)
            text(d, (px0, 1018), f"tempo {int((t * 8) / 60 // 60)}:{int((t * 8) / 60 % 60):02d}", SMALL, DIM)
            if a.rate:
                text(d, (W - 48, 1018), f"passa desta fase em {a.rate} das tentativas", SMALL, GOOD, "ra")
            cache_t, cache_img = t, img
        out = cache_img.copy()
        out.paste(Image.fromarray(frames[fi]).resize((1024, 896), Image.NEAREST), (48, 84))
        ff.stdin.write(np.asarray(out).tobytes())
    ff.stdin.close(); ff.wait()


if __name__ == "__main__":
    main()
