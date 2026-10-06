#!/usr/bin/env python3
"""Render the 'all attempts at once' replay from the JSON written by examples/ghosts.rs.

    python3 scripts/ghosts.py runs/ghosts/L3.json out.mp4 [speed]

Top: a scrolling view of the level with every attempt as a coloured ghost with a short trail; a red X marks
each death. Bottom: the whole level with every ghost, a tick for every death so far, and a live histogram of
where attempts end. `speed` is emulator frames per video frame at 30 fps (default 3 = 1.5x real time).
"""
import colorsys, json, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
MAIN_H, MINI_Y = 700, 740
BG, SOLID, SOLID_EDGE, COIN, OBJ = (22, 26, 40), (84, 96, 128), (130, 144, 180), (240, 200, 60), (150, 112, 60)
FONT = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 26)
SMALL = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans.ttf", 18)

def main():
    src, out = sys.argv[1], sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    d = json.load(open(src))
    eps, tiles = d["episodes"], d["tiles"]
    n = len(eps)
    xs_all = np.concatenate([np.array(e["xs"]) for e in eps]); ys_all = np.concatenate([np.array(e["ys"]) for e in eps])
    level_w = int(min(xs_all.max() + 320, len(tiles[0]) * 16))
    y0 = int(max(0, ys_all.min() - 96)); y1 = int(min(27 * 16, ys_all.max() + 96))
    S = min(3.0, MAIN_H / (y1 - y0))
    M = min((W - 60) / level_w, 190 / (y1 - y0))  # minimap scale: fits the width and a 190 px strip

    def terrain(scale):
        """The tile map from row y0 to y1 and column 0 to level_w, as an image."""
        ncols = level_w // 16 + 1
        img = Image.new("RGB", (int(ncols * 16 * scale), int((y1 - y0) * scale)), BG)
        dr = ImageDraw.Draw(img)
        for r, row in enumerate(tiles):
            ty = r * 16
            if ty + 16 < y0 or ty > y1:
                continue
            for c in range(min(ncols, len(row))):
                ch = row[c]
                if ch == ".":
                    continue
                x0, yy = c * 16 * scale, (ty - y0) * scale
                x1, y2 = x0 + 16 * scale, yy + 16 * scale
                if ch == "#":
                    dr.rectangle([x0, yy, x1 - 1, y2 - 1], fill=SOLID, outline=SOLID_EDGE if scale > 1 else None)
                elif ch == "o":
                    dr.ellipse([x0 + 4 * scale, yy + 3 * scale, x1 - 4 * scale, y2 - 3 * scale], fill=COIN)
                else:
                    dr.rectangle([x0 + 1, yy + 1, x1 - 2, y2 - 2], fill=OBJ)
        return img

    big, mini = terrain(S), terrain(M)
    colors = [tuple(int(255 * v) for v in colorsys.hsv_to_rgb(i / n, 0.55, 1.0)) for i in range(n)]
    total = max(len(e["xs"]) for e in eps)
    total = min(total, 60 * 70)  # cap at 70 s of game time
    bucket = 64
    hist = np.zeros(level_w // bucket + 2, dtype=int)
    deaths_marked = set()
    cam = 0.0
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
                           "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    dead_xs = []  # (x, y) of every death so far, for the persistent X marks
    frame = 0
    t = 0
    hold = 0
    while True:
        alive = dead = cleared = timeout = 0
        pos = []
        for i, e in enumerate(eps):
            L = len(e["xs"])
            k = min(t, L - 1)
            finished = t >= L - 1
            if finished and i not in deaths_marked:
                deaths_marked.add(i)
                if e["out"] == "died":
                    hist[min(e["xs"][-1] // bucket, len(hist) - 1)] += 1
                    dead_xs.append((e["xs"][-1], e["ys"][-1]))
            if finished:
                if e["out"] == "died": dead += 1
                elif e["out"] == "cleared": cleared += 1
                else: timeout += 1
            else:
                alive += 1
            pos.append((i, k, finished))
        lead = [e["xs"][min(t, len(e["xs"]) - 1)] for e in eps if t < len(e["xs"]) - 1]
        target = (np.mean(lead) if lead else cam + W / S / 2) - W / S / 2
        cam += (target - cam) * 0.08
        cam = float(np.clip(cam, 0, max(0, level_w - W / S)))

        canvas = Image.new("RGB", (W, H), (12, 14, 22))
        cx = int(cam * S)
        canvas.paste(big.crop((cx, 0, cx + W, big.height)), (0, 0))
        dr = ImageDraw.Draw(canvas)
        sx = lambda x: (x + 8 - cam) * S
        sy = lambda y: (y + 14 - y0) * S
        for (px, py) in dead_xs:
            X, Y = sx(px), sy(py)
            if -20 < X < W + 20:
                dr.line([X - 7, Y - 7, X + 7, Y + 7], fill=(150, 40, 40), width=3); dr.line([X - 7, Y + 7, X + 7, Y - 7], fill=(150, 40, 40), width=3)
        for i, k, finished in pos:
            e = eps[i]; col = colors[i]
            if not finished:
                for back in range(1, 12):
                    kk = max(0, k - back * K)
                    f = 1 - back / 12
                    c = tuple(int(BG[j] + (col[j] - BG[j]) * f * 0.7) for j in range(3))
                    X, Y = sx(e["xs"][kk]), sy(e["ys"][kk])
                    dr.ellipse([X - 3, Y - 3, X + 3, Y + 3], fill=c)
                X, Y = sx(e["xs"][k]), sy(e["ys"][k])
                dr.ellipse([X - 8, Y - 8, X + 8, Y + 8], fill=col, outline=(255, 255, 255))
            else:
                X, Y = sx(e["xs"][-1]), sy(e["ys"][-1])
                if e["out"] == "died":
                    ring = min(1.0, (t - (len(e["xs"]) - 1)) / 24)
                    if ring < 1.0:
                        dr.ellipse([X - 8 - 26 * ring, Y - 8 - 26 * ring, X + 8 + 26 * ring, Y + 8 + 26 * ring], outline=(255, 90, 90), width=3)
                elif e["out"] == "cleared":
                    dr.regular_polygon((X, Y, 14), 5, fill=(255, 215, 0))
        # minimap
        my = lambda y: MINI_Y + (y - y0) * M
        canvas.paste(mini.crop((0, 0, mini.width, mini.height)), (30, MINI_Y))
        dr.rectangle([30 + cam * M, MINI_Y - 2, 30 + (cam + W / S) * M, MINI_Y + mini.height + 2], outline=(255, 255, 255), width=2)
        for (px, py) in dead_xs:
            dr.line([30 + px * M, MINI_Y, 30 + px * M, MINI_Y + mini.height], fill=(190, 60, 60), width=1)
        for i, k, finished in pos:
            e = eps[i]
            X, Y = 30 + e["xs"][k] * M, my(e["ys"][k] + 14)
            if not finished:
                dr.ellipse([X - 4, Y - 4, X + 4, Y + 4], fill=colors[i])
        # death histogram
        hy1 = H - 20; hh = 150
        dr.text((30, hy1 - hh - 28), "where attempts ended (deaths per 64 px)", font=SMALL, fill=(200, 200, 215))
        peak = max(1, hist.max())
        for b, v in enumerate(hist):
            if v:
                bx = 30 + b * bucket * M
                dr.rectangle([bx, hy1 - v / peak * hh, bx + max(2, bucket * M - 1), hy1], fill=(220, 70, 70))
                if v == peak:
                    dr.text((bx - 6, hy1 - v / peak * hh - 24), f"{v}", font=SMALL, fill=(255, 200, 200))
        dr.line([30, hy1, 30 + level_w * M, hy1], fill=(120, 120, 140), width=1)
        # HUD
        secs = t / 60
        hud = f"Level {d['level']}    {n} attempts    alive {alive}    died {dead}    cleared {cleared}" + (f"    stalled {timeout}" if timeout else "") + f"    {secs:5.1f}s"
        dr.rectangle([0, 0, W, 44], fill=(10, 12, 20))
        dr.text((16, 8), hud, font=FONT, fill=(235, 235, 245))
        ff.stdin.write(canvas.tobytes())
        frame += 1
        t += K
        if t > total and alive == 0:
            hold += 1
            if hold > 90:
                break
        elif t > total + 60 * 70:
            break
    ff.stdin.close(); ff.wait()
    print(f"wrote {out}: {frame} frames, {cleared} cleared, {dead} died, {timeout} stalled")

main()
