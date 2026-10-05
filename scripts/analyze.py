#!/usr/bin/env python3
"""Summarise the telemetry a training run writes: python3 scripts/analyze.py runs/ext1 [last_iters]

episodes.jsonl: one line per finished episode (level, from-start or frontier, outcome, where and why it ended,
reward split, action counts). telemetry.jsonl: one line per iteration (losses, KL, explained variance, action mix).
"""
import json, sys, collections

ACTIONS = ["none", "right", "right+B", "right+Y", "right+Y+B", "B", "left", "down", "up", "right+A", "right+Y+A"]

def load(path):
    out = []
    try:
        for l in open(path):
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass  # a line cut off while the run was still writing it
    except FileNotFoundError:
        pass
    return out

def main():
    run = sys.argv[1]
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    eps, its = load(f"{run}/episodes.jsonl"), load(f"{run}/telemetry.jsonl")
    if not its:
        sys.exit("no telemetry yet")
    last = its[-1]["iter"]
    print(f"{run}: {its[-1]['steps']:,} steps, {last} iterations, {len(eps):,} episodes")
    recent = [e for e in eps if e["iter"] > last - window]
    print(f"\n== per level, last {window} iterations (S = from level start, F = from saved state)")
    print("level  S: n  clear  died  tmo | F: n  clear | max_x(S)  mean steps  deaths by cause (pit/enemy/other)  top death x")
    for lv in sorted({e["level"] for e in recent}):
        s = [e for e in recent if e["level"] == lv and e["fs"]]
        f = [e for e in recent if e["level"] == lv and not e["fs"]]
        pct = lambda xs, k: 100 * sum(e["out"] == k for e in xs) / max(len(xs), 1)
        causes = collections.Counter(e["cause"] for e in s + f if e["out"] == "died")
        deaths = collections.Counter((e["end_x"] // 64) * 64 for e in s + f if e["out"] == "died")
        top = ", ".join(f"x~{x}({n})" for x, n in deaths.most_common(3))
        mx = sum(e["max_x"] for e in s) / max(len(s), 1)
        st = sum(e["steps"] for e in s) / max(len(s), 1)
        print(f"L{lv:<4} {len(s):4d} {pct(s,'clear'):5.0f}% {pct(s,'died'):4.0f}% {pct(s,'timeout'):3.0f}% | {len(f):4d} {pct(f,'clear'):5.0f}% | {mx:7.0f} {st:9.0f}   {causes['pit']}/{causes['enemy']}/{causes['other']}   {top}")
    print("\n== action mix of recent episodes (share of steps)")
    tot = [0] * len(ACTIONS)
    for e in recent:
        for i, a in enumerate(e["acts"]):
            tot[i] += a
    n = sum(tot) or 1
    print("  " + "  ".join(f"{ACTIONS[i]} {100*tot[i]/n:.0f}%" for i in range(len(e['acts'])) if tot[i]))
    print("\n== reward split per episode (mean over recent)")
    keys = ["r_prog", "r_coin", "r_time", "r_death", "r_timeout", "r_clear", "r_explore"]
    print("  " + "  ".join(f"{k} {sum(e.get(k, 0) for e in recent)/max(len(recent),1):+.1f}" for k in keys))
    print("\n== training health (last 5 iterations; ent = policy entropy, ev = critic explained variance)")
    for t in its[-5:]:
        print(f"  iter {t['iter']:5d} sps {t['sps']:5.0f}  ent {t['ent']:.2f}  kl {t['kl']:.4f}  clip {t['clip_frac']:.2f}  ev {t['explained_var']:+.2f}  adv_std {t['adv_std']:.2f}  value {t['value_mean']:.1f}")

main()
