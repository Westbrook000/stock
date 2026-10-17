#!/usr/bin/env python3
"""从本地A股/ETF真实行情中挑选供需区教学片段，生成游戏关卡 JSON。

选段标准（与 js/engine/zones.js 同规则）：
- 窗口内存在一个"干净"的供需区：基带(1-3根静态K线) + 暴力出线；
- 决策点 decision_idx 位于出线之后、价格回踩之前或之中（未来仍有≥25根走势）；
- 回踩结果明确：反弹（成功）或收盘破竭线（失败），二者只居其一；
- 区域在窗口内可定位大级别供给/需求区以计算 L.T.S.D 方位。

输出: supply-demand-game/data/episodes/<id>.json 与 index.json
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIRS = [ROOT.parent / "data" / "stocks", ROOT.parent / "data" / "etfs"]
OUT_DIR = ROOT / "data" / "episodes"

import csv
from datetime import datetime


def load_bars(path):
    bars = []
    name = path.stem
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                if row.get("名称"):
                    name = row["名称"]
                bars.append({
                    "date": row["日期"][:10],
                    "open": float(row["开盘"]), "high": float(row["最高"]),
                    "low": float(row["最低"]), "close": float(row["收盘"]),
                })
            except (KeyError, ValueError):
                continue
    return bars, name


def atr(bars, i, period=14):
    if i < period - 1:
        return None
    tot = 0.0
    for k in range(i - period + 1, i + 1):
        pc = bars[k - 1]["close"] if k > 0 else bars[k]["open"]
        tot += max(bars[k]["high"] - bars[k]["low"],
                   abs(bars[k]["high"] - pc), abs(bars[k]["low"] - pc))
    return tot / period


def is_static(b):
    rng = b["high"] - b["low"]
    if rng <= 1e-9:
        return True
    return abs(b["close"] - b["open"]) / rng < 0.5


def detect_zones(bars, start, end):
    """与JS引擎一致的因果检测。"""
    zones = []
    n = len(bars)
    for i in range(max(start, 14), min(end, n)):
        b = bars[i]
        if is_static(b):
            continue
        a = atr(bars, i)
        if not a:
            continue
        body = abs(b["close"] - b["open"])
        prev_close = bars[i - 1]["close"]
        if body < 1.5 * a or abs(b["close"] / prev_close - 1) < 0.05:
            continue
        d = 1 if b["close"] > b["open"] else -1
        j = i - 1
        while j >= max(0, i - 4) and is_static(bars[j]):
            j -= 1
        base_start, base_end = j + 1, i - 1
        if base_start > base_end or base_start - 5 < 0:
            continue
        pre = "up" if bars[base_start - 1]["close"] >= bars[base_start - 5]["close"] else "down"
        out_dir = "up" if d > 0 else "down"
        ztype = "D" if d > 0 else "S"
        if any(z["type"] == ztype and z["baseStart"] <= base_end and z["baseEnd"] >= base_start for z in zones):
            continue
        base = bars[base_start:base_end + 1]
        pattern = {"downup": "drop-base-rally", "upup": "rally-base-rally",
                   "updown": "rally-base-drop", "downdown": "drop-base-drop"}[pre + out_dir]
        zones.append({
            "type": ztype, "pattern": pattern,
            "baseStart": base_start, "baseEnd": base_end, "impulseIdx": i,
            "createdIdx": i,
            "jieLine": min(x["low"] for x in base) if ztype == "D" else max(x["high"] for x in base),
            "keWick": max(x["high"] for x in base) if ztype == "D" else min(x["low"] for x in base),
            "keBodyTop": max(max(x["open"], x["close"]) for x in base),
            "keBodyBottom": min(min(x["open"], x["close"]) for x in base),
            "impulseBodyAtr": round(body / a, 2), "atr": round(a, 4),
            "baseBars": len(base),
        })
    return zones


def outcome_of(bars, zone, decision_idx, horizon=40):
    """回踩结果：bounce(克线守住反弹≥2R) / busted(收盘破竭线) / None(未发生)"""
    ke = zone["keBodyBottom"]
    risk = abs(ke - zone["jieLine"])
    touched = False
    for i in range(decision_idx, min(len(bars), decision_idx + horizon)):
        b = bars[i]
        if zone["type"] == "D":
            if b["low"] <= ke:
                touched = True
            if touched and b["high"] >= ke + 2 * risk:
                return "bounce"
            if b["close"] < zone["jieLine"]:
                return "busted"
        else:
            if b["high"] >= ke:
                touched = True
            if touched and b["low"] <= ke - 2 * risk:
                return "bounce"
            if b["close"] > zone["jieLine"]:
                return "busted"
    return None


def make_episode(code, name, bars, decision_idx, zone, big_d, big_s, rr):
    lo = max(0, decision_idx - 90)
    hi = min(len(bars), decision_idx + 45)
    window = bars[lo:hi]
    shift = lambda idx: idx - lo
    zones = detect_zones(window, 14, len(window))
    keep = [z for z in zones if abs(z["impulseIdx"] - shift(zone["impulseIdx"])) <= 2]
    if not keep:
        return None
    z = keep[0]
    for k, zz in enumerate(zones):
        zz["id"] = f"z{k+1}"
    # 决策点前触碰次数（按推荐克线）
    ke = z["keBodyBottom"]
    touches = sum(1 for i in range(z["createdIdx"] + 1, shift(decision_idx))
                  if (window[i]["low"] <= ke if z["type"] == "D" else window[i]["high"] >= ke))
    z["touches"] = touches
    z["rr"] = rr
    d_idx = shift(decision_idx)
    if d_idx <= z["createdIdx"] or d_idx >= len(window) - 20:
        return None
    return {
        "code": code, "name": name,
        "bars": [{k: v for k, v in b.items()} for b in window],
        "decisionIdx": d_idx,
        "zones": zones,
        "targetZone": z["id"],
        "big": {"demandKe": big_d, "supplyKe": big_s},
        "outcome": outcome_of(window, z, d_idx),
    }


def pick_and_build(paths, want, seed=7):
    rng = random.Random(seed)
    episodes = []
    tried = 0
    for p in paths:
        if len(episodes) >= want:
            break
        tried += 1
        if tried % 40 == 0:
            print(f"  扫描 {tried} 个文件, 已选 {len(episodes)} 关", flush=True)
        try:
            bars, name = load_bars(p)
        except Exception:
            continue
        if len(bars) < 400:
            continue
        code = p.stem
        bars = bars[-800:]
        zones = detect_zones(bars, 14, len(bars) - 40)
        rng.shuffle(zones)
        for zone in zones[:6]:
            # 决策点：出线后 3~25 根
            d = zone["impulseIdx"] + rng.randint(3, 25)
            if d >= len(bars) - 30:
                continue
            # 大级别区域：现价两侧最近的大区（供给在上方、需求在下方）
            price = bars[d]["close"]
            prior = [z for z in zones if z["createdIdx"] < d - 5 and z is not zone]
            above = [z["keBodyTop"] for z in prior if z["type"] == "S" and z["keBodyTop"] > price * 1.01]
            below = [z["keBodyBottom"] for z in prior if z["type"] == "D" and z["keBodyBottom"] < price * 0.99]
            big_s = min(above, default=None)
            big_d = max(below, default=None)
            if not big_s or not big_d:
                continue
            ke = zone["keBodyBottom"]
            risk = abs(ke - zone["jieLine"])
            target = big_s if zone["type"] == "D" else big_d
            rr = round(abs(target - ke) / risk, 1) if risk > 0 else 0
            if rr < 2:
                continue
            ep = make_episode(code, name, bars, d, zone, big_d, big_s, rr)
            if ep and ep["outcome"] in ("bounce", "busted"):
                ep["id"] = f"{len(episodes)+1:02d}_{code}"
                episodes.append(ep)
                break
    return episodes


def main():
    files = []
    for d in DATA_DIRS:
        if d.exists():
            files += sorted(d.glob("*.csv"))
    random.Random(42).shuffle(files)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    episodes = pick_and_build(files, 12, seed=2026)
    index = []
    for i, ep in enumerate(episodes, 1):
        z = next(z for z in ep["zones"] if z["id"] == ep["targetZone"])
        ep["id"] = f"{i:02d}_{ep['code']}"
        (OUT_DIR / f"{ep['id']}.json").write_text(
            json.dumps(ep, ensure_ascii=False), encoding="utf-8")
        index.append({"id": ep["id"], "code": ep["code"], "name": ep["name"],
                      "type": z["type"], "pattern": z["pattern"],
                      "touches": z["touches"], "outcome": ep["outcome"]})
    (OUT_DIR / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"生成 {len(index)} 关:")
    for row in index:
        print("  ", row)


if __name__ == "__main__":
    main()
