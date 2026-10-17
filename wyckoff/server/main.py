"""FastAPI 应用：信号榜单 / 个股分析 / 回测 / 数据更新。"""
from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import __version__
from ..config import REPO_ROOT, StrategyConfig
from ..data_store import load_frame
from ..downloader import download_all_etfs, update_all_stocks
from ..scanner import build_universe, load_latest_scan, save_scan, scan
from . import jobs

app = FastAPI(title="威科夫分析服务", version=__version__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# ---------------- 工具 ----------------

def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6) if f else f
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


_analyze_cache: "OrderedDict[str, object]" = OrderedDict()
_CACHE_MAX = 128


def _cached_analysis(code: str, kind: str, cfg_overrides: Optional[Dict]):
    from ..analysis.engine import run_analysis
    if kind == "auto":
        kind = "stock" if (REPO_ROOT / "data" / "stocks" / f"{code}.csv").exists() else "etf"
    key = f"{kind}:{code}:{sorted((cfg_overrides or {}).items())}"
    hit = _analyze_cache.get(key)
    if hit is not None:
        path = REPO_ROOT / "data" / ("stocks" if kind == "stock" else "etfs") / f"{code}.csv"
        if hit[1] == path.stat().st_mtime_ns:
            _analyze_cache.move_to_end(key)
            return hit[0]
    df = load_frame(code, kind)
    if df is None:
        raise HTTPException(404, f"未找到 {code} 的本地数据，请先在数据页下载/更新")
    cfg = StrategyConfig.load()
    if cfg_overrides:
        cfg = cfg.with_overrides(**cfg_overrides)
    res = run_analysis(df, cfg, code=code)
    path = REPO_ROOT / "data" / ("stocks" if kind == "stock" else "etfs") / f"{code}.csv"
    _analyze_cache[key] = (res, path.stat().st_mtime_ns)
    if len(_analyze_cache) > _CACHE_MAX:
        _analyze_cache.popitem(last=False)
    return res


# ---------------- 模型 ----------------

class ScanReq(BaseModel):
    kind: str = "both"          # stock / etf / both
    sample: int = 0             # 0=全部
    workers: int = 8
    params: Optional[Dict] = None


class BacktestReq(BaseModel):
    code: str
    kind: str = "stock"
    since: Optional[str] = None
    until: Optional[str] = None
    params: Optional[Dict] = None


class BatchBacktestReq(BaseModel):
    kind: str = "both"
    sample: int = 100
    since: Optional[str] = "2015-01-01"
    until: Optional[str] = None
    workers: int = 8
    params: Optional[Dict] = None
    tag: str = "api"


class GridReq(BaseModel):
    kind: str = "stock"
    sample: int = 60
    since: Optional[str] = "2015-01-01"
    workers: int = 8
    grid: Dict[str, List[float]]


class UpdateReq(BaseModel):
    what: str = "stocks"        # stocks / etfs
    workers: int = 10


# ---------------- API ----------------

@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/api/scan")
def get_scan():
    rows = load_latest_scan()
    if rows is None:
        return {"available": False, "rows": [], "hint": "尚未生成扫描结果，请先 POST /api/scan/run"}
    rows = sorted(rows, key=lambda r: (-r.get("buy_score", 0)))
    return {"available": True, "count": len(rows), "rows": _clean(rows[:500])}


@app.post("/api/scan/run")
def run_scan_job(req: ScanReq):
    kinds = ("stock", "etf") if req.kind == "both" else (req.kind,)

    def work():
        rows = scan(kinds=kinds, workers=req.workers, sample=req.sample, cfg_overrides=req.params)
        rows = [r for r in rows if "error" not in r]
        save_scan(rows)
        return {"count": len(rows)}
    return {"job_id": jobs.submit("scan", work, req.dict())}


@app.get("/api/symbol/{code}")
def symbol_detail(code: str, kind: str = "stock", bars: int = 500,
                  params: Optional[str] = None):
    import json
    overrides = json.loads(params) if params else None
    resolved = "stock" if (REPO_ROOT / "data" / "stocks" / f"{code}.csv").exists() else "etf"
    kind = kind if kind in ("stock", "etf") else resolved
    res = _cached_analysis(code, kind, overrides)
    feats = res.features
    sig = res.signals
    n = len(feats)
    lo = max(0, n - bars)
    dates = [str(d.date()) for d in feats["date"].iloc[lo:]]
    candles = [[float(o), float(c), float(l), float(h)] for o, c, l, h in zip(
        feats["open"].iloc[lo:], feats["close"].iloc[lo:],
        feats["low"].iloc[lo:], feats["high"].iloc[lo:])]
    vols = [{"d": dates[i - lo], "v": float(feats["volume"].iloc[i]),
             "up": bool(feats["up"].iloc[i])} for i in range(lo, n)]
    events = []
    for e in res.events:
        if e.confirm_idx >= lo:
            events.append({
                "type": e.etype, "kind": e.kind, "quality": e.quality,
                "date": str(feats["date"].iloc[e.confirm_idx].date()),
                "price": e.price, "failed": e.failed_idx is not None,
                "bar": e.confirm_idx - lo,
            })
    zones = []
    for z in res.zones:
        if z.created_idx >= lo:
            zones.append({"type": z.ztype, "top": z.top, "bottom": z.bottom,
                          "created": str(feats["date"].iloc[z.created_idx].date()),
                          "broken": str(feats["date"].iloc[z.broken_idx].date()) if z.broken_idx else None})
    signals_tbl = [
        {"date": dates[i - lo], "action": sig["action"].iloc[i], "reason": sig["reason"].iloc[i],
         "buy": float(sig["buy_score"].iloc[i]), "sell": float(sig["sell_score"].iloc[i])}
        for i in range(lo, n) if sig["action"].iloc[i] != "HOLD" or sig["reason"].iloc[i]]
    return _clean({
        "code": res.code, "name": res.name, "kind": kind,
        "dates": dates, "candles": candles, "volumes": vols,
        "events": events, "zones": zones,
        "signals": signals_tbl[-80:],
        "latest": res.latest,
        "supports": [p.price for p in res.pivots if p.kind == "low" and p.price_idx >= lo][:10],
        # 合并后的支撑带（宽度≤zone_max_width，只向下合并），未破位的才展示
        "support_zones": [[round(lv.bottom, 4), round(lv.top, 4)] for lv in res.levels
                          if lv.kind == "support" and lv.active and lv.created_idx >= lo],
    })


_name_index: Optional[List[Dict]] = None


def _get_name_index() -> List[Dict]:
    global _name_index
    if _name_index is not None:
        return _name_index
    from ..data_store import list_symbols, universe_dir
    rows = []
    for kind in ("stock", "etf"):
        for code in list_symbols(kind):
            name = ""
            p = universe_dir(kind) / f"{code}.csv"
            try:
                with open(p, encoding="utf-8-sig") as f:
                    header = f.readline()
                    row = f.readline()
                parts = row.split(",")
                if len(parts) >= 3 and "名称" in header:
                    name = parts[2]
            except Exception:  # noqa: BLE001
                pass
            rows.append({"code": code, "name": name, "kind": kind})
    _name_index = rows
    return rows


@app.get("/api/search")
def search(q: str):
    """按代码前缀或名称模糊搜索（股票+ETF）。"""
    q = q.strip()
    out = [r for r in _get_name_index() if not q or r["code"].startswith(q) or q in r["name"]]
    return out[:30]


@app.post("/api/backtest/single")
def backtest_single(req: BacktestReq):
    from ..backtest.runner import run_symbol
    r = run_symbol(req.code, req.kind, req.params, req.since, req.until)
    if not r:
        raise HTTPException(404, "无足够数据")
    return _clean(r)


@app.post("/api/backtest/batch")
def backtest_batch(req: BatchBacktestReq):
    from ..backtest.runner import run_batch, save_report, summarize

    def work():
        codes = build_universe(
            ("stock", "etf") if req.kind == "both" else (req.kind,), req.sample)
        rows = run_batch(codes, req.params, workers=req.workers, start=req.since, end=req.until)
        summary = summarize(rows)
        path = save_report(rows, summary, req.tag)
        return {"summary": summary, "report": path,
                "symbols": len([r for r in rows if r])}
    return {"job_id": jobs.submit("backtest_batch", work, req.dict())}


@app.post("/api/backtest/grid")
def backtest_grid(req: GridReq):
    from ..backtest.optimizer import grid_search

    def work():
        codes = build_universe(
            ("stock", "etf") if req.kind == "both" else (req.kind,), req.sample)
        df = grid_search(req.grid, codes, workers=req.workers, start=req.since)
        recs = df.to_dict("records")
        return {"combos": len(recs), "path": df.attrs.get("path"), "rows": _clean(recs[:40])}
    return {"job_id": jobs.submit("grid", work, req.dict())}


@app.post("/api/data/update")
def data_update(req: UpdateReq):
    def work():
        if req.what == "etfs":
            res = download_all_etfs(workers=req.workers)
        else:
            res = update_all_stocks(workers=req.workers)
        ok = sum(1 for r in res if r["status"] == "ok")
        bad = [r for r in res if str(r["status"]).startswith("error")]
        return {"total": len(res), "ok": ok,
                "failed": [{"code": r["code"], "status": r["status"]} for r in bad][:20]}
    return {"job_id": jobs.submit("data_update", work, req.dict())}


@app.get("/api/jobs")
def list_jobs(kind: Optional[str] = None):
    return _clean(jobs.list_jobs(kind))


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return _clean(j)


# ---------------- 静态页面 ----------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))
