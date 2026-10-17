"""本地行情数据的统一读写层。

CSV schema（与既有 data/stocks 兼容）：日期,代码,名称,开盘,最高,最低,收盘,成交量
内存中统一为英文列：date/open/high/low/close/volume + code/name。
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .config import ETFS_DIR, STOCKS_DIR

CN_TO_EN = {
    "日期": "date", "代码": "code", "名称": "name",
    "开盘": "open", "最高": "high", "最低": "low",
    "收盘": "close", "成交量": "volume",
}
EN_TO_CN = {v: k for k, v in CN_TO_EN.items()}
EN_COLS = list(CN_TO_EN.values())


def universe_dir(kind: str) -> Path:
    if kind == "stock":
        return STOCKS_DIR
    if kind == "etf":
        return ETFS_DIR
    raise ValueError(f"unknown kind: {kind}")


def list_symbols(kind: str, min_rows: int = 0) -> List[str]:
    base = universe_dir(kind)
    if not base.exists():
        return []
    codes = sorted(p.stem for p in base.glob("*.csv"))
    if min_rows <= 0:
        return codes
    out = []
    for code in codes:
        try:
            n = sum(1 for _ in open(base / f"{code}.csv", "rb")) - 1
            if n >= min_rows:
                out.append(code)
        except OSError:
            continue
    return out


def all_symbols() -> List[tuple]:
    """[(code, kind), ...] 全市场（股票+ETF）。"""
    return [(c, "stock") for c in list_symbols("stock")] + [
        (c, "etf") for c in list_symbols("etf")
    ]


_cache: Dict[str, pd.DataFrame] = {}
_cache_lock = threading.Lock()


def load_frame(code: str, kind: str, use_cache: bool = False) -> Optional[pd.DataFrame]:
    path = universe_dir(kind) / f"{code}.csv"
    if not path.exists():
        return None
    if use_cache:
        with _cache_lock:
            hit = _cache.get(str(path))
            if hit is not None:
                return hit.copy()
    df = pd.read_csv(path, encoding="utf-8-sig")
    rename = {c: CN_TO_EN[c] for c in df.columns if c in CN_TO_EN}
    df = df.rename(columns=rename)
    missing = [c for c in ("date", "open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["code"] = code
    if "name" not in df.columns:
        df["name"] = code
    df = df[["date", "code", "name", "open", "high", "low", "close", "volume"]]
    if use_cache:
        with _cache_lock:
            _cache[str(path)] = df.copy()
    return df


def load_symbol(code: str, kind: Optional[str] = None, **kw) -> Optional[pd.DataFrame]:
    """按代码加载，kind 缺省时自动探测股票/ETF目录。"""
    kinds = [kind] if kind else ["stock", "etf"]
    for k in kinds:
        df = load_frame(code, k, **kw)
        if df is not None:
            return df
    return None


def save_frame(df: pd.DataFrame, code: str, kind: str) -> Path:
    out = universe_dir(kind) / f"{code}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cn = df.rename(columns={c: EN_TO_CN[c] for c in df.columns if c in EN_TO_CN})
    cols = [EN_TO_CN[c] for c in EN_COLS if c in df.columns]
    cn[cols].to_csv(out, index=False, encoding="utf-8-sig")
    return out


def append_history(code: str, kind: str, new_rows: pd.DataFrame) -> int:
    """增量追加，按日期去重；返回新增行数。"""
    old = load_frame(code, kind)
    if old is None:
        base = new_rows.copy()
    else:
        base = pd.concat([old, new_rows], ignore_index=True)
    base = base.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    before = 0 if old is None else len(old)
    save_frame(base, code, kind)
    return len(base) - before
