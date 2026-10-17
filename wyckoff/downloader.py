"""行情下载：A股 + ETF，全部使用新浪数据源。

本机代理会屏蔽东方财富域名，因此强制直连（NO_PROXY）。
"""
from __future__ import annotations

import os

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

warnings.filterwarnings("ignore")

from .data_store import (
    append_history,
    list_symbols,
    load_frame,
    save_frame,
    universe_dir,
)

log = logging.getLogger("wyckoff.downloader")

MAX_RETRIES = 3
RETRY_DELAY = 2.0


# ---------------------------------------------------------------- ETF

def fetch_etf_list() -> pd.DataFrame:
    """场内ETF清单：code(6位)/name/market。"""
    import akshare as ak
    raw = ak.fund_etf_category_sina(symbol="ETF基金")
    df = raw.rename(columns={"代码": "full_code", "名称": "name"})
    df["market"] = df["full_code"].str[:2]
    df["code"] = df["full_code"].str[2:]
    return df[["code", "name", "market"]]


def _normalize(df: pd.DataFrame, code: str, name: str) -> pd.DataFrame:
    out = df.rename(
        columns={"date": "date", "open": "open", "high": "high", "low": "low",
                 "close": "close", "volume": "volume"}
    ).copy()
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = code
    out["name"] = name
    cols = ["date", "code", "name", "open", "high", "low", "close", "volume"]
    return out[cols].sort_values("date").reset_index(drop=True)


def download_etf(code: str, name: str, market: str, retry: int = 0) -> Dict:
    import akshare as ak
    symbol = f"{market}{code}"
    try:
        df = ak.fund_etf_hist_sina(symbol=symbol)
        if df is None or len(df) < 30:
            return {"code": code, "status": "too_short", "rows": 0}
        frame = _normalize(df, code, name)
        save_frame(frame, code, "etf")
        return {"code": code, "status": "ok", "rows": len(frame)}
    except Exception as exc:  # noqa: BLE001
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY * (retry + 1))
            return download_etf(code, name, market, retry + 1)
        return {"code": code, "status": f"error:{str(exc)[:40]}", "rows": 0}


def download_all_etfs(workers: int = 10, force: bool = False,
                      progress_cb: Optional[Callable[[int, int, Dict], None]] = None) -> List[Dict]:
    etf_list = fetch_etf_list()
    have = set(list_symbols("etf")) if not force else set()
    todo = [r for r in etf_list.itertuples() if r.code not in have]
    results: List[Dict] = []
    total = len(todo)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_etf, row.code, row.name, row.market): row
            for row in todo
        }
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if progress_cb:
                progress_cb(i, total, res)
    return results


def update_etf(code: str, name: str, market: str, retry: int = 0) -> Dict:
    """ETF增量：新浪接口只给全量，重新拉取后按日期合并。"""
    return download_etf(code, name, market, retry)


# ---------------------------------------------------------------- 股票

def _stock_symbol_prefix(code: str) -> str:
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def download_stock(code: str, name: str, retry: int = 0) -> Dict:
    import akshare as ak
    try:
        df = ak.stock_zh_a_daily(symbol=_stock_symbol_prefix(code), start_date="19900101",
                                 end_date=datetime.now().strftime("%Y%m%d"))
        if df is None or len(df) < 30:
            return {"code": code, "status": "too_short", "rows": 0}
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"] if "date" in out.columns else out.index)
        frame = _normalize(out, code, name)
        save_frame(frame, code, "stock")
        return {"code": code, "status": "ok", "rows": len(frame)}
    except Exception as exc:  # noqa: BLE001
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY * (retry + 1))
            return download_stock(code, name, retry + 1)
        return {"code": code, "status": f"error:{str(exc)[:40]}", "rows": 0}


def update_stock(code: str, retry: int = 0) -> Dict:
    import akshare as ak
    old = load_frame(code, "stock")
    name = old["name"].iloc[0] if old is not None and "name" in old else code
    if old is None or old.empty:
        return download_stock(code, str(name))
    last_date = old["date"].max()
    if (datetime.now() - last_date).days <= 3:
        return {"code": code, "status": "fresh", "rows": 0}
    try:
        start = (last_date - timedelta(days=10)).strftime("%Y%m%d")
        df = ak.stock_zh_a_daily(symbol=_stock_symbol_prefix(code), start_date=start,
                                 end_date=datetime.now().strftime("%Y%m%d"))
        if df is None or df.empty:
            return {"code": code, "status": "no_new", "rows": 0}
        df["date"] = pd.to_datetime(df["date"] if "date" in df.columns else df.index)
        new_rows = _normalize(df, code, str(name))
        added = append_history(code, "stock", new_rows)
        return {"code": code, "status": "ok", "rows": added}
    except Exception as exc:  # noqa: BLE001
        if retry < MAX_RETRIES:
            time.sleep(RETRY_DELAY * (retry + 1))
            return update_stock(code, retry + 1)
        return {"code": code, "status": f"error:{str(exc)[:40]}", "rows": 0}


def stock_universe_from_csv(path: Optional[str] = None) -> List[Tuple[str, str]]:
    """从 a_stock_codes.csv 读取股票池（沿用旧过滤：排除ST/创业板/北交所）。"""
    from .config import REPO_ROOT
    path = path or str(REPO_ROOT / "a_stock_codes.csv")
    df = pd.read_csv(path)
    df["code"] = df["code"].astype(str).str.zfill(6)
    name_col = "name" if "name" in df.columns else "名称"
    df = df[~df[name_col].astype(str).str.contains("ST|退", na=False)]
    df = df[df["code"].str.startswith(("0", "6"))]
    return list(zip(df["code"], df[name_col].astype(str)))


def update_all_stocks(workers: int = 10,
                      progress_cb: Optional[Callable[[int, int, Dict], None]] = None) -> List[Dict]:
    codes = list_symbols("stock")
    results: List[Dict] = []
    total = len(codes)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(update_stock, c): c for c in codes}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if progress_cb:
                progress_cb(i, total, res)
    return results
