"""Explicit, auditable refresh of one local stock or ETF history file."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import urllib.error
from datetime import date
from pathlib import Path

import pandas as pd


SOURCE_NAME = "AkShare Sina daily history"
_REFRESH_LOCK = threading.Lock()
PROVIDER_RETRY_ATTEMPTS = 3
PROVIDER_RETRY_BASE_SECONDS = 0.5


class ProviderRetriesExhausted(RuntimeError):
    def __init__(self, attempts: int, cause: Exception):
        self.attempts = attempts
        self.cause = cause
        super().__init__(f"行情源请求重试 {attempts} 次后仍失败：{cause}")


def fetch_history(code: str, kind: str) -> pd.DataFrame:
    """Fetch raw provider rows. Imported lazily so offline analysis still works."""
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover - depends on optional runtime install
        raise RuntimeError("行情更新需要安装 AkShare：pip install -r requirements.txt") from exc
    if kind == "stock":
        if not re.fullmatch(r"\d{6}", code) or code.startswith(("4", "8", "689")):
            raise ValueError("当前数据源暂不支持此股票代码；未修改本地文件")
        symbol = ("sh" if code.startswith("6") else "sz") + code
        return ak.stock_zh_a_daily(symbol=symbol, start_date="19900101",
                                   end_date=date.today().strftime("%Y%m%d"), adjust="qfq")
    if kind == "etf":
        if not re.fullmatch(r"\d{6}", code) or not code.startswith(("1", "5")):
            raise ValueError("当前数据源要求ETF代码为6位且以1（深市）或5（沪市）开头；未修改本地文件")
        symbol = ("sh" if code.startswith("5") else "sz") + code
        return ak.fund_etf_hist_sina(symbol=symbol)
    raise ValueError("kind must be stock or etf")


def _is_transient_provider_error(error: Exception) -> bool:
    """Retry temporary transport and upstream throttling failures only."""
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    status = status or getattr(response, "status_code", None) or getattr(error, "code", None)
    try:
        if int(status) in {408, 425, 429, 500, 502, 503, 504}:
            return True
    except (TypeError, ValueError):
        pass

    error_name = type(error).__name__.lower()
    if isinstance(error, (TimeoutError, ConnectionError, urllib.error.URLError)):
        return True
    if any(token in error_name for token in (
        "timeout", "connectionerror", "connectionreset", "chunkedencoding", "retryerror"
    )):
        return True
    message = str(error).lower()
    return any(token in message for token in (
        "rate limit", "too many requests", "429", "timed out", "timeout",
        "connection reset", "connection aborted", "temporarily unavailable",
        "service unavailable", "bad gateway", "gateway timeout",
    ))


def _fetch_with_retry(fetch, code: str, kind: str) -> pd.DataFrame:
    for attempt in range(PROVIDER_RETRY_ATTEMPTS):
        try:
            return fetch(code, kind), attempt + 1
        except Exception as error:
            if not _is_transient_provider_error(error):
                raise
            if attempt + 1 >= PROVIDER_RETRY_ATTEMPTS:
                raise ProviderRetriesExhausted(attempt + 1, error) from error
            time.sleep(PROVIDER_RETRY_BASE_SECONDS * (2 ** attempt))
    raise RuntimeError("行情源重试意外结束")  # pragma: no cover


def _normalize(frame: pd.DataFrame, code: str, kind: str) -> pd.DataFrame:
    aliases = {
        "date": ("date", "日期", "交易日期"), "open": ("open", "开盘"),
        "high": ("high", "最高"), "low": ("low", "最低"),
        "close": ("close", "收盘"), "volume": ("volume", "成交量", "成交股数"),
        "name": ("name", "名称", "证券简称"), "code": ("code", "代码", "证券代码"),
    }
    source_columns = {str(column).strip().lower(): column for column in frame.columns}
    renamed = {}
    for target, variants in aliases.items():
        match = next((source_columns[v.lower()] for v in variants if v.lower() in source_columns), None)
        if match is not None:
            renamed[match] = target
    bars = frame.rename(columns=renamed).copy()
    bars["date"] = pd.to_datetime(bars["date"]).dt.strftime("%Y-%m-%d")
    bars["code"] = code
    if "name" not in bars:
        bars["name"] = code
    return bars[["date", "code", "name", "open", "high", "low", "close", "volume"]]


def _refresh_one_unlocked(root: str | Path, code: str, kind: str, provider=None) -> dict:
    """Fetch provider history and replace one local CSV without inspecting bar values."""
    if not re.fullmatch(r"\d{6}", str(code)) or kind not in ("stock", "etf"):
        raise ValueError("invalid instrument code or kind")
    root = Path(root)
    path = root / "data" / ("stocks" if kind == "stock" else "etfs") / (code + ".csv")
    if not path.is_file():
        raise FileNotFoundError("本地标的文件不存在")
    fetch = provider or fetch_history
    provider_frame, fetch_attempts = _fetch_with_retry(fetch, code, kind)
    normalized = _normalize(provider_frame, code, kind)
    new_bytes = normalized.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8-sig")
    current_dir = root / "data" / "market_data_current"
    current_dir.mkdir(parents=True, exist_ok=True)
    basis_path = current_dir / ".market-data-basis.json"
    basis = {}
    if basis_path.is_file():
        try:
            basis = json.loads(basis_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            basis = {}
    basis.pop("refresh_manifest", None)
    basis.setdefault("adjustment_by_kind", {})[kind] = "qfq" if kind == "stock" else "raw"
    basis.setdefault("source_apis", {})[kind] = "akshare.stock_zh_a_daily" if kind == "stock" else "akshare.fund_etf_hist_sina"
    basis_bytes = json.dumps(basis, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write(basis_path, basis_bytes)
    _atomic_write(path, new_bytes)
    dates = pd.to_datetime(normalized["date"])
    return {"code": code, "kind": kind, "rows": len(normalized),
            "first_date": str(dates.min().date()) if len(dates) else None,
            "last_date": str(dates.max().date()) if len(dates) else None,
            "adjustment": "qfq" if kind == "stock" else "raw",
            "source": SOURCE_NAME, "provider_attempts": fetch_attempts}


def _atomic_write(path: Path, content: bytes) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as temp:
            temp.write(content)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def refresh_one(root: str | Path, code: str, kind: str, provider=None) -> dict:
    """Serialize market and source-basis writes so concurrent refreshes cannot clobber metadata."""
    with _REFRESH_LOCK:
        return _refresh_one_unlocked(root, code, kind, provider)
