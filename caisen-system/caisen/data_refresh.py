"""Explicit, auditable refresh of one local stock or ETF history file."""

from __future__ import annotations

import hashlib
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
    if frame is None or frame.empty:
        raise ValueError("行情源返回空数据，拒绝覆盖本地历史")
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
    if "code" in bars:
        expected_code = str(code).zfill(6)
        provided_codes = bars["code"].map(_canonical_code)
        if provided_codes.isna().any() or not provided_codes.eq(expected_code).all():
            mismatched = sorted(set(provided_codes.dropna()))
            raise ValueError(
                f"行情代码与请求标的不一致：请求 {expected_code}，上游包含 {', '.join(mismatched) or '空代码'}"
            )
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in bars]
    if missing:
        raise ValueError("行情缺少字段：" + ", ".join(missing))
    bars = bars[required + (["name"] if "name" in bars else [])]
    bars["date"] = pd.to_datetime(bars.date, errors="coerce")
    for column in required[1:]:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    valid = bars[required].notna().all(axis=1)
    valid &= (bars[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid &= bars.volume >= 0
    valid &= bars.high >= bars[["open", "low", "close"]].max(axis=1)
    valid &= bars.low <= bars[["open", "high", "close"]].min(axis=1)
    bars = bars.loc[valid].sort_values("date", kind="mergesort").drop_duplicates("date", keep="last")
    if bars.empty:
        raise ValueError("行情没有通过 OHLCV 校验的记录")
    if (bars.date.dt.date > date.today()).any():
        raise ValueError("行情含未来日期记录")
    bars.insert(1, "code", code)
    if "name" not in bars:
        bars["name"] = code
    return bars.reset_index(drop=True)


def _refresh_one_unlocked(root: str | Path, code: str, kind: str, provider=None) -> dict:
    """Validate a complete adjusted history before atomically replacing one CSV."""
    if not re.fullmatch(r"\d{6}", str(code)) or kind not in ("stock", "etf"):
        raise ValueError("invalid instrument code or kind")
    root = Path(root)
    path = root / "data" / ("stocks" if kind == "stock" else "etfs") / (code + ".csv")
    if not path.is_file():
        raise FileNotFoundError("本地标的文件不存在")
    old_bytes = path.read_bytes()
    fetch = provider or fetch_history
    provider_frame, fetch_attempts = _fetch_with_retry(fetch, code, kind)
    normalized = _normalize(provider_frame, code, kind)
    if len(normalized) < 20:
        raise ValueError("新数据少于20根有效K线，拒绝覆盖本地历史")
    try:
        previous = _normalize(pd.read_csv(path), code, kind)
    except Exception as exc:
        raise ValueError("无法验证现有行情历史，拒绝覆盖本地文件") from exc
    if previous.empty:
        raise ValueError("无法验证现有行情历史，拒绝覆盖本地文件")
    previous_dates = set(previous.date.dt.date)
    refreshed_dates = set(normalized.date.dt.date)
    missing_dates = previous_dates - refreshed_dates
    if (normalized.date.min() > previous.date.min()
            or normalized.date.max() < previous.date.max()
            or missing_dates):
        raise ValueError(
            "新数据未覆盖本地历史，拒绝覆盖本地文件 "
            f"(本地 {previous.date.min().date()} 至 {previous.date.max().date()}，"
            f"上游 {normalized.date.min().date()} 至 {normalized.date.max().date()}，"
            f"缺少本地交易日 {len(missing_dates)} 个)"
        )
    old_hash = hashlib.sha256(old_bytes).hexdigest()
    new_bytes = normalized.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8-sig")
    new_hash = hashlib.sha256(new_bytes).hexdigest()
    snapshot_path = root / "data" / "market_data_versions" / "snapshots" / kind / (code + "-" + old_hash + ".csv")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot_path.exists():
        _atomic_write(snapshot_path, old_bytes)

    current_dir = root / "data" / "market_data_current"
    current_dir.mkdir(parents=True, exist_ok=True)
    basis_path = current_dir / ".market-data-basis.json"
    previous_basis = basis_path.read_bytes() if basis_path.is_file() else None
    basis = {}
    if basis_path.is_file():
        try:
            basis = json.loads(basis_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            basis = {}
    basis.setdefault("adjustment_by_kind", {})[kind] = "qfq" if kind == "stock" else "raw"
    basis.setdefault("source_apis", {})[kind] = "akshare.stock_zh_a_daily" if kind == "stock" else "akshare.fund_etf_hist_sina"
    basis["refresh_manifest"] = "market_data_versions/latest-refresh.json"
    basis_bytes = json.dumps(basis, ensure_ascii=False, indent=2).encode("utf-8")
    manifest_path = root / "data" / "market_data_versions" / "latest-refresh.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    previous_manifest = manifest_path.read_bytes() if manifest_path.is_file() else None
    manifest = {"source": SOURCE_NAME, "symbols": {}}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    refreshed_at = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    manifest["updated_at"] = refreshed_at
    manifest.setdefault("symbols", {})[f"{kind}:{code}"] = {
        "source": SOURCE_NAME, "adjustment": "qfq" if kind == "stock" else "raw",
        "source_api": basis["source_apis"][kind],
        "volume_unit": "provider_native (ETF documented in 手; stock unit requires verification)",
        "provider_attempts": fetch_attempts,
        "updated_at": refreshed_at,
        "first_date": str(normalized.date.iloc[0].date()), "last_date": str(normalized.date.iloc[-1].date()),
        "rows": len(normalized), "previous_sha256": old_hash, "file_sha256": new_hash,
        "previous_snapshot": str(snapshot_path.relative_to(root / "data")),
        "corporate_actions_verified": False,
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    committed = []
    try:
        _atomic_write(path, new_bytes)
        committed.append((path, old_bytes))
        _atomic_write(basis_path, basis_bytes)
        committed.append((basis_path, previous_basis))
        _atomic_write(manifest_path, manifest_bytes)
        committed.append((manifest_path, previous_manifest))
    except Exception as commit_error:
        rollback_errors = []
        for target, previous in reversed(committed):
            try:
                if previous is None:
                    target.unlink(missing_ok=True)
                else:
                    _atomic_write(target, previous)
            except Exception as rollback_error:  # pragma: no cover - disk failure recovery
                rollback_errors.append(f"{target}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError("刷新提交失败，且回滚未完全成功：" + "; ".join(rollback_errors)) from commit_error
        raise
    return {"code": code, "kind": kind, "rows": len(normalized), "first_date": str(normalized.date.iloc[0].date()),
            "last_date": str(normalized.date.iloc[-1].date()), "previous_sha256": old_hash,
            "file_sha256": new_hash, "adjustment": "qfq" if kind == "stock" else "raw",
            "source": SOURCE_NAME, "previous_snapshot": str(snapshot_path),
            "provider_attempts": fetch_attempts}


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


def _canonical_code(value):
    if pd.isna(value):
        return None
    text = re.sub(r"\.0+$", "", str(value).strip())
    digits = "".join(character for character in text if character.isdigit())
    if not digits:
        return None
    return digits[-6:].zfill(6)


def refresh_one(root: str | Path, code: str, kind: str, provider=None) -> dict:
    """Serialize file/manifest writes so concurrent refreshes cannot clobber metadata."""
    with _REFRESH_LOCK:
        return _refresh_one_unlocked(root, code, kind, provider)
