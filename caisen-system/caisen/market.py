from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, timedelta
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


ALIASES = {"date": ["date", "日期", "交易日期"], "code": ["code", "代码", "证券代码"],
           "name": ["name", "名称", "证券简称"], "open": ["open", "开盘"], "high": ["high", "最高"],
           "low": ["low", "最低"], "close": ["close", "收盘"], "volume": ["volume", "成交量", "成交股数"]}


@dataclass(frozen=True)
class InstrumentFile:
    """One local instrument CSV discovered for analysis."""

    code: str
    kind: str
    path: str
    status: str = "discovered"


@dataclass
class MarketDataResult:
    instrument: InstrumentFile
    bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    source: str = "local_csv"
    file_sha256: Optional[str] = None
    data_version: str = "unknown"
    adjustment_basis: str = "unknown"
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    quality_status: str = "unavailable"
    warnings: list[str] = field(default_factory=list)
    rejected_rows: int = 0
    segments: list[dict] = field(default_factory=list)
    calendar_version: Optional[str] = None
    provisional_bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    source_rows: int = 0
    normalized_rows: int = 0


def discover_instruments(root: str | Path = ".") -> list[InstrumentFile]:
    """List current stock and ETF CSV inputs in deterministic order."""
    root_path = Path(root)
    instruments = []
    for kind, folder in (("stock", "stocks"), ("etf", "etfs")):
        directory = root_path / "data" / folder
        if not directory.is_dir():
            continue
        for path in directory.glob("*.csv"):
            instruments.append(InstrumentFile(code=path.stem, kind=kind, path=str(path)))
    order = {"stock": 0, "etf": 1}
    return sorted(instruments, key=lambda item: (order[item.kind], item.code, item.path))


def load_daily_bars(instrument: InstrumentFile, root: str | Path = ".") -> MarketDataResult:
    """Load a local instrument file with row-level quality and provenance."""
    path = Path(instrument.path)
    warnings = []
    try:
        raw_bytes = path.read_bytes()
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return MarketDataResult(instrument, warnings=["file unavailable: %s" % exc])

    names = {str(column).lstrip("\ufeff").lower(): column for column in frame.columns}
    rename = {}
    for canonical, aliases in ALIASES.items():
        found = next((names[alias.lower()] for alias in aliases if alias.lower() in names), None)
        if found is not None:
            rename[found] = canonical
    normalized = frame.rename(columns=rename).copy()
    source_rows = int(len(normalized))
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in normalized]
    if missing:
        return MarketDataResult(
            instrument,
            source="local_csv",
            file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            quality_status="unavailable",
            warnings=["missing required columns: " + ", ".join(missing)],
            source_rows=source_rows,
        )

    parsed_dates = pd.to_datetime(normalized["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["date"] = parsed_dates
    valid = normalized[required].notna().all(axis=1)
    prices = normalized[["open", "high", "low", "close"]]
    valid &= (prices > 0).all(axis=1)
    valid &= normalized["volume"] >= 0
    valid &= normalized["high"] >= normalized[["open", "low", "close"]].max(axis=1)
    valid &= normalized["low"] <= normalized[["open", "high", "close"]].min(axis=1)
    rejected_rows = int((~valid).sum())
    if rejected_rows:
        warnings.append("rejected %d malformed OHLCV row(s)" % rejected_rows)
    normalized = normalized.loc[valid].copy()

    if "code" in normalized.columns:
        expected = str(instrument.code).strip()
        row_codes = normalized["code"].map(lambda value: _normalize_code(value, expected))
        if (row_codes != expected).any():
            warnings.append("row code mismatch for instrument %s" % expected)
    if "name" in normalized.columns:
        names_in_file = normalized["name"].dropna().astype(str).str.strip()
        if names_in_file.nunique() > 1:
            warnings.append("multiple instrument names in source file")

    duplicate_rows = int(normalized["date"].duplicated(keep="last").sum())
    if duplicate_rows:
        warnings.append("removed %d duplicate date row(s), keeping last" % duplicate_rows)
    normalized = normalized.sort_values("date", kind="mergesort").drop_duplicates("date", keep="last").reset_index(drop=True)
    if normalized.empty:
        return MarketDataResult(
            instrument,
            source="local_csv",
            file_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            quality_status="insufficient",
            warnings=warnings + ["no usable bars"],
            rejected_rows=rejected_rows,
            source_rows=source_rows,
        )

    adjustment, source_api, registry_quality, refresh_metadata = _instrument_metadata(root, instrument, raw_bytes)
    for quality_key, warning in (
        ("volume_unit_verified", "volume unit is not verified for this file hash"),
        ("corporate_actions_verified", "corporate-action adjustment is not verified for this file hash"),
    ):
        if not registry_quality.get(quality_key, False):
            warnings.append(warning)

    prior_close = normalized["close"].shift(1)
    overnight_move = (normalized["open"] / prior_close - 1.0).abs()
    data_breaks = [int(index) for index in normalized.index[(overnight_move > 0.35).fillna(False)]]
    normalized["segment_id"] = 0
    for index in data_breaks:
        normalized.loc[index:, "segment_id"] += 1
    if data_breaks:
        warnings.append("unverified price discontinuity split %d segment(s)" % len(data_breaks))

    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    version_source = file_sha256
    version_material = "market-normalize-v2|%s|%s|%s|%s" % (
        version_source, instrument.kind, instrument.code, adjustment
    )
    data_version = hashlib.sha256(version_material.encode("utf-8")).hexdigest()[:20]
    segments = []
    for segment_id, group in normalized.groupby("segment_id", sort=True):
        segments.append({
            "segment_id": int(segment_id),
            "start_date": str(group["date"].iloc[0].date()),
            "end_date": str(group["date"].iloc[-1].date()),
            "bar_count": int(len(group)),
        })
    if not registry_quality.get("registry_hash_bound", False):
        warnings.append("instrument registry is not bound to the current file hash")
    status = "usable" if not warnings else "usable_with_warnings"
    normalized.attrs["quality"] = {
        **registry_quality,
        "volume_unit_verified": bool(registry_quality.get("volume_unit_verified")),
        "corporate_actions_verified": bool(registry_quality.get("corporate_actions_verified")),
        "data_breaks": data_breaks,
        "quality_status": status,
        "warnings": list(warnings),
        "rejected_rows": rejected_rows,
        "duplicate_rows": duplicate_rows,
    }
    normalized.attrs["provenance"] = {
        "source": "local_csv",
        "source_api": source_api,
        "path": str(path),
        "file_sha256": file_sha256,
        "data_version": data_version,
        "history_basis": "current_snapshot",
        "adjustment_basis": adjustment,
        "latest_date": str(normalized["date"].iloc[-1].date()),
        "refresh": refresh_metadata,
    }
    return MarketDataResult(
        instrument=instrument,
        bars=normalized,
        source="local_csv",
        file_sha256=file_sha256,
        data_version=data_version,
        adjustment_basis=adjustment,
        first_date=str(normalized["date"].iloc[0].date()),
        last_date=str(normalized["date"].iloc[-1].date()),
        quality_status=status,
        warnings=warnings,
        rejected_rows=rejected_rows,
        segments=segments,
        source_rows=source_rows,
        normalized_rows=len(normalized),
    )




def _normalize_code(value, expected):
    if pd.isna(value):
        return ""
    code = re.sub(r"\.0$", "", str(value)).strip()
    if code.isdigit() and expected.isdigit():
        return code.zfill(len(expected))
    return code


def _instrument_metadata(root, instrument, raw_bytes):
    root_path = Path(root)
    basis_path = root_path / "data" / "market_data_current" / ".market-data-basis.json"
    adjustment = "unknown"
    source_api = "unknown"
    refresh_metadata = None
    if basis_path.is_file():
        try:
            basis = json.loads(basis_path.read_text(encoding="utf-8"))
            adjustment = basis.get("adjustment_by_kind", {}).get(instrument.kind, adjustment)
            source_api = basis.get("source_apis", {}).get(instrument.kind, source_api)
        except (OSError, ValueError):
            pass
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    registry_path = Path(os.environ.get("CAISEN_INSTRUMENTS", root_path / "caisen-system" / "config" / "instruments.json"))
    quality = {}
    if registry_path.is_file():
        try:
            entries = json.loads(registry_path.read_text(encoding="utf-8")).get("instruments", {})
            entry = entries.get("%s:%s" % (instrument.kind, instrument.code), {})
            bound = entry.get("data_file_sha256") == raw_hash
            quality = {
                "registry_hash_bound": bool(bound),
                "volume_unit_verified": bool(bound and entry.get("volume_unit_verified")),
                "corporate_actions_verified": bool(bound and entry.get("corporate_actions_verified")),
                "trading_rules_verified": bool(bound and entry.get("trading_rules_verified")),
                "category": entry.get("category", "unclassified"),
                "tick_size": float(entry.get("tick_size", 0.01)),
                "lot_size": int(entry.get("lot_size", 100)),
            }
        except (OSError, ValueError, TypeError):
            quality = {"registry_hash_bound": False}
    return adjustment, source_api, quality, refresh_metadata


def normalize_bars(frame: pd.DataFrame, code: str = "", kind: str = "stock", source: str = "unknown") -> pd.DataFrame:
    names = {str(c).lstrip("\ufeff").lower(): c for c in frame.columns}
    rename = {}
    for canonical, aliases in ALIASES.items():
        found = next((names[a.lower()] for a in aliases if a.lower() in names), None)
        if found is not None:
            rename[found] = canonical
    out = frame.rename(columns=rename).copy()
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out]
    if missing:
        raise ValueError("missing required bar columns: " + ", ".join(missing))
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required).sort_values("date",kind="mergesort").drop_duplicates("date", keep="last").reset_index(drop=True)
    bad = (out["high"] < out[["open", "low", "close"]].max(axis=1)) | (out["low"] > out[["open", "high", "close"]].min(axis=1)) | (out["volume"] < 0)
    if bad.any():
        raise ValueError("invalid OHLCV rows")
    # Large unverified overnight repricings can be splits, bad adjustments, or corrupt rows.
    # Keep the bars for inspection, but isolate both sides so geometry cannot bridge them.
    prior_close = out["close"].shift(1)
    overnight_move = (out["open"] / prior_close - 1.0).abs()
    data_breaks = [int(i) for i in out.index[(overnight_move > 0.35).fillna(False)]]
    out["segment_id"] = 0
    for i in data_breaks:
        out.loc[i:, "segment_id"] += 1
    out.attrs["quality"] = {"volume_unit_verified": False, "corporate_actions_verified": False,
                            "data_breaks": data_breaks, "data_break_rule": "overnight_gap_gt_35pct_v1"}
    out.attrs["provenance"] = {"source": source, "code": code, "kind": kind,
                                 "data_version": hashlib.sha256(pd.util.hash_pandas_object(out, index=True).values.tobytes()).hexdigest()[:16]}
    return out


def load_local_history(code: str, kind: str, root: str = ".") -> Optional[pd.DataFrame]:
    if kind not in ("stock","etf") or not re.fullmatch(r"[A-Za-z0-9_-]{1,16}",str(code)):
        raise ValueError("unsupported instrument identifier or kind")
    folder = "stocks" if kind == "stock" else "etfs"
    path = os.path.join(root, "data", folder, "%s.csv" % code)
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    out = normalize_bars(frame, code=code, kind=kind, source="legacy_csv")
    with open(path, "rb") as handle:
        out.attrs["provenance"]["file_sha256"] = hashlib.sha256(handle.read()).hexdigest()
    out.attrs["provenance"]["cutoff"] = str(out["date"].max().date()) if len(out) else None
    registry=os.environ.get("CAISEN_INSTRUMENTS",os.path.join(root,"caisen-system","config","instruments.json"))
    if os.path.isfile(registry):
        with open(registry,"r",encoding="utf-8") as handle:
            entries=json.load(handle).get("instruments",{})
        entry=entries.get("%s:%s"%(kind,code),{})
        bound=entry.get("data_file_sha256")==out.attrs["provenance"]["file_sha256"]
        out.attrs["quality"].update({
            "volume_unit_verified":bool(bound and entry.get("volume_unit_verified")),
            "corporate_actions_verified":bool(bound and entry.get("corporate_actions_verified")),
            "registry_hash_bound":bool(bound),
            "tick_size":float(entry.get("tick_size",0.01)),
            "lot_size":int(entry.get("lot_size",100)),
            "settlement_days":int(entry.get("settlement_days",1)),
            "trading_rules_verified":bool(bound and entry.get("trading_rules_verified")),
            "price_limit_pct":entry.get("price_limit_pct"),
            "category":entry.get("category","unclassified"),
        })
    return out


class ChinaXSHGCalendar:
    """Official XSHG closure dates for verified years; library dates are previews only."""

    def __init__(self):
        data_path = Path(__file__).parent / "data" / "xshg-official-closures.json"
        self._official_data = json.loads(data_path.read_text(encoding="utf-8"))
        self._verified_years = {int(year) for year in self._official_data["verified_years"]}
        self._closed_by_year = {
            int(year): tuple((date.fromisoformat(first), date.fromisoformat(last))
                             for first, last in details["closed_ranges"])
            for year, details in self._official_data["years"].items()
        }
        try:
            import exchange_calendars as exchange_calendars
            self._library_version = exchange_calendars.__version__
            self._calendar = exchange_calendars.get_calendar(
                "XSHG", start="2000-01-01", end="2025-12-31"
            )
        except ImportError:  # pragma: no cover - optional preview dependency
            self._library_version = "unavailable"
            self._calendar = None
        self._week_cache = {}
        self.version = "xshg-sse-official-2020-2026-v1+preview-library-%s" % self._library_version

    def last_session_of_week(self, week_start):
        sessions = self.sessions_of_week(week_start)
        return sessions[-1] if sessions else None

    def first_session_of_week(self, week_start):
        sessions = self.sessions_of_week(week_start)
        return sessions[0] if sessions else None

    def sessions_of_week(self, week_start):
        start = pd.Timestamp(week_start).date()
        if start in self._week_cache:
            return self._week_cache[start]
        sessions = []
        for offset in range(5):
            current = start + timedelta(days=offset)
            if current.year < 2000 or current.year > 2026:
                return None
            if current.year in self._verified_years:
                is_session = current.weekday() < 5 and not any(
                    first <= current <= last for first, last in self._closed_by_year[current.year]
                )
            else:
                if self._calendar is None:
                    return None
                is_session = bool(self._calendar.is_session(pd.Timestamp(current)))
            if is_session:
                sessions.append(pd.Timestamp(current))
        result = tuple(sessions)
        self._week_cache[start] = result
        return result

    def is_week_verified(self, week_start):
        return pd.Timestamp(week_start).year in self._verified_years


_DEFAULT_CHINA_CALENDAR = None


def default_china_calendar():
    """Return the configured calendar, or None when its dependency is unavailable."""
    global _DEFAULT_CHINA_CALENDAR
    if _DEFAULT_CHINA_CALENDAR is None:
        try:
            _DEFAULT_CHINA_CALENDAR = ChinaXSHGCalendar()
        except (ImportError, RuntimeError, ValueError):
            return None
    return _DEFAULT_CHINA_CALENDAR


class ObservedFridayCalendar:
    """Conservative fallback; short holiday weeks stay provisional."""

    version = "observed-friday-fallback-v1"

    def last_session_of_week(self, week_start):
        return pd.Timestamp(week_start) + pd.Timedelta(days=4)


def aggregate_weekly(bars, calendar=None):
    """Aggregate complete weeks; preserve unfinished weeks as preview bars.

    DataFrame input returns a DataFrame for existing callers. MarketDataResult
    input returns a MarketDataResult with completed and provisional bars.
    """
    is_result = isinstance(bars, MarketDataResult)
    daily_result = bars if is_result else None
    frame = (bars.bars if is_result else bars).copy()
    if frame.empty:
        return _weekly_result(daily_result, frame, frame, calendar, []) if is_result else frame
    frame["date"] = pd.to_datetime(frame["date"])
    calendar_version = getattr(calendar, "version", "unknown") if calendar is not None else "unknown"
    if isinstance(bars, pd.DataFrame):
        source_attrs = bars.attrs
    else:
        source_attrs = bars.bars.attrs
    latest_available = min(frame["date"].max(), pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None))
    grouping = frame.copy()
    grouping["week_start"] = grouping["date"].dt.normalize() - pd.to_timedelta(grouping["date"].dt.weekday, unit="D")
    if "segment_id" not in grouping:
        grouping["segment_id"] = 0
    first_available_by_week = grouping.groupby("week_start", sort=False)["date"].min().dt.normalize()
    columns = ["week_start", "segment_id", "date", "open", "high", "low", "close", "volume"]
    if calendar is None:
        weekly = grouping.sort_values("date", kind="mergesort").groupby(["week_start", "segment_id"], sort=True).agg(
            date=("date", "max"), open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
        ).reset_index()
        weekly["segment_id"] = weekly["segment_id"].astype(int)
        completed = pd.DataFrame(columns=columns)
        preview = weekly[columns].reset_index(drop=True)
    else:
        completed_rows, preview_rows = [], []
        unverified_calendar_weeks = False
        for (week_start, segment_id), group in grouping.groupby(["week_start", "segment_id"], sort=True):
            try:
                final_session = calendar.last_session_of_week(pd.Timestamp(week_start).date())
                final_session = pd.Timestamp(final_session).normalize() if final_session is not None else None
                first_fn = getattr(calendar, "first_session_of_week", None)
                first_value = first_fn(pd.Timestamp(week_start).date()) if first_fn else week_start
                first_session = pd.Timestamp(first_value).normalize() if first_value is not None else None
            except (TypeError, ValueError, KeyError):
                final_session = first_session = None
            row = {
                "week_start": pd.Timestamp(week_start),
                "segment_id": int(segment_id),
                "date": pd.Timestamp(group["date"].max()),
                "open": float(group.iloc[0]["open"]),
                "high": float(group["high"].max()),
                "low": float(group["low"].min()),
                "close": float(group.iloc[-1]["close"]),
                "volume": float(group["volume"].sum()),
            }
            first_available_date = first_available_by_week.loc[week_start]
            verified = getattr(calendar, "is_week_verified", lambda _: True)(week_start)
            if not verified:
                unverified_calendar_weeks = True
            complete = (verified and final_session is not None
                        and first_session is not None
                        and first_available_date == first_session
                        and final_session <= latest_available.normalize()
                        and (group["date"].dt.normalize() == final_session).any())
            (completed_rows if complete else preview_rows).append(row)
        completed = pd.DataFrame(completed_rows, columns=columns)
        preview = pd.DataFrame(preview_rows, columns=columns)
    completed_rows = len(completed)
    for output in (completed, preview):
        output.attrs.update(source_attrs)
    warnings = []
    if calendar is None:
        warnings.append("verified exchange calendar unavailable; weekly bars are preview-only")
    elif calendar_version == ObservedFridayCalendar.version:
        warnings.append("weekly calendar uses observed-Friday fallback; holiday sessions are not verified")
    if calendar is not None and not isinstance(bars, pd.DataFrame) and unverified_calendar_weeks:
        warnings.append("historical exchange-calendar sessions are not independently verified; those weeks are preview-only")
    if not completed_rows:
        warnings.append("no completed weekly bars; latest week is preview-only")
    quality = dict(source_attrs.get("quality", {}))
    quality.update({
        "weekly_status": "usable_with_warnings" if completed_rows else "incomplete_week_preview",
        "weekly_calendar_verified": bool(completed_rows) and calendar is not None and calendar_version != ObservedFridayCalendar.version and calendar_version != "unknown",
        "calendar_version": calendar_version,
        "warnings": list(quality.get("warnings", [])) + warnings,
    })
    completed.attrs["quality"] = quality
    completed.attrs["preview_bars"] = preview
    completed.attrs["calendar_version"] = calendar_version
    preview.attrs["quality"] = quality
    if is_result:
        return _weekly_result(daily_result, completed, preview, calendar, warnings)
    return completed


def _weekly_result(daily_result, completed, preview, calendar, warnings):
    if daily_result is None:
        return completed
    calendar_version = getattr(calendar, "version", "unknown") if calendar is not None else "unknown"
    all_warnings = list(daily_result.warnings) + list(warnings)
    if calendar is None:
        all_warnings.append("verified exchange calendar unavailable; weekly bars are preview-only")
    elif calendar_version == ObservedFridayCalendar.version:
        all_warnings.append("weekly calendar uses observed-Friday fallback; holiday sessions are not verified")
    if (calendar is not None and hasattr(calendar, "is_week_verified")
            and not completed.empty and any(
                not calendar.is_week_verified(start)
                for start in completed.get("week_start", [])
            )):
        all_warnings.append("historical exchange-calendar sessions are not independently verified; those weeks are preview-only")
    if completed.empty and "no completed weekly bars; latest week is preview-only" not in all_warnings:
        all_warnings.append("no completed weekly bars; latest week is preview-only")
    status = "insufficient" if completed.empty else ("usable_with_warnings" if all_warnings else "usable")
    digest_source = daily_result.data_version + "|weekly-v2|" + str(calendar_version) + "|" + (_frame_hash(completed) if not completed.empty else "empty")
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:20]
    return MarketDataResult(
        instrument=daily_result.instrument,
        bars=completed,
        source=daily_result.source,
        file_sha256=daily_result.file_sha256,
        data_version=digest,
        adjustment_basis=daily_result.adjustment_basis,
        first_date=str(completed.date.min().date()) if not completed.empty else None,
        last_date=str(completed.date.max().date()) if not completed.empty else None,
        quality_status=status,
        warnings=all_warnings,
        rejected_rows=daily_result.rejected_rows,
        segments=daily_result.segments,
        calendar_version=calendar_version,
        provisional_bars=preview,
        source_rows=daily_result.source_rows,
        normalized_rows=len(completed),
    )


def _frame_hash(frame: pd.DataFrame) -> str:
    """Hash the normalized analysis bars used to derive an aggregate version."""
    columns = [column for column in
               ("week_start", "date", "open", "high", "low", "close", "volume", "segment_id")
               if column in frame.columns]
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes()
    ).hexdigest()


def audit_market_inventory(root: str | Path = ".") -> dict:
    """Read every discovered local CSV and return a reproducible audit summary."""
    instruments = discover_instruments(root)
    files = []
    counts = {"readable": 0, "warning": 0, "insufficient": 0, "failed": 0}
    for item in instruments:
        result = load_daily_bars(item, root=root)
        readable = not result.bars.empty
        warned = result.quality_status == "usable_with_warnings"
        if readable:
            counts["readable"] += 1
        if warned:
            counts["warning"] += 1
        if result.quality_status == "insufficient":
            counts["insufficient"] += 1
        elif result.quality_status == "unavailable":
            counts["failed"] += 1
        files.append({
            "code": item.code,
            "kind": item.kind,
            "path": item.path,
            "quality_status": result.quality_status,
            "readable": readable,
            "warning": warned,
            "first_date": result.first_date,
            "last_date": result.last_date,
            "source_rows": result.source_rows,
            "normalized_rows": result.normalized_rows,
            "rejected_rows": result.rejected_rows,
            "file_sha256": result.file_sha256,
            "data_version": result.data_version,
            "warnings": list(result.warnings),
        })
    return {
        "inventory_at": datetime.now(timezone.utc).isoformat(),
        "discovered": len(instruments),
        "stock_count": sum(item.kind == "stock" for item in instruments),
        "etf_count": sum(item.kind == "etf" for item in instruments),
        **counts,
        "files": files,
    }
