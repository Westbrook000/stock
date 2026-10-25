import json
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

from .backtest import run_backtest
from .market import (InstrumentFile, aggregate_weekly, default_china_calendar,
                     discover_instruments, load_daily_bars)
from .patterns import BOOK_PATTERNS, PATTERN_LABELS, current_pattern_snapshot, detect_patterns
from .decision import rank_scan_results, synthesize_decision
from .plans import build_trade_plan
from .store import connect, get_pattern, save_pattern, save_patterns
from . import RULE_VERSION
from .data_refresh import refresh_one

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEB = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def _lifespan(_app):
    _recover_interrupted_refresh_jobs()
    yield


app = FastAPI(title="蔡森量价结构研究系统", version="0.1.0",
              description="形态研究、回放和模拟，不提供券商下单。", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")
_jobs = {}


def _persist_job(job_id,payload):
    with connect() as db:
        db.execute("INSERT OR REPLACE INTO jobs(job_id,status,payload) VALUES(?,?,?)",
                   (job_id,payload.get("status","unknown"),json.dumps(payload,ensure_ascii=False)))


def _persist_refresh_job(job_id, payload, item_update=None, initialize=False):
    """Persist bounded job metadata and O(1) per-symbol refresh checkpoints."""
    summary = {key: value for key, value in payload.items()
               if key not in ("inventory", "results", "failures", "latest_results", "latest_failures")}
    with connect() as db:
        if initialize:
            for instrument in payload.get("inventory", []):
                key = f'{instrument["kind"]}:{instrument["code"]}'
                db.execute(
                    "INSERT OR IGNORE INTO refresh_job_items(job_id,instrument_key,code,kind,path,status,payload) VALUES(?,?,?,?,?,?,?)",
                    (job_id, key, instrument["code"], instrument["kind"], instrument["path"],
                     "pending", json.dumps(instrument, ensure_ascii=False)))
            for status, records in (("success", payload.get("results", [])),
                                    ("failure", payload.get("failures", []))):
                for record in records:
                    key = f'{record["kind"]}:{record["code"]}'
                    db.execute(
                        "UPDATE refresh_job_items SET status=?,payload=? WHERE job_id=? AND instrument_key=?",
                        (status, json.dumps(record, ensure_ascii=False), job_id, key))
        if item_update:
            instrument = item_update["instrument"]
            record = item_update["record"]
            status = item_update["status"]
            key = f'{instrument["kind"]}:{instrument["code"]}'
            db.execute(
                "INSERT INTO refresh_job_items(job_id,instrument_key,code,kind,path,status,payload) VALUES(?,?,?,?,?,?,?) "
                "ON CONFLICT(job_id,instrument_key) DO UPDATE SET status=excluded.status,payload=excluded.payload,path=excluded.path",
                (job_id, key, instrument["code"], instrument["kind"], instrument["path"],
                 status, json.dumps(record, ensure_ascii=False)))
        db.execute("INSERT OR REPLACE INTO jobs(job_id,status,payload) VALUES(?,?,?)",
                   (job_id, summary.get("status", "unknown"), json.dumps(summary, ensure_ascii=False)))


def _load_job_payload(job_id):
    payload = dict(_jobs[job_id]) if job_id in _jobs else None
    with connect() as db:
        if payload is None:
            row = db.execute("SELECT payload FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row:
                return None
            payload = json.loads(row[0])
        items = []
        if payload.get("job_type") == "data_refresh":
            items = db.execute("SELECT code,kind,path,status,payload FROM refresh_job_items WHERE job_id=? ORDER BY rowid",
                               (job_id,)).fetchall()
    if payload.get("job_type") == "data_refresh" and items:
        inventory, results, failures = [], [], []
        for item in items:
            record = json.loads(item["payload"])
            inventory.append({"code": item["code"], "kind": item["kind"], "path": item["path"]})
            if item["status"] == "success":
                results.append(record)
            elif item["status"] == "failure":
                failures.append(record)
        payload.update({"inventory": inventory, "results": results, "failures": failures,
                        "success_count": len(results), "failure_count": len(failures),
                        "latest_results": results[-20:], "latest_failures": failures[-20:]})
        payload["completed"] = len(results) + len(failures)
    return payload


def _recover_interrupted_refresh_jobs():
    """Mark refresh workers interrupted after a process restart, preserving checkpoints."""
    recovered = 0
    with connect() as db:
        rows = db.execute("SELECT job_id,payload FROM jobs WHERE status IN ('queued','running')").fetchall()
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except (TypeError, json.JSONDecodeError):
                continue
            if payload.get("job_type") != "data_refresh":
                continue
            # Migrate refresh jobs written by older versions, which kept the
            # whole inventory and accumulated outcomes in jobs.payload.
            for instrument in payload.get("inventory", []):
                key = f'{instrument["kind"]}:{instrument["code"]}'
                db.execute(
                    "INSERT OR IGNORE INTO refresh_job_items(job_id,instrument_key,code,kind,path,status,payload) VALUES(?,?,?,?,?,?,?)",
                    (row["job_id"], key, instrument["code"], instrument["kind"], instrument["path"],
                     "pending", json.dumps(instrument, ensure_ascii=False)))
            for status, records in (("success", payload.get("results", [])),
                                    ("failure", payload.get("failures", []))):
                for record in records:
                    key = f'{record["kind"]}:{record["code"]}'
                    db.execute(
                        "UPDATE refresh_job_items SET status=?,payload=? WHERE job_id=? AND instrument_key=?",
                        (status, json.dumps(record, ensure_ascii=False), row["job_id"], key))
            payload["status"] = "interrupted"
            payload["error"] = "服务重启时行情更新仍在运行；可恢复未成功的标的"
            summary = {key: value for key, value in payload.items()
                       if key not in ("inventory", "results", "failures", "latest_results", "latest_failures")}
            db.execute("UPDATE jobs SET status=?,payload=? WHERE job_id=?",
                       ("interrupted", json.dumps(summary, ensure_ascii=False), row["job_id"]))
            _jobs[row["job_id"]] = summary
            recovered += 1
    return recovered


class Review(BaseModel):
    verdict: str
    corrected: dict = None
    note: str = ""


class BacktestRequest(BaseModel):
    code: str
    kind: str = "stock"
    timeframe: str = "1d"
    start: str = None
    end: str = None
    cash: float = 100000


def _bars(code, kind, timeframe, asof=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,16}",str(code)):
        raise HTTPException(400,"invalid instrument code")
    folder = "stocks" if kind == "stock" else "etfs"
    path = Path(ROOT) / "data" / folder / ("%s.csv" % code)
    if not path.is_file():
        raise HTTPException(404, "本地历史文件不存在")
    if asof:
        try: date.fromisoformat(asof)
        except ValueError: raise HTTPException(400,"asof must be YYYY-MM-DD")
    result=load_daily_bars(InstrumentFile(code,kind,str(path)),root=ROOT,as_of=asof)
    if timeframe == "1w":
        weekly=aggregate_weekly(result,asof=asof,calendar=default_china_calendar())
        return weekly.bars
    return result.bars


def _market_data(code, kind, asof=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,16}",str(code)) or kind not in ("stock","etf"):
        raise HTTPException(400,"invalid instrument code or kind")
    folder="stocks" if kind=="stock" else "etfs"
    path=Path(ROOT)/"data"/folder/("%s.csv"%code)
    if not path.is_file():
        raise HTTPException(404,"本地历史文件不存在")
    return load_daily_bars(InstrumentFile(code,kind,str(path)),root=ROOT,as_of=asof)


def _quality_status(bars):
    if bars is None or bars.empty:
        return "insufficient"
    quality=bars.attrs.get("quality",{})
    verified=bool(quality.get("volume_unit_verified") and quality.get("corporate_actions_verified") and quality.get("trading_rules_verified"))
    return "usable" if verified else "usable_with_warnings"


def _mini_bars(bars, size=48):
    if bars is None or bars.empty:
        return []
    return [{"date":str(row.date.date()),"open":float(row.open),"high":float(row.high),
             "low":float(row.low),"close":float(row.close),"volume":float(row.volume)}
            for row in bars.tail(size).itertuples()]


def _scan_progress(job_id, count, total):
    _jobs[job_id]={"status":"running","progress":(count+1)/max(total,1),"scanned":count+1,"total":total}


def _refresh_worker(job_id, params):
    try:
        specs = params.get("inventory")
        if specs is None:
            instruments = discover_instruments(ROOT)
            if params.get("code"):
                instruments = [item for item in instruments if item.code == params["code"] and item.kind == params["kind"]]
            specs = [{"code": item.code, "kind": item.kind, "path": item.path} for item in instruments]
        instruments = [InstrumentFile(item["code"], item["kind"], item["path"]) for item in specs]
        results = {f"{item.get('kind')}:{item.get('code')}": item
                   for item in params.get("results", [])}
        failures = {f"{item.get('kind')}:{item.get('code')}": item
                    for item in params.get("failures", [])}
        pending_keys = set(failures) - set(results)
        total = len(instruments)
        inventory = [{"code": item.code, "kind": item.kind, "path": str(item.path)} for item in instruments]
        payload = {**params, "job_type": "data_refresh", "inventory": inventory,
                   "status": "running", "total": total}

        def checkpoint(status="running", error=None, changed_instrument=None):
            ordered_results = [results[f"{item.kind}:{item.code}"] for item in instruments
                               if f"{item.kind}:{item.code}" in results]
            ordered_failures = [failures[f"{item.kind}:{item.code}"] for item in instruments
                                if f"{item.kind}:{item.code}" in failures]
            completed = len(ordered_results) + sum(
                1 for item in ordered_failures if f"{item.get('kind')}:{item.get('code')}" not in pending_keys)
            progress = 1.0 if status == "completed" else completed / max(total, 1)
            payload.update({"status": status, "progress": progress,
                            "completed": completed, "success_count": len(ordered_results),
                            "failure_count": len(ordered_failures), "results": ordered_results,
                            "failures": ordered_failures, "latest_results": ordered_results[-20:],
                            "latest_failures": ordered_failures[-20:]})
            if error is not None:
                payload["error"] = str(error)
            else:
                payload.pop("error", None)
            _jobs[job_id] = dict(payload)
            item_update = None
            if changed_instrument is not None:
                key = f"{changed_instrument.kind}:{changed_instrument.code}"
                if key in results:
                    item_update = {"instrument": {"code": changed_instrument.code,
                                                    "kind": changed_instrument.kind,
                                                    "path": str(changed_instrument.path)},
                                   "status": "success", "record": results[key]}
                elif key in failures:
                    item_update = {"instrument": {"code": changed_instrument.code,
                                                    "kind": changed_instrument.kind,
                                                    "path": str(changed_instrument.path)},
                                   "status": "failure", "record": failures[key]}
            _persist_refresh_job(job_id, payload, item_update=item_update,
                                 initialize=changed_instrument is None and not results and not failures)

        checkpoint()
        for index, instrument in enumerate(instruments, 1):
            key = f"{instrument.kind}:{instrument.code}"
            if key in results:
                continue
            try:
                results[key] = refresh_one(ROOT, instrument.code, instrument.kind)
                failures.pop(key, None)
            except Exception as exc:
                failures[key] = {"code": instrument.code, "kind": instrument.kind, "error": str(exc)}
            pending_keys.discard(key)
            checkpoint(changed_instrument=instrument)
            if params.get("scope") == "all" and index < total:
                time.sleep(0.25)
        checkpoint("completed")
        payload["limitations"] = ["每个标的独立下载与原子替换；失败时保留该标的旧文件",
                                  "股票复权为前复权快照，ETF保持原始价格；系统不会据此声称复权因子、成交量单位或企业行动已核验"]
        _jobs[job_id] = dict(payload)
        _persist_refresh_job(job_id, payload)
    except Exception as exc:
        payload = {**_jobs.get(job_id, params), "job_type": "data_refresh",
                   "status": "failed", "error": str(exc)}
        _jobs[job_id] = payload
        _persist_refresh_job(job_id, payload)


@app.get("/")
def home():
    return FileResponse(os.path.join(WEB, "index.html"))


@app.get("/api/v1/health")
def health():
    return {"status":"ok","system":"caisen-volume-price","rule_version":RULE_VERSION,"broker_orders":False}


@app.get("/api/v1/analysis")
def analysis(code: str, kind: str = "stock", timeframe: str = "1d", asof: str = None):
    if kind not in ("stock","etf") or timeframe not in ("1d","1w"):
        raise HTTPException(400,"kind must be stock/etf and timeframe must be 1d/1w")
    daily_result=_market_data(code,kind,asof)
    daily=daily_result.bars
    weekly_result=aggregate_weekly(daily_result,asof=asof,calendar=default_china_calendar())
    weekly=weekly_result.bars
    bars_by_tf={"1d":daily,"1w":weekly}
    pattern_bars_by_tf={"1d":daily.tail(500).copy(),"1w":weekly.tail(520).copy()}
    quality_by_tf={"1d":daily_result.quality_status,"1w":weekly_result.quality_status}
    warnings_by_tf={"1d":daily_result.warnings,"1w":weekly_result.warnings}
    price_by_tf={tf:float(frame.close.iloc[-1]) for tf,frame in bars_by_tf.items() if not frame.empty}
    patterns_by_tf={}
    for tf,frame in pattern_bars_by_tf.items():
        patterns_by_tf[tf]=current_pattern_snapshot(detect_patterns(frame,code,kind,tf,asof)) if not frame.empty else []
    patterns = patterns_by_tf[timeframe]
    bars=bars_by_tf[timeframe]
    results=[]
    for pattern in patterns:
        payload=pattern.to_dict(); save_pattern(payload)
        if pattern.direction == "bullish" and pattern.state == "confirmed" and "manual_review_required" not in pattern.flags:
            payload["trade_plan"]=build_trade_plan(pattern,float(bars.close.iloc[-1]),bars=bars,
                tick_size=bars.attrs.get("quality",{}).get("tick_size",0.01),lot_size=bars.attrs.get("quality",{}).get("lot_size",100))
        results.append(payload)
    chart_is_preview = False
    chart_frame = bars
    if timeframe == "1w" and bars.empty and not weekly_result.provisional_bars.empty:
        chart_frame = weekly_result.provisional_bars
        chart_is_preview = True
    chart_offset = max(0, len(bars) - len(pattern_bars_by_tf[timeframe])) if not chart_is_preview else 0
    chart_bars=[{"date":str(row.date.date()),"open":float(row.open),"high":float(row.high),"low":float(row.low),"close":float(row.close),"volume":float(row.volume)} for row in chart_frame.itertuples()]
    other_tf="1w" if timeframe=="1d" else "1d"
    other_bars=bars_by_tf[other_tf]
    other_patterns=patterns_by_tf[other_tf]
    directions={p.direction for p in patterns if p.state=="confirmed"}
    other_directions={p.direction for p in other_patterns if p.state=="confirmed"}
    decision=synthesize_decision(code,kind,patterns_by_tf,price_by_tf,quality_by_tf,bars_by_timeframe=pattern_bars_by_tf,
                                 warnings_by_timeframe=warnings_by_tf)
    latest_date=str(daily.date.iloc[-1].date()) if not daily.empty else None
    return {"code":code,"kind":kind,"timeframe":timeframe,"asof":asof or latest_date,"_bars":chart_bars,"_chart_offset":chart_offset,
            "chart_is_preview":chart_is_preview,
            "bar_count":len(bars),"data_provenance":daily.attrs.get("provenance",{}),
            "data_quality":bars.attrs.get("quality",{}),"patterns":results,"decision":decision,
            "book_pattern_catalog":BOOK_PATTERNS,"timeframe_comparison":{"requested":timeframe,
              "other_timeframe":other_tf,"other_confirmed_patterns":[{"pattern_type":p.pattern_type,"state":p.state,"direction":p.direction,"confirmed_at":p.confirmed_at} for p in other_patterns if p.state=="confirmed"],
              "conflict":bool(directions and other_directions and directions.isdisjoint(other_directions))}}


def _scan_worker(job_id, params):
    try:
        instruments=discover_instruments(ROOT)
        if params.get("kind") not in (None,"all"):
            instruments=[item for item in instruments if item.kind==params.get("kind")]
        limit=int(params.get("limit",10000)); instruments=instruments[:max(1,min(limit,10000))]
        records=[]; requested=params.get("timeframes",["1d","1w"]); pending_patterns=[]
        for count,instrument in enumerate(instruments):
            kind,code=instrument.kind,instrument.code
            try:
                daily_result=load_daily_bars(instrument,root=ROOT)
                daily=daily_result.bars
                weekly_result=aggregate_weekly(daily_result,calendar=default_china_calendar())
                weekly=weekly_result.bars
                category=daily.attrs.get("quality",{}).get("category","unclassified")
                if params.get("category") and category!=params["category"]:
                    _scan_progress(job_id,count,len(instruments))
                    continue
                bars_by_tf={"1d":daily,"1w":weekly}
                quality_by_tf={"1d":daily_result.quality_status,"1w":weekly_result.quality_status}
                warnings_by_tf={"1d":daily_result.warnings,"1w":weekly_result.warnings}
                prices={tf:float(frame.close.iloc[-1]) for tf,frame in bars_by_tf.items() if not frame.empty}
                patterns_by_tf={}
                for tf in requested:
                    frame=bars_by_tf[tf].tail(500 if tf=="1d" else 520).copy()
                    patterns_by_tf[tf]=current_pattern_snapshot(detect_patterns(frame,code,kind,tf)) if not frame.empty else []
                decision=synthesize_decision(code,kind,patterns_by_tf,
                    {tf:prices[tf] for tf in requested if tf in prices},
                    {tf:quality_by_tf[tf] for tf in requested},
                    bars_by_timeframe={tf:bars_by_tf[tf].tail(500 if tf=="1d" else 520) for tf in requested},
                    warnings_by_timeframe={tf:warnings_by_tf[tf] for tf in requested})
                all_patterns=[p for plist in patterns_by_tf.values() for p in plist]
                if params.get("pattern_type") and not any(p.pattern_type==params["pattern_type"] for p in all_patterns):
                    _scan_progress(job_id,count,len(instruments))
                    continue
                if params.get("state") and not any(p.state==params["state"] for p in all_patterns):
                    _scan_progress(job_id,count,len(instruments))
                    continue
                confirmation_dates=[str(p.confirmation_date or "") for p in all_patterns if p.state=="confirmed"]
                if params.get("confirmed_after") and not any(d>=params["confirmed_after"] for d in confirmation_dates):
                    _scan_progress(job_id,count,len(instruments))
                    continue
                if params.get("confirmed_before") and not any(d<=params["confirmed_before"] for d in confirmation_dates):
                    _scan_progress(job_id,count,len(instruments))
                    continue
                pattern_rows=[]
                for p in all_patterns:
                    label=PATTERN_LABELS.get(p.pattern_type,p.pattern_type)
                    payload=p.to_dict(); payload["display_label"]=label+ ("，未确认" if p.state=="forming" else ("，价格确认，量能未核验" if p.state=="confirmed" and not p.volume_verified else "，已失效" if p.state=="invalidated" else "，目标已达" if p.state=="target_reached" else "，箱体后续边界已确认（历史）" if p.state=="followthrough_confirmed" else "，已确认")) + ("，待人工复核" if "manual_review_required" in p.flags else "")
                    pending_patterns.append(payload)
                    pattern_rows.append({"pattern_id":p.pattern_id,"pattern_type":p.pattern_type,"direction":p.direction,
                                         "timeframe":p.timeframe,"state":p.state,"display_label":payload["display_label"],
                                         "formed_at":p.formed_at,"confirmation_date":p.confirmation_date,
                                         "volume_verified":p.volume_verified})
                records.append({**decision,"patterns":pattern_rows,"category":category,
                                "name":str(daily.name.iloc[-1]) if "name" in daily and len(daily) else code,
                                "latest_date":str(daily.date.iloc[-1].date()) if len(daily) else None,
                                "latest_close":float(daily.close.iloc[-1]) if len(daily) else None,
                                "data_version":daily_result.data_version,"data_warnings":daily_result.warnings,
                                "quality_status":daily_result.quality_status,"mini_bars":_mini_bars(daily)})
            except Exception as exc:
                records.append({"code":code,"kind":kind,"action":"INSUFFICIENT_DATA","state":"data_error",
                                "reason_codes":["file_analysis_failed"],"error":str(exc),
                                "rank":{"action_priority":4,"confirmed_timeframes":0,"reward_risk":0,"risk_pct":1,
                                        "confirmation_ordinal":0,"tie_break":{"kind":kind,"code":code}}})
            _scan_progress(job_id,count,len(instruments))
            if len(pending_patterns)>=500:
                save_patterns(pending_patterns)
                pending_patterns=[]
        if pending_patterns:
            save_patterns(pending_patterns)
        records=rank_scan_results(records)
        _jobs[job_id]={"status":"completed","progress":1,"scanned":len(instruments),"total":len(instruments),"results":records,
                       "inventory":{"inventory_at":datetime.now(timezone.utc).isoformat(),
                                    "stock_count":sum(item.kind=="stock" for item in instruments),
                                    "etf_count":sum(item.kind=="etf" for item in instruments),"source":"local_csv",
                                    "warning_count":sum(bool(row.get("data_warnings")) for row in records),
                                    "failed_count":sum(row.get("state")=="data_error" for row in records)},
                       "limitations":["当前本地存续标的回放，不代表完整市场历史成分", "量能/企业行动信息未验证；候选仍需人工复核", "无交易日历时周线只作预览，数据不足不参与信号"]}
        _persist_job(job_id,_jobs[job_id])
    except Exception as exc:
        _jobs[job_id]={"status":"failed","error":str(exc)}
        _persist_job(job_id,_jobs[job_id])


@app.post("/api/v1/scans",status_code=202)
def start_scan(params: dict):
    if params.get("kind") not in (None,"all","stock","etf"):
        raise HTTPException(400,"kind must be all/stock/etf")
    if not params.get("timeframes"):
        params["timeframes"]=["1d","1w"]
    if any(tf not in ("1d","1w") for tf in params["timeframes"]):
        raise HTTPException(400,"timeframes may contain only 1d and 1w")
    job_id=uuid.uuid4().hex
    _jobs[job_id]={"status":"queued","progress":0}
    _persist_job(job_id,_jobs[job_id])
    threading.Thread(target=_scan_worker,args=(job_id,params),daemon=True).start()
    return {"job_id":job_id,"status":"queued"}


@app.post("/api/v1/data/refresh", status_code=202)
def start_data_refresh(params: dict):
    code = params.get("code")
    kind = params.get("kind")
    if code is not None and (kind not in ("stock", "etf") or not re.fullmatch(r"\d{6}", str(code))):
        raise HTTPException(400, "单标的更新需要6位代码及 stock/etf 类型")
    if code is not None:
        folder = "stocks" if kind == "stock" else "etfs"
        if not (Path(ROOT) / "data" / folder / f"{code}.csv").is_file():
            raise HTTPException(404, "本地行情目录中不存在该标的，无法更新")
    if code is None and params.get("scope") != "all":
        raise HTTPException(400, "请指定单个代码，或 scope=all 扫描全部本地标的")
    if code is not None:
        path = Path(ROOT) / "data" / ("stocks" if kind == "stock" else "etfs") / f"{code}.csv"
        instruments = [InstrumentFile(str(code), kind, str(path))]
        scope = "single"
    else:
        instruments = discover_instruments(ROOT)
        scope = "all"
    inventory = [{"code": item.code, "kind": item.kind, "path": str(item.path)} for item in instruments]
    job_id = uuid.uuid4().hex
    job_payload = {"job_type": "data_refresh", "status": "queued", "progress": 0,
                   "completed": 0, "total": len(inventory), "scope": scope,
                   "inventory": inventory, "results": [], "failures": [],
                   "success_count": 0, "failure_count": 0}
    _jobs[job_id] = job_payload
    _persist_refresh_job(job_id, job_payload, initialize=True)
    threading.Thread(target=_refresh_worker, args=(job_id, job_payload), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "scope": scope}


@app.post("/api/v1/jobs/{job_id}/resume", status_code=202)
def resume_data_refresh(job_id: str):
    payload = _load_job_payload(job_id)
    if payload is None:
        raise HTTPException(404, "job not found")
    if payload.get("job_type") != "data_refresh":
        raise HTTPException(409, "only data refresh jobs can be resumed")
    resumable = payload.get("status") in ("interrupted", "failed") or (
        payload.get("status") == "completed" and int(payload.get("failure_count", 0)) > 0
    )
    if not resumable:
        raise HTTPException(409, "data refresh job has no interrupted or failed symbols")
    if not payload.get("inventory"):
        raise HTTPException(409, "job has no saved inventory checkpoint")
    payload["status"] = "queued"
    payload.pop("error", None)
    _jobs[job_id] = payload
    _persist_refresh_job(job_id, payload)
    threading.Thread(target=_refresh_worker, args=(job_id, payload), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "scope": payload.get("scope"), "resumed": True}


@app.get("/api/v1/jobs/{job_id}")
def job(job_id: str):
    payload = _load_job_payload(job_id)
    if payload is None:
        raise HTTPException(404, "job not found")
    return payload


def recoverable_refresh_jobs():
    """List compact refresh-job summaries for recovery without sessionStorage."""
    with connect() as db:
        rows = db.execute(
            "SELECT job_id,status,payload,created_at FROM jobs "
            "WHERE json_extract(payload,'$.job_type')='data_refresh' "
            "AND status IN ('queued','running','interrupted','failed','completed') "
            "ORDER BY created_at DESC LIMIT 100").fetchall()
    found = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except (TypeError, json.JSONDecodeError):
            continue
        status = row["status"]
        failures = int(payload.get("failure_count", 0))
        if status == "completed" and failures == 0:
            continue
        found.append({"job_id": row["job_id"], "job_type": "data_refresh",
                      "status": status, "scope": payload.get("scope", "all"),
                      "progress": payload.get("progress", 0),
                      "completed": payload.get("completed", 0),
                      "total": payload.get("total", 0),
                      "success_count": int(payload.get("success_count", 0)),
                      "failure_count": failures, "error": payload.get("error"),
                      "created_at": row["created_at"]})
    return {"jobs": found}


@app.get("/api/v1/data/refresh/jobs")
def list_recoverable_refresh_jobs():
    return recoverable_refresh_jobs()


@app.get("/api/v1/patterns/{pattern_id}")
def pattern_detail(pattern_id: str):
    pattern=get_pattern(pattern_id)
    if pattern is None: raise HTTPException(404,"pattern not found")
    return pattern


@app.put("/api/v1/patterns/{pattern_id}/review")
def review(pattern_id: str, review_data: Review):
    pattern=get_pattern(pattern_id)
    if pattern is None: raise HTTPException(404,"pattern not found")
    if review_data.verdict not in ("accepted","false_positive","corrected","needs_review"):
        raise HTTPException(400,"unsupported review verdict")
    with connect() as db:
        db.execute("INSERT INTO reviews(pattern_id,verdict,corrected,note) VALUES(?,?,?,?)",
                   (pattern_id,review_data.verdict,json.dumps(review_data.corrected,ensure_ascii=False) if review_data.corrected else None,review_data.note))
    return {"saved":True,"pattern_id":pattern_id,"verdict":review_data.verdict,"review_version":"manual-v1"}


@app.post("/api/v1/backtests")
def backtest(request: BacktestRequest):
    if request.kind not in ("stock","etf") or request.timeframe not in ("1d","1w"):
        raise HTTPException(400,"kind must be stock/etf and timeframe must be 1d/1w")
    for value in (request.start,request.end):
        if value:
            try: date.fromisoformat(value)
            except ValueError: raise HTTPException(400,"start/end must use YYYY-MM-DD")
    market_result=_market_data(request.code,request.kind,request.end)
    bars=market_result.bars
    result=run_backtest(bars,request.code,request.kind,request.timeframe,request.cash,start=request.start)
    bt_id=uuid.uuid4().hex
    with connect() as db: db.execute("INSERT INTO backtests(backtest_id,payload) VALUES(?,?)",(bt_id,json.dumps(result,ensure_ascii=False)))
    result["backtest_id"]=bt_id
    return result
