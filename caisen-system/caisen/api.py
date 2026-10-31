import json
import math
import os
import re
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from multiprocessing import get_context

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import pandas as pd

from .backtest import run_backtest
from .fixed_batch import run_fixed_batch
from .fixed_backtest import STRATEGY_VERSION, SUPPORTED_STRATEGY_VERSIONS
from .market import (InstrumentFile, aggregate_weekly, default_china_calendar,
                     discover_instruments, load_daily_bars)
from .patterns import BOOK_PATTERNS, PATTERN_LABELS, current_pattern_snapshot, detect_patterns
from .main_bottoms import detect_main_bottoms
from .migration import SUPPORTED_NEW_PATH_TYPES, migration_status
from .decision import _label as _pattern_display_label, rank_scan_results, synthesize_decision
from .plans import BUY_CANDIDATE_PATTERN_TYPES, build_trade_plan
from .store import (connect, create_scan_run, get_pattern, get_scan_run, interrupt_active_scan_runs,
                    import_legacy_scan_jobs, list_scan_runs, query_scan_results, save_pattern, save_patterns,
                    save_scan_result, update_scan_run)
from . import RULE_VERSION, __version__
from .data_refresh import refresh_one

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEB = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def _lifespan(_app):
    import_legacy_scan_jobs()
    interrupt_active_scan_runs()
    _recover_fixed_backtests()
    try:
        yield
    finally:
        _stop_scan_executors()


app = FastAPI(title="蔡森量价结构研究系统", version=__version__,
              description="市场扫描、形态图表和模拟，不提供券商下单。", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")
_jobs = {}
_fixed_backtest_jobs = {}
_active_scan_executors = {}
_scan_executors_lock = threading.Lock()


def _stop_scan_executors():
    with _scan_executors_lock:
        executors = tuple(_active_scan_executors.items())
    for executor, stop_event in executors:
        stop_event.set()
        executor.shutdown(wait=False, cancel_futures=True)


def _persist_job(job_id,payload):
    with connect() as db:
        db.execute("INSERT INTO jobs(job_id,status,payload) VALUES(?,?,?) "
                   "ON CONFLICT(job_id) DO UPDATE SET status=excluded.status,payload=excluded.payload",
                   (job_id,payload.get("status","unknown"),json.dumps(payload,ensure_ascii=False,allow_nan=False)))


def _load_job_payload(job_id):
    if job_id in _jobs:
        return _jobs[job_id]
    with connect() as db:
        row=db.execute("SELECT payload FROM jobs WHERE job_id=?",(job_id,)).fetchone()
    return json.loads(row[0]) if row else None


class BacktestRequest(BaseModel):
    code: str
    kind: str = "stock"
    timeframe: str = "1d"
    start: str = None
    end: str = None
    cash: float = 100000


def _bars(code, kind, timeframe):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,16}",str(code)):
        raise HTTPException(400,"invalid instrument code")
    folder = "stocks" if kind == "stock" else "etfs"
    path = Path(ROOT) / "data" / folder / ("%s.csv" % code)
    if not path.is_file():
        raise HTTPException(404, "本地历史文件不存在")
    result=load_daily_bars(InstrumentFile(code,kind,str(path)),root=ROOT)
    if timeframe == "1w":
        weekly=aggregate_weekly(result,calendar=default_china_calendar())
        return weekly.bars
    return result.bars


def _market_data(code, kind):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,16}",str(code)) or kind not in ("stock","etf"):
        raise HTTPException(400,"invalid instrument code or kind")
    folder="stocks" if kind=="stock" else "etfs"
    path=Path(ROOT)/"data"/folder/("%s.csv"%code)
    if not path.is_file():
        raise HTTPException(404,"本地历史文件不存在")
    return load_daily_bars(InstrumentFile(code,kind,str(path)),root=ROOT)


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

def _main_bottom_chart_offset(bars, timeframe):
    if bars is None or bars.empty:
        return 0
    years = 10 if timeframe == "1w" else 5
    dates = pd.to_datetime(bars.date)
    cutoff = dates.max() - pd.DateOffset(years=years)
    eligible = [index for index, value in enumerate(dates) if value >= cutoff]
    return eligible[0] if eligible else 0

def _is_main_bottom(pattern):
    evidence=pattern.get("evidence",[]) if isinstance(pattern,dict) else getattr(pattern,"evidence",[])
    return any(item.get("code")=="main_structure" and isinstance(item.get("value"),dict)
               and item["value"].get("role")=="main" for item in evidence)

def _main_selection(pattern):
    evidence=pattern.get("evidence",[]) if isinstance(pattern,dict) else getattr(pattern,"evidence",[])
    return next((item.get("value") for item in evidence if item.get("code")=="main_selection"
                 and isinstance(item.get("value"),dict)),{})

def _decision_patterns_by_timeframe(patterns_by_timeframe):
    """Use only selected events recognized by the causal main-structure engine."""
    terminal={"invalidated","target_reached","followthrough_confirmed"}
    selected={}
    for timeframe,patterns in patterns_by_timeframe.items():
        selected[timeframe]=[pattern for pattern in patterns
                              if getattr(pattern,"pattern_type",None) in SUPPORTED_NEW_PATH_TYPES
                              and _is_main_bottom(pattern)
                              and _main_selection(pattern).get("selected") is True
                              and getattr(pattern,"state",None) not in terminal]
    return selected


def _supported_main_patterns(patterns):
    return [pattern for pattern in patterns
            if getattr(pattern, "pattern_type", None) in SUPPORTED_NEW_PATH_TYPES
            and _is_main_bottom(pattern)]

def _main_bottom_windows(daily_bars, weekly_bars):
    """Limit main-bottom structure search to five daily or ten weekly years."""
    def trailing(frame, years):
        if frame is None or frame.empty:
            return frame.copy() if frame is not None else pd.DataFrame()
        dates = pd.to_datetime(frame.date)
        cutoff = dates.max() - pd.DateOffset(years=years)
        return frame.loc[dates >= cutoff].copy()

    return trailing(daily_bars, 5), trailing(weekly_bars, 10)


def _slice_scan_window(result, params):
    """Apply the requested date window before pattern detection and weekly aggregation."""
    bars=result.bars.copy()
    if bars.empty:
        return result, {"requested_from":params.get("date_from"),"requested_to":params.get("date_to"),
                        "from":None,"to":None,"latest_available":None,"historical":False}
    bars["date"]=pd.to_datetime(bars["date"])
    latest=pd.Timestamp(bars["date"].max()).date().isoformat()
    if params.get("date_from"):
        bars=bars.loc[bars["date"].dt.date >= date.fromisoformat(params["date_from"])].copy()
    if params.get("date_to"):
        bars=bars.loc[bars["date"].dt.date <= date.fromisoformat(params["date_to"])].copy()
    bars.attrs.update(result.bars.attrs)
    start=str(bars["date"].min().date()) if not bars.empty else None
    end=str(bars["date"].max().date()) if not bars.empty else None
    window={"requested_from":params.get("date_from"),"requested_to":params.get("date_to"),
            "from":start,"to":end,"latest_available":latest,
            "historical":bool(end and end < latest)}
    sliced=replace(result,bars=bars,first_date=start,last_date=end)
    return sliced,window


def _history_through(result, cutoff):
    """Keep all historical context through the observation date, excluding future bars."""
    bars=result.bars.copy()
    if cutoff:
        bars["date"]=pd.to_datetime(bars["date"])
        bars=bars.loc[bars["date"].dt.date<=date.fromisoformat(cutoff)].copy()
    bars.attrs.update(result.bars.attrs)
    return replace(result,bars=bars)


def _scan_progress(job_id, count, total, persist_history=False):
    _jobs[job_id]={"status":"running","progress":(count+1)/max(total,1),"scanned":count+1,"total":total}
    if persist_history:
        update_scan_run(job_id, status="running", progress=(count+1)/max(total,1),
                         scanned=count+1, total=total)


def _analyze_scan_instrument(instrument, params, root=None):
    """Analyze one instrument without writing shared scan state or SQLite."""
    root = ROOT if root is None else root
    kind,code=instrument.kind,instrument.code
    try:
        daily_result=load_daily_bars(instrument,root=root)
        daily_background=_history_through(daily_result,params.get("date_to")).bars
        daily_result,analysis_window=_slice_scan_window(daily_result,params)
        daily=daily_result.bars
        if daily.empty:
            return ({"code":code,"kind":kind,"name":code,"category":"unclassified",
                     "action":"INSUFFICIENT_DATA","state":"no_rows_in_date_range",
                     "reason_codes":["no_market_data_in_selected_date_range"],
                     "analysis_window":analysis_window,
                     "pattern_migration":migration_status(),
                     "rank":{"action_priority":4,"confirmed_timeframes":0,"reward_risk":0,"risk_pct":1,
                             "confirmation_ordinal":0,"tie_break":{"kind":kind,"code":code}}}, [])
        weekly_result=aggregate_weekly(daily_result,calendar=default_china_calendar())
        weekly=weekly_result.bars
        category=daily.attrs.get("quality",{}).get("category","unclassified")
        bars_by_tf={"1d":daily,"1w":weekly}
        backgrounds_by_tf={"1d":daily_background,"1w":daily_background}
        quality_by_tf={"1d":daily_result.quality_status,"1w":weekly_result.quality_status}
        warnings_by_tf={"1d":daily_result.warnings,"1w":weekly_result.warnings}
        prices={tf:float(frame.close.iloc[-1]) for tf,frame in bars_by_tf.items() if not frame.empty}
        requested=params.get("timeframes",["1d","1w"])
        main_patterns_by_tf={}
        for tf in requested:
            main_patterns_by_tf[tf]=detect_main_bottoms(
                bars_by_tf[tf],code,kind,tf,
                tick_size=bars_by_tf[tf].attrs.get("quality",{}).get("tick_size",0.01),
                background_bars=backgrounds_by_tf["1d"],
            ) if not bars_by_tf[tf].empty else []
        patterns_by_tf={tf:current_pattern_snapshot(_supported_main_patterns(main_patterns_by_tf[tf]))
                        for tf in requested}
        decision_patterns_by_tf=_decision_patterns_by_timeframe(patterns_by_tf)
        decision=synthesize_decision(code,kind,decision_patterns_by_tf,
            {tf:prices[tf] for tf in requested if tf in prices},
            {tf:quality_by_tf[tf] for tf in requested},
            bars_by_timeframe={tf:bars_by_tf[tf].tail(500 if tf=="1d" else 520) for tf in requested},
            warnings_by_timeframe={tf:warnings_by_tf[tf] for tf in requested})
        all_patterns=[p for plist in patterns_by_tf.values() for p in plist]
        pattern_rows=[]
        main_events=[]
        pending_patterns=[]
        for p in all_patterns:
            structure=next((item.get("value") for item in p.evidence
                            if item.get("code")=="main_structure" and isinstance(item.get("value"),dict)),None)
            is_main=bool(structure and structure.get("role")=="main")
            label=PATTERN_LABELS.get(p.pattern_type,p.pattern_type)
            if is_main:
                label="主结构 · "+label
            state_label=("，未确认" if p.state=="forming" else
                         "，价格确认，量能仅作辅助" if p.state=="confirmed" and is_main and not p.volume_verified else
                         "，价格确认，量能未核验" if p.state=="confirmed" and not p.volume_verified else
                         "，已失效" if p.state=="invalidated" else
                         "，目标已达" if p.state=="target_reached" else
                         "，箱体后续边界已确认（历史）" if p.state=="followthrough_confirmed" else "，已确认")
            payload=p.to_dict()
            payload["display_label"]=label+state_label+("，结构存在歧义" if "ambiguous_structure" in p.flags else "")
            pending_patterns.append(payload)
            summary={"pattern_id":p.pattern_id,"pattern_type":p.pattern_type,"direction":p.direction,
                     "timeframe":p.timeframe,"state":p.state,"display_label":payload["display_label"],
                     "formed_at":p.formed_at,"confirmation_date":p.confirmation_date,
                     "volume_verified":p.volume_verified,"parent_id":p.parent_id,
                     "is_main_structure":is_main,
                     "selected_main":bool(is_main and next((item.get("value",{}).get("selected")
                                                            for item in p.evidence if item.get("code")=="main_selection"),False)),
                     "main_scale":structure.get("scale") if is_main else None,
                     "execution_suppressed":any(item.get("code")=="execution_suppression"
                                                 and isinstance(item.get("value"),dict)
                                                 and item["value"].get("suppressed") is True for item in p.evidence)}
            pattern_rows.append(summary)
            if is_main:
                main_events.append(payload)
        selected_mains=[event for event in main_events if next(
            (item.get("value",{}).get("selected") for item in event.get("evidence",[]) if item.get("code")=="main_selection"),False)]
        selected_main_ids={event["pattern_id"] for event in selected_mains}
        main_children=[summary for summary in pattern_rows if summary.get("parent_id") in selected_main_ids]
        ended_patterns=[summary for summary in pattern_rows if summary.get("state") in
                        ("invalidated","target_reached","followthrough_confirmed")]
        row={**decision,"patterns":pattern_rows,"main_structure_events":main_events,
             "selected_main_structures":selected_mains,"main_children":main_children,
             "ended_patterns":ended_patterns,"category":category,
             "name":str(daily.name.iloc[-1]) if "name" in daily and len(daily) else code,
             "latest_date":str(daily.date.iloc[-1].date()) if len(daily) else None,
             "latest_close":float(daily.close.iloc[-1]) if len(daily) else None,
             "analysis_window":analysis_window,"data_version":daily_result.data_version,
             "data_warnings":daily_result.warnings,
             "data_versions":{"1d":daily_result.data_version,"1w":weekly_result.data_version},
             "quality_status":daily_result.quality_status,"mini_bars":_mini_bars(daily),
             "pattern_migration":migration_status()}
        json.dumps(row,ensure_ascii=False,allow_nan=False)
        return row,pending_patterns
    except Exception as exc:
        return ({"code":code,"kind":kind,"name":code,"category":"unclassified",
                 "action":"INSUFFICIENT_DATA","state":"data_error",
                 "reason_codes":["file_analysis_failed"],"error":str(exc),
                 "pattern_migration":migration_status(),
                 "rank":{"action_priority":4,"confirmed_timeframes":0,"reward_risk":0,"risk_pct":1,
                         "confirmation_ordinal":0,"tie_break":{"kind":kind,"code":code}}}, [])


def _refresh_worker(job_id, params):
    payload={**params,"job_type":"data_refresh","status":"running","progress":0,
             "completed":0,"success_count":0,"failure_count":0,"results":[],"failures":[]}
    _jobs[job_id]=payload
    try:
        inventory=params.get("inventory")
        if inventory is None:
            inventory=[{"code":item.code,"kind":item.kind,"path":str(item.path)}
                       for item in discover_instruments(ROOT)
                       if not params.get("code") or (item.code==params["code"] and item.kind==params["kind"])]
        payload["total"]=len(inventory)
        for index,item in enumerate(inventory):
            try:
                payload["results"].append(refresh_one(ROOT,item["code"],item["kind"]))
            except Exception as error:
                payload["failures"].append({"code":item["code"],"kind":item["kind"],"error":str(error)})
            payload.update(completed=index+1,progress=(index+1)/max(len(inventory),1),
                           success_count=len(payload["results"]),failure_count=len(payload["failures"]))
            if params.get("scope")=="all" and index+1<len(inventory):
                time.sleep(.25)
        payload.update(status="completed",progress=1)
    except Exception as error:
        payload.update(status="failed",error=str(error))


def _fixed_backtest_worker(job_id, params):
    job=_fixed_backtest_jobs[job_id]
    try:
        job.update(status="running",progress=0,completed=0)
        _persist_job(job_id,job)
        instruments=discover_instruments(ROOT)
        instruments=[item for item in instruments if item.kind in params["kinds"]]
        if params.get("limit") is not None:
            instruments=instruments[:params["limit"]]
        requested=params["timeframes"]
        job.update(status="running",total=len(instruments)*len(requested),progress=0,
                   completed=0,failures=[])
        items=[]
        eligible_dates=set()
        trade_starts=[]
        for index,instrument in enumerate(instruments):
            try:
                result=load_daily_bars(instrument,root=ROOT)
                daily=result.bars.copy()
                if params.get("date_to"):
                    daily=daily.loc[pd.to_datetime(daily.date).dt.date<=date.fromisoformat(params["date_to"])].copy()
                daily.attrs.update(result.bars.attrs)
                dates=pd.to_datetime(daily.date,errors="coerce").dropna().drop_duplicates().sort_values().reset_index(drop=True)
                if params.get("date_from"):
                    trade_start=params["date_from"]
                elif params.get("scope")=="recent300" and not dates.empty:
                    trade_start=dates.iloc[max(0,len(dates)-300)].date().isoformat()
                else:
                    trade_start=None
                if trade_start:
                    trade_starts.append(trade_start)
                    eligible_dates.update(value.date().isoformat() for value in dates if value.date().isoformat()>=trade_start)
                for timeframe in requested:
                    items.append({"code":instrument.code,"kind":instrument.kind,"timeframe":timeframe,
                                  "bars":daily.copy(),"cash":100000.0,"calendar":default_china_calendar(),
                                  "trade_start":trade_start})
            except Exception as error:
                for timeframe in requested:
                    job["failures"].append({"code":instrument.code,"kind":instrument.kind,
                                             "timeframe":timeframe,"error":str(error)})
            job.update(completed=min((index+1)*len(requested),job["total"]),
                       progress=.2*(index+1)/max(len(instruments),1))
            if (index+1)%10==0 or index+1==len(instruments):
                _persist_job(job_id,job)
        if not items:
            job.update(status="failed",progress=1,error="所选标的没有可用于回测的行情",completed=job["total"])
            return
        development_cutoff=params.get("development_cutoff")
        recent_window=bool(params.get("date_from") or params.get("scope")=="recent300")
        if development_cutoff is None and eligible_dates and recent_window:
            ordered_dates=sorted(eligible_dates)
            development_cutoff=ordered_dates[min(len(ordered_dates)-1,math.floor(len(ordered_dates)*.7))]
        job.update(development_cutoff=development_cutoff,
                   trade_start_policy=("explicit" if params.get("date_from") else
                                       "last_300_observations" if params.get("scope")=="recent300" else "all_history"),
                   trade_window={"from":min(trade_starts) if trade_starts else None,
                                 "to":max(pd.to_datetime(item["bars"].date).max().date().isoformat()
                                          for item in items if not item["bars"].empty) if any(not item["bars"].empty for item in items) else None})
        _persist_job(job_id,job)
        output_dir=Path(ROOT)/"state"/"fixed_backtests"/job_id
        def report_progress(snapshot):
            total=max(int(snapshot.get("total",0)),1)
            completed=int(snapshot.get("completed",0))
            job.update(completed=min(job["total"],int(job["total"]*.2)+completed),
                       progress=.2+.8*completed/total,
                       current={"code":snapshot.get("current"),"timeframe":snapshot.get("timeframe")})
            paths=_find_fixed_backtest_artifacts(job.get("output_dir"))
            if paths:
                job["output"]=paths
            _persist_job(job_id,job)
        strategy_version=params.get("strategy_version", STRATEGY_VERSION)
        report=run_fixed_batch(items,output_dir,development_cutoff=development_cutoff,
                               max_workers=params["max_workers"],progress_callback=report_progress,
                               strategy_version=strategy_version)
        failures=report.get("batch_execution",{}).get("errors",[])
        job["failures"].extend(failures)
        status="partial_failed" if job["failures"] or failures else "completed"
        job.update(status=status,progress=1,completed=job["total"],
                   strategy_version=report.get("strategy_version"),development_cutoff=report.get("development_cutoff"),
                   summary=report.get("summary",{}),groups=report.get("groups",[]),
                   output=report.get("output",{}))
        _persist_job(job_id,job)
    except Exception as error:
        job.update(status="failed",progress=1,error=str(error))
        paths=_find_fixed_backtest_artifacts(job.get("output_dir"))
        if paths:
            job["output"]=paths
            job["status"]="partial_failed"
        _persist_job(job_id,job)


def _find_fixed_backtest_artifacts(output_dir):
    if not output_dir:
        return None
    root=Path(output_dir)
    if not root.is_dir():
        return None
    html_paths=sorted(root.glob("fixed-backtest-*/fixed-backtest-report.html"))
    if not html_paths:
        return None
    directory=html_paths[-1].parent
    return {"html":str(html_paths[-1].resolve()),
            "json":str((directory/"fixed-backtest-results.json").resolve()),
            "javascript":str((directory/"fixed-backtest-report.js").resolve())}


def _recover_fixed_backtests():
    with connect() as db:
        rows=db.execute("SELECT job_id,payload FROM jobs WHERE status IN ('queued','running')").fetchall()
    for row in rows:
        try:
            job=json.loads(row["payload"])
        except (TypeError,ValueError):
            continue
        if job.get("job_type")!="fixed_backtest":
            continue
        job.update(status="interrupted",error="服务重启时任务仍在运行，已保留已完成账户，不自动续跑")
        paths=_find_fixed_backtest_artifacts(job.get("output_dir"))
        if paths:
            job["output"]=paths
        _fixed_backtest_jobs[row["job_id"]]=job
        _persist_job(row["job_id"],job)


@app.get("/")
def home():
    return FileResponse(os.path.join(WEB, "index.html"))


@app.get("/api/v1/health")
def health():
    return {"status":"ok","system":"caisen-volume-price","rule_version":RULE_VERSION,"app_version":__version__,"broker_orders":False}


@app.get("/api/v1/analysis")
def analysis(code: str, kind: str = "stock", timeframe: str = "1d", request: Request = None):
    if request is not None and any(key in request.query_params for key in ("asof", "as_of")):
        raise HTTPException(422,"历史截止日期分析已移除，请查看当前分析中的历史形态")
    if kind not in ("stock","etf") or timeframe not in ("1d","1w"):
        raise HTTPException(400,"kind must be stock/etf and timeframe must be 1d/1w")
    daily_result=_market_data(code,kind)
    daily=daily_result.bars
    weekly_result=aggregate_weekly(daily_result,calendar=default_china_calendar())
    weekly=weekly_result.bars
    bars_by_tf={"1d":daily,"1w":weekly}
    pattern_bars_by_tf={"1d":daily.tail(500).copy(),"1w":weekly.tail(520).copy()}
    backgrounds_by_tf={"1d":daily,"1w":daily}
    quality_by_tf={"1d":daily_result.quality_status,"1w":weekly_result.quality_status}
    warnings_by_tf={"1d":daily_result.warnings,"1w":weekly_result.warnings}
    price_by_tf={tf:float(frame.close.iloc[-1]) for tf,frame in bars_by_tf.items() if not frame.empty}
    main_patterns_by_tf={}
    for tf,frame in pattern_bars_by_tf.items():
        main_patterns_by_tf[tf]=detect_main_bottoms(
            bars_by_tf[tf],code,kind,tf,
            tick_size=bars_by_tf[tf].attrs.get("quality",{}).get("tick_size",0.01),
                        background_bars=daily,
        ) if not bars_by_tf[tf].empty else []
    patterns_by_tf={tf:current_pattern_snapshot(_supported_main_patterns(main_patterns_by_tf[tf]))
                    for tf in main_patterns_by_tf}
    decision_patterns_by_tf=_decision_patterns_by_timeframe(patterns_by_tf)
    chart_offsets={}
    for tf,patterns_for_tf in patterns_by_tf.items():
        legacy_offset=max(0,len(bars_by_tf[tf])-len(pattern_bars_by_tf[tf]))
        main_offset=_main_bottom_chart_offset(bars_by_tf[tf],tf)
        for pattern in patterns_for_tf:
            is_main=any(item.get("code")=="main_structure" and isinstance(item.get("value"),dict)
                        and item["value"].get("role")=="main" for item in pattern.evidence)
            chart_offsets[pattern.pattern_id]=main_offset if is_main else legacy_offset
    patterns = patterns_by_tf[timeframe]
    bars=bars_by_tf[timeframe]
    results=[]
    for pattern in patterns:
        payload=pattern.to_dict(); save_pattern(payload)
        payload["chart_offset"]=chart_offsets.get(pattern.pattern_id,0)
        payload["display_label"]=_pattern_display_label(pattern)
        if pattern.pattern_type in BUY_CANDIDATE_PATTERN_TYPES and pattern.direction == "bullish" and pattern.state == "confirmed" and "ambiguous_structure" not in pattern.flags:
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
    other_patterns=decision_patterns_by_tf[other_tf]
    directions={p.direction for p in decision_patterns_by_tf[timeframe] if p.state=="confirmed"}
    other_directions={p.direction for p in decision_patterns_by_tf[other_tf] if p.state=="confirmed"}
    decision=synthesize_decision(code,kind,decision_patterns_by_tf,price_by_tf,quality_by_tf,bars_by_timeframe=pattern_bars_by_tf,
                                 warnings_by_timeframe=warnings_by_tf)
    decision["patterns"]=[{**pattern.to_dict(),"display_label":_pattern_display_label(pattern),
                           "chart_offset":chart_offsets.get(pattern.pattern_id,0)}
                          for tf in ("1d","1w") for pattern in patterns_by_tf.get(tf,[])]
    latest_date=str(daily.date.iloc[-1].date()) if not daily.empty else None
    return {"code":code,"kind":kind,"timeframe":timeframe,"latest_date":latest_date,"_bars":chart_bars,"_chart_offset":chart_offset,
            "chart_is_preview":chart_is_preview,
            "bar_count":len(bars),"data_provenance":daily.attrs.get("provenance",{}),
            "data_quality":bars.attrs.get("quality",{}),"patterns":results,"decision":decision,
            "pattern_migration":migration_status(),
            "book_pattern_catalog":BOOK_PATTERNS,"timeframe_comparison":{"requested":timeframe,
              "other_timeframe":other_tf,"other_confirmed_patterns":[{"pattern_type":p.pattern_type,"state":p.state,"direction":p.direction,"confirmed_at":p.confirmed_at} for p in other_patterns if p.state=="confirmed"],
              "conflict":bool(directions and other_directions and directions.isdisjoint(other_directions))}}


def _create_scan_executor(max_workers):
    """Use spawn so scan workers do not inherit Uvicorn's threaded process state."""
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn"))


def _scan_worker(job_id, params, root=None, executor_factory=None, analysis_fn=None):
    root = ROOT if root is None else root
    executor_factory = _create_scan_executor if executor_factory is None else executor_factory
    analysis_fn = _analyze_scan_instrument if analysis_fn is None else analysis_fn
    persist_history=False
    try:
        persist_history=get_scan_run(job_id) is not None
        if persist_history:
            update_scan_run(job_id,status="running")
        instruments=discover_instruments(root)
        selected_kinds=params.get("kinds", ["stock", "etf"])
        instruments=[item for item in instruments if item.kind in selected_kinds]
        if params.get("limit") is not None:
            instruments=instruments[:params["limit"]]
        records=[None]*len(instruments)
        pending_patterns=[]
        completed=0
        max_workers=max(1,min(10,len(instruments)))
        executor=executor_factory(max_workers=max_workers)
        stop_event=threading.Event()
        with _scan_executors_lock:
            _active_scan_executors[executor]=stop_event
        try:
            with executor:
                instrument_iter=iter(enumerate(instruments))
                futures={}
                interrupted=False
                for _ in range(min(len(instruments),max_workers*2)):
                    if stop_event.is_set():
                        interrupted=True
                        break
                    index,instrument=next(instrument_iter)
                    try:
                        futures[executor.submit(analysis_fn,instrument,params,root)]=(index,instrument)
                    except RuntimeError:
                        if not stop_event.is_set():
                            raise
                        interrupted=True
                        break
                while futures:
                    for future in tuple(futures):
                        if future.cancelled():
                            futures.pop(future)
                            interrupted=True
                    if not futures:
                        break
                    completed_futures,_=wait(futures,timeout=0.2,return_when=FIRST_COMPLETED)
                    if not completed_futures:
                        continue
                    for future in sorted(completed_futures,key=lambda item:futures[item][0]):
                        index,instrument=futures.pop(future)
                        if future.cancelled():
                            interrupted=True
                            continue
                        row,patterns=future.result()
                        records[index]=row
                        if persist_history:
                            save_scan_result(job_id,index,row)
                        pending_patterns.extend(patterns)
                        completed+=1
                        _scan_progress(job_id,completed-1,len(instruments),persist_history)
                        if len(pending_patterns)>=500:
                            save_patterns(pending_patterns)
                            pending_patterns=[]
                        if interrupted or stop_event.is_set():
                            interrupted=True
                            continue
                        try:
                            next_index,next_instrument=next(instrument_iter)
                        except StopIteration:
                            continue
                        try:
                            futures[executor.submit(analysis_fn,next_instrument,params,root)]=(next_index,next_instrument)
                        except RuntimeError:
                            if not stop_event.is_set():
                                raise
                            interrupted=True
                    if stop_event.is_set():
                        interrupted=True
                if pending_patterns:
                    save_patterns(pending_patterns)
                    pending_patterns=[]
                if interrupted:
                    partial_records=[row for row in records if row is not None]
                    progress=completed/max(len(instruments),1)
                    _jobs[job_id]={"status":"interrupted","progress":progress,"scanned":completed,
                                   "total":len(instruments),"results":partial_records}
                    _persist_job(job_id,_jobs[job_id])
                    if persist_history:
                        update_scan_run(job_id,status="interrupted",progress=progress,
                                         scanned=completed,total=len(instruments))
                    return
        finally:
            with _scan_executors_lock:
                _active_scan_executors.pop(executor,None)
        if pending_patterns:
            save_patterns(pending_patterns)
        records=rank_scan_results(records)
        if persist_history:
            for ordinal,row in enumerate(records):
                save_scan_result(job_id,ordinal,row)
        failed_count=sum(row.get("state")=="data_error" for row in records)
        run_status=("failed" if records and failed_count==len(records) else
                    ("partial_failed" if failed_count else "completed"))
        _jobs[job_id]={"status":run_status,"progress":1,"scanned":len(instruments),"total":len(instruments),"results":records,
                       "inventory":{"inventory_at":datetime.now(timezone.utc).isoformat(),
                                    "stock_count":sum(item.kind=="stock" for item in instruments),
                                    "etf_count":sum(item.kind=="etf" for item in instruments),"source":"local_csv",
                                    "requested_window":{"from":params.get("date_from"),"to":params.get("date_to")},
                                    "warning_count":sum(bool(row.get("data_warnings")) for row in records),
                                    "failed_count":failed_count},
                       "limitations":["当前本地存续标的分析，不代表完整市场历史成分", "量能/企业行动信息未验证；候选仅供观察", "无交易日历时周线只作预览，数据不足不参与信号"]}
        _persist_job(job_id,_jobs[job_id])
        if persist_history:
            update_scan_run(job_id,status=run_status,progress=1,scanned=len(instruments),total=len(instruments),
                             inventory=_jobs[job_id]["inventory"])
    except Exception as exc:
        _jobs[job_id]={"status":"failed","error":str(exc)}
        _persist_job(job_id,_jobs[job_id])
        if persist_history:
            update_scan_run(job_id,status="failed",error=str(exc))


@app.post("/api/v1/scans",status_code=202)
def start_scan(params: dict):
    if not params.get("timeframes"):
        params["timeframes"]=["1d","1w"]
    if any(tf not in ("1d","1w") for tf in params["timeframes"]):
        raise HTTPException(400,"timeframes may contain only 1d and 1w")
    kinds=params.get("kinds",["stock","etf"])
    if not isinstance(kinds,list) or not kinds or any(kind not in ("stock","etf") for kind in kinds):
        raise HTTPException(400,"kinds must contain stock and/or etf")
    params["kinds"]=list(dict.fromkeys(kinds))
    limit=params.get("limit")
    if limit is not None and (not isinstance(limit,int) or isinstance(limit,bool) or limit < 1 or limit > 10000):
        raise HTTPException(400,"limit must be an integer from 1 to 10000")
    for key in ("date_from","date_to"):
        value=params.get(key)
        if value:
            try: date.fromisoformat(value)
            except (TypeError,ValueError): raise HTTPException(400,"date_from/date_to must use YYYY-MM-DD")
    if params.get("date_from") and params.get("date_to") and params["date_from"] > params["date_to"]:
        raise HTTPException(400,"date_from must not be later than date_to")
    job_id=uuid.uuid4().hex
    _jobs[job_id]={"status":"queued","progress":0,"rule_version":RULE_VERSION}
    all_instruments=discover_instruments(ROOT)
    total=sum(item.kind in params["kinds"] for item in all_instruments)
    if params.get("limit") is not None:
        total=min(total,params["limit"])
    create_scan_run(job_id,dict(params),total=total,rule_version=RULE_VERSION,app_version=__version__)
    _persist_job(job_id,_jobs[job_id])
    threading.Thread(target=_scan_worker,args=(job_id,params),daemon=True).start()
    return {"job_id":job_id,"status":"queued"}


@app.get("/api/v1/scans")
def list_scans(page: int = 1):
    if page < 1:
        raise HTTPException(400,"page must be a positive integer")
    items,total=list_scan_runs(limit=50,offset=(page-1)*50)
    return {"items":items,"total":total,"page":page,"page_size":50}


@app.get("/api/v1/scans/{scan_id}/results")
def scan_results(scan_id: str, q: str = None, kind: str = None, category: str = None, action: str = None,
                 pattern_types: str = None, timeframes: str = None, page: int = 1,
                 sort: str = "rank", order: str = "asc"):
    if page < 1:
        raise HTTPException(400,"page must be a positive integer")
    if kind and kind not in ("stock","etf"):
        raise HTTPException(400,"kind must be stock or etf")
    if get_scan_run(scan_id) is None:
        raise HTTPException(404,"scan not found")
    patterns=[value for value in (pattern_types or "").split(",") if value]
    periods=[value for value in (timeframes or "").split(",") if value]
    if any(value not in ("1d","1w") for value in periods):
        raise HTTPException(400,"timeframes may contain only 1d and 1w")
    try:
        items,total=query_scan_results(scan_id,search=q,kind=kind,category=category,action=action,
                                       pattern_types=patterns,timeframes=periods,sort_by=sort,
                                       sort_order=order,limit=50,offset=(page-1)*50)
    except ValueError as exc:
        raise HTTPException(400,str(exc))
    return {"items":items,"total":total,"page":page,"page_size":50,
            "detail_notice":"以下是任务运行时保存的结果；打开标的详情会加载最新行情。"}


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
    threading.Thread(target=_refresh_worker, args=(job_id, job_payload), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "scope": scope}


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str):
    payload=_load_job_payload(job_id)
    if payload is None: raise HTTPException(404,"job not found")
    return payload


@app.get("/api/v1/patterns/{pattern_id}")
def pattern_detail(pattern_id: str):
    pattern=get_pattern(pattern_id)
    if pattern is None: raise HTTPException(404,"pattern not found")
    return pattern


@app.post("/api/v1/backtests")
def backtest(request: BacktestRequest):
    if request.kind not in ("stock","etf") or request.timeframe not in ("1d","1w"):
        raise HTTPException(400,"kind must be stock/etf and timeframe must be 1d/1w")
    if not math.isfinite(request.cash) or request.cash <= 0:
        raise HTTPException(400,"cash must be a positive finite amount")
    for value in (request.start,request.end):
        if value:
            try: date.fromisoformat(value)
            except ValueError: raise HTTPException(400,"start/end must use YYYY-MM-DD")
    if request.start and request.end and request.start > request.end:
        raise HTTPException(400,"start must not be later than end")
    market_result=_market_data(request.code,request.kind)
    bars=market_result.bars.copy()
    latest_available=str(pd.to_datetime(bars.date).max().date()) if not bars.empty else None
    if request.start:
        bars=bars.loc[pd.to_datetime(bars.date).dt.date >= date.fromisoformat(request.start)].copy()
    if request.end:
        bars=bars.loc[pd.to_datetime(bars.date).dt.date <= date.fromisoformat(request.end)].copy()
    analysis_window={"from":str(pd.to_datetime(bars.date).min().date()) if not bars.empty else None,
                     "to":str(pd.to_datetime(bars.date).max().date()) if not bars.empty else None,
                     "latest_available":latest_available,
                     "historical":bool(not bars.empty and latest_available and str(pd.to_datetime(bars.date).max().date()) < latest_available)}
    result=run_backtest(bars,request.code,request.kind,request.timeframe,request.cash,start=request.start)
    result["analysis_window"]=analysis_window
    bt_id=uuid.uuid4().hex
    with connect() as db: db.execute("INSERT INTO backtests(backtest_id,payload) VALUES(?,?)",(bt_id,json.dumps(result,ensure_ascii=False)))
    result["backtest_id"]=bt_id
    return result


@app.post("/api/v1/fixed-backtests",status_code=202)
def start_fixed_backtest(params: dict):
    strategy_version=params.get("strategy_version", STRATEGY_VERSION)
    if strategy_version not in SUPPORTED_STRATEGY_VERSIONS:
        raise HTTPException(400,"unsupported fixed strategy version")
    kinds=params.get("kinds",["stock","etf"])
    if not isinstance(kinds,list) or not kinds or any(kind not in ("stock","etf") for kind in kinds):
        raise HTTPException(400,"kinds must contain stock and/or etf")
    kinds=list(dict.fromkeys(kinds))
    timeframes=params.get("timeframes",["1d","1w"])
    if not isinstance(timeframes,list) or not timeframes or any(tf not in ("1d","1w") for tf in timeframes):
        raise HTTPException(400,"timeframes may contain only 1d and 1w")
    timeframes=list(dict.fromkeys(timeframes))
    limit=params.get("limit")
    if limit is not None and (not isinstance(limit,int) or isinstance(limit,bool) or limit<1 or limit>10000):
        raise HTTPException(400,"limit must be an integer from 1 to 10000")
    for key in ("date_from","date_to","development_cutoff"):
        value=params.get(key)
        if value:
            try: date.fromisoformat(value)
            except (TypeError,ValueError): raise HTTPException(400,f"{key} must use YYYY-MM-DD")
    if params.get("date_from") and params.get("date_to") and params["date_from"]>params["date_to"]:
        raise HTTPException(400,"date_from must not be later than date_to")
    cash=params.get("cash",100000)
    if not isinstance(cash,(int,float)) or isinstance(cash,bool) or not math.isfinite(cash) or cash!=100000:
        raise HTTPException(400,"fixed strategy uses exactly 100000 per independent account")
    max_workers=params.get("max_workers",1)
    if not isinstance(max_workers,int) or isinstance(max_workers,bool) or max_workers!=1:
        raise HTTPException(400,"fixed strategy currently runs serially with one worker")
    scope=params.get("scope","all_history")
    if scope not in ("all_history","recent300"):
        raise HTTPException(400,"scope must be all_history or recent300")
    normalized={"kinds":kinds,"timeframes":timeframes,"cash":100000.0,
                "max_workers":1,"scope":scope,
                "trade_start_policy":"explicit" if params.get("date_from") else scope,
                "strategy_version":strategy_version}
    for key in ("limit","date_from","date_to","development_cutoff"):
        if params.get(key) is not None:
            normalized[key]=params[key]
    job_id=uuid.uuid4().hex
    output_dir=Path(ROOT)/"state"/"fixed_backtests"/job_id
    _fixed_backtest_jobs[job_id]={"job_id":job_id,"job_type":"fixed_backtest","status":"queued","progress":0,
                                  "completed":0,"total":0,"failures":[],"params":normalized,
                                  "strategy_version":strategy_version,
                                  "output_dir":str(output_dir),"created_at":datetime.now(timezone.utc).isoformat()}
    _persist_job(job_id,_fixed_backtest_jobs[job_id])
    threading.Thread(target=_fixed_backtest_worker,args=(job_id,normalized),daemon=True).start()
    return {"job_id":job_id,"status":"queued"}


@app.get("/api/v1/fixed-backtests")
def list_fixed_backtests(page: int = 1):
    if page < 1:
        raise HTTPException(400,"page must be a positive integer")
    with connect() as db:
        rows=db.execute("SELECT job_id,payload FROM jobs WHERE json_valid(payload) "
                         "AND json_extract(payload,'$.job_type')='fixed_backtest' "
                         "ORDER BY created_at DESC,job_id DESC LIMIT 50 OFFSET ?",
                         ((page-1)*50,)).fetchall()
        total=db.execute("SELECT COUNT(*) FROM jobs WHERE json_valid(payload) "
                         "AND json_extract(payload,'$.job_type')='fixed_backtest'").fetchone()[0]
    items=[]
    for row in rows:
        try: job=json.loads(row["payload"])
        except (TypeError,ValueError): continue
        item={key:job.get(key) for key in ("job_id","status","progress","completed","total",
                                            "params","created_at","error","summary",
                                            "strategy_version","development_cutoff","failures")}
        if job.get("output",{}).get("html"):
            item["report_url"]="/api/v1/fixed-backtests/%s/report"%row["job_id"]
        items.append(item)
    return {"items":items,"total":total,"page":page,"page_size":50}


@app.get("/api/v1/fixed-backtests/{job_id}")
def get_fixed_backtest_job(job_id: str):
    job=_fixed_backtest_jobs.get(job_id)
    if job is None:
        try: job=_load_job_payload(job_id)
        except (TypeError,ValueError): job=None
        if job and job.get("job_type")=="fixed_backtest":
            _fixed_backtest_jobs[job_id]=job
        else:
            job=None
    if job is None:
        raise HTTPException(404,"fixed backtest task not found")
    result={key:value for key,value in job.items() if key not in ("output_dir",)}
    if job.get("output"):
        result["report_url"]="/api/v1/fixed-backtests/%s/report"%job_id
    return result


@app.get("/api/v1/fixed-backtests/{job_id}/report")
def fixed_backtest_report(job_id: str):
    job=_fixed_backtest_jobs.get(job_id)
    if job is None:
        try: job=_load_job_payload(job_id)
        except (TypeError,ValueError): job=None
        if job and job.get("job_type")=="fixed_backtest":
            _fixed_backtest_jobs[job_id]=job
        else:
            job=None
    if job is None:
        raise HTTPException(404,"fixed backtest task not found")
    if job.get("status") not in ("completed","partial_failed","interrupted") or not job.get("output",{}).get("html"):
        raise HTTPException(409,"fixed backtest report is not ready")
    path=Path(job["output"]["html"])
    try:
        content=path.read_text(encoding="utf-8")
    except OSError:
        raise HTTPException(404,"fixed backtest report is unavailable")
    content=content.replace('src="fixed-backtest-report.js"',
                            'src="/static/fixed-backtest-report.js?v=20261029v3"')
    return HTMLResponse(content)
