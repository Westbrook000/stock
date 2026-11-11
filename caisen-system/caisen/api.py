import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from multiprocessing import get_context

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd

from .market import (InstrumentFile, aggregate_weekly, default_china_calendar,
                     discover_instruments, load_daily_bars)
from .patterns import BOOK_PATTERNS, PATTERN_LABELS, current_pattern_snapshot, detect_patterns
from .main_bottoms import detect_main_bottoms
from .daily_incremental import recognize_daily_instances
from .migration import SUPPORTED_NEW_PATH_TYPES, migration_status
from .decision import _label as _pattern_display_label, synthesize_decision
from .assessment import assess_scan
from .plans import BUY_CANDIDATE_PATTERN_TYPES, build_trade_plan
from .store import (connect, create_scan_run, get_pattern, get_scan_run, interrupt_active_scan_runs,
                    import_legacy_scan_jobs, list_scan_runs, query_scan_results,
                    save_pattern, save_patterns, start_saved_scan_assessment,
                    save_scan_result, update_scan_run)
from . import RULE_VERSION, __version__
from .data_refresh import refresh_one

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def _lifespan(_app):
    import_legacy_scan_jobs()
    interrupt_active_scan_runs()
    _recover_fixed_backtests()
    _recover_daily_backtests()
    try:
        yield
    finally:
        _stop_scan_executors()


app = FastAPI(title="蔡森量价结构研究系统", version=__version__,
              description="市场扫描、形态图表和模拟，不提供券商下单。", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")
_jobs = {}
_fixed_backtest_jobs = {}
_daily_backtest_jobs = {}
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


def _assessment_bars(bars_by_timeframe, quality_by_timeframe):
    """Keep incomplete or unavailable OHLC out of formal scan assessments."""
    result = {}
    for timeframe, bars in bars_by_timeframe.items():
        if quality_by_timeframe.get(timeframe) in ("insufficient", "unavailable"):
            result[timeframe] = bars.iloc[0:0].copy()
        else:
            result[timeframe] = bars
    return result


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
                              and ((getattr(pattern,"pattern_type",None) == "bull_flag"
                                    and getattr(pattern,"state",None) == "confirmed"
                                    and any(item.get("code") == "flag_breakout"
                                            for item in getattr(pattern,"evidence",[])))
                                   or (_is_main_bottom(pattern)
                                       and _main_selection(pattern).get("selected") is True))
                              and getattr(pattern,"state",None) not in terminal]
    return selected


def _apply_assessment_to_decision(decision, assessment):
    """Use the shared scan assessment for display while retaining the old plan for reference."""
    decision = dict(decision)
    previous_action = decision.get("action")
    status = assessment.get("status")
    reason = assessment.get("reason")
    if reason is None:
        selected_tf = assessment.get("selected_timeframe")
        reason = (assessment.get("timeframes", {}).get(selected_tf) or {}).get("reason")
    decision["legacy_action"] = previous_action
    decision["assessment"] = assessment
    decision["reason_codes"] = [reason] if reason else []
    if status == "BUY":
        decision["action"] = "CONSIDER_BUY"
        decision["explanation"] = "扫描评估确认有效突破或满足突破回试条件；原交易计划仅作参考。"
    elif status == "FOCUS_NECKLINE":
        decision["action"] = "FOCUS_NECKLINE"
        decision["explanation"] = "形态已成型，等待颈线有效突破。"
    elif status == "NONE":
        decision["action"] = "WAIT" if previous_action == "CONSIDER_BUY" else previous_action
        decision["explanation"] = "当前评估未触发买入或颈线关注条件；保留其他原行动提示。"
    elif status == "INSUFFICIENT_EVIDENCE":
        decision["assessment_notice"] = "缺少必要的扫描时点证据，保留原行动；旧交易计划仅作参考。"
        decision["explanation"] = decision["assessment_notice"]
    return decision


def _supported_main_patterns(patterns):
    return [pattern for pattern in patterns
            if getattr(pattern, "pattern_type", None) in SUPPORTED_NEW_PATH_TYPES
            and (getattr(pattern, "pattern_type", None) == "bull_flag"
                 or _is_main_bottom(pattern))]

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
        instance_events_by_tf={}
        for tf in requested:
            if bars_by_tf[tf].empty:
                main_patterns_by_tf[tf],instance_events_by_tf[tf]=[],[]
            elif tf == "1d":
                # Recognition identity and index anchors depend on the full
                # causal prefix through date_to. date_from limits scan display
                # context only; it must not resample pivots or instance IDs.
                shared=recognize_daily_instances(daily_background,code,kind,
                                                  recognizer="boundary_first")
                main_patterns_by_tf[tf]=shared["instances"]
                instance_events_by_tf[tf]=shared["instance_events"]
            else:
                main_patterns_by_tf[tf]=detect_main_bottoms(
                    bars_by_tf[tf],code,kind,tf,
                    tick_size=bars_by_tf[tf].attrs.get("quality",{}).get("tick_size",0.01),
                    background_bars=backgrounds_by_tf["1d"])
                instance_events_by_tf[tf]=[]
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
            state_label=("，已成型，等待颈线有效突破" if p.state=="forming" and is_main else
                         "，未确认" if p.state=="forming" else
                         "，价格确认，量能仅作辅助" if p.state=="confirmed" and is_main and not p.volume_verified else
                         "，价格确认，量能未核验" if p.state=="confirmed" and not p.volume_verified else
                         "，已失效" if p.state=="invalidated" else
                         "，目标已达" if p.state=="target_reached" else
                         "，箱体后续边界已确认（历史）" if p.state=="followthrough_confirmed" else "，已确认")
            payload=p.to_dict()
            if p.timeframe == "1d":
                events=[event for event in instance_events_by_tf.get("1d", [])
                        if event.get("instance_id") == p.pattern_id]
                latest_event=events[-1] if events else {}
                payload.update({"pattern_instance_id":p.pattern_id,
                                "instance_state_events":events,
                                "entry_eligible":latest_event.get("entry_eligible") is True,
                                "has_entry_event":any(event.get("entry_eligible") is True
                                                      for event in events),
                                **{key:latest_event.get(key) for key in (
                                    "structure_key","boundary_version_id","paired_at","state_at",
                                    "selection_reason","recognition_schema_version","state_schema_version")}})
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
                     "ambiguous_structure":"ambiguous_structure" in p.flags,
                     "execution_suppressed":any(item.get("code")=="execution_suppression"
                                                 and isinstance(item.get("value"),dict)
                                                 and item["value"].get("suppressed") is True for item in p.evidence)}
            if p.timeframe == "1d":
                summary.update({key: payload.get(key) for key in (
                    "pattern_instance_id","structure_key","boundary_version_id","paired_at","state_at",
                    "selection_reason","recognition_schema_version","state_schema_version",
                    "entry_eligible","has_entry_event")})
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
             "pattern_instance_events":instance_events_by_tf.get("1d",[]),
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
        row["_assessment"] = assess_scan(
            all_patterns, _assessment_bars(bars_by_tf, quality_by_tf), analysis_window.get("to"),
            tick_size_by_timeframe={tf: bars_by_tf[tf].attrs.get("quality", {}).get("tick_size", 0.01)
                                    for tf in bars_by_tf},
            background_bars_by_timeframe=backgrounds_by_tf,
            instance_events_by_timeframe={"1d":instance_events_by_tf.get("1d",[])})
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
    return {"status":"ok","system":"caisen-volume-price","rule_version":RULE_VERSION,"app_version":__version__,"broker_orders":False,
            "pattern_migration":migration_status()}


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
    pattern_bars_by_tf={"1d":daily.copy(),"1w":weekly.tail(520).copy()}
    backgrounds_by_tf={"1d":daily,"1w":daily}
    quality_by_tf={"1d":daily_result.quality_status,"1w":weekly_result.quality_status}
    warnings_by_tf={"1d":daily_result.warnings,"1w":weekly_result.warnings}
    price_by_tf={tf:float(frame.close.iloc[-1]) for tf,frame in bars_by_tf.items() if not frame.empty}
    main_patterns_by_tf={}
    instance_events_by_tf={}
    for tf,frame in pattern_bars_by_tf.items():
        if bars_by_tf[tf].empty:
            main_patterns_by_tf[tf], instance_events_by_tf[tf] = [], []
        elif tf == "1d":
            shared = recognize_daily_instances(
                bars_by_tf[tf], code, kind,
                recognizer="boundary_first",
                performance=None)
            main_patterns_by_tf[tf] = shared["instances"]
            instance_events_by_tf[tf] = shared["instance_events"]
        else:
            main_patterns_by_tf[tf]=detect_main_bottoms(
                bars_by_tf[tf],code,kind,tf,
                tick_size=bars_by_tf[tf].attrs.get("quality",{}).get("tick_size",0.01),
                background_bars=daily)
            instance_events_by_tf[tf] = []
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
        if timeframe == "1d":
            payload["pattern_instance_id"] = pattern.pattern_id
            payload["instance_state_events"] = [event for event in instance_events_by_tf.get("1d", [])
                                                  if event.get("instance_id") == pattern.pattern_id]
            latest_event = (payload["instance_state_events"][-1]
                            if payload["instance_state_events"] else {})
            payload["entry_eligible"] = latest_event.get("entry_eligible") is True
            payload["has_entry_event"] = any(event.get("entry_eligible") is True
                                              for event in payload["instance_state_events"])
            payload.update({key:latest_event.get(key) for key in (
                "structure_key","boundary_version_id","paired_at","state_at",
                "selection_reason","recognition_schema_version","state_schema_version")})
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
    as_of_date=str(daily.date.iloc[-1].date()) if not daily.empty else None
    assessment=assess_scan([pattern for items in patterns_by_tf.values() for pattern in items],
                           _assessment_bars(bars_by_tf,quality_by_tf),as_of_date,
                           tick_size_by_timeframe={tf: bars_by_tf[tf].attrs.get("quality", {}).get("tick_size", 0.01)
                                                   for tf in bars_by_tf},
                           background_bars_by_timeframe=backgrounds_by_tf,
                           instance_events_by_timeframe={"1d":instance_events_by_tf.get("1d",[])})
    decision=_apply_assessment_to_decision(decision,assessment)
    decision["patterns"]=[{**pattern.to_dict(),"display_label":_pattern_display_label(pattern),
                           "chart_offset":chart_offsets.get(pattern.pattern_id,0)}
                          for tf in ("1d","1w") for pattern in patterns_by_tf.get(tf,[])]
    latest_date=as_of_date
    return {"code":code,"kind":kind,"timeframe":timeframe,"latest_date":latest_date,"_bars":chart_bars,"_chart_offset":chart_offset,
            "chart_is_preview":chart_is_preview,
            "bar_count":len(bars),"data_provenance":daily.attrs.get("provenance",{}),
            "data_quality":bars.attrs.get("quality",{}),"patterns":results,"decision":decision,
            "assessment":assessment,"pattern_instance_events":instance_events_by_tf.get(timeframe, []),
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
        pending_patterns=[]
        completed=0
        failed_count=0
        warning_count=0
        max_workers=max(1,min(8,len(instruments)))
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
                        assessment=row.pop("_assessment",None)
                        if persist_history:
                            save_scan_result(job_id,index,row,assessment=assessment)
                        pending_patterns.extend(patterns)
                        completed+=1
                        failed_count += int(row.get("state")=="data_error")
                        warning_count += int(bool(row.get("data_warnings")))
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
                    progress=completed/max(len(instruments),1)
                    _jobs[job_id]={"status":"interrupted","progress":progress,"scanned":completed,
                                   "total":len(instruments)}
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
        run_status=("failed" if instruments and failed_count==len(instruments) else
                    ("partial_failed" if failed_count else "completed"))
        _jobs[job_id]={"status":run_status,"progress":1,"scanned":len(instruments),"total":len(instruments),
                       "inventory":{"inventory_at":datetime.now(timezone.utc).isoformat(),
                                    "stock_count":sum(item.kind=="stock" for item in instruments),
                                    "etf_count":sum(item.kind=="etf" for item in instruments),"source":"local_csv",
                                    "requested_window":{"from":params.get("date_from"),"to":params.get("date_to")},
                                    "warning_count":warning_count,
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
    assessment_progress=start_saved_scan_assessment(scan_id)
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
            "assessment_progress":assessment_progress,
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
    run=get_scan_run(job_id)
    if run is not None:
        return {key:run[key] for key in ("run_id","status","progress","scanned","total","inventory","rule_version","app_version","error","created_at","updated_at")}
    payload=_load_job_payload(job_id)
    if payload is None: raise HTTPException(404,"job not found")
    return payload


@app.get("/api/v1/patterns/{pattern_id}")
def pattern_detail(pattern_id: str):
    pattern=get_pattern(pattern_id)
    if pattern is None: raise HTTPException(404,"pattern not found")
    return pattern


def _daily_backtest_worker(job_id, params):
    job=_daily_backtest_jobs[job_id]
    try:
        from .daily_research import run_daily_research
        job.update(status="running",progress=0)
        _persist_job(job_id,job)
        output_dir=Path(params["output_dir"])
        def on_progress(snapshot):
            total=max(int(snapshot.get("total",0)),1)
            completed=max(0,int(snapshot.get("completed",0)))
            stage=snapshot.get("stage")
            progress=min(.98, .05 + .9*completed/total) if total else .05
            active=[dict(item) for item in snapshot.get("active_symbols", [])
                    if isinstance(item, dict)]
            current=(dict(active[0], symbol=active[0].get("code")) if active else
                     ({"symbol": snapshot.get("symbol")} if snapshot.get("symbol") else {}))
            job.update(status="running",progress=progress,stage=stage,
                       completed=completed,total=int(snapshot.get("total",0)),
                       current=current,active_symbols=active,
                       elapsed_seconds=snapshot.get("elapsed_seconds"))
            _persist_job(job_id,job)
        result=run_daily_research(
            universe_path=PROJECT_ROOT/"research"/"universes"/"daily_800_stock_200_etf_v1.json",
            output_dir=output_dir,date_to=params["date_to"],trade_start=params["trade_start"],
            workers=params["workers"],
            progress_callback=on_progress,recognizer=params["recognizer"],
            execution_mode=params["execution_mode"],worker_source_override=params["worker_source"])
        for field in ("strategy_version","recognizer_version","engine_version","source_rule_version","versions"):
            if result.get(field) != params.get(field):
                raise RuntimeError("effective backtest configuration changed before execution: %s" % field)
        artifacts=result.get("artifacts") or {}
        summary=result.get("summary",{})
        universe=result.get("universe") or {}
        total=max(0,int(universe.get("selected_account_count",job.get("total",0))))
        reported_completed=max(0,int(summary.get("account_count",job.get("completed",0))))
        completed=min(reported_completed,total)
        status={"complete":"completed"}.get(result.get("status"),result.get("status","completed"))
        if reported_completed != total and status == "completed":
            status="partial_failed"
        job.update(status=status,progress=1,completed=completed,total=total,
                   current={},active_symbols=[],
                   universe=universe,parameters=result.get("parameters"),
                   summary=summary,artifacts=artifacts,timings=result.get("timings",{}),
                   run_id=result.get("run_id"),strategy_version=result.get("strategy_version"),
                   recognizer_version=result.get("recognizer_version"),engine_version=result.get("engine_version"),
                   source_rule_version=result.get("source_rule_version"),
                   versions=result.get("versions"),
                   recognition_rules=result.get("recognition_rules"),
                   trading_rules=result.get("trading_rules"),
                   effective_parameters=result.get("parameters"),
                   account_files=result.get("account_files",[]))
        _persist_job(job_id,job)
    except Exception as error:
        job.update(status="failed",progress=1,error="%s: %s" % (type(error).__name__, error),
                   current={},active_symbols=[])
        _persist_job(job_id,job)


def _recover_daily_backtests():
    with connect() as db:
        rows=db.execute("SELECT job_id,payload FROM jobs WHERE status IN ('queued','running')").fetchall()
    for row in rows:
        try: job=json.loads(row["payload"])
        except (TypeError,ValueError): continue
        if job.get("job_type")!="daily_backtest": continue
        job.update(status="interrupted",error="服务重启时任务仍在运行；已保留任务记录，需重新运行")
        _daily_backtest_jobs[row["job_id"]]=job
        _persist_job(row["job_id"],job)


def _daily_job_or_404(job_id):
    job=_daily_backtest_jobs.get(job_id)
    if job is None:
        try: job=_load_job_payload(job_id)
        except (TypeError,ValueError): job=None
        if not isinstance(job,dict) or job.get("job_type")!="daily_backtest": job=None
        if job: _daily_backtest_jobs[job_id]=job
    if job is None: raise HTTPException(404,"daily backtest task not found")
    return job


@app.post("/api/v1/backtests",status_code=202)
def start_daily_backtest(params: dict):
    if params:
        raise HTTPException(400,"daily backtest configuration is fixed")
    from .daily_research import resolve_daily_research_config
    try:
        effective = resolve_daily_research_config(execution_mode="spawn")
    except Exception as error:
        raise HTTPException(503,"cannot resolve current daily research configuration: %s: %s" %
                            (type(error).__name__, error)) from error
    job_id=uuid.uuid4().hex
    output_dir=PROJECT_ROOT/"state"/"daily_backtests"/job_id
    normalized={"universe_id":"daily_800_stock_200_etf_v1",
                "date_to":datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
                "trade_start":None,"trade_start_policy":"all_history",
                **effective,
                "strategy":effective["strategy_version"],
                "initial_cash_per_symbol":effective["trading_rules"]["initial_cash_per_symbol"],
                "stop_pct":effective["trading_rules"]["stop_fraction"],
                "target_pct":effective["trading_rules"]["target_fraction"],
                "commission":effective["trading_rules"]["commission"],
                "slippage":effective["trading_rules"]["slippage"],
                "max_hold":effective["trading_rules"]["max_hold_sessions"],
                "end_close":effective["trading_rules"]["end_of_period_close"],
                "output_dir":str(output_dir.resolve())}
    job={"job_id":job_id,"job_type":"daily_backtest","status":"queued","progress":0,
         "completed":0,"total":1000,"params":normalized,
         "strategy_version":effective["strategy_version"],
         "recognizer_version":effective["recognizer_version"],
         "engine_version":effective["engine_version"],
         "source_rule_version":effective["source_rule_version"],
         "versions":effective["versions"],
         "recognition_rules":effective["recognition_rules"],
         "trading_rules":effective["trading_rules"],
         "created_at":datetime.now(timezone.utc).isoformat()}
    _daily_backtest_jobs[job_id]=job
    _persist_job(job_id,job)
    threading.Thread(target=_daily_backtest_worker,args=(job_id,normalized),daemon=True).start()
    return {"job_id":job_id,"status":"queued","params":normalized,
            "strategy_version":effective["strategy_version"],
            "recognizer_version":effective["recognizer_version"],
            "engine_version":effective["engine_version"],
            "source_rule_version":effective["source_rule_version"],
            "versions":effective["versions"]}


@app.get("/api/v1/backtests")
def list_daily_backtests(page: int = 1):
    if page<1: raise HTTPException(400,"page must be a positive integer")
    with connect() as db:
        rows=db.execute("SELECT job_id,payload,created_at FROM jobs WHERE json_valid(payload) "
                        "AND json_extract(payload,'$.job_type')='daily_backtest' "
                        "ORDER BY created_at DESC,job_id DESC LIMIT 50 OFFSET ?",((page-1)*50,)).fetchall()
        total=db.execute("SELECT COUNT(*) FROM jobs WHERE json_valid(payload) "
                         "AND json_extract(payload,'$.job_type')='daily_backtest'").fetchone()[0]
    items=[]
    for row in rows:
        try: job=json.loads(row["payload"])
        except (TypeError,ValueError): continue
        item={key:job.get(key) for key in ("job_id","status","progress","completed","total","params",
                                             "created_at","universe","summary","error","strategy_version",
                                             "recognizer_version","engine_version","source_rule_version",
                                             "recognition_rules","trading_rules","effective_parameters","versions")}
        item["created_at"]=item.get("created_at") or row["created_at"]
        if job.get("artifacts",{}).get("html_path"):
            item["report_url"]="/api/v1/backtests/%s/report"%row["job_id"]
        items.append(item)
    return {"items":items,"total":total,"page":page,"page_size":50}


@app.get("/api/v1/backtests/{job_id}")
def get_daily_backtest(job_id: str):
    job=_daily_job_or_404(job_id)
    result={key:value for key,value in job.items() if key!="job_type"}
    if job.get("artifacts",{}).get("html_path"):
        result["report_url"]="/api/v1/backtests/%s/report"%job_id
    return result


@app.get("/api/v1/backtests/{job_id}/accounts/{filename}")
def daily_backtest_account(job_id: str, filename: str):
    job=_daily_job_or_404(job_id)
    if not re.fullmatch(r"(?:stock|etf)-\d{6}\.json",filename):
        raise HTTPException(404,"daily backtest account not found")
    artifacts=job.get("artifacts") or {}
    account_directory=artifacts.get("account_directory")
    if not account_directory:
        raise HTTPException(404,"daily backtest account not found")
    expected_root=(PROJECT_ROOT/"state"/"daily_backtests"/job_id/"accounts").resolve()
    try: root=Path(account_directory).resolve(strict=True)
    except OSError: raise HTTPException(404,"daily backtest account not found")
    if root!=expected_root or not root.is_dir():
        raise HTTPException(404,"daily backtest account not found")
    path=root/filename
    try: resolved=path.resolve(strict=True)
    except OSError: raise HTTPException(404,"daily backtest account not found")
    if resolved.parent!=root or not resolved.is_file():
        raise HTTPException(404,"daily backtest account not found")
    if "account_files" in job:
        members=set()
        for item in job.get("account_files") or []:
            member=Path(item)
            if not member.is_absolute(): member=root/member
            try: members.add(member.resolve(strict=True))
            except OSError: continue
        if resolved not in members:
            raise HTTPException(404,"daily backtest account not found")
    return FileResponse(str(resolved),media_type="application/json")


@app.get("/api/v1/backtests/{job_id}/report")
def daily_backtest_report(job_id: str):
    job=_daily_job_or_404(job_id)
    if job.get("status") not in ("completed","partial_failed","interrupted") or not job.get("artifacts",{}).get("html_path"):
        raise HTTPException(409,"daily backtest report is not ready")
    try: return HTMLResponse(Path(job["artifacts"]["html_path"]).read_text(encoding="utf-8"))
    except OSError: raise HTTPException(404,"daily backtest report is unavailable")


@app.post("/api/v1/fixed-backtests",status_code=410)
def start_fixed_backtest(_params: dict):
    raise HTTPException(410,"fixed backtest submissions are retired; use /api/v1/backtests")


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
