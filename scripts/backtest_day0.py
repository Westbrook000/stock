#!/usr/bin/env python3
"""Fixed-cohort event-entry backtest; optional exit-only profit optimization.

python3 scripts/backtest_day0.py --workers 4
python3 scripts/backtest_day0.py --workers 4 --optimize
No score thresholds, no reuse of old signal caches, no automatic config overwrite.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/stock-day0-matplotlib')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from wyckoff.analysis.engine import run_analysis, STRATEGY_VERSION
from wyckoff.backtest.daily import prepare_daily, simulate_daily
from wyckoff.config import StrategyConfig, REPO_ROOT
from wyckoff.data_store import load_frame, universe_dir

PERIODS = {'full': ('2019-01-01', None), 'train': ('2019-01-01', '2022-12-31'),
           'validation': ('2023-01-01', '2024-12-31'), 'test': ('2025-01-01', None)}
UPSTREAM_COMMIT = 'fdd4a7f0fdccf6c4007da79fd6f3879164462c3c'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(x) for x in value]
    if isinstance(value, (float,np.floating)) and not np.isfinite(value): return None
    if isinstance(value, np.generic): return value.item()
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding='utf-8')


def worker(row, cfg, grid):
    code,kind = str(row['code']), row['kind']
    frame=load_frame(code,kind)
    if frame is None: raise RuntimeError(f'missing data {kind}:{code}')
    first=int(frame.date.searchsorted(pd.Timestamp('2019-01-01')))
    # Full bounded SR lookback plus reversal confirmation/EMA warmup before the report period.
    frame=frame.iloc[max(0,first-600):].reset_index(drop=True)
    if not len(frame) or (frame.close<=0).any(): raise RuntimeError(f'invalid close {kind}:{code}')
    res=run_analysis(frame,cfg,code=code)
    data=prepare_daily(res.features,res.signals)
    outcomes=[]
    for period,(start,end) in PERIODS.items():
        metrics=simulate_daily(data,None,cfg.trade,start=start,end=end,include_equity=True)
        outcomes.append({'code':code,'kind':kind,'period':period,**metrics})
    candidates=[]
    for ident,params in enumerate(grid):
        rule=replace(cfg.trade,**params)
        for period in ('train','validation'):
            start,end=PERIODS[period]
            m=simulate_daily(data,None,rule,start=start,end=end)
            candidates.append({'id':ident,'period':period,'code':code,'kind':kind,
                               **{k:v for k,v in m.items() if k!='trade_list'}})
    counts=pd.Series([e.etype for e in res.events],dtype=str).value_counts().to_dict()
    return outcomes,candidates,{'code':code,'kind':kind,**counts}


def aggregate(frame):
    gp=float(frame.gross_profit.sum()); gl=float(frame.gross_loss.sum())
    return {'symbols':len(frame), 'trades':int(frame.n_trades.sum()),
            'net_pnl':float(frame.net_pnl.sum()), 'profit_factor':gp/gl if gl else None,
            'avg_total_ret':float(frame.total_ret.mean()), 'median_total_ret':float(frame.total_ret.median()),
            'avg_max_dd':float(frame.max_dd.mean())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample',default='data/backtests/spring_st_300_20261017_v2/sample.csv')
    parser.add_argument('--out',default='data/backtests/day0_022_compatible_300_20261018')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int,default=0,help='development subset; recorded in manifest')
    parser.add_argument('--optimize',action='store_true',help='train/validation exit-rule grid, test untouched')
    args=parser.parse_args()
    cohort=pd.read_csv(args.sample,dtype={'code':str})
    if args.limit: cohort=cohort.iloc[:args.limit]
    if cohort.duplicated(['code','kind']).any(): raise ValueError('duplicate cohort symbols')
    cfg=StrategyConfig.load()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cohort.to_csv(out/'sample.csv',index=False,encoding='utf-8-sig')
    cfg.save(out/'strategy.yaml')
    grid=[dict(take_profit=tp,stop_loss=sl,max_hold_days=hold) for tp,sl,hold in
          itertools.product([.10,.15,.20],[.04,.08],[10,20])] if args.optimize else []
    rows,candidates,event_counts=[],[],[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(worker,row,cfg,grid):(row['kind'],row['code']) for row in cohort.to_dict('records')}
        for i,future in enumerate(as_completed(pending),1):
            try: results,search,counts=future.result()
            except Exception as exc:
                raise RuntimeError(f'Failed {pending[future]} after {i-1}/{len(cohort)}') from exc
            rows.extend(results);candidates.extend(search);event_counts.append(counts)
            if i%25==0 or i==len(cohort): print(f'{i}/{len(cohort)} completed',flush=True)
    assert len(rows)==len(cohort)*len(PERIODS)
    flat=pd.DataFrame([{k:v for k,v in row.items() if k not in ('trade_list','equity','dates')} for row in rows])
    flat.to_csv(out/'symbols.csv',index=False,encoding='utf-8-sig')
    trades=pd.DataFrame([{'code':row['code'],'kind':row['kind'],'period':row['period'],**trade}
                         for row in rows for trade in row['trade_list']])
    trades.to_csv(out/'trades.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(event_counts).fillna(0).to_csv(out/'events.csv',index=False,encoding='utf-8-sig')
    summaries={p:aggregate(flat[flat.period==p]) for p in PERIODS}
    for p in PERIODS:
        if not trades.empty:
            t=trades[trades.period==p]
            summaries[p]['win_rate']=float((t.pnl>0).mean()) if len(t) else None
    write_json(out/'summary.json',summaries)
    if not trades.empty:
        by=trades.groupby(['period','entry_type']).agg(trades=('pnl','size'),net_pnl=('pnl','sum'),
              win_rate=('pnl',lambda x:float((x>0).mean())),avg_ret=('ret','mean'))
        by.to_csv(out/'entry_types.csv',encoding='utf-8-sig')
    # Equal independent accounts; this is not a single account holding 300 simultaneous positions.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,8))
    for ax,period in zip(axes,('full','test')):
        curves=[pd.Series(r['equity'],index=pd.to_datetime(r['dates']),name=r['kind']+r['code'])
                for r in rows if r['period']==period]
        matrix=pd.concat(curves,axis=1).sort_index().ffill().fillna(cfg.trade.initial_cash)
        curve=matrix.mean(axis=1)/cfg.trade.initial_cash
        curve.to_csv(out/f'equity_{period}.csv',header=['equal_account_nav'])
        curve.plot(ax=ax,label='Mean of independent event-entry accounts')
        ax.set_title(f'{period} | {len(cohort)} symbols | {STRATEGY_VERSION}')
        ax.set_ylabel('Net asset value');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'equity.png',dpi=140);plt.close(fig)
    if candidates:
        search=pd.DataFrame(candidates)
        search.to_csv(out/'exit_grid_symbols.csv',index=False,encoding='utf-8-sig')
        ranking=[]
        for ident,params in enumerate(grid):
            train=aggregate(search[(search.id==ident)&(search.period=='train')])
            validation=aggregate(search[(search.id==ident)&(search.period=='validation')])
            ranking.append({'id':ident,**params,'train_pnl':train['net_pnl'],'validation_pnl':validation['net_pnl'],
                            'validation_pf':validation['profit_factor']})
        ranking=pd.DataFrame(ranking).sort_values(['validation_pnl','train_pnl'],ascending=False)
        ranking.to_csv(out/'exit_grid.csv',index=False,encoding='utf-8-sig')
        # Research only: never silently overwrite strategy config or fit the test period.
    sources=list((REPO_ROOT/'wyckoff'/'analysis').glob('*.py'))+list((REPO_ROOT/'wyckoff'/'backtest').glob('*.py'))+[Path(__file__),REPO_ROOT/'wyckoff'/'config.py']
    manifest={'strategy_version':STRATEGY_VERSION,'upstream_commit':UPSTREAM_COMMIT,
              'config':asdict(cfg),'symbols':len(cohort),'kind_counts':cohort.kind.value_counts().to_dict(),
              'periods':PERIODS,'selection':'reused fixed prior cohort; existing-universe survivorship bias',
              'execution':'next open, T+1, independent accounts, fees and slippage, no score gates',
              'data_sha256':{r.kind+':'+r.code:digest(universe_dir(r.kind)/(r.code+'.csv')) for r in cohort.itertuples()},
              'source_sha256':{str(p.relative_to(REPO_ROOT)):digest(p) for p in sources},
              'local_port_sha256':digest(REPO_ROOT/'wyckoff/analysis/day0_backend.py'),
              'provenance_sha256':digest(REPO_ROOT/'third_party/DAY0_NOTICE.md')}
    write_json(out/'manifest.json',manifest)
    print(json.dumps(clean(summaries),ensure_ascii=False,indent=2),flush=True)
    print(f'Report: {out.resolve()}',flush=True)

if __name__=='__main__':main()
