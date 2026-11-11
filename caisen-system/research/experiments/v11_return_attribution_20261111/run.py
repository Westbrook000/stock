from pathlib import Path
import sys,json,subprocess,concurrent.futures,time,gzip,random,math,traceback,html,os
PROJECT=Path(__file__).resolve().parents[3]
OUT=PROJECT/'state/v11-return-attribution-20261111'
BASE=PROJECT/'reports/daily-research'
CURRENT=BASE/'rollback-33ca8475-paired100-process8-20261111'
SNAPSHOT=BASE/'paired-input-snapshot-20261105-v1'
SCRIPT=Path(__file__).resolve()
PYTHON=sys.executable

def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str));temp.replace(path)

def worker(variant,code):
 sys.path.insert(0,str(OUT/'variants'/variant))
 from caisen.market import discover_instruments
 from caisen.daily_research import _complete_account_task
 instrument=next(x for x in discover_instruments(SNAPSHOT) if x.kind=='stock' and x.code==code)
 folder=OUT/'accounts'/variant;folder.mkdir(parents=True,exist_ok=True);path=folder/(code+'.json')
 started=time.perf_counter();_complete_account_task(instrument,str(SNAPSHOT),'2026-10-30',None,str(path),'boundary_first')
 a=json.loads(path.read_text());assert a['status']=='complete',(variant,code,a.get('error'))
 assert all(a['accounting_checks'].values()),(variant,code,a['accounting_checks'])
 a['attribution_variant']=variant
 with gzip.open(path.with_suffix('.json.gz'),'wt',compresslevel=1) as f:json.dump(a,f,ensure_ascii=False,default=str)
 path.unlink()
 brief={'variant':variant,'code':code,'seconds':time.perf_counter()-started,'timings':a['timings'],'signals':len(a['signals']),'trades':len(a['trades']),'ending_equity':a['ending_equity'],'metrics':a['metrics'],'accounting_checks':a['accounting_checks']}
 write(OUT/'summaries'/variant/(code+'.json'),brief)

def task(item):
 variant,code=item;summary=OUT/'summaries'/variant/(code+'.json')
 if summary.exists():return json.loads(summary.read_text())
 log=OUT/'logs'/variant/(code+'.log');log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('wb') as f:result=subprocess.run([PYTHON,str(SCRIPT),'worker',variant,code],stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(str(log)+'\n'+log.read_text()[-2000:])
 return json.loads(summary.read_text())

def summarize(variant,codes):
 sys.path.insert(0,str(PROJECT))
 from caisen.daily_research import _aggregate_equity_curve,_summarize
 rows=[];trades=[];equity={};checks=[];missing=[]
 for code in codes:
  if variant=='current':a=json.loads((CURRENT/'accounts'/('stock-'+code+'.json')).read_text())
  else:
   path=OUT/'accounts'/variant/(code+'.json.gz')
   if not path.exists():missing.append(code);continue
   with gzip.open(path,'rt') as f:a=json.load(f)
  rows.append({'code':code,'kind':'stock','status':a['status'],'ending_equity':a['ending_equity'],'signal_count':len(a['signals'])})
  trades.extend(dict(t,code=code,kind='stock') for t in a['trades']);_aggregate_equity_curve.add_account(equity,100000,a['daily_equity'])
  checks.append(all(a['accounting_checks'].values()))
 capital=len(rows)*100000;ending=math.fsum(a['ending_equity'] for a in rows)
 result=_summarize(rows,trades,capital,ending,_aggregate_equity_curve.build(equity,capital))
 result.update(variant=variant,accounting_passed=all(checks),missing=missing)
 return result

def run_queue(tasks,phase):
 started=time.monotonic();total=len(tasks);completed=0;errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  pending={pool.submit(task,t):t for t in tasks}
  for future in concurrent.futures.as_completed(pending):
   item=pending[future]
   try:result=future.result();print('DONE',phase,item,round(result['seconds']),flush=True)
   except Exception as e:errors.append({'task':item,'error':str(e)});print('FAILED',item,str(e),flush=True)
   completed+=1;write(OUT/'progress.json',{'status':'running','phase':phase,'completed':completed,'total':total,'elapsed_seconds':time.monotonic()-started,'errors':errors})
 if errors:raise RuntimeError(json.dumps(errors))

def main():
 reference=json.loads((BASE/'paired-100-v8-v12-20261109/comparison.json').read_text());codes=sorted(x['code'] for x in reference['sample'])
 panel=sorted(set(random.Random(20261111).sample(codes,10))|{'000538','600633'})
 variants=list(json.loads((OUT/'variants.json').read_text()))
 write(OUT/'protocol.json',{'cohort':codes,'panel':panel,'seed':20261111,'variants':variants,'workers':8,'parallel_backend':'8 independent Python subprocesses','date_to':'2026-10-30','input':str(SNAPSHOT),'baseline':str(CURRENT),'production_modified':False,'selection':'Among precise_boundary_identity, no_global_premature_failure, scoped_phase_repair: select greatest panel equal-capital return, then win rate. Exits remain fixed. Screening is development only; all100 previously inspected.'})
 run_queue([(v,c) for v in variants for c in panel],'single_factor_panel')
 panel_results={v:summarize(v,panel) for v in ['current']+variants};write(OUT/'panel-results.json',panel_results)
 valid=['precise_boundary_identity','no_global_premature_failure','scoped_phase_repair']
 ranked=sorted(valid,key=lambda v:(panel_results[v]['summary']['cumulative_return'],panel_results[v]['summary']['win_rate']),reverse=True)
 winner=ranked[0];write(OUT/'selected-repair.json',{'variant':winner,'ranking':ranked,'panel':panel,'not_promoted_to_production':True})
 run_queue([(winner,c) for c in codes if c not in panel],'repair_full_100')
 full=summarize(winner,codes);baseline=summarize('current',codes)
 comparison={'status':'complete','selected_repair':winner,'reference_v8':reference['versions']['old']['summary'],'current':baseline,'repair':full,'panel_results':panel_results,'note':'Experimental variants do not change production. Six listed changes are tested separately; combined repair is also screened. Cache control must preserve signals. A return increase is not proof of pattern accuracy.'}
 # Compare cache control signals and economic ledgers with the unchanged baseline.
 cache_parity=[]
 for c in panel:
  old=json.loads((CURRENT/'accounts'/('stock-'+c+'.json')).read_text())
  with gzip.open(OUT/'accounts/no_prefix_cache'/(c+'.json.gz'),'rt') as f:new=json.load(f)
  cache_parity.append({'code':c,'signals_equal':old['signals']==new['signals'],'trades_equal':old['trades']==new['trades'],'equity_equal':old['daily_equity']==new['daily_equity']})
 comparison['cache_control_parity']=cache_parity
 write(OUT/'comparison.json',comparison)
 section='<pre>'+html.escape(json.dumps(comparison,ensure_ascii=False,indent=2))+'</pre>'
 report='<!doctype html><meta charset="utf-8"><title>V11 return attribution</title><style>body{font:15px/1.6 system-ui;max-width:1150px;margin:35px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>V11 return attribution and repair backtest</h1><p>Frozen 100 stocks, 2026-10-30, Backtrader fixed 8% stop / 20% target. Eight independent processes.</p>'+section
 (OUT/'report.html').write_text(report)
 write(OUT/'progress.json',{'status':'complete','phase':'report_written','report':str(OUT/'report.html'),'selected_repair':winner,'full_accounts':100})
 print('COMPLETE',winner,full['summary'],flush=True)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='worker':worker(sys.argv[2],sys.argv[3])
 else:
  try:main()
  except Exception:
   write(OUT/'progress.json',{'status':'failed','error':traceback.format_exc()});raise
