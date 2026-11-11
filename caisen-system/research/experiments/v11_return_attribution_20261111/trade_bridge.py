from pathlib import Path
import json,collections,math
PROJECT=Path(__file__).resolve().parents[3];BASE=PROJECT/'reports/daily-research';CURRENT=BASE/'rollback-33ca8475-paired100-process8-20261111';REF=BASE/'paired-100-v8-v12-20261109';OUT=PROJECT/'state/v11-return-attribution-20261111'
rows=[];bridge=collections.Counter();counts=collections.Counter();shape=collections.defaultdict(collections.Counter)
for f in sorted((CURRENT/'accounts').glob('*.json')):
 a=json.loads(f.read_text());code=a['code'];old=json.loads((REF/('account-'+code+'.json')).read_text())['versions']['old']['replay'];om={t['entry_date']:t for t in old['trades']};nm={t['entry_date']:t for t in a['trades']}
 for day in sorted(om.keys()|nm.keys()):
  o,n=om.get(day),nm.get(day)
  if o and n:category='shared_entry_size_or_exit_delta';delta=n['pnl']-o['pnl'];types=sorted(set(o['pattern_types'])|set(n['pattern_types']))
  elif o:category='removed_old_trade';delta=-o['pnl'];types=o['pattern_types']
  else:category='added_new_trade';delta=n['pnl'];types=n['pattern_types']
  bridge[category]+=delta;counts[category]+=1
  for t in types:shape[t][category]+=delta
  rows.append({'code':code,'entry_date':day,'category':category,'pnl_delta':delta,'old':o,'new':n})
net=math.fsum(json.loads(f.read_text())['ending_equity']-json.loads((REF/('account-'+f.stem.split('-')[-1]+'.json')).read_text())['versions']['old']['replay']['ending_equity'] for f in (CURRENT/'accounts').glob('*.json'))
assert abs(net-sum(bridge.values()))<.01,(net,bridge)
r={'net_pnl_delta':net,'bridge':dict(bridge),'counts':dict(counts),'shape_contributions_nonadditive':dict(shape),'largest_losses':sorted(rows,key=lambda r:r['pnl_delta'])[:20],'largest_gains':sorted(rows,key=lambda r:r['pnl_delta'],reverse=True)[:20],'note':'Entry-date matched accounting decomposition, not a rule-level causal attribution. Shared entries reflect different position sizes and/or exits; future opportunities also depend on position occupancy.'}
(OUT/'trade-bridge.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ('largest_losses','largest_gains','shape_contributions_nonadditive')},ensure_ascii=False,indent=2))
