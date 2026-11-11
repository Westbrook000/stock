from pathlib import Path
import subprocess,shutil,json,hashlib,ast
PROJECT=Path(__file__).resolve().parents[3]
OUT=PROJECT/'state/v11-return-attribution-20261111'
VARIANTS=('no_phase_consumption','no_global_premature_failure','v9_selection','moving_support','no_right_shoulder_guard','no_prefix_cache','precise_boundary_identity','scoped_phase_repair')
def replace_once(s,old,new):
 assert s.count(old)==1,(old[:90],s.count(old))
 return s.replace(old,new,1)
def transform(main,daily,name):
 if name=='no_phase_consumption':
  main=replace_once(main,'    left_line, right_line = consumed.get("line"), candidate.get("line")','    return False\n    left_line, right_line = consumed.get("line"), candidate.get("line")')
 if name in ('no_global_premature_failure','scoped_phase_repair'):
  start=main.index('        failed_phase = _failed_premature_neckline_phase(consumed_phases, close, index)')
  end=main.index('        available = [candidate for candidate in neckline_candidates',start)
  main=main[:start]+main[end:]
 if name=='v9_selection':
  old=subprocess.check_output(['git','show','748d3d01:caisen-system/caisen/main_bottoms.py'],cwd=PROJECT.parent,text=True)
  begin=old.index('        chosen_candidate = min(available, key=lambda candidate: candidate["score"])',old.index('def _make_main_pattern('))
  end=old.index('    if chosen_candidate is None:',begin)
  a=main.index('        available.sort(key=lambda candidate: candidate["score"])',main.index('def _make_main_pattern('));b=main.index('    if chosen_candidate is None:',a)
  main=main[:a]+old[begin:end]+main[b:]
 if name=='moving_support':
  main=replace_once(main,'            prior_support = float(pending_breakout["item"]["frozen_neckline"])','            prior_support = float(_at(prior_candidate["line"], index))')
  main=replace_once(main,'            if close[index] < post_breakout_support_level:','            if close[index] < _at(neckline, index):')
 if name=='no_right_shoulder_guard':
  tree=ast.parse(main);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_ihs_right_shoulder_intact')
  lines=main.splitlines(keepends=True);lines.insert(node.body[0].lineno-1,'    return True\n');main=''.join(lines)
 if name=='no_prefix_cache':
  daily=replace_once(daily,'    def _frame_for_current_prefix(self):','    def _frame_for_current_prefix(self):\n        self._prefix_frame_cache = None')
 if name in ('precise_boundary_identity','scoped_phase_repair'):
  tree=ast.parse(main);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_same_neckline_boundary_phase')
  lines=main.splitlines(keepends=True);start=node.body[0].lineno-1;end=node.end_lineno
  lines[start:end]=['    left, right = consumed.get("line"), candidate.get("line")\n','    return bool(left and right and\n','                abs(float(left["slope"]) - float(right["slope"])) <= 1e-12 and\n','                abs(float(left["intercept"]) - float(right["intercept"])) <= 1e-10)\n']
  main=''.join(lines)
 compile(main,'main_bottoms.py','exec');compile(daily,'daily_incremental.py','exec')
 return main,daily

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 main=subprocess.check_output(['git','show','33ca8475:caisen-system/caisen/main_bottoms.py'],cwd=PROJECT.parent,text=True);daily=subprocess.check_output(['git','show','33ca8475:caisen-system/caisen/daily_incremental.py'],cwd=PROJECT.parent,text=True)
 registry={}
 for name in VARIANTS:
  folder=OUT/'variants'/name
  if folder.exists():raise FileExistsError(folder)
  shutil.copytree(PROJECT/'caisen',folder/'caisen',ignore=shutil.ignore_patterns('__pycache__'))
  m,d=transform(main,daily,name)
  (folder/'caisen/main_bottoms.py').write_text(m);(folder/'caisen/daily_incremental.py').write_text(d)
  registry[name]={'main_sha256':hashlib.sha256(m.encode()).hexdigest(),'daily_sha256':hashlib.sha256(d.encode()).hexdigest()}
 (OUT/'variants.json').write_text(json.dumps(registry,indent=2));print(OUT)
if __name__=='__main__':prepare()
