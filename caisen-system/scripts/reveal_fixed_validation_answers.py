"""Create a second-stage machine overlay after blind answers are submitted.

This tool validates that the answers refer to the exact frozen chart prefixes,
then displays both sides for visual review. It deliberately computes no score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


KNOWN_SHAPES = {
    "w_bottom": "W底", "spring": "破底翻", "bull_flag": "下飘旗形上涨中继",
    "inverse_head_shoulders": "头肩底", "triangle_bottom": "收敛三角形底部",
    "m_top": "M头", "false_breakout": "假突破", "bear_flag": "上飘旗形下跌中继",
    "head_shoulders": "头肩顶", "triangle_top": "收敛三角形头部",
    "other_shape": "其他形态", "no_book_pattern": "没有任何书本形态",
}


def _fingerprint_bars(bars):
    canonical = json.dumps(bars, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_answers(answer_path, machine_sidecar_path):
    answers = json.loads(Path(answer_path).read_text(encoding="utf-8"))
    sidecar = json.loads(Path(machine_sidecar_path).read_text(encoding="utf-8"))
    if answers.get("validation_version") != sidecar.get("validation_version"):
        raise ValueError("validation version mismatch")
    if len(answers.get("cases") or []) != 20:
        raise ValueError("a submitted batch must contain exactly 20 cases")
    machine_cases = sidecar.get("cases") or {}
    checked = []
    seen = set()
    for answer in answers.get("cases") or []:
        case_id = str(answer.get("id"))
        if case_id in seen:
            raise ValueError("duplicate answered case: " + case_id)
        seen.add(case_id)
        machine = machine_cases.get(case_id)
        if machine is None:
            raise ValueError("machine sidecar has no case " + case_id)
        if str(answers.get("batch")) != str(machine.get("batch")):
            raise ValueError("batch mismatch for case " + case_id)
        if str(answer.get("code")) != str(machine.get("code")):
            raise ValueError("code mismatch for case " + case_id)
        if str(answer.get("timeframe")) != str(machine.get("timeframe")):
            raise ValueError("timeframe mismatch for case " + case_id)
        if str(answer.get("as_of")) != str(machine.get("as_of")):
            raise ValueError("as_of mismatch for case " + case_id)
        fingerprint = str(answer.get("data_fingerprint") or "")
        if not fingerprint or fingerprint != str(machine.get("source_data_sha256") or ""):
            raise ValueError("data fingerprint mismatch for case " + case_id)
        chart_range = answer.get("chart_range") or {}
        bars = machine.get("bars") or []
        if fingerprint != _fingerprint_bars(bars):
            raise ValueError("frozen chart fingerprint mismatch for case " + case_id)
        if not bars or chart_range.get("from") != bars[0][0] or chart_range.get("to") != bars[-1][0]:
            raise ValueError("chart range mismatch for case " + case_id)
        chart_start, chart_end = str(chart_range["from"]), str(chart_range["to"])
        if not isinstance(answer.get("overall_uncertain"), bool):
            raise ValueError("overall_uncertain must be a boolean for case " + case_id)
        shapes = answer.get("shapes") or {}
        if "no_book_pattern" in shapes and len(shapes) != 1:
            raise ValueError("no_book_pattern cannot be combined with another shape for case " + case_id)
        for shape, value in shapes.items():
            if shape not in KNOWN_SHAPES:
                raise ValueError("unknown human label for case %s: %s" % (case_id, shape))
            if value.get("state") not in {"已确认", "未确认", "已失效", "不确定"}:
                raise ValueError("invalid per-shape state for case " + case_id)
            for key in ("start_date", "end_date"):
                if value.get(key):
                    date_value = str(value[key])
                    if date_value > str(answer["as_of"]):
                        raise ValueError("human structure date after as_of for case " + case_id)
                    if not chart_start <= date_value <= chart_end:
                        raise ValueError("human structure date outside chart range for case " + case_id)
            if (value.get("start_date") and value.get("end_date")
                    and str(value["start_date"]) > str(value["end_date"])):
                raise ValueError("human structure start after end for case " + case_id)
            for line in value.get("lines", []):
                for point in (line.get("start", {}), line.get("end", {})):
                    if str(point.get("date", "")) > str(answer["as_of"]):
                        raise ValueError("human line after as_of for case " + case_id)
                    if not chart_start <= str(point.get("date", "")) <= chart_end:
                        raise ValueError("human line outside chart range for case " + case_id)
        checked.append({"answer": answer, "machine": machine})
    if not checked:
        raise ValueError("submission contains no answered cases")
    return {"validation_version": answers["validation_version"], "batch": answers.get("batch"),
            "cases": checked, "overall_uncertain_count": sum(
                bool(item["answer"].get("overall_uncertain")) for item in checked)}


def _matching_counts(checked_cases):
    """Return descriptive case/label agreement counts without an accuracy rate."""
    evaluable = 0
    exact_matches = 0
    excluded_uncertain = 0
    excluded_other = 0
    excluded_unknown_machine = 0
    unlabeled = 0
    per_shape = {shape: {"human_present": 0, "machine_present": 0, "matched": 0,
                         "state_matched": 0, "state_mismatched": 0, "state_unknown": 0}
                 for shape in KNOWN_SHAPES if shape not in {"other_shape", "no_book_pattern"}}
    unknown_machine_types = {}
    for item in checked_cases:
        answer, machine = item["answer"], item["machine"]
        human = answer.get("shapes") or {}
        machine_events = machine.get("linked_execution_events") or []
        for event in machine_events:
            pattern_type = event.get("pattern_type")
            if pattern_type and pattern_type not in per_shape:
                unknown_machine_types[pattern_type] = unknown_machine_types.get(pattern_type, 0) + 1
        if answer.get("overall_uncertain"):
            excluded_uncertain += 1
            continue
        if "other_shape" in human:
            excluded_other += 1
            continue
        if not human:
            unlabeled += 1
            continue
        human_types = {shape for shape in human if shape in per_shape}
        explicitly_none = "no_book_pattern" in human
        machine_types = set()
        machine_states = {}
        case_has_unknown_machine = False
        for event in machine_events:
            shape = event.get("pattern_type")
            if shape not in per_shape:
                if shape:
                    case_has_unknown_machine = True
                continue
            machine_types.add(shape)
            state = event.get("state")
            machine_state = {"confirmed": "已确认", "forming": "未确认", "invalidated": "已失效"}.get(state)
            if machine_state:
                machine_states.setdefault(shape, set()).add(machine_state)
        if case_has_unknown_machine:
            excluded_unknown_machine += 1
            continue
        if explicitly_none and human_types:
            raise ValueError("no_book_pattern cannot be combined with a book shape")
        evaluable += 1
        expected = set() if explicitly_none else human_types
        if expected == machine_types:
            exact_matches += 1
        for shape, counts in per_shape.items():
            if shape in expected:
                counts["human_present"] += 1
            if shape in machine_types:
                counts["machine_present"] += 1
            if shape in expected and shape in machine_types:
                counts["matched"] += 1
                human_state = human[shape].get("state")
                if human_state == "不确定" or not machine_states.get(shape):
                    counts["state_unknown"] += 1
                elif human_state in machine_states[shape]:
                    counts["state_matched"] += 1
                else:
                    counts["state_mismatched"] += 1
    return {
        "evaluable_cases": evaluable,
        "exact_multilabel_match_count": exact_matches,
        "excluded_overall_uncertain_cases": excluded_uncertain,
        "excluded_other_shape_cases": excluded_other,
        "excluded_unknown_machine_cases": excluded_unknown_machine,
        "unlabeled_cases": unlabeled,
        "per_shape_descriptive_counts": per_shape,
        "unknown_machine_type_counts": unknown_machine_types,
        "note": "Counts describe this purposefully selected batch only; no population accuracy rate is calculated.",
    }


def _report_html(payload):
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>主结构盲评 · 机器叠图</title>
<style>body{margin:20px auto;padding:0 16px;max-width:1200px;background:#0a1220;color:#e7eef8;font:14px/1.5 system-ui,sans-serif}h1{font-size:20px}.note{padding:12px;background:#3a2a19;border-radius:8px;color:#ffd39b}.case{padding:14px;margin:12px 0;background:#111c2d;border:1px solid #33445b;border-radius:10px}.split{display:grid;grid-template-columns:1fr 1fr;gap:12px}.side{padding:9px;background:#0d1726;border-radius:8px}canvas{width:100%;height:350px;display:block;background:#09111c}.muted{color:#9aabc1}@media(max-width:800px){.split{grid-template-columns:1fr}}</style>
<h1>主结构策略盲评 · 机器叠图</h1><p class="note">这是提交答案后的第二阶段结构对照。全局不确定案例：<b id="uncertain"></b>（这些案例不参与标签匹配统计）。本报告只展示本批逐例对照，不代表市场总体效果。</p><div id="matching" class="case"></div><div id="cases"></div>
<script>window.DATA=__DATA__;</script><script>
(()=>{const data=window.DATA,names=__SHAPES__,root=document.querySelector('#cases'),counts=data.matching_counts;document.querySelector('#uncertain').textContent=data.overall_uncertain_count;document.querySelector('#matching').innerHTML=`<b>本批标签匹配计数（不是总体准确率）</b><p>可比较案例 ${counts.evaluable_cases} · 多标签完全一致 ${counts.exact_multilabel_match_count} · 整体不确定排除 ${counts.excluded_overall_uncertain_cases} · 其他形态排除 ${counts.excluded_other_shape_cases} · 未知机器类型排除 ${counts.excluded_unknown_machine_cases} · 未标注 ${counts.unlabeled_cases}</p><p>${counts.note}</p><p>机器未知类型：${esc(JSON.stringify(counts.unknown_machine_type_counts))}</p><p>每形态：${Object.entries(counts.per_shape_descriptive_counts).map(([id,v])=>`${label(id)} 人机同现${v.matched}·状态相同${v.state_matched}·状态不一致${v.state_mismatched}·状态未知${v.state_unknown}`).join("；")}</p>`;function esc(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function label(type){return names[type]||('未知形态：'+type)}function machineEvents(m){return m.linked_execution_events||m.main_parent_events||m.legacy_detector_events||[]}function linesHuman(a){const out=[];for(const [shape,v] of Object.entries(a.shapes||{}))for(const line of v.lines||[])out.push({kind:shape,start:line.start,end:line.end,color:'#58d6b0'});return out}function linesMachine(m){return machineEvents(m).flatMap(e=>(e.machine_lines||[]).map(line=>({...line,color:'#ff9f68'})))}function draw(canvas,bars,lines){const ctx=canvas.getContext('2d'),w=canvas.width=canvas.clientWidth*devicePixelRatio,h=canvas.height=canvas.clientHeight*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);const W=canvas.clientWidth,H=canvas.clientHeight,p={l:54,r:12,t:12,b:22},pw=W-p.l-p.r,ph=H-p.t-p.b,hi=Math.max(...bars.map(x=>x[2])),lo=Math.min(...bars.map(x=>x[3])),step=pw/bars.length,y=v=>p.t+(hi-v)/(hi-lo||1)*ph;ctx.fillStyle='#09111c';ctx.fillRect(0,0,W,H);for(let g=0;g<=4;g++){let yy=p.t+ph*g/4;ctx.strokeStyle='#25364a';ctx.beginPath();ctx.moveTo(p.l,yy);ctx.lineTo(W-p.r,yy);ctx.stroke()}bars.forEach((b,i)=>{let x=p.l+(i+.5)*step;ctx.strokeStyle=b[4]>=b[1]?'#48c9a9':'#ed7474';ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.moveTo(x,y(b[2]));ctx.lineTo(x,y(b[3]));ctx.stroke();ctx.fillRect(x-step*.28,Math.min(y(b[1]),y(b[4])),Math.max(1,step*.56),Math.max(1,Math.abs(y(b[1])-y(b[4]))))});for(const line of lines){const at=pt=>{let i=bars.findIndex(b=>b[0]>=pt.date);if(i<0)i=bars.length-1;return {x:p.l+(i+.5)*step,y:y(pt.price)}};const a=at(line.start),b=at(line.end);ctx.strokeStyle=line.color;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}ctx.fillStyle='#9aabc1';ctx.font='11px sans-serif';ctx.fillText(bars[0][0],p.l,H-5);ctx.fillText(bars[bars.length-1][0],W-95,H-5)}
for(const row of data.cases){const a=row.answer,m=row.machine,events=machineEvents(m),human=Object.entries(a.shapes||{}).map(([id,v])=>`<li>${esc(label(id))}：${esc(v.state)} · ${esc(v.start_date||'起始未知')} 至 ${esc(v.end_date||'结束未知')}<br>${esc(v.notes||'')}</li>`).join('')||'<li>未标注书本形态</li>',machine=events.map(e=>`<li>${esc(label(e.pattern_type))}：${esc(e.state)} · 确认 ${esc(e.confirmation_date||'无')} ${e.execution_suppression?.suppressed?'· 子形态不单独执行':''}</li>`).join('')||'<li>机器未输出相关结构</li>';const section=document.createElement('section');section.className='case';section.innerHTML=`<h2>${esc(a.id)} · ${esc(a.code)} · ${esc(a.as_of)}</h2><div class="muted">人工 ${a.overall_uncertain?'整体不确定（排除标签匹配）':'整体可判'} · 数据指纹 ${esc(a.data_fingerprint)}</div><div class="split"><canvas></canvas><div><div class="side"><b>人工标注</b><ul>${human}</ul></div><div class="side"><b>机器标注</b><ul>${machine}</ul></div><p class="muted">绿线：人工线段 · 橙线：机器结构线</p></div></div>`;root.appendChild(section);draw(section.querySelector('canvas'),m.bars||[],[...linesHuman(a),...linesMachine(m)])}window.onresize=()=>location.reload()})();</script></html>'''.replace("__DATA__", data).replace("__SHAPES__", json.dumps(KNOWN_SHAPES, ensure_ascii=False))


def reveal_answers(answer_path, machine_sidecar_path, output_path):
    payload = validate_answers(answer_path, machine_sidecar_path)
    payload["matching_counts"] = _matching_counts(payload["cases"])
    path = Path(output_path)
    if path.exists():
        raise FileExistsError("refusing to overwrite existing reveal report: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_report_html(payload), encoding="utf-8")
    return str(path.resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers", type=Path)
    parser.add_argument("machine_sidecar", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(reveal_answers(args.answers, args.machine_sidecar, args.output))


if __name__ == "__main__":
    main()
