"""Create a cautious post-review comparison from frozen blind-review artifacts.

This script is evaluation-only: it reads the frozen HTML cases, machine sidecar,
uploaded human answers, and current CSVs. It never writes to those source files.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
STOCK_ROOT = PROJECT.parent
ANSWER_PATH = Path("/Users/sotawest/Downloads/blind-pattern-review.json")
OUT = PROJECT / "reports" / "blind-pattern-review-evaluation"
sys.path.insert(0, str(PROJECT))
from caisen.patterns import detect_patterns  # noqa: E402

TYPE_ZH = {"w_bottom": "W底", "inverse_head_shoulders": "头肩底",
           "triangle_bottom": "收敛三角底", "bull_flag": "下飘旗形", "spring": "破底翻",
           "other_or_none": "其他/无书本形态"}
MACHINE_ZH = {"detector accepted at signal cutoff": "机器在目标截止点接受",
              "forming candidate without confirmation": "机器候选未确认",
              "rejected gate diagnostic": "机器诊断候选被门槛拒绝",
              "no selected bullish pattern emitted at cutoff": "该截止点没有输出选中的看多形态"}
HUMAN_NOTES = {
    "R01": "人工画的是2023–2024宏观W：突破颈线后又跌回，先成立后失效。机器报告短结构拒绝；时间尺度/门槛不同，需看全事件而非把拒绝直接判为漏检。",
    "R02": "用户澄清：无书本形态。",
    "R03": "备注明确为M头；other_or_none在此代表未选列表内看多形态，不能读作无结构。",
    "R04": "人工W区间约2024-01至2024-09，备注称8月底突破；机器单一目标是2025-08至09的下飘旗形。区间不同，不据此判W的识别对错。",
    "R05": "人工备注为头部三角（看空）且右侧小W未确认；机器目标是形成中的W。目标抽样与人工所指的小W可能尺度相近，但类别/区间需看图。",
    "R06": "保留unknown：人工明确不确定。",
    "R07": "备注明确破底翻已确立；机器目标另报W底已确认。单一机器目标不覆盖全窗其他形态。",
    "R08": "备注只写破底翻，未填总体判断/确认；仅一条长颈线不足以断定完整破底翻，保留为待澄清。机器W候选因高位门槛被拒。",
    "R09": "人工标收敛三角底潜力、未确认；机器目标头肩底已确认。结构家族及确认状态有分歧，需按图和规则口径复核。",
    "R10": "备注明确M头看空；机器输出的是另一个未确认W候选。",
    "R11": "用户澄清：无书本形态；机器此截止点未输出看多形态，方向上相符。",
    "R12": "用户澄清：无书本形态；机器只有被短结构门槛拒绝的诊断候选。",
    "R13": "用户澄清：无书本形态；机器只有被高位门槛拒绝的头肩底诊断候选。",
    "R14": "备注明确破底翻已确认；机器目标是已确认的收敛三角底，类型存在差异。",
    "R15": "保留unknown：人工明确不确定。",
    "R16": "保留unknown：人工明确不确定，虽勾选other_or_none也不视为确定的无形态。",
    "R17": "保留unknown：人工明确不确定，other_or_none不作为确定负例。",
    "R18": "人工确认的W约2025-08至2026-04；机器形成中的W约2026-07，属不同尺度/区间，不能直接算确认错误。",
    "R19": "备注明确破底翻；机器头肩底候选被高位门槛拒绝。画线是一条长水平边界，本身不能证明完整破底翻。",
    "R20": "原始答案全空属于未答；用户后续单独澄清为无书本形态。机器W候选被高位门槛拒绝。",
}
FOLLOW_UP_CLARIFICATIONS = {
    "source": "用户在分析对话中的后续澄清；原始下载文件保持未改动。",
    "cases": {
        "R02": "无任何书本形态。", "R11": "无任何书本形态。",
        "R12": "无任何书本形态。", "R13": "无任何书本形态。",
        "R20": "原始答案为空是未作答；后续澄清为无任何书本形态。",
        "unknown_cases": ["R06", "R09", "R15", "R16", "R17"],
        "R05": {"triangle_top": "看空三角形，已确认", "w_bottom": "右侧小W，未确认"},
        "structure_priority": "用户偏好：只有尚有效且在时间/结构范围上包含当前子结构的父级大结构，才优先于已确认内部小结构；历史上已经结束的大结构不永久压制后续新形态。R04与R18的人机结构是先后区间，不构成父子关系；R09可能是同一窗口内的父子错位，但人工总体判断为unknown，不能当作算法错误真值。此项暂作产品规则记录，不改生产代码。",
    },
}


def read_frozen():
    page = (PROJECT / "reports" / "pattern-blind-review.html").read_text(encoding="utf-8")
    match = re.search(r"window\.REVIEW_CASES=(.*?);</script>", page)
    if not match:
        raise ValueError("Frozen case payload not found in blind review HTML")
    return json.loads(match.group(1))


def read_csv(code: str, kind: str) -> pd.DataFrame:
    path = STOCK_ROOT / "data" / ("etfs" if kind == "etf" else "stocks") / f"{code}.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    rename = {"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交股数": "volume"}
    frame = frame.rename(columns=rename)
    frame.columns = [str(c).lower() for c in frame.columns]
    frame["date"] = pd.to_datetime(frame.date, errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def align_check(case, human, machine):
    checks = {
        "id": case["id"] == human["id"] == machine["id"],
        "code": case["code"] == human["code"] == machine["code"],
        "as_of": case["as_of"] == human["as_of"] == machine["as_of"],
        "range": (case["bars"][0][0] == human["chart_range"]["from"] == machine["data_window"]["start"]
                  and case["bars"][-1][0] == human["chart_range"]["to"] == machine["data_window"]["end"]),
    }
    return checks


def data_parity(case):
    current = read_csv(case["code"], case["kind"])
    lo, hi = case["bars"][0][0], case["bars"][-1][0]
    src = current[(current.date >= lo) & (current.date <= hi)].reset_index(drop=True)
    bars = case["bars"]
    if len(src) != len(bars):
        return False, f"bars {len(bars)} frozen vs {len(src)} CSV"
    cols = ["open", "high", "low", "close", "volume"]
    for idx, row in enumerate(bars):
        if src.iloc[idx].date != row[0]:
            return False, f"date mismatch at {idx}"
        if any(not np.isclose(float(src.iloc[idx][col]), float(row[j]), rtol=0, atol=0.00011 if col != "volume" else 0.011)
               for j, col in enumerate(cols, 1)):
            return False, f"OHLCV mismatch at {row[0]}"
    return True, "date/OHLCV exact to frozen precision"


def bars_frame(case):
    return pd.DataFrame(case["bars"], columns=["date", "open", "high", "low", "close", "volume"]).assign(date=lambda x: pd.to_datetime(x.date))


def run_current(case, machine, parity):
    if not parity:
        return [], None
    frame = bars_frame(case)
    window = 520 if case["kind"] == "etf" else 500
    found = detect_patterns(frame.tail(window), code=case["code"], kind=case["kind"], background_bars=frame,
                            structure_window=window)
    target = next((p for p in found if p.pattern_id == machine.get("pattern_id")), None)
    return found, target


def y_machine_dates(pattern, case):
    if not pattern:
        return []
    frame = bars_frame(case)
    window = 520 if case["kind"] == "etf" else 500
    offset = max(0, len(frame) - window)
    pts = []
    for i in range(1, 6):
        idx = pattern.key_points.get(f"pivot_{i}_index")
        absolute_idx = int(idx) + offset if idx is not None else len(frame)
        if idx is None or absolute_idx >= len(frame):
            continue
        pts.append((frame.date.iloc[absolute_idx], float(pattern.key_points.get(f"pivot_{i}", np.nan))))
    return pts


def plot_case(case, human, machine, target, parity):
    frame = bars_frame(case)
    dates = mdates.date2num(frame.date)
    fig, (ax, vol) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
    fig.patch.set_facecolor("#fbfcfe")
    for a in (ax, vol):
        a.set_facecolor("#fbfcfe")
        a.grid(True, color="#e6ebf1", linewidth=.65)
        a.spines[["top", "right"]].set_visible(False)
    up = frame.close >= frame.open
    for i, row in frame.iterrows():
        color = "#168b72" if up.iloc[i] else "#d35b55"
        ax.vlines(dates[i], row.low, row.high, color=color, linewidth=.7)
        body_low, body_hi = min(row.open, row.close), max(row.open, row.close)
        ax.add_patch(plt.Rectangle((dates[i] - .33, body_low), .66, max(body_hi - body_low, 0.001), color=color, linewidth=0))
        vol.bar(dates[i], row.volume, width=.66, color=color, alpha=.55)
    machine_label = MACHINE_ZH.get(machine["assessment"]["result"], machine["assessment"]["result"])
    topology = machine.get("assessment", {}).get("candidate_topology", [])
    machine_type = TYPE_ZH.get(machine.get("pattern_type"), TYPE_ZH.get(topology[0], topology[0]) if topology else "无单一目标")
    title = f"{case['id']}  {case['code']}  截至 {case['as_of']}  |  人工：{human.get('judgment') or '未填'} / {','.join(TYPE_ZH.get(x,x) for x in human.get('pattern_types',[])) or '未选'}  |  机器：{machine_type}（{machine_label}）"
    fig.suptitle(title, fontsize=12, x=.05, ha="left", y=.975)
    # Human annotations are drawn verbatim as date/price coordinates; no snapping or correction.
    for n, line in enumerate(human.get("lines", []), 1):
        x = mdates.date2num(pd.to_datetime([line["start"]["date"], line["end"]["date"]]))
        y = [line["start"]["price"], line["end"]["price"]]
        ax.plot(x, y, color="#2868c7", lw=1.8, alpha=.85, label="人工原线" if n == 1 else None)
        ax.scatter(x, y, s=13, color="#2868c7", zorder=4)
        ax.text(x[0], y[0], f"H{n}", fontsize=8, color="#174a95", va="bottom")
    if machine.get("formed_at"):
        span_end = machine.get("confirmation_date") or case["as_of"]
        ax.axvspan(pd.Timestamp(machine["formed_at"]), pd.Timestamp(span_end), color="#f3a847", alpha=.10,
                   label="机器目标形成→确认/截止区间")
    elif machine.get("formation_start"):
        span_end = machine.get("last_structure_point") or case["as_of"]
        ax.axvspan(pd.Timestamp(machine["formation_start"]), pd.Timestamp(span_end), color="#f3a847", alpha=.16,
                   label="机器短结构诊断区间（无几何锚点）")
        ax.text(pd.Timestamp(machine["formation_start"]), .97,
                f"诊断 {machine['formation_start']}→{span_end}（非机器连线）",
                transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8, color="#a95d08")
    if target:
        pts = y_machine_dates(target, case)
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, color="#e58a23", lw=1.8, marker="o", ms=4, label="机器目标结构（当前源重算同ID）")
            for n, (xx, yy) in enumerate(pts, 1):
                ax.annotate(f"M{n}", (xx, yy), xytext=(2, 5), textcoords="offset points", fontsize=8, color="#a95d08")
        if target.neckline:
            first_index = min((int(v) for k, v in target.key_points.items()
                               if k.startswith("pivot_") and k.endswith("_index")), default=0)
            window = 520 if case["kind"] == "etf" else 500
            offset = max(0, len(frame) - window)
            local_idx = np.arange(first_index, len(frame) - offset)
            global_idx = local_idx + offset
            neck = target.neckline["slope"] * local_idx + target.neckline["intercept"]
            span_label = f"机器颈线 {frame.date.iloc[first_index + offset]:%Y-%m-%d}→{frame.date.iloc[-1]:%Y-%m-%d}"
            ax.plot(frame.date.iloc[global_idx], neck, color="#e58a23", lw=1.2, ls="--", alpha=.9, label=span_label)
            ax.scatter([frame.date.iloc[first_index + offset], frame.date.iloc[-1]], [neck[0], neck[-1]],
                       color="#e58a23", s=17, marker="s", zorder=5)
    elif machine.get("formed_at"):
        ax.axvline(pd.Timestamp(machine["formed_at"]), color="#e58a23", lw=1.1, ls="--", label="sidecar形成日（无可复核锚点）")
    machine_span = "机器几何锚点已用冻结数据与当前代码重算同ID" if target else "机器侧栏无可核对几何锚点" if parity else "冻结行情与当前CSV不一致；未混用机器锚点"
    ax.text(.01, .02, machine_span, transform=ax.transAxes, fontsize=8, color="#5e6978", va="bottom")
    ax.set_ylabel("价格")
    vol.set_ylabel("成交量")
    vol.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=8))
    vol.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=.9)
    note = HUMAN_NOTES.get(case["id"], "")
    fig.text(.05, .015, note, fontsize=8.5, color="#414b59", wrap=True)
    fig.tight_layout(rect=[0, .06, 1, .94])
    path = OUT / f"{case['id']}-comparison.png"
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    return path


def interpret(human, machine, found, target, parity, case_id):
    assessment = machine["assessment"]
    mtype = machine.get("pattern_type")
    if case_id == "R20":
        h = "无书本形态（后续澄清；原答案未答）"
    elif case_id in {"R02", "R11", "R12", "R13"}:
        h = "无书本形态（后续澄清）"
    elif case_id == "R09":
        h = "总体unknown；已选收敛三角底候选，突破未确认"
    elif human.get("judgment") == "不确定":
        h = "unknown（保留，不计对错）"
    elif human.get("notes"):
        h = human["notes"]
    elif human.get("pattern_types") == ["other_or_none"]:
        h = "未标列表内看多形态；结合备注/澄清见人工解释"
    elif human.get("pattern_types"):
        h = ", ".join(TYPE_ZH.get(x, x) for x in human["pattern_types"])
    else:
        h = "未答/未选"
    if mtype:
        m = f"{TYPE_ZH.get(mtype,mtype)} · {assessment['result']} · {machine.get('formed_at')}至{machine.get('confirmation_date') or '截止日未确认'}"
    elif assessment.get("candidate_topology"):
        m = f"诊断拓扑 {','.join(TYPE_ZH.get(x,x) for x in assessment['candidate_topology'])} · {assessment.get('reason')}门槛拒绝"
    else:
        m = assessment["result"]
    if case_id in {"R06", "R15", "R16", "R17"}:
        status = "unknown保留"
    elif case_id == "R09":
        status = "总体判断unknown；另保留已填的三角底候选与未确认状态，不折算为对错"
    elif case_id == "R20":
        status = "人工后续澄清无形态；机器拒绝W诊断候选"
    elif case_id in {"R01", "R04", "R18"}:
        status = "时间尺度/区间错位，暂不判对错"
    elif target and mtype:
        status = "机器sidecar目标可按当前代码同ID重现；仅代表单一抽样目标"
    elif mtype and not parity:
        status = "行情未对齐，机器几何不重算"
    else:
        status = "仅描述并列结果，不折算准确率"
    if case_id in {"R03", "R05", "R07", "R08", "R09", "R10", "R14", "R19"}:
        status += "；备注含非列表类别/其他方向，不能按other_or_none直读"
    return h, m, status


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    human_doc = json.loads(ANSWER_PATH.read_text(encoding="utf-8"))
    frozen = read_frozen()
    machine_doc = json.loads((PROJECT / "reports" / "pattern-blind-review.machine-answers.json").read_text(encoding="utf-8"))
    human_by = {x["id"]: x for x in human_doc["cases"]}
    machine_by = {x["id"]: x for x in machine_doc["cases"]}
    output, payloads = [], {}
    global_integrity = {"case_count": len(frozen) == len(human_doc["cases"]) == len(machine_doc["cases"]),
                        "rule_version": human_doc["rule_version"] == machine_doc["rule_version"]}
    for case in frozen:
        h, m = human_by[case["id"]], machine_by[case["id"]]
        checks = align_check(case, h, m)
        ok, parity_detail = data_parity(case)
        patterns, target = run_current(case, m, ok)
        target_pivots = y_machine_dates(target, case)
        anchor_match = (not target or (bool(target_pivots) and str(pd.Timestamp(target_pivots[0][0]).date()) == str(target.formed_at)))
        paragraph, machine_text, status = interpret(h, m, patterns, target, ok, case["id"])
        record = {"id": case["id"], "code": case["code"], "kind": case["kind"], "as_of": case["as_of"],
                  "data_range": [case["bars"][0][0], case["bars"][-1][0]], "bars": len(case["bars"]),
                  "alignment_checks": checks, "csv_parity": ok, "csv_parity_detail": parity_detail,
                  "human_interpretation": paragraph, "human_raw": {k: h.get(k) for k in ("judgment", "pattern_types", "confirmation_status", "notes", "lines")},
                  "machine_interpretation": machine_text, "machine_raw": m,
                  "current_recomputed_ids": [p.pattern_id for p in patterns],
                  "current_matching_target": target.pattern_id if target else None,
                  "current_matching_target_state": target.state if target else None,
                  "target_first_pivot_date": str(pd.Timestamp(target_pivots[0][0]).date()) if target_pivots else None,
                  "target_formed_at_anchor_matches": anchor_match,
                  "current_w_events": [{"id": p.pattern_id, "formed_at": p.formed_at,
                                         "confirmation_date": p.confirmation_date, "state": p.state,
                                         "last_observed_date": p.last_observed_date,
                                         "pivot_dates": [str(bars_date) for bars_date, _ in y_machine_dates(p, case)]}
                                        for p in patterns if p.pattern_type == "w_bottom"],
                  "current_all_events": [{"id": p.pattern_id, "pattern_type": p.pattern_type,
                                          "formed_at": p.formed_at, "confirmation_date": p.confirmation_date,
                                          "state": p.state,
                                          "pivot_dates": [str(d) for d, _ in y_machine_dates(p, case)]}
                                         for p in patterns],
                  "human_per_pattern": FOLLOW_UP_CLARIFICATIONS["cases"].get(case["id"]),
                  "cautious_read": status, "human_note_interpretation": HUMAN_NOTES[case["id"]]}
        output.append(record)
        payloads[case["id"]] = (case, h, m, target, ok)
    errors = [f"{r['id']} {key}" for r in output for key, value in r["alignment_checks"].items() if not value]
    if not global_integrity["case_count"] or not global_integrity["rule_version"]:
        errors.append("count/rule_version")
    report = {"title": "盲评答案对照（谨慎解读版）", "generated_from": {"frozen_html": "reports/pattern-blind-review.html",
             "machine_sidecar": "reports/pattern-blind-review.machine-answers.json", "uploaded_answers": str(ANSWER_PATH)},
             "integrity": {**global_integrity, "alignment_errors": errors,
             "all_current_csv_windows_match_frozen": all(x["csv_parity"] for x in output),
             "all_recomputed_target_anchor_dates_match_formed_at": all(x["target_formed_at_anchor_matches"] for x in output)},
             "implementation_risk": "待单独证实：当前检测器质量/时间门槛中，结构形成前的价格若参与颈线突破扫描，可能把后续才定义的颈线回投到更早K线，影响首次有效突破与低位背景判断。该风险不代表已在某一例确认触发；本报告不据此调整结果。",
             "user_structure_preference": "用户明确偏好：同图大结构未确认而内部小结构已确认时，以主要大结构为准，不让内部小结构生成买点。这是候选业务规则，尚未写入生产代码；不要推演为任何历史大结构永久压制新结构，作用范围与失效条件待后续决策。",
             "interpretation_policy": ["无准确率统计：样本为目的性抽样，机器答案是单一目标。",
                 "other_or_none不自动视为无形态；备注中的M头、破底翻、看空三角明确按内容解释。",
                 "R06/15/16/17作为unknown保留；R20原始空白另列后续澄清。",
                 "若冻结OHLCV与当前CSV逐日一致，当前detect_patterns全窗重算结果只用于目标可复现性及同区间W事件检查，不代表原机器版本。"],
             "cases": output}
    (OUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "clarifications.json").write_text(json.dumps(FOLLOW_UP_CLARIFICATIONS, ensure_ascii=False, indent=2), encoding="utf-8")
    for cid in ("R01", "R02", "R04", "R05", "R09", "R18"):
        case, h, m, target, ok = payloads[cid]
        plot_case(case, h, m, target, ok)
    make_html(report)
    print(json.dumps({"output_dir": str(OUT), "integrity": report["integrity"],
                      "charts": [f"{cid}-comparison.png" for cid in ("R01", "R02", "R04", "R05", "R09", "R18")],
                      "w_events": {r["id"]: r["current_w_events"] for r in output if r["id"] in {"R01", "R04", "R18"}},
                      "target_reproduced": {r["id"]: r["current_matching_target"] is not None for r in output if r["machine_raw"].get("pattern_id")}},
                      ensure_ascii=False, indent=2))


def make_html(report):
    rows = []
    for r in report["cases"]:
        lines = r["human_raw"].get("lines") or []
        image = f'<a href="{r["id"]}-comparison.png">图</a>' if (OUT / f"{r['id']}-comparison.png").exists() else "—"
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in (
            r["id"], r["code"], f"{r['data_range'][0]}–{r['data_range'][1]}",
            r["human_interpretation"], r["machine_interpretation"], r["cautious_read"],
            f"{len(lines)}条人工线", "一致" if r["csv_parity"] else r["csv_parity_detail"], image)) + "</tr>")
    policy = "".join(f"<li>{html.escape(s)}</li>" for s in report["interpretation_policy"])
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>盲评答案对照</title>
<style>body{{font:14px/1.55 system-ui,-apple-system,sans-serif;margin:28px;color:#1c2430}}h1{{font-size:24px}}.hint{{background:#f2f6fa;padding:12px 18px;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #d9e0e7;padding:7px;vertical-align:top}}th{{background:#eef3f8;position:sticky;top:0}}td:nth-child(-n+3){{white-space:nowrap}}tr:nth-child(even){{background:#fafbfd}}a{{color:#165eb3}}.fine{{color:#546271}}</style>
<h1>盲评答案对照（谨慎解读版）</h1><p>冻结样本 {len(report['cases'])} 例。用户澄清后15例有明确判断、5例保留unknown（含R09）。窗口一致性检查：{'通过' if not report['integrity']['alignment_errors'] else '有差异：'+str(report['integrity']['alignment_errors'])}。CSV一致性：{'全部一致' if report['integrity']['all_current_csv_windows_match_frozen'] else '存在不一致，见逐例记录'}。</p>
<div class="hint"><b>解读口径</b><ul>{policy}</ul><p><b>用户新增的结构偏好：</b>{html.escape(report["user_structure_preference"])}</p><p><b>待单独证实的实现风险：</b>{html.escape(report["implementation_risk"])}</p></div><p class="fine">机器选择是抽样target，不等于全窗口全部形态。彩色图中的人工线按原始日期/价格原样绘出，不吸附到K线；机器锚点仅在当前代码用冻结行情按pattern_id重算成功时绘制。</p>
<table><thead><tr><th>例</th><th>代码</th><th>冻结窗口</th><th>人工解读</th><th>机器目标/诊断</th><th>谨慎判断</th><th>原线</th><th>CSV校验</th><th>图</th></tr></thead><tbody>{''.join(rows)}</tbody></table></html>'''
    (OUT / "summary.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
