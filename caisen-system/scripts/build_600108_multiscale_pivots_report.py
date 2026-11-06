#!/usr/bin/env python3
"""Build a local, self-contained report of confirmed continuous-engine pivots."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
STOCK_ROOT = REPO.parent
sys.path.insert(0, str(REPO))

from caisen.daily_incremental import (  # noqa: E402
    BOUNDARY_FIRST_ENGINE_VERSION,
    DailyRecognitionSession,
    ENGINE_VERSION,
    SOURCE_RULE_VERSION,
)
from caisen.daily_research import load_research_bars  # noqa: E402
from caisen.main_bottoms import PIVOT_SCALES  # noqa: E402

INPUT_PATH = STOCK_ROOT / "data" / "stocks" / "600108.csv"
OUT_HTML = REPO / "reports" / "600108-multiscale-pivots-full-history-20261106.html"
OUT_JSON = REPO / "reports" / "600108-multiscale-pivots-full-history-20261106.json"
SCALE_LABELS = {
    "2atr_5pct": "小尺度 · 2ATR / 5%",
    "3atr_8pct": "中尺度 · 3ATR / 8%",
    "5atr_12pct": "大尺度 · 5ATR / 12%",
}


def _extract_stream(instrument, root):
    loaded = load_research_bars(instrument, root=root)
    bars = loaded["bars"]
    if bars.empty:
        raise ValueError("600108 history is empty")
    session = DailyRecognitionSession(
        instrument.code,
        instrument.kind,
        tick_size=loaded["tick_size"],
        total_bars=len(bars),
        boundary_first=True,
    )
    events = {name: [] for name, _, _ in PIVOT_SCALES}
    atr_checks = []
    for index, row in enumerate(bars.to_dict("records")):
        date = pd.Timestamp(row["date"])
        bar = {key: row[key] for key in ("open", "high", "low", "close", "volume")}
        bar["date"] = date
        session.bars.append(bar)
        session.date_strings.append(str(date.date()))
        previous_close = float(session.bars[index - 1]["close"]) if index else float(bar["close"])
        true_range = max(
            float(bar["high"]) - float(bar["low"]),
            abs(float(bar["high"]) - previous_close),
            abs(float(bar["low"]) - previous_close),
        )
        atr = (true_range if index == 0 else
               session.atr_values[-1] + (true_range - session.atr_values[-1]) / 14.0)
        session.atr_values.append(float(atr))
        found = session._append_scale_pivots(index)
        for scale, pivot in found:
            events[scale].append({
                "scale": scale,
                "kind": pivot.kind,
                "index": int(pivot.index),
                "date": str(pivot.date),
                "price": float(pivot.price),
                "confirmed_at": int(pivot.confirmed_at),
                "confirmed_date": str(pivot.confirmed_date),
                "confirmation_lag_bars": int(pivot.confirmed_at - pivot.index),
            })
            if pivot.confirmed_at != index or pivot.confirmed_date != str(date.date()):
                raise AssertionError("pivot was not emitted on its right-confirmation bar")
        if index in (0, 13, 14, 27, len(bars) - 1):
            atr_checks.append({"index": index, "date": str(date.date()), "atr14": float(atr)})

    crosscheck_bars = min(300, len(bars))
    reference = DailyRecognitionSession(
        instrument.code, instrument.kind, tick_size=loaded["tick_size"],
        total_bars=crosscheck_bars, boundary_first=True,
    )
    for row in bars.iloc[:crosscheck_bars].to_dict("records"):
        reference.append_bar(row)
    crosscheck_scales = {}
    for name in events:
        expected = [
            (item["index"], item["kind"], item["price"], item["confirmed_at"], item["date"], item["confirmed_date"])
            for item in events[name] if item["confirmed_at"] < crosscheck_bars
        ]
        actual = [
            (int(p.index), p.kind, float(p.price), int(p.confirmed_at), str(p.date), str(p.confirmed_date))
            for p in reference.scale_pivots[name]
        ]
        if actual != expected:
            raise AssertionError(f"300-bar full-session pivot cross-check failed for {name}")
        crosscheck_scales[name] = len(actual)

    preview = []
    for name, _, _ in PIVOT_SCALES:
        state = session.scale_states[name]
        if state["direction"] > 0:
            index, kind = int(state["high_index"]), "high"
        elif state["direction"] < 0:
            index, kind = int(state["low_index"]), "low"
        else:
            continue
        preview.append({
            "scale": name,
            "kind": kind,
            "index": index,
            "date": str(pd.Timestamp(session.bars[index]["date"]).date()),
            "price": float(session.bars[index]["high" if kind == "high" else "low"]),
            "state_as_of": str(pd.Timestamp(session.bars[-1]["date"]).date()),
            "confirmed": False,
        })

    all_pivots = [point for points in events.values() for point in points]
    if any(point["confirmed_at"] >= len(bars) for point in all_pivots):
        raise AssertionError("a confirmed pivot points beyond available history")
    input_hash = hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest()
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    versions = {
        "source_rule_version": SOURCE_RULE_VERSION,
        "engine_version": ENGINE_VERSION,
        "boundary_first_engine_version": BOUNDARY_FIRST_ENGINE_VERSION,
        "pivot_scales": [
            {"scale": name, "atr_multiple": float(atr), "percent_multiple": float(percent),
             "minimum_ticks": 2}
            for name, atr, percent in PIVOT_SCALES
        ],
        "confirmation": "right-confirmed continuous zigzag; occurrence date=index, confirmed_date=append bar",
        "atr": "Wilder ATR14 recurrence used by DailyRecognitionSession: first TR, then previous + (TR - previous)/14",
        "tick_size": float(loaded["tick_size"]),
        "tick_size_source": loaded["tick_size_source"],
    }
    return {
        "report_version": "600108_multiscale_pivots_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": "600108",
        "kind": "stock",
        "input": {
            "path": str(INPUT_PATH.resolve()),
            "sha256": input_hash,
            "data_revision": loaded["data_revision"],
            "first_date": loaded["first_date"],
            "last_date": loaded["last_date"],
            "bar_count": len(bars),
        },
        "code_sha256": code_hash,
        "versions": versions,
        "confirmed_pivots": events,
        "bars": [
            {"date": pd.Timestamp(row.date).date().isoformat(),
             "open": float(row.open), "high": float(row.high), "low": float(row.low),
             "close": float(row.close), "volume": float(row.volume)}
            for row in bars.itertuples(index=False)
        ],
        "unconfirmed_endpoints_preview_only": preview,
        "atr_sample_checks": atr_checks,
        "validation": {
            "method": "each point is recorded only when _append_scale_pivots emits it at its confirmed_at index",
            "300_bar_full_session_crosscheck": {
                "bars": crosscheck_bars,
                "passed": True,
                "counts_by_scale": crosscheck_scales,
                "comparison": "direct scale_pivots from DailyRecognitionSession.append_bar versus this isolated primitive stream on the same OHLC/ATR prefix",
            },
            "confirmed_point_count": len(all_pivots),
            "strictly_causal": all(point["confirmed_at"] >= point["index"] for point in all_pivots),
            "same_day_cross_scale_points_preserved": True,
            "pressure_pivots_included": False,
            "shape_or_buy_engine_run": False,
        },
    }


def _report_html(data):
    embedded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return HTML.replace("__DATA_JSON__", embedded)


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>600108 多尺度因果枢轴 · 全历史</title>
<style>
:root{color-scheme:light;--ink:#17212b;--muted:#5b6775;--line:#d5dde6;--paper:#f3f6f8;--surface:#fff;--small:#147d73;--medium:#c27612;--large:#3d68b2}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1680px;margin:0 auto;padding:20px 22px 40px}h1{font-size:27px;line-height:1.2;margin:0 0 6px}h2{font-size:18px;margin:0 0 12px}h3{font-size:15px;margin:0 0 8px}.muted{color:var(--muted)}.top{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;margin-bottom:14px}.top p{margin:6px 0}.badge{display:inline-block;border:1px solid #c3d4d0;background:#eff8f5;color:#185b51;padding:4px 8px;border-radius:4px;font-size:12px}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:14px;margin:12px 0}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px}.meta div{padding:8px 10px;background:#f7f9fb;border-left:2px solid #cbd5df}.meta b{display:block;font-size:11px;color:var(--muted);font-weight:600}.meta span{overflow-wrap:anywhere}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px}.control-group{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.control-group label{display:inline-flex;gap:5px;align-items:center}.control-group select{font:inherit;padding:6px 8px;border:1px solid #aeb9c5;border-radius:4px;background:white}button{font:inherit;border:1px solid #aeb9c5;border-radius:4px;background:#fff;padding:6px 10px;cursor:pointer;color:#1f2937}button:hover{background:#f0f5f8}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #75a7e8;outline-offset:1px}.preset.active{border-color:#275b91;background:#edf5fc;color:#173d65}.slider-row{display:grid;grid-template-columns:auto minmax(180px,1fr) auto minmax(180px,1fr) minmax(170px,auto);gap:10px;align-items:center;margin-top:12px}.slider-row input{width:100%;accent-color:#315f8c}.date-value{font-variant-numeric:tabular-nums;white-space:nowrap}.chart-tools{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap}.legend{display:flex;gap:12px;flex-wrap:wrap}.legend span{display:inline-flex;gap:5px;align-items:center}.swatch{width:11px;height:11px;border:1px solid #243342;border-radius:50%;display:inline-block}.chart-stack{display:grid;grid-template-columns:1fr;gap:8px}.chart-panel{border:1px solid var(--line);background:#fff;border-radius:5px;overflow:hidden}.chart-panel[hidden]{display:none}.chart-title{display:flex;justify-content:space-between;padding:5px 9px;border-bottom:1px solid #e7ebef;font-size:12px;color:var(--muted)}canvas{display:block;width:100%;height:248px;touch-action:none}.chart-panel.overlay canvas{height:540px}.tooltip{min-height:92px;display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px}.tooltip div{background:#f7f9fb;padding:7px 9px;border-radius:4px}.tooltip b{display:block;font-size:11px;color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:10px}.metric{border:1px solid var(--line);border-top:3px solid var(--accent);background:#fff;padding:12px;border-radius:5px}.metric h3{font-size:14px}.metric-main{font-size:21px;font-weight:700;font-variant-numeric:tabular-nums}.metric-sub{font-size:12px;color:var(--muted);margin-top:3px}.tables{display:grid;grid-template-columns:1fr 1fr;gap:14px}.table-wrap{max-height:360px;overflow:auto;border:1px solid var(--line);border-radius:4px}table{border-collapse:collapse;width:100%;font-size:12px}thead{position:sticky;top:0;background:#eef2f5}th,td{padding:6px 8px;text-align:right;border-bottom:1px solid #e7ebef}th:first-child,td:first-child{text-align:left}.preview-list{display:flex;gap:8px;flex-wrap:wrap}.preview-item{padding:8px 10px;background:#fff8e8;border:1px dashed #c99b46;border-radius:4px}.note{border-left:3px solid #c27612;background:#fff9ef;padding:10px 12px}.empty{padding:10px;color:var(--muted)}details summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:350px;overflow:auto;background:#f7f9fb;padding:10px;border-radius:4px}.foot{font-size:12px;color:var(--muted)}
@media(max-width:850px){main{padding:14px 10px}.top{display:block}.cards,.tables{grid-template-columns:1fr}.slider-row{grid-template-columns:auto 1fr;}.date-value{grid-column:span 2}.chart-panel canvas{height:220px}.chart-panel.overlay canvas{height:370px}}
</style></head><body><main>
<header class="top"><div><h1>600108 · 多尺度因果枢轴</h1><p class="muted">完整本地行情；每个已确认枢轴显示发生日与右侧确认日。默认全历史，并保留同日跨尺度点。</p></div><span class="badge">仅结构枢轴 · 不含形态买点</span></header>
<section class="panel"><h2>数据与规则</h2><div id="meta" class="meta"></div><p class="note">三种主图枢轴分别是 2ATR/5%、3ATR/8%、5ATR/12%，阈值均至少为 2 个报价单位。Wilder ATR14 使用引擎逐日递推。右侧端点只在下方单独预览，未确认，不计入枢轴数量。压力候选点属于另一条流，本图不混入。</p></section>
<section class="panel"><div class="controls">
<div class="control-group" aria-label="日期范围预设"><button class="preset active" data-range="all">全历史</button><button class="preset" data-range="5y">近5年</button><button class="preset" data-range="1y">近1年</button><button class="preset" data-range="bottom2024">2024大底</button></div>
<div class="control-group"><label for="mode">视图</label><select id="mode"><option value="separate">三层分图</option><option value="overlay">合并叠加</option></select><button id="zoomIn" title="以区间中心放大">放大</button><button id="zoomOut" title="以区间中心缩小">缩小</button><button id="reset">重置</button></div>
<div class="control-group" id="scaleChecks"></div>
</div><div class="slider-row"><span>开始</span><input id="startRange" type="range" min="0" value="0" step="1"><span class="date-value" id="startDate"></span><span>结束</span><input id="endRange" type="range" min="0" value="0" step="1"><span class="date-value" id="endDate"></span></div>
<div class="chart-tools"><div class="legend"><span><i class="swatch" style="background:var(--small)"></i>小尺度</span><span><i class="swatch" style="background:var(--medium)"></i>中尺度</span><span><i class="swatch" style="background:var(--large)"></i>大尺度</span><span>▲ 高点　● 低点</span></div><span class="muted">拖动画面平移；滚轮缩放；悬停查看 OHLC / 枢轴确认详情</span></div>
<div id="chartStack" class="chart-stack"></div></section>
<section class="panel"><h2>当前区间统计</h2><div id="metrics" class="cards"></div><p class="foot">数量按发生日落在当前可视区间统计。中位间隔按同尺度连续确认点的发生日bar索引差计算；密度为每100根bar点数。同日跨尺度仍分别计数。</p></section>
<section class="panel"><h2>年 / 月发生数</h2><div class="tables"><div><h3>按年（含零计数）</h3><div class="table-wrap"><table id="yearTable"></table></div></div><div><h3>按月（当前区间，含零计数）</h3><div class="table-wrap"><table id="monthTable"></table></div></div></div></section>
<section class="panel"><h2>未确认端点预览（不纳入统计）</h2><div id="preview" class="preview-list"></div></section>
<section class="panel"><details><summary>ATR递推校验样例 / 数据源审计</summary><pre id="audit"></pre></details></section>
<p class="foot">数据只在本文件和配套JSON中；交互无需联网、CDN或服务端。枢轴发生时间与确认时间不同，任何确认日前行情都不能使用该点。</p>
<div id="tooltip" class="panel tooltip" role="status" aria-live="polite"></div>
</main><script id="report-data" type="application/json">__DATA_JSON__</script><script>
(()=>{
  'use strict';

  const DATA = JSON.parse(document.getElementById('report-data').textContent);
  const bars = DATA.bars || [];
  const N = bars.length;
  const SCALE_ORDER = ['2atr_5pct', '3atr_8pct', '5atr_12pct'];
  const COLORS = { '2atr_5pct': '#147d73', '3atr_8pct': '#c27612', '5atr_12pct': '#3d68b2' };
  const LABELS = { '2atr_5pct': '小尺度', '3atr_8pct': '中尺度', '5atr_12pct': '大尺度' };
  const PIVOTS = DATA.confirmed_pivots;
  const enabled = new Set(SCALE_ORDER);
  const state = { start: 0, end: Math.max(0, N - 1), mode: 'separate' };
  const stack = document.getElementById('chartStack');
  const canvases = {};
  const $ = id => document.getElementById(id);

  function date(index) {
    return bars[Math.max(0, Math.min(N - 1, index))]?.date || '—';
  }

  function dateIndex(value) {
    let low = 0;
    let high = N;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (bars[middle].date < value) low = middle + 1;
      else high = middle;
    }
    return low;
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, character => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[character]));
  }

  function addMetadata() {
    const input = DATA.input;
    const versions = DATA.versions;
    const rows = [
      ['本地数据', input.path],
      ['SHA-256', input.sha256],
      ['行情版本', input.data_revision],
      ['日期 / bar', `${input.first_date} → ${input.last_date} · ${input.bar_count.toLocaleString()}`],
      ['实际 tick', `${versions.tick_size} (${versions.tick_size_source})`],
      ['规则 / 引擎', `${versions.source_rule_version} · ${versions.engine_version} · ${versions.boundary_first_engine_version}`]
    ];
    $('meta').innerHTML = rows.map(([label, value]) =>
      `<div><b>${label}</b><span>${escapeHtml(String(value))}</span></div>`).join('');
    $('scaleChecks').innerHTML = SCALE_ORDER.map(scale =>
      `<label><input type="checkbox" checked data-scale="${scale}"><i class="swatch" style="background:${COLORS[scale]}"></i>${LABELS[scale]}</label>`
    ).join('');
    $('audit').textContent = JSON.stringify({
      versions: DATA.versions,
      input: DATA.input,
      validation: DATA.validation,
      atr_sample_checks: DATA.atr_sample_checks
    }, null, 2);
  }

  function createCharts() {
    for (const scale of SCALE_ORDER) {
      const section = document.createElement('section');
      section.className = 'chart-panel';
      section.dataset.scale = scale;
      section.innerHTML = `<div class="chart-title"><span>${LABELS[scale]} · 已确认高低点</span><span data-count></span></div><canvas aria-label="${LABELS[scale]}枢轴K线图" tabindex="0"></canvas>`;
      stack.appendChild(section);
      canvases[scale] = section.querySelector('canvas');
      bindCanvas(canvases[scale]);
    }
  }

  function visibleScales() {
    return SCALE_ORDER.filter(scale => enabled.has(scale));
  }

  function resizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return { width, height, ratio };
  }

  function drawChart(canvas, scales) {
    const dimensions = resizeCanvas(canvas);
    const context = canvas.getContext('2d');
    context.setTransform(dimensions.ratio, 0, 0, dimensions.ratio, 0, 0);
    const width = dimensions.width / dimensions.ratio;
    const height = dimensions.height / dimensions.ratio;
    context.clearRect(0, 0, width, height);
    context.fillStyle = '#fff';
    context.fillRect(0, 0, width, height);
    if (!N) {
      context.fillStyle = '#5b6775';
      context.fillText('无行情数据', 20, 30);
      return;
    }

    const margin = { left: 58, right: 14, top: 14, bottom: 36 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const start = state.start;
    const end = state.end;
    const visibleCount = Math.max(1, end - start + 1);
    const values = [];
    for (let index = start; index <= end; index++) {
      values.push(Number(bars[index].low), Number(bars[index].high));
    }
    for (const scale of scales) {
      for (const point of PIVOTS[scale] || []) {
        if (point.index >= start && point.index <= end) values.push(point.price);
      }
    }
    let minimum = Math.min(...values);
    let maximum = Math.max(...values);
    if (!Number.isFinite(minimum) || !Number.isFinite(maximum)) {
      minimum = 0;
      maximum = 1;
    }
    const span = Math.max(maximum - minimum, Math.abs(maximum) * 0.01, 1e-6);
    minimum -= span * 0.06;
    maximum += span * 0.06;
    const x = index => margin.left + (index - start) / Math.max(1, visibleCount - 1) * plotWidth;
    const y = value => margin.top + (maximum - value) / (maximum - minimum) * plotHeight;

    context.strokeStyle = '#e5eaef';
    context.lineWidth = 1;
    context.fillStyle = '#687582';
    context.font = '11px system-ui,sans-serif';
    context.textAlign = 'right';
    for (let tick = 0; tick < 5; tick++) {
      const value = minimum + (maximum - minimum) * tick / 4;
      const yTick = y(value);
      context.beginPath();
      context.moveTo(margin.left, yTick);
      context.lineTo(width - margin.right, yTick);
      context.stroke();
      context.fillText(value.toFixed(2), margin.left - 7, yTick + 4);
    }

    const candleStep = plotWidth / visibleCount;
    const candleWidth = Math.max(0.6, Math.min(5, candleStep * 0.58));
    context.lineWidth = Math.max(0.7, Math.min(1.2, candleStep * 0.18));
    for (let index = start; index <= end; index++) {
      const bar = bars[index];
      const xx = x(index);
      const open = Number(bar.open);
      const close = Number(bar.close);
      const color = close >= open ? '#16835a' : '#ce4848';
      context.strokeStyle = color;
      context.fillStyle = color;
      context.beginPath();
      context.moveTo(xx, y(Number(bar.high)));
      context.lineTo(xx, y(Number(bar.low)));
      context.stroke();
      const bodyTop = Math.min(y(open), y(close));
      const bodyHeight = Math.max(0.8, Math.abs(y(open) - y(close)));
      context.fillRect(xx - candleWidth / 2, bodyTop, candleWidth, bodyHeight);
    }

    for (const scale of scales) {
      context.fillStyle = COLORS[scale];
      const radius = scale === '5atr_12pct' ? 4.8 : scale === '3atr_8pct' ? 3.8 : 3;
      for (const point of PIVOTS[scale] || []) {
        if (point.index < start || point.index > end) continue;
        const xx = x(point.index);
        const yy = y(point.price);
        context.beginPath();
        if (point.kind === 'high') {
          context.moveTo(xx, yy - radius);
          context.lineTo(xx + radius, yy + radius);
          context.lineTo(xx - radius, yy + radius);
          context.closePath();
        } else {
          context.arc(xx, yy, radius, 0, Math.PI * 2);
        }
        context.fill();
      }
    }

    context.fillStyle = '#5b6775';
    context.font = '11px system-ui,sans-serif';
    context.textAlign = 'center';
    for (let tick = 0; tick < 6; tick++) {
      const index = Math.round(start + (end - start) * tick / 5);
      context.fillText(date(index), x(index), height - 11);
    }
    canvas._plot = { ...margin, plotWidth, plotHeight, width, height, start, end, minimum, maximum, x, y };
  }

  function render() {
    const scales = visibleScales();
    document.querySelectorAll('.chart-panel').forEach(section => {
      const scale = section.dataset.scale;
      section.hidden = state.mode === 'overlay' ? scale !== scales[0] : !enabled.has(scale);
      section.classList.toggle('overlay', state.mode === 'overlay' && scale === scales[0]);
    });
    for (const scale of SCALE_ORDER) {
      const section = canvases[scale].closest('.chart-panel');
      if (section.hidden) continue;
      const visible = state.mode === 'overlay' ? scales : [scale];
      drawChart(canvases[scale], visible);
      const count = visible.reduce((sum, key) => sum +
        (PIVOTS[key] || []).filter(point => point.index >= state.start && point.index <= state.end).length, 0);
      section.querySelector('[data-count]').textContent = `${count.toLocaleString()} 点`;
    }
    $('startRange').value = state.start;
    $('endRange').value = state.end;
    $('startDate').textContent = date(state.start);
    $('endDate').textContent = date(state.end);
    document.querySelectorAll('.preset').forEach(button => button.classList.remove('active'));
    updateStatistics();
  }

  function rowHtml(label, counts) {
    const total = SCALE_ORDER.reduce((sum, scale) => sum + counts[scale], 0);
    return `<tr><td>${label}</td>${SCALE_ORDER.map(scale => `<td>${counts[scale]}</td>`).join('')}<td><b>${total}</b></td></tr>`;
  }

  function periodKeys(startDate, endDate, monthly) {
    const startParts = startDate.slice(0, 7).split('-').map(Number);
    const endParts = endDate.slice(0, 7).split('-').map(Number);
    let year = startParts[0];
    let month = monthly ? startParts[1] : 1;
    const lastYear = endParts[0];
    const lastMonth = monthly ? endParts[1] : 1;
    const keys = [];
    while (year < lastYear || (year === lastYear && month <= lastMonth)) {
      keys.push(monthly ? `${year}-${String(month).padStart(2, '0')}` : String(year));
      if (!monthly) {
        year++;
      } else {
        month++;
        if (month > 12) { month = 1; year++; }
      }
    }
    return keys;
  }

  function updateStatistics() {
    const cards = [];
    for (const scale of SCALE_ORDER) {
      const points = (PIVOTS[scale] || []).filter(point =>
        point.index >= state.start && point.index <= state.end).slice().sort((a, b) => a.index - b.index);
      const highs = points.filter(point => point.kind === 'high').length;
      const lows = points.length - highs;
      const gaps = [];
      for (let index = 1; index < points.length; index++) {
        gaps.push(points[index].index - points[index - 1].index);
      }
      gaps.sort((a, b) => a - b);
      const median = gaps.length === 0 ? null : gaps.length % 2
        ? gaps[(gaps.length - 1) / 2]
        : (gaps[gaps.length / 2 - 1] + gaps[gaps.length / 2]) / 2;
      const span = Math.max(1, state.end - state.start + 1);
      const density = points.length / span * 100;
      cards.push(`<article class="metric" style="--accent:${COLORS[scale]}"><h3>${LABELS[scale]}</h3><div class="metric-main">${points.length.toLocaleString()} 点</div><div class="metric-sub">高 ${highs} · 低 ${lows} · 中位间隔 ${median === null ? '—' : `${median} bar`}</div><div class="metric-sub">密度 ${density.toFixed(2)} 点 / 100 bar</div></article>`);
    }
    $('metrics').innerHTML = cards.join('');

    const yearCounts = new Map();
    const monthCounts = new Map();
    for (const scale of SCALE_ORDER) {
      for (const point of PIVOTS[scale] || []) {
        if (point.index < state.start || point.index > state.end) continue;
        const year = point.date.slice(0, 4);
        const month = point.date.slice(0, 7);
        if (!yearCounts.has(year)) yearCounts.set(year, Object.fromEntries(SCALE_ORDER.map(key => [key, 0])));
        if (!monthCounts.has(month)) monthCounts.set(month, Object.fromEntries(SCALE_ORDER.map(key => [key, 0])));
        yearCounts.get(year)[scale]++;
        monthCounts.get(month)[scale]++;
      }
    }
    const header = `<thead><tr><th>期间</th>${SCALE_ORDER.map(scale => `<th>${LABELS[scale]}</th>`).join('')}<th>合计</th></tr></thead>`;
    const zeros = () => Object.fromEntries(SCALE_ORDER.map(scale => [scale, 0]));
    const yearRows = periodKeys(date(state.start), date(state.end), false).reverse().map(key =>
      rowHtml(key, yearCounts.get(key) || zeros()));
    const monthRows = periodKeys(date(state.start), date(state.end), true).reverse().map(key =>
      rowHtml(key, monthCounts.get(key) || zeros()));
    $('yearTable').innerHTML = header + `<tbody>${yearRows.join('')}</tbody>`;
    $('monthTable').innerHTML = header + `<tbody>${monthRows.join('')}</tbody>`;

    const endpoints = DATA.unconfirmed_endpoints_preview_only || [];
    $('preview').innerHTML = endpoints.length ? endpoints.map(point =>
      `<div class="preview-item"><b>${LABELS[point.scale]}</b> · 未确认${point.kind === 'high' ? '高点' : '低点'} ${point.price.toFixed(2)} · 发生 ${point.date} · 截至 ${point.state_as_of}仍为预览</div>`
    ).join('') : '<span class="muted">当前行情末端没有待确认端点。</span>';
  }

  function showPreset(name) {
    let start = 0;
    let end = N - 1;
    const lastDate = bars[N - 1]?.date || '';
    const lastYear = Number(lastDate.slice(0, 4));
    const monthDay = lastDate.slice(4);
    if (name === '5y') start = dateIndex(`${lastYear - 5}${monthDay}`);
    if (name === '1y') start = dateIndex(`${lastYear - 1}${monthDay}`);
    if (name === 'bottom2024') {
      start = dateIndex('2023-10-01');
      end = Math.max(start, dateIndex('2025-03-31') - 1);
    }
    state.start = Math.max(0, Math.min(start, N - 1));
    state.end = Math.max(state.start + 1, Math.min(end, N - 1));
    render();
    document.querySelector(`.preset[data-range="${name}"]`)?.classList.add('active');
  }

  function setRange(start, end) {
    if (N < 2) return;
    const minimumSpan = Math.min(20, N - 1);
    start = Math.max(0, Math.min(N - 1, Math.round(start)));
    end = Math.max(0, Math.min(N - 1, Math.round(end)));
    if (end - start < minimumSpan) {
      if (start + minimumSpan < N) end = start + minimumSpan;
      else start = Math.max(0, end - minimumSpan);
    }
    state.start = start;
    state.end = end;
    render();
  }

  function zoom(factor, anchor = (state.start + state.end) / 2) {
    const oldSpan = state.end - state.start + 1;
    const newSpan = Math.max(20, Math.min(N, Math.round(oldSpan * factor)));
    const ratio = (anchor - state.start) / Math.max(1, oldSpan - 1);
    let start = Math.round(anchor - ratio * (newSpan - 1));
    start = Math.max(0, Math.min(N - newSpan, start));
    setRange(start, start + newSpan - 1);
  }

  function bindCanvas(canvas) {
    canvas.addEventListener('wheel', event => {
      event.preventDefault();
      const plot = canvas._plot;
      if (!plot) return;
      const rect = canvas.getBoundingClientRect();
      const relative = Math.max(0, Math.min(1,
        (event.clientX - rect.left - plot.left) / plot.plotWidth));
      const anchor = state.start + relative * (state.end - state.start);
      zoom(event.deltaY < 0 ? 0.8 : 1.25, anchor);
    }, { passive: false });

    canvas.addEventListener('pointerdown', event => {
      canvas.setPointerCapture(event.pointerId);
      canvas._drag = { x: event.clientX, start: state.start, end: state.end };
    });
    canvas.addEventListener('pointermove', event => {
      if (!canvas._drag) {
        hover(canvas, event);
        return;
      }
      const plot = canvas._plot;
      if (!plot) return;
      const span = canvas._drag.end - canvas._drag.start + 1;
      const delta = Math.round(-(event.clientX - canvas._drag.x) /
        Math.max(1, plot.plotWidth) * span);
      const start = Math.max(0, Math.min(N - span, canvas._drag.start + delta));
      setRange(start, start + span - 1);
    });
    canvas.addEventListener('pointerup', () => { canvas._drag = null; });
    canvas.addEventListener('pointercancel', () => { canvas._drag = null; });
  }

  function hover(canvas, event) {
    const plot = canvas._plot;
    if (!plot) return;
    const rect = canvas.getBoundingClientRect();
    const pointerX = event.clientX - rect.left;
    const pointerY = event.clientY - rect.top;
    const visible = visibleScales();
    const candidates = [];
    for (const scale of visible) {
      for (const point of PIVOTS[scale] || []) {
        if (point.index < state.start || point.index > state.end) continue;
        const dx = plot.x(point.index) - pointerX;
        const dy = plot.y(point.price) - pointerY;
        const distance = Math.hypot(dx, dy);
        if (Math.abs(dx) <= 8 && Math.abs(dy) <= 8) candidates.push({ point, distance });
      }
    }
    candidates.sort((a, b) => a.distance - b.distance);
    const nearest = candidates[0]?.point || null;
    const barIndex = nearest
      ? nearest.index
      : Math.max(plot.start, Math.min(plot.end,
        Math.round(plot.start + (pointerX - plot.left) / plot.plotWidth * (plot.end - plot.start))));
    const bar = bars[barIndex];
    const related = nearest
      ? visible.flatMap(scale => (PIVOTS[scale] || []).filter(point => point.index === nearest.index))
      : [];
    const cells = [
      ['日期', date(barIndex)],
      ['OHLC', `开 ${Number(bar.open).toFixed(2)} · 高 ${Number(bar.high).toFixed(2)} · 低 ${Number(bar.low).toFixed(2)} · 收 ${Number(bar.close).toFixed(2)}`],
      ['枢轴命中', related.length ? `${related.length} 个（跨尺度分别计数）` : '无已确认点']
    ];
    related.forEach((point, index) => {
      cells.push([`点 ${index + 1}`, `${LABELS[point.scale]} ${point.kind === 'high' ? '高点' : '低点'} ${point.price.toFixed(2)} · ${point.date} → ${point.confirmed_date} · lag ${point.confirmation_lag_bars} bar`]);
    });
    $('tooltip').innerHTML = cells.map(([label, value]) =>
      `<div><b>${escapeHtml(label)}</b>${escapeHtml(value)}</div>`).join('');
  }

  function init() {
    if (!N) {
      stack.innerHTML = '<p class="empty">行情为空，不能生成图表。</p>';
      return;
    }
    $('startRange').max = N - 1;
    $('endRange').max = N - 1;
    $('endRange').value = N - 1;
    createCharts();
    addMetadata();
    $('startRange').addEventListener('input', event => setRange(Number(event.target.value), state.end));
    $('endRange').addEventListener('input', event => setRange(state.start, Number(event.target.value)));
    document.querySelectorAll('.preset').forEach(button =>
      button.addEventListener('click', () => showPreset(button.dataset.range)));
    $('mode').addEventListener('change', event => { state.mode = event.target.value; render(); });
    $('scaleChecks').addEventListener('change', event => {
      const scale = event.target.dataset.scale;
      if (!scale) return;
      if (event.target.checked) enabled.add(scale);
      else enabled.delete(scale);
      if (!enabled.size) { enabled.add(scale); event.target.checked = true; }
      render();
    });
    $('zoomIn').addEventListener('click', () => zoom(0.7));
    $('zoomOut').addEventListener('click', () => zoom(1.43));
    $('reset').addEventListener('click', () => showPreset('all'));
    $('tooltip').innerHTML = '<div><b>选择提示</b>将鼠标移到图上查看日期、OHLC和同日枢轴。</div>';
    render();
  }

  window.addEventListener('resize', render);
  init();
})();
</script>
</body></html>'''


def main():
    instrument = SimpleNamespace(code="600108", kind="stock", path=str(INPUT_PATH))
    data = _extract_stream(instrument, STOCK_ROOT)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
                        encoding="utf-8")
    OUT_HTML.write_text(_report_html(data), encoding="utf-8")
    summary = {scale: {"count": len(points),
                       "high": sum(point["kind"] == "high" for point in points),
                       "low": sum(point["kind"] == "low" for point in points),
                       "median_gap_bars": (int(np.median(np.diff([p["index"] for p in points])))
                                           if len(points) > 1 else None)}
               for scale, points in data["confirmed_pivots"].items()}
    print(json.dumps({"status": "complete", "input": data["input"],
                      "tick_size": data["versions"]["tick_size"], "summary": summary,
                      "preview_only": data["unconfirmed_endpoints_preview_only"],
                      "validation": data["validation"], "html": str(OUT_HTML.resolve()),
                      "json": str(OUT_JSON.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
