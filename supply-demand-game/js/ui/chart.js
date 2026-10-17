// 纯Canvas K线引擎：蜡烛、区域、线位、交易标记、十字光标
export class Chart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.bars = [];
    this.visibleTo = 0;        // 已揭示到第几根（不含）
    this.from = 0;             // 视窗起始
    this.count = 90;           // 视窗内K线数
    this.zones = [];
    this.big = null;
    this.trade = null;         // {entry, stop, targets[], events[]}
    this.selection = null;     // {start,end} 玩家框选
    this.decisionIdx = -1;
    this.hoverIdx = -1;
    this.onSelect = null;
    this._bind();
  }

  setData(bars, decisionIdx) {
    this.bars = bars;
    this.decisionIdx = decisionIdx;
    this.visibleTo = decisionIdx + 1;
    this.from = Math.max(0, this.visibleTo - this.count);
    this.draw();
  }

  reveal(n = 1) {
    this.visibleTo = Math.min(this.bars.length, this.visibleTo + n);
    if (this.visibleTo - this.from > this.count) this.from = this.visibleTo - this.count;
    this.draw();
  }

  _layout() {
    const { width: W, height: H } = this.canvas;
    const padR = 64, padB = 26, padT = 10;
    return { W, H, padR, padB, padT, plotW: W - padR, plotH: H - padB - padT };
  }

  _priceRange() {
    const slice = this.bars.slice(this.from, this.visibleTo);
    let lo = Infinity, hi = -Infinity;
    for (const b of slice) { lo = Math.min(lo, b.low); hi = Math.max(hi, b.high); }
    for (const z of this.zones) {
      if (z.baseStart >= this.from && z.baseStart < this.visibleTo) {
        lo = Math.min(lo, z.jieLine); hi = Math.max(hi, z.keWick);
      }
    }
    if (this.trade) {
      for (const p of [this.trade.entry, this.trade.stop, ...(this.trade.targets || [])]) {
        if (p != null) { lo = Math.min(lo, p); hi = Math.max(hi, p); }
      }
    }
    const pad = (hi - lo) * 0.08 || 1;
    return [lo - pad, hi + pad];
  }

  draw() {
    const ctx = this.ctx;
    const { W, H, padT, plotW, plotH, padR, padB } = this._layout();
    ctx.clearRect(0, 0, W, H);
    if (!this.bars.length) return;
    const [lo, hi] = this._priceRange();
    const n = this.visibleTo - this.from;
    const bw = plotW / n;
    const x = i => (i - this.from) * bw + bw / 2;
    const y = p => padT + (hi - p) / (hi - lo) * plotH;
    this._x = x; this._y = y; this._bw = bw;

    // 网格与价格轴
    ctx.strokeStyle = "#eef0f6"; ctx.fillStyle = "#98a"; ctx.font = "11px sans-serif";
    for (let g = 0; g <= 5; g++) {
      const p = lo + (hi - lo) * g / 5, yy = y(p);
      ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(plotW, yy); ctx.stroke();
      ctx.fillText(p.toFixed(2), plotW + 6, yy + 3);
    }

    // 决策线
    if (this.decisionIdx >= this.from && this.decisionIdx < this.visibleTo) {
      ctx.strokeStyle = "#b8bfea"; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x(this.decisionIdx), padT); ctx.lineTo(x(this.decisionIdx), H - padB); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#7a83c9"; ctx.fillText("决策点", x(this.decisionIdx) + 4, padT + 10);
    }

    // 大级别区域线
    if (this.big) {
      for (const [p, label] of [[this.big.supplyKe, "大供给"], [this.big.demandKe, "大需求"]]) {
        if (p == null) continue;
        ctx.strokeStyle = "#c9ccd8"; ctx.setLineDash([2, 5]);
        ctx.beginPath(); ctx.moveTo(0, y(p)); ctx.lineTo(plotW, y(p)); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#aab"; ctx.fillText(label, 4, y(p) - 3);
      }
    }

    // 供需区
    for (const z of this.zones) {
      if (z.baseEnd < this.from || z.baseStart > this.visibleTo) continue;
      const ke = z._ke ?? (z.type === "D" ? z.keBodyBottom : z.keBodyTop);
      const top = Math.max(ke, z.jieLine), bot = Math.min(ke, z.jieLine);
      ctx.fillStyle = z.type === "D" ? "rgba(61,155,122,.16)" : "rgba(225,91,91,.16)";
      const x0 = x(z.baseStart) - bw / 2;
      ctx.fillRect(x0, y(top), (this.visibleTo - z.baseStart) * bw, y(bot) - y(top));
      ctx.strokeStyle = z.type === "D" ? "#3d9b7a" : "#e15b5b";
      ctx.setLineDash([6, 3]);
      ctx.beginPath(); ctx.moveTo(x0, y(ke)); ctx.lineTo(plotW, y(ke)); ctx.stroke();
      ctx.setLineDash([2, 4]);
      ctx.strokeStyle = "#889";
      ctx.beginPath(); ctx.moveTo(x0, y(z.jieLine)); ctx.lineTo(plotW, y(z.jieLine)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = z.type === "D" ? "#2c7a5f" : "#b64747";
      ctx.fillText(z.type === "D" ? "克线" : "克线", plotW - 30, y(ke) - 3);
      ctx.fillStyle = "#889"; ctx.fillText("竭线", plotW - 30, y(z.jieLine) + 10);
    }

    // 玩家框选
    if (this.selection) {
      const { start, end } = this.selection;
      ctx.fillStyle = "rgba(122,131,201,.22)";
      ctx.fillRect(x(start) - bw / 2, padT, (end - start + 1) * bw, plotH);
    }

    // 蜡烛
    for (let i = this.from; i < this.visibleTo; i++) {
      const b = this.bars[i];
      const up = b.close >= b.open;
      ctx.strokeStyle = ctx.fillStyle = up ? "#e15b5b" : "#3d9b7a";
      const cx = x(i);
      ctx.beginPath(); ctx.moveTo(cx, y(b.high)); ctx.lineTo(cx, y(b.low)); ctx.stroke();
      const top = y(Math.max(b.open, b.close));
      const h = Math.max(1, y(Math.min(b.open, b.close)) - top);
      ctx.fillRect(cx - bw * 0.32, top, bw * 0.64, h);
    }

    // 交易线
    if (this.trade) {
      const t = this.trade;
      const line = (p, color, label) => {
        if (p == null) return;
        ctx.strokeStyle = color; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(0, y(p)); ctx.lineTo(plotW, y(p)); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = color; ctx.fillText(label, 4, y(p) - 3);
      };
      line(t.entry, "#4a5bd0", `入场 ${t.entry?.toFixed(2)}`);
      line(t.stop, "#e6a23c", `止损 ${t.stop?.toFixed(2)}`);
      (t.targets || []).forEach((p, i) => line(p, "#909399", `目标${i + 1}`));
      for (const ev of t.events || []) {
        const cx = x(ev.idx ?? this.visibleTo - 1);
        ctx.fillStyle = ev.type === "stop" ? "#e6a23c" : ev.type === "target" ? "#3d9b7a" : ev.type === "protect" ? "#909399" : "#4a5bd0";
        ctx.beginPath(); ctx.arc(cx, y(ev.price), 4, 0, 7); ctx.fill();
        ctx.fillStyle = "#333"; ctx.fillText(evLabel(ev), cx + 6, y(ev.price) + 3);
      }
    }

    // 十字光标
    if (this.hoverIdx >= this.from && this.hoverIdx < this.visibleTo) {
      const b = this.bars[this.hoverIdx];
      ctx.fillStyle = "rgba(38,40,58,.85)";
      ctx.fillRect(this.canvas.width - padR, 20, padR, 74);
      ctx.fillStyle = "#fff";
      ctx.fillText(b.date, this.canvas.width - padR + 4, 34);
      ctx.fillText(`开${b.open.toFixed(2)} 收${b.close.toFixed(2)}`, this.canvas.width - padR + 4, 50);
      ctx.fillText(`高${b.high.toFixed(2)} 低${b.low.toFixed(2)}`, this.canvas.width - padR + 4, 66);
      const cls = Math.abs(b.close - b.open) / Math.max(1e-9, b.high - b.low) < 0.5 ? "静态·积累" : "动态·消耗";
      ctx.fillStyle = cls.startsWith("静态") ? "#8fd0b5" : "#f2a0a0";
      ctx.fillText(cls, this.canvas.width - padR + 4, 84);
    }
  }

  _bind() {
    const c = this.canvas;
    let dragStart = null;
    c.addEventListener("mousedown", e => {
      const i = this._idxAt(e);
      if (i == null) return;
      dragStart = i;
      this.selection = { start: i, end: i };
      this.draw();
    });
    c.addEventListener("mousemove", e => {
      const i = this._idxAt(e);
      this.hoverIdx = i ?? -1;
      if (dragStart != null && i != null) {
        this.selection = { start: Math.min(dragStart, i), end: Math.max(dragStart, i) };
      }
      this.draw();
    });
    c.addEventListener("mouseup", e => {
      if (dragStart != null && this.onSelect && this.selection) {
        this.onSelect(this.selection.start, this.selection.end);
      }
      dragStart = null;
    });
    c.addEventListener("wheel", e => {
      e.preventDefault();
      this.count = Math.max(30, Math.min(160, this.count + (e.deltaY > 0 ? 10 : -10)));
      this.from = Math.max(0, this.visibleTo - this.count);
      this.draw();
    }, { passive: false });
  }

  _idxAt(e) {
    const r = this.canvas.getBoundingClientRect();
    const px = (e.clientX - r.left) * (this.canvas.width / r.width);
    const { plotW } = this._layout();
    if (px > plotW) return null;
    const n = this.visibleTo - this.from;
    const i = this.from + Math.floor(px / (plotW / n));
    return i >= this.from && i < this.visibleTo ? i : null;
  }
}

function evLabel(ev) {
  return { fill: "成交", stop: "止损", target: "止盈", protect: "推保护", exit: "离场" }[ev.type] || ev.type;
}
