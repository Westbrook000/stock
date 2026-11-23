(function (root) {
  function buildQuery({page = 1, action = "", patternTypes = []} = {}) {
    const query = new URLSearchParams({page: String(page), sort: "rank", order: "asc"});
    if (action) query.set("action", action);
    if (patternTypes.length) query.set("pattern_types", patternTypes.join(","));
    return query.toString();
  }

  function historyPatterns(row) {
    const relevant = row.action === "WATCH" ? (row.watch || []).map(item => item.pattern_id)
      : row.action === "CONSIDER_BUY" ? [row.selected_plan?.pattern_id].filter(Boolean) : [];
    const ended = new Set(["invalidated", "expired", "target_reached", "followthrough_confirmed"]);
    return ["1d", "1w"].flatMap(timeframe => {
      const matches = (row.patterns || []).filter(item => item.timeframe === timeframe && !ended.has(item.state));
      const triggered = matches.find(item => relevant.includes(item.pattern_id));
      const primary = matches.filter(item => !item.parent_id && !item.execution_suppressed);
      const main = primary.filter(item => item.selected_main || item.is_main_structure);
      const latest = (main.length ? main : primary).reduce((best, item) =>
        !best || (item.confirmation_date || item.formed_at || "") >
          (best.confirmation_date || best.formed_at || "") ? item : best, null);
      return triggered || latest ? [triggered || latest] : [];
    });
  }

  function decodeMiniCandles(encoded) {
    if (!encoded) return [];
    try {
      const bytes = atob(encoded);
      if (bytes.length > 96 || bytes.length % 4) return [];
      return Array.from({length: bytes.length / 4}, (_, index) => {
        const start = index * 4;
        return {open:bytes.charCodeAt(start), close:bytes.charCodeAt(start+1),
                low:bytes.charCodeAt(start+2), high:bytes.charCodeAt(start+3)};
      });
    } catch { return []; }
  }

  root.CaisenScanResults = {buildQuery, historyPatterns, decodeMiniCandles};
  if (typeof module !== "undefined" && module.exports) module.exports = root.CaisenScanResults;
})(typeof window === "undefined" ? globalThis : window);
