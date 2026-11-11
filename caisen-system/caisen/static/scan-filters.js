(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.CaisenScanFilters = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const TERMINAL_STATES = new Set(['invalidated', 'target_reached', 'followthrough_confirmed']);

  function filterScanRows(rows, filters = {}) {
    const { action = '', patternType = '' } = filters;
    const patternTypes = filters.patternTypes || (patternType ? [patternType] : []);
    return (rows || []).filter(row => {
      if (action && row.action !== action) return false;
      if (!patternTypes.length) return true;
      return (row.patterns || []).some(pattern =>
        patternTypes.includes(pattern.pattern_type) && !TERMINAL_STATES.has(pattern.state)
      );
    });
  }

  function mainPatterns(row) {
    const patterns = row?.patterns || [];
    const selected = patterns.filter(pattern => {
      const structure = (pattern.evidence || []).some(item => item.code === 'main_structure' && item.value?.role === 'main');
      const chosen = (pattern.evidence || []).some(item => item.code === 'main_selection' && item.value?.selected === true);
      return (pattern.is_main_structure === true || structure) && (pattern.selected_main === true || chosen);
    });
    const byTimeframe = {};
    for (const timeframe of ['1d', '1w']) {
      const candidates = selected.filter(pattern => pattern.timeframe === timeframe);
      candidates.sort((a, b) => String(b.formed_at || '').localeCompare(String(a.formed_at || '')));
      if (candidates[0]) byTimeframe[timeframe] = candidates[0];
    }
    return byTimeframe;
  }

  function patternSupportLabel(label, status, historical = false) {
    if (status === 'supported') return label;
    if (status === 'unsupported') return `${label}（当前扫描暂未支持${historical ? '，历史结果可查' : ''}）`;
    return `${label}（正在读取支持状态）`;
  }

  function patternChipMarkup(value, label, selected = false, disabled = false) {
    const escape = text => String(text ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
    return `<button type="button" class="pattern-filter-chip${selected ? ' selected' : ''}" data-value="${escape(value)}" aria-pressed="${selected}"${disabled ? ' disabled aria-disabled="true"' : ''}>${escape(label)}</button>`;
  }

  return { filterScanRows, mainPatterns, patternChipMarkup, patternSupportLabel };
});
