(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.CaisenScanFilters = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  const TERMINAL_STATES = new Set(['invalidated', 'target_reached', 'followthrough_confirmed']);

  function filterScanRows(rows, filters = {}) {
    const { action = '', patternType = '' } = filters;
    return (rows || []).filter(row => {
      if (action && row.action !== action) return false;
      if (!patternType) return true;
      return (row.patterns || []).some(pattern =>
        pattern.pattern_type === patternType && !TERMINAL_STATES.has(pattern.state)
      );
    });
  }

  return { filterScanRows };
});
