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

  return { filterScanRows };
});
