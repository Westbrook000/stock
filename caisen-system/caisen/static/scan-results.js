(function (root) {
  function buildQuery({page = 1, action = "", patternTypes = []} = {}) {
    const query = new URLSearchParams({page: String(page), sort: "rank", order: "asc"});
    if (action) query.set("action", action);
    if (patternTypes.length) query.set("pattern_types", patternTypes.join(","));
    return query.toString();
  }

  root.CaisenScanResults = {buildQuery};
  if (typeof module !== "undefined" && module.exports) module.exports = root.CaisenScanResults;
})(typeof window === "undefined" ? globalThis : window);
