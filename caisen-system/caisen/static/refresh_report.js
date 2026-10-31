(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.CaisenRefreshReport = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, character => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[character]);
  }

  function summary(job, single) {
    const successes = Number(job.success_count || 0);
    const failures = Number(job.failure_count || 0);
    let message = `行情更新完成：成功 ${successes} 个，失败 ${failures} 个。`;
    if (single && Array.isArray(job.results) && job.results.length) {
      const item = job.results[job.results.length - 1];
      const range = item.first_date && item.last_date
        ? `；覆盖 ${item.first_date} 至 ${item.last_date}` : '';
      const latest = item.last_date ? `；最新数据截至 ${item.last_date}` : '';
      const source = item.source ? `；来源 ${item.source}` : '';
      const attempts = item.provider_attempts ? `；请求 ${item.provider_attempts} 次` : '';
      message += `${item.code || ''}${range}${latest}${source}${attempts}`;
    }
    if (failures) message += '请查看失败标的和原因。';
    return message;
  }

  function render(job, limit = 20) {
    const results = Array.isArray(job.results) ? job.results : [];
    const failures = Array.isArray(job.failures) ? job.failures : [];
    const successRows = results.slice(0, limit).map(item => {
      const kind = item.kind === 'etf' ? 'ETF' : '股票';
      const range = item.first_date && item.last_date
        ? `${escapeHtml(item.first_date)} 至 ${escapeHtml(item.last_date)}` : '日期范围未知';
      const source = item.source ? escapeHtml(item.source) : '来源未知';
      const attempts = Number(item.provider_attempts || 0);
      return `<li><strong>${kind} ${escapeHtml(item.code || '未知代码')}</strong><span>${range} · ${source} · 请求 ${attempts} 次 · 已覆盖本地行情</span></li>`;
    }).join('');
    const failureRows = failures.slice(0, limit).map(item => {
      const kind = item.kind === 'etf' ? 'ETF' : '股票';
      return `<li><strong>${kind} ${escapeHtml(item.code || '未知代码')}</strong><span>${escapeHtml(item.error || '未知错误')}</span></li>`;
    }).join('');
    const extraSuccesses = Math.max(0, results.length - limit);
    const extraFailures = Math.max(0, failures.length - limit);
    const jobId = escapeHtml(job.job_id || '未提供');
    return `<details class="refresh-report"${results.length + failures.length <= 1 ? ' open' : ''}>
      <summary>更新明细：成功 ${Number(job.success_count || 0)} 个，失败 ${Number(job.failure_count || 0)} 个</summary>
      <div class="refresh-report-body"><p>任务 ${jobId}。结果记录来源、日期和请求次数。</p>
      <h2>成功</h2>${results.length ? `<ul>${successRows}</ul>` : '<p>没有成功更新的标的。</p>'}
      ${extraSuccesses ? `<p>另有 ${extraSuccesses} 个成功标的；完整结果可按任务 ID 查询。</p>` : ''}
      <h2>失败</h2>${failures.length ? `<ul class="refresh-failures">${failureRows}</ul>` : '<p>没有失败标的。</p>'}
      ${extraFailures ? `<p>另有 ${extraFailures} 个失败标的；完整结果可按任务 ID 查询。</p>` : ''}</div>
    </details>`;
  }

  return { formatRefreshSummary: summary, renderRefreshReport: render };
});
