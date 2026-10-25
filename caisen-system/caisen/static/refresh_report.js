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
      const unchanged = item.previous_sha256 && item.previous_sha256 === item.file_sha256
        ? '；文件内容未变化' : '';
      message += `${item.code || ''}${range}${latest}${source}${attempts}${unchanged}`;
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
      const changed = item.previous_sha256 && item.previous_sha256 === item.file_sha256
        ? '文件内容未变化' : '文件已写入';
      const hash = item.file_sha256 ? ` · SHA-256 ${escapeHtml(item.file_sha256.slice(0, 12))}…` : '';
      return `<li><strong>${kind} ${escapeHtml(item.code || '未知代码')}</strong><span>${range} · ${source} · 请求 ${attempts} 次 · ${changed}${hash}</span></li>`;
    }).join('');
    const failureRows = failures.slice(0, limit).map(item => {
      const kind = item.kind === 'etf' ? 'ETF' : '股票';
      return `<li><strong>${kind} ${escapeHtml(item.code || '未知代码')}</strong><span>${escapeHtml(item.error || '未知错误')}</span></li>`;
    }).join('');
    const extraSuccesses = Math.max(0, results.length - limit);
    const extraFailures = Math.max(0, failures.length - limit);
    const jobId = escapeHtml(job.job_id || '未提供');
    const hasFailedSymbols = Number(job.failure_count || 0) > 0;
    const canResume = ['interrupted', 'failed'].includes(job.status)
      || (job.status === 'completed' && hasFailedSymbols);
    const recoveryLabel = job.status === 'completed' ? '重试失败标的' : '恢复未完成的更新';
    const recovery = canResume
      ? `<button type="button" class="resume-refresh" data-resume-refresh="${jobId}">${recoveryLabel}</button>` : '';
    return `<details class="refresh-report"${results.length + failures.length <= 1 ? ' open' : ''}>
      <summary>更新明细：成功 ${Number(job.success_count || 0)} 个，失败 ${Number(job.failure_count || 0)} 个</summary>
      <div class="refresh-report-body"><p>任务 ${jobId}。结果保留来源、日期、请求次数和文件校验信息。</p>${recovery}
      <h2>成功</h2>${results.length ? `<ul>${successRows}</ul>` : '<p>没有成功更新的标的。</p>'}
      ${extraSuccesses ? `<p>另有 ${extraSuccesses} 个成功标的；完整结果可按任务 ID 查询。</p>` : ''}
      <h2>失败</h2>${failures.length ? `<ul class="refresh-failures">${failureRows}</ul>` : '<p>没有失败标的。</p>'}
      ${extraFailures ? `<p>另有 ${extraFailures} 个失败标的；完整结果可按任务 ID 查询。</p>` : ''}</div>
    </details>`;
  }

  function renderRecoveryList(jobs) {
    const recoverable = Array.isArray(jobs) ? jobs : [];
    if (!recoverable.length) return '';
    const rows = recoverable.map(job => {
      const jobId = escapeHtml(job.job_id || '未提供');
      const single = job.scope === 'single';
      const status = job.status === 'completed' ? '部分标的失败' :
        job.status === 'interrupted' ? '服务重启时中断' :
        job.status === 'failed' ? '任务失败' : '更新仍在运行';
      const progress = `${Number(job.completed || 0)}/${Number(job.total || 0)}`;
      const retryable = ['interrupted', 'failed'].includes(job.status) ||
        (job.status === 'completed' && Number(job.failure_count || 0) > 0);
      const action = retryable
        ? `<button type="button" data-resume-refresh="${jobId}">${job.status === 'completed' ? '重试失败标的' : '恢复未完成的更新'}</button>`
        : `<button type="button" data-watch-refresh="${jobId}">查看进度</button>`;
      return `<li><strong>${single ? '单标的' : '全市场'}行情更新 · ${status}</strong><span>任务 ${jobId} · ${progress} · 成功 ${Number(job.success_count || 0)} · 失败 ${Number(job.failure_count || 0)}</span>${action}</li>`;
    }).join('');
    return `<section class="refresh-recovery"><h2>可恢复的行情更新</h2><ul>${rows}</ul></section>`;
  }

  return { formatRefreshSummary: summary, renderRefreshReport: render,
    renderRefreshRecoveryList: renderRecoveryList };
});
