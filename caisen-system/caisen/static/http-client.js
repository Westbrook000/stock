(function (global) {
  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const body = await response.text();
    let data;
    try {
      data = JSON.parse(body);
    } catch (_) {
      const plain = body.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 240);
      const genericServerError = /^(internal server error|bad gateway|service unavailable)$/i.test(plain);
      const message = plain && !genericServerError ? plain : '服务器暂时无法完成请求，请稍后重试。';
      throw new Error(response.ok
        ? '服务器返回了无法识别的响应。'
        : `请求失败（${response.status}）：${message}`);
    }
    if (!response.ok) {
      const detail = data && (data.detail || data.error);
      throw new Error(typeof detail === 'string' ? detail : `请求失败（${response.status}）`);
    }
    return data;
  }

  global.CaisenHttpClient = { fetchJson };
})(window);
