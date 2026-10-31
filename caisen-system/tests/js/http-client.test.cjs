const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../../caisen/static/http-client.js'), 'utf8');

function clientWith(response) {
  const window = {};
  vm.runInNewContext(source, { window, fetch: async () => response });
  return window.CaisenHttpClient;
}

test('HTTP JSON errors use the server detail', async () => {
  const client = clientWith({ ok: false, status: 422, text: async () => '{"detail":"日期无效"}' });
  await assert.rejects(client.fetchJson('/api'), /日期无效/);
});

test('plain text server errors become readable messages', async () => {
  const client = clientWith({ ok: false, status: 500, text: async () => 'Internal Server Error' });
  await assert.rejects(client.fetchJson('/api'), /请求失败（500）：服务器暂时无法完成请求/);
});

test('non-standard NaN bodies are rejected as invalid JSON', async () => {
  const client = clientWith({ ok: true, status: 200, text: async () => '{"value":NaN}' });
  await assert.rejects(client.fetchJson('/api'), /无法识别的响应/);
});
