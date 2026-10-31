# 蔡森量价结构分析系统

本地工作台，默认 http://127.0.0.1:8301 。提供市场扫描、当前形态与已结束形态图表、买卖提示和 Backtrader 模拟；不连接券商。

## 使用

从仓库根目录启动：

```sh
.venv/bin/python caisen-system/scripts/serve.py
```

- **市场扫描**：扫描全部本地股票与 ETF 的活动十招，支持日线和周线；结果每个标的一行，默认买入优先。完成后按行动或形态多选筛选，不重新扫描。
- **形态详情**：显示当前形态及已结束形态。点击“查看该形态”后自动切换对应日/周线，定位到形态区间，标出结构点、颈线/边界、确认点和结束位置。“回到最新”清除历史选择。
- **交易提示**：四种底部/上涨中继结构可形成买入计划；风险上限7%、首目标至少2R。结构歧义、未突破或空间不足时返回等待。卖出仅提示已有持仓者考虑退出。
- **模拟执行**：使用 Backtrader、零费用，验收撮合与账目；策略效果验证暂缓。

工作台只保留市场扫描和形态详情两个入口。人工复核、盲标/人工校准工具和记录、任意历史截止日期回放均已移除。当前规则阈值继续使用固定工程参数，不代表已验证的准确率。

## 行情更新

详情可更新单个标的，市场扫描页可更新全部本地标的。AkShare 获取股票前复权日线和 ETF 原始日线；仅整理字段和日期格式，不校验 OHLCV 数值、历史覆盖或行数，不保存旧文件哈希、快照或刷新清单。采用临时文件原子替换；下载、格式转换或写入失败时保留对应旧行情。临时网络错误最多请求3次，间隔0.5/1秒。

刷新仅在当前进程中保留进度和结果；没有恢复接口、续跑检查点或服务重启恢复。中断后重新发起整次更新。每天上海时间05:00的本地自动化调用：

```sh
.venv/bin/python caisen-system/scripts/daily_refresh.py
```

运行需要电脑可用且 Codex 桌面端保持运行。分析和扫描本身离线，不触发行情更新。

## 接口

- `GET /api/v1/analysis?code=000001&kind=stock&timeframe=1d`：最新分析、全部图表行情与形态证据；不支持历史截止日期。
- `POST /api/v1/scans`：`{"timeframes":["1d","1w"]}`，完整本地库存扫描，形态筛选只在结果页进行。
- `GET /api/v1/jobs/{job_id}`：读取当前任务进度。
- `POST /api/v1/data/refresh`：`{"scope":"all"}` 或 `{"code":"000001","kind":"stock"}`。
- `GET /api/v1/patterns/{pattern_id}`：机器形态记录。
- `POST /api/v1/backtests`：Backtrader模拟，零费用。

状态库默认 `caisen-system/state/caisen.sqlite3`，可用 `CAISEN_DB` 指定。行情目录为仓库的 `data/stocks`、`data/etfs`。分析结果继续保留数据版本标识。

## 验证

```sh
.venv/bin/python -m pytest -q caisen-system/tests
node --test caisen-system/tests/js/*.test.cjs
node --check caisen-system/caisen/static/app.js
```

前缀因果性检查保留：未来 K 线不能改写过去已知的拐点与确认事件。周线按已核验的交易日历判断完整周；未完成周只作预览。现有本地库存、复权和量能口径不等于完整历史市场，功能运行不代表策略盈利有效。
