# 蔡森量价结构分析系统

本地工作台，默认 http://127.0.0.1:8301 。提供市场扫描、当前形态与已结束形态图表、买卖提示和 Backtrader 模拟；不连接券商。

## 使用

从仓库根目录启动：

```sh
.venv/bin/python caisen-system/scripts/serve.py
```

- **市场扫描**：扫描全部本地股票与 ETF 的活动十招，支持日线和周线；结果每个标的一行，默认买入优先。完成后按行动或形态多选筛选，不重新扫描。
- **形态详情**：显示当前形态及已结束形态。点击“查看该形态”后自动切换对应日/周线，定位到形态区间，标出结构点、颈线/边界、确认点和结束位置。“回到最新”清除历史选择。
- **交易提示**：四种底部/上涨中继结构可形成买入计划；有效突破后按下一交易日开盘介入，按实际成交价止损8%、止盈20%；结构歧义或尚未突破时返回等待。卖出仅提示已有持仓者考虑退出。
- **日线回测**：固定800只股票、200只ETF；连续ATR/波段与五年形态搜索，8个 spawn 子进程识别、Backtrader逐账户撮合。当前规则为2倍ATR压力候选、突破前一根已完成K线的1倍ATR缓冲；同结构等价冻结颈线按边界阶段消费。基线每账户10万元、100股/份整手、止损8%、止盈20%、零费用、不加仓、无最长持有期、期末收盘退出。仅保留W底、头肩底、底部三角形、下飘旗形上涨中继四类买入信号，不启用 min3 或 open-admission 实验。

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
- `POST /api/v1/backtests`：提交固定样本日线回测，返回任务ID；GET同路径查询历史任务，GET `/api/v1/backtests/{id}` 查询进度，GET `/api/v1/backtests/{id}/report` 查看报告。旧固定回测POST返回410，旧报告继续可读。

状态库默认 `caisen-system/state/caisen.sqlite3`，可用 `CAISEN_DB` 指定。行情目录为仓库的 `data/stocks`、`data/etfs`。分析结果继续保留数据版本标识。

## 验证

```sh
.venv/bin/python -m pytest -q caisen-system/tests
node --test caisen-system/tests/js/*.test.cjs
node --check caisen-system/caisen/static/app.js
```

前缀因果性检查保留：未来 K 线不能改写过去已知的拐点与确认事件。周线按已核验的交易日历判断完整周；未完成周只作预览。现有本地库存、复权和量能口径不等于完整历史市场，功能运行不代表策略盈利有效。


## 固定日线研究样本

成员清单：`research/universes/daily_800_stock_200_etf_v1.json`，固定种子20261101；成员随行情更新保持不变，数据使用每次任务截止日的本地行情。规则快照：`research/rules/daily_four_patterns_fixed_8_20_v1.json`。样本缺失或失败保留原代码及原因，不补抽。

```sh
# 在 caisen-system 目录查看当前 CLI 参数
python3 scripts/run_daily_research.py --help
# 运行固定1000标的日线研究；默认使用已验证的执行策略
python3 scripts/run_daily_research.py
```

当前版本标识为 `caisen-main-bottoms-v9`、`caisen-daily-continuous-v5`、`caisen-daily-boundary-first-v3`；主形态规则为 `caisen-v22-main-bottom-v1`，颈线选择规则为 `pattern_matched_pressure_envelope_v7`。已验证的 spawn-8 执行策略见 `reports/daily-research/benchmark/benchmark-execution-v5-current-rules-36.json`：同一冻结输入和当前代码下，36个账户的单进程与8进程完整账本签名一致，8进程耗时约340秒，峰值进程树内存约1.91GB。该结论仅覆盖这36个账户，不代表1000账户的当前版本验收，也不代表策略盈利。

历史 `reports/daily-research/full-20261104-v8-thread2-boundary/` 是 v8 的1000账户冷运行，耗时约84分钟；它不是当前 v9 规则的1000账户结果。当前 `--benchmark` CLI 选项仍引用 v8/2ATR归档，并测量1、2、4、6、8、10个 spawn worker；这不是上文 v9 的36账户验证。查看实际参数请运行 `--help`。请检查每次任务的实际版本和完成报告，不要把单案例、36账户策略验证或部分完成任务当作全量效果结论。
