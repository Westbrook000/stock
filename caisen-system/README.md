# 蔡森量价结构研究系统

独立本地服务，默认 `127.0.0.1:8301`。含市场扫描、单标的形态图、截至日期回放、人工复核和下一交易日模拟回测接口；不下单。详情图默认显示最近 120 根，可拖动、滚轮缩放、用范围按钮调整或展开全部历史；失效、目标到达、箱体后续边界确认的终止形态单独折叠，不参与当前交易计划。当前应用规则版本为 v15，活动目录为 `caisen-rules-v6`，启用十招；原书来源审计仍保留十二个候选的记录。系统跟踪确认后的目标到达、形成阶段的极值失效与颈线外推失效；ATR 缺失时跳过该拐点，不以未来 ATR 回填；突破确认只从全套结构点均已可知后开始计算。左右肩相差不超过头部至颈线高度 7% 的头肩几何候选会进入识别；测幅目标无效时视为规则不成立，不输出形态候选。7%是开发标签导出的工程阈值，须经独立留出验证。破底翻/假突破要求先有整理箱体，再确认越界和收回。W/M 与收敛三角形候选要求相邻结构段时长比不高于 2.5；W底两底高低差还需不高于较高底部的15%。这些阈值根据十个开发标注及对688216的后续澄清提出，既非原书数值也非准确性结论，尚待独立盲评。箱体参数同样是工程初值并强制人工复核；W底候选详情显示两段持续K线数、时长比例和两底差。W底破底翻与假突破（头肩顶）已停用，不进入当前识别；旧历史标签和原书审计记录保留。旗形要求实际拐点和完整回撤结构；同一组拐点产生竞争标签时会阻止直接买入并提示人工复核。量能或企业行动来源未核验时，形态继续展示为研究候选，但交易计划返回等待。周线日历采用上交所 2020–2026 年官方休市公告；2000–2019 年的第三方历史日历只作预览，2027 年后等官方公告录入再启用。官方日期见 `caisen/data/xshg-official-closures.json`。

## 启动

从仓库根目录运行：

```sh
.venv/bin/python caisen-system/scripts/serve.py
```

打开 http://127.0.0.1:8301 。状态库默认写到 `caisen-system/state/caisen.sqlite3`；可用 `CAISEN_DB=/path/to/db.sqlite3` 改位置。数据读取 `../data/stocks` 与 `../data/etfs`。服务只绑定本机，不提供券商接口。

## 自动化验证

在仓库根目录运行 `.venv/bin/python -m pytest caisen-system/tests`，或先进入 `caisen-system/` 再运行 `../.venv/bin/python -m pytest`。

## 盲标校准

从当前本地 CSV 生成可复现的盲标复核页，不触发联网更新：

```sh
.venv/bin/python caisen-system/scripts/create_blind_calibration_round.py --count 10 --seed 20261026
```

默认写入 `caisen-system/calibration/round_03_v15.html`。新问卷只列当前活动十招及 N，导出答案会携带校准轮次、引擎与规则集版本；页面按股票/ETF、日/周线分层抽样，不展示系统形态提名。完成后可保存为 `caisen-system/calibration/labels/round_03_v15.user.json`，再运行：

```sh
.venv/bin/python caisen-system/scripts/evaluate_calibration_labels.py \
  --labels caisen-system/calibration/labels/round_03_v15.user.json \
  --output caisen-system/calibration/reports/round_03_v15.machine-review.json
```

评估器按答案中的品种和周期重放截至日期的行情，报告分别记录标注版本与本次评估版本。round_01/round_02、旧答案与 `round_01_v14` 历史报告不覆盖。样本不以机器提名为筛选条件，因此可发现漏识别；小样本不能估计总体准确率或策略收益。

详情页“更新当前行情”会显式请求单个股票/ETF；市场扫描页“更新全部行情”逐只刷新当前本地库存。更新使用 AkShare，股票拉取前复权日线；ETF 按代码前缀映射沪市 `sh`、深市 `sz`，保存原始日线。对超时、连接中断、HTTP 408/425/429/5xx 最多重试三次，间隔 0.5/1 秒；校验失败不重试，结果与清单记录尝试次数。旧 CSV 原文会先按哈希归档，行情和元数据分步原子提交；元数据提交失败会回滚已替换的行情文件。刷新任务显示成功/失败计数，并记录新旧哈希。2026-10-25 已通过详情页分别真实更新 000001 与 510300：股票从 8,453 根增至 8,458 根，ETF 从 3,495 根增至 3,500 根；两者最新日期都从 2026-10-16 推进到 2026-10-23。两个旧文件的快照和哈希都已核验，保存在 `../data/market_data_versions/snapshots/`。刷新成功不代表企业行动、复权因子或成交量单位已验证。全市场刷新可能被上游限流；分析和扫描本身仍保持离线。

## API

- `GET /api/v1/analysis?code=000001&kind=stock&timeframe=1d&asof=2025-01-03`
- `POST /api/v1/scans` JSON: `{"kind":"all","limit":10000,"timeframes":["1d","1w"]}`
- `GET /api/v1/jobs/{job_id}`
- `POST /api/v1/data/refresh` JSON: `{"code":"000001","kind":"stock"}` 单标的；`{"scope":"all"}` 更新全部本地标的。
- `GET /api/v1/patterns/{pattern_id}` / `PUT /api/v1/patterns/{pattern_id}/review`
- `POST /api/v1/backtests` JSON: `{"code":"000001","kind":"stock","timeframe":"1d","cash":100000}`（Backtrader 撮合、零费用；只验收订单/账目，不代表策略有效）

## 当前边界

本地 CSV 是当前存续行情文件，不代表完整历史市场；价格复权、成交量单位、企业行动断点和退市/清盘史均未完成验证。量能未验证的模式只能作为研究候选，计划返回等待。算法没有经过240例盲评；功能可运行不等于策略有盈利效力。参见文档中的剩余验收项。
