# Task List: 蔡森量价股票分析系统

## 用户提出的系统重构（2026-10-25）

- [x] 默认最近120根详情K线，支持缩放按钮、范围切换、滚轮和滑块；缩放范围变化时同步调整图表数据窗口。
- [x] 新增单标的/全部行情更新任务、AkShare数据源、校验、原子文件替换、哈希清单与部分失败报告。
- [x] 形态识别规则升级至 v2，去除缺少越界/收复证据的自动破底翻/假突破，旗形改用真实摆动拐点；竞争形态触发人工复核并抑制直接买入计划。
- [x] `pytest`: 76 passed；JavaScript/Python语法检查通过；浏览器确认默认窗口显示120/8,453根且放大按钮降至84根。
- [x] 通过详情页真实用户触发验证单标的上游下载、覆盖范围校验、快照与哈希清单写入；本次数据已是最新交易日，因此文件内容不变。
- [x] 批量刷新在启动时一次保存标的清单，逐标的更新规范化检查点行，任务摘要维持有界；服务启动标记过期任务为中断；恢复跳过成功项、重试失败/未处理标的；含部分失败任务重试。新增覆盖检查点、内存丢失后重建、摘要有界、进程内异常保留成功状态、服务重启恢复和续跑跳过逻辑的测试。
- [x] 受控浏览器“服务重启中断→页面自动发现→点击续跑→结果回显”端到端验证通过；仅用不存在的测试代码 999999，失败发生在行情源调用之前，无外部请求或行情文件更改。完整 5,294 个本地标的全量联网更新仍不执行。
- [ ] 继续观察上游限流恢复与长期运行稳定性；策略效果验证暂缓。
- [x] 接入 XSHG 交易日历：2020–2026 使用上交所官方休市公告生成正式周线；2000–2019 第三方日历、2027+ 日期保持预览。Backtrader 功能验收维持零费用，策略效果验证仍待后续。

## 新增需求（2026-10-25）：十招范围、买入候选与扫描后筛选

状态：用户已批准并授权自动实施；Task 1、Task 2 已完成。

- [x] Task 2：同步规格、README、规则卡和数据验证说明；明确来源审计保留12项，而活动目录是十招；历史 v14 报告按旧口径解释。
- [x] Task 3：decision-engine 只允许 W底、头肩底、下飘旗形上涨中继、收敛三角形底部创建买入计划。六个非白名单 ID 仍保留结构展示，不能进入 `candidate_plans`；既有空头 SELL_IF_HELD 测试通过。
- [x] Checkpoint 1：后端活动十招目录、四种买入候选白名单和未突破等待契约测试通过；旧版本名称映射保留。
- [x] Task 5：扫描完成后加入行动与形态组合筛选，保留完整库存计数及默认排序；筛选仅作用于现有结果，不重新扫描。Node 过滤单测 4 项、工作台回归 22 项通过；8302 隔离服务实测 3/3、买入零结果空态、WAIT+W底组合得到 1 项，并可直达详情。8301 用户服务及其行情更新未受影响。
- [ ] Task 7：端到端验收。全套 pytest 161项、Node筛选4项、前端/问卷 JS 语法检查通过；8302 临时隔离服务以3只标的完成扫描、零买入空态、WAIT+W底组合及详情导航实测。当前版本全市场复扫在90/5,294处因串行运行耗时停止；未重启/干扰8301的行情刷新。当前版本5,294只复扫和最终 Checkpoint 2 尚未完成。依赖 Tasks 1–6。
- [ ] Checkpoint 2：全量测试和浏览器流程通过；买入候选严格限四招；筛选不改变后台扫描结果和排序依据。

## 当前模块：`source-rules`

- [x] Task 1: 建立 12 招来源审计索引
- [x] Task 2: 审计多头反转规则
- [x] Task 3: 审计空头/顶部规则
- [x] Checkpoint: 原书审计
- [x] Task 4: 建立规则目录契约与校验器
- [x] Task 5: 编码双底与破底翻规则组
- [x] Task 6: 编码旗形与收敛三角形规则组
- [x] Task 7: 编码 M头、头肩与假突破规则组
- [x] Checkpoint: 活动规则与公式
- [x] Task 8: 同步规则卡并完成模块验收
- [x] Checkpoint: `source-rules` 完成

## 后续系统模块：依赖完成后规格化、审批、再实施

- [x] Gate M1: 规格化 `market-data`
- [x] M1.1: 库存发现与数据结果契约
- [x] M1.2: 单文件规范化和质量审计
- [x] M1.3: 交易日历周线契约
- [x] M1.4: 全本地库存验收
- [x] Checkpoint: `market-data` 完成；周线日历按已核验覆盖范围区分正式数据与预览
- [x] Gate M2: 规格化 `pattern-engine`；用户已授权 build auto
- [x] M2.1: 统一形态事件与状态契约
- [x] M2.2: 修复拐点和形态的前缀因果性
- [x] M2.3: 规则目录驱动识别
- [x] M2.4: 数据段、日周周期、量价证据接入
- [x] Gate M3: 规格化 `decision-engine`；用户已授权 build auto
- [x] M3.1: 实现动作语义、风险/2R计划和零费用基线
- [x] M3.2: 实现日周合并、冲突及数据不足表达
- [x] M3.3: 实现买入优先可解释排序
- [x] M3.4: 决策与排序测试验收
- [x] Checkpoint: `decision-engine` 首轮完成
- [x] Gate M4: 规格化 `simulation-validation`；用户指定 Backtrader、零费用
- [x] M4.1: Backtrader零费用账户与订单账本
- [x] M4.2: 下一日开盘一次尝试、结构止损/首目标和拒单规则
- [x] M4.3: 日线撮合、周线输入、预览周拦截与期末估值
- [x] M4.4: 账目一致性测试
- [x] Checkpoint: `simulation-validation` 首轮完成；仅功能验收，不验证策略效果
- [x] Gate M5: 规格化 `analysis-workbench`；用户已明确扫描、排序、缩略图、详情流程
- [x] M5.1: 全量扫描每标的一行及稳定排序/进度（5,294/5,294，0 失败）
- [x] M5.2: 扫描列表易读状态、迷你K线和点击直达详情
- [x] M5.3: 详情页默认当前形态/买卖分析、证据/数据质量/日周冲突
- [x] M5.4: 页面/API集成验收与本地运行核对（浏览器、pytest、Backtrader烟测）
- [x] Gate M2: 规格化 `pattern-engine`
- [x] Gate M3: 规格化 `decision-engine`（定义可解释的买入优先级及 SELL/持仓语义）
- [x] Gate M4: 规格化 `simulation-validation`
- [x] Gate M5: 规格化 `analysis-workbench`（一键全量扫描、买入优先排序、迷你 K 线、点击直达详情）
- [x] Checkpoint: 所有模块规格批准，并为每个模块建立细化任务/验证步骤



## 2026-10-27 approved implementation

- [x] Chart: historical event overlay, timeframe selection, automatic window and latest reset; verify geometry/window Node tests and browser.
- [x] Retirement: remove review/calibration tools and all records including original answer; verify routes/storage and ambiguity WAIT.
- [x] Replay: remove arbitrary cutoff code and dependencies; verify remaining calendar and prefix causality tests.
- [x] Scan/update: full-universe scan with post-result filters; remove refresh resumability and persistence.
- [x] Final: complete pytest/Node checks, local browser and service verification.

Verification: 132 pytest cases passed; 5 Node test cases passed; JavaScript syntax and git diff checks passed. Browser verified 000001 daily W anchors/invalidation, weekly event automatic timeframe navigation, and latest-window reset; console errors/warnings were empty. Project labels/calibration artifacts, SQLite reviews/checkpoints and the authorized Downloads JSON were removed. Full-market online refresh was not invoked.

Plan boundary regression: ambiguous structures, non-allowlisted shapes, and locked flag targets cannot generate buys through direct calls or simulation; all callers share the same plan gate.

## 2026-10-28 approved implementation

- [x] Add pre-scan stock/ETF scope, optional stable-order symbol cap, and inclusive analysis start/end dates. Slice daily bars before weekly aggregation and pattern analysis; retain insufficient-data rows for empty windows and disclose actual window/latest date.
- [x] Add optional multi-shape selection as a saved post-scan display filter only. All patterns still feed decisions and risk gates; this selector does not reduce detector work.
- [x] Mark historical scan rows with their interval end and explain that row navigation opens latest detail analysis.
- [x] Add a Backtrader simulation panel for code, kind, timeframe, dates and starting cash; asynchronously show fills/trades, orders, rejections, cash, marked position, equity and accounting checks. Keep strategy-effectiveness verification deferred.
- [x] Prevent weekly bars cut by a selected midweek start or end from entering formal weekly analysis; they remain preview-only.
- [x] Validate the Backtest API's inclusive date window and positive cash; resolve the existing `_market_data` call signature mismatch.

Verification: 140 Caisen pytest tests pass when run from the stock workspace root (the legacy local-CSV test resolves `data/` relative to cwd); 5 Node tests pass; JavaScript syntax and Python compile checks pass. The HTTP page and `/api/v1/backtests` were exercised on an isolated local server with a temporary SQLite path; the response reported zero fees and both accounting invariants true. No network refresh or strategy-performance run is part of this work.
