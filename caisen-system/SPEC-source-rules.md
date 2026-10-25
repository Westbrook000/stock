# Spec: `source-rules` 蔡森原书规则库

## Objective

为 `caisen-system` 建立可追溯、可版本化的蔡森量价形态规则来源。规则内容以项目内《多空转折一手抓》PDF 的正文和图例为一手依据；现有 `docs/rule-cards.md` 与 `.agents/skills/caisen-volume-price/` 仅作为待核对的研究草稿，不能代替原书。

本模块逐项核对技术篇 12 招候选；仅将原书依据足以明确支持的形态写入可执行规则目录，并记录纳入或排除理由。纳入规则记录形态拓扑、形成中证据、确认条件、失效条件、目标/测幅公式、量价条件、适用限制和出处。它提供后续 `pattern-engine` 与 `decision-engine` 消费的规则数据，不负责从行情识别图形，也不直接给出买入或卖出动作。

## Assumptions

1. 原书 PDF 是本模块的权威来源；依据不足的候选不推测补齐，也不进入可执行规则目录。
2. 首版限定为技术篇 12 招。时间波、全书一般交易心法和 A 股交易制度不作为本模块的形态规则。
3. 现有规则卡的 12 招清单作为逐项核对的起点；原书标题、技能叫法与项目命名有差异时保留别名和解释，不静默合并。
4. 只有原书依据足以明确支持规则的招式才进入可执行规则目录。依据不足的候选不以“待人工复核”规则占位，也不推导自动条件；来源审计记录其排除原因，但该记录不提供识别或决策行为。
5. 规则正文以 JSON 作为机器可读的规范数据，使用 Python 标准库读取，避免为规则文件新增运行依赖；说明文档链接到规则 ID。
6. “买入/持有/卖出/等待”的动作映射、持仓语义和风险门槛由后续 `decision-engine` 规格决定。

## Tech Stack

- Python 3.11+ 项目，现有 FastAPI/pandas/NumPy 应用保持兼容。
- 规则目录使用 UTF-8 JSON 和标准库 `json`；不引入新的运行时依赖。
- 规则数据独立于 Python 识别算法；`rule_set_version` 与应用/API 版本分别管理。

## Commands

从仓库根目录执行：

```sh
# 运行规则库测试
PYTHONPATH=caisen-system .venv/bin/python -m pytest -q caisen-system/tests/test_rule_catalog.py

# 运行蔡森项目完整测试集
PYTHONPATH=caisen-system .venv/bin/python -m pytest -q caisen-system/tests

# 启动现有本地分析服务（用于检查既有页面/API兼容性）
.venv/bin/python caisen-system/scripts/serve.py
```

若虚拟环境尚未创建，安装项目依赖的命令为：

```sh
.venv/bin/python -m pip install -r caisen-system/requirements.txt pytest
```

## Project Structure

```text
caisen-system/
├── caisen/
│   └── rules/
│       ├── patterns.v2.json       # 当前 12 招规范；v1 保留历史审计
│       └── catalog.py             # 加载、校验和只读访问规则
├── docs/
│   ├── rule-cards.md              # 面向研究者的规则说明，规则 ID 与 JSON 对应
│   └── rule-source-audit.md       # 原书页码/图号与规则字段的核对记录
└── tests/
    └── test_rule_catalog.py       # 完整性、来源、公式和版本测试
```

本模块不改写识别算法；后续模块可以逐步迁移 `patterns.py` 中的名称和常量，但在规格获批前不做迁移或行为变更。

## Rule Data Shape

每条已纳入规则至少包含：稳定的 `rule_id`、规范名称、别名、多空方向、拓扑摘要、候选/形成中条件、价格确认、量价条件（必需/增强/不适用）、失效条件、目标公式及触发限制、出处和规则版本。书中叙述、实现解释和数值工程参数必须有明确来源类型标签，不能混写成作者原话。来源审计还应覆盖现有清单中的全部 12 个候选，记录纳入或排除结论；被排除项不得出现在可执行 JSON 规则目录中。

出处至少记录原书印刷页码；若扫描页码不同，同时记录 PDF 页码。图例规则还应记录图号或图注。已纳入规则的出处必须完成核实；依据不足的候选只在审计记录标为 `excluded_insufficient_source`，不生成虚构页码或引文。

JSON 示例：

```json
{
  "rule_id": "w_bottom",
  "name": "W底",
  "aliases": ["双底"],
  "direction": "bullish",
  "lifecycle": {
    "forming": ["两个低点由反弹高点分隔"],
    "confirmed_by": "收盘价突破按已知锚点冻结的颈线",
    "invalidated_by": "确认后收盘重新跌破颈线"
  },
  "volume": {"required": [], "supporting": ["突破时量能增强"]},
  "targets": [{"formula_id": "neckline_height_projection", "available_after": "confirmed"}],
  "engineering_parameters": [],
  "source_refs": [{"printed_page": 42, "pdf_page": 51, "figure": "图例编号", "source_status": "verified"}],
  "rule_set_version": "caisen-rules-v4"
}
```

## Code Style

- 规则 ID 使用稳定的 `snake_case`；字段使用有明确含义的英文键，面向用户的术语使用简体中文。
- 规则文件只存领域事实和参数，不放可执行 Python 表达式或动态代码。
- `catalog.py` 提供只读读取/校验接口；校验失败时指出规则 ID 和字段路径，不静默回退到旧规则。
- 原文事实、研究者解释、工程参数分别标记为 `source_text`、`interpretation`、`engineering_parameters`，需要引用时通过 `source_refs` 追溯。

示例读取方式：

```python
from pathlib import Path
import json

RULES_PATH = Path(__file__).parent / "patterns.v2.json"


def load_rule_catalog() -> dict[str, dict]:
    payload = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    return {item["rule_id"]: item for item in payload["patterns"]}
```

## Testing Strategy

- 使用 pytest；规则库测试位于 `caisen-system/tests/test_rule_catalog.py`。
- 完整性测试：来源审计覆盖现有 12 个候选并各有纳入/排除结论；活动规则 ID 唯一，每条有必需生命周期、已核实来源和版本字段。
- 出处测试：每条活动规则至少有一个可定位的印刷页或 PDF 页；来源不足的排除项不出现在可执行目录中。
- 公式测试：对 W 底、M 头/头肩顶、旗形中继和收敛三角形的测幅规则使用独立数值案例验证方向、基准和启用阶段；未满足书中触发条件时不得输出目标。
- 边界测试：当前 `BOOK_PATTERNS` 的 12 个 ID 均出现在审计候选清单中；排除项不会通过规则目录提供给下游。现有识别算法的迁移和兼容处理由后续 `pattern-engine` 规格决定。
- 完成规则数据和校验器后，运行规则库测试及 `PYTHONPATH=caisen-system .venv/bin/python -m pytest -q caisen-system/tests`。

## Boundaries

- **Always:** 对每条纳入规则保留已核实的原书位置；分开记录书中原意、解释和工程阈值；对来源不足的候选记录排除原因且不生成占位规则；保持稳定规则 ID 和可审计版本。
- **Ask first:** 改变 12 招覆盖范围；把规则扩展成原书未明确提出的新型态；新增运行依赖；把作者的经验描述升级为系统硬性买卖规则或统计概率。
- **Never:** 编造书页、图号或作者引文；把技能摘要当作完整原书；把未确认形态写成已确认；在本模块中把作者经验胜率宣称为验证结果；连接券商或提交真实订单。

## Success Criteria

1. 现有清单中的 12 招候选均完成原书核对，并在审计表中记录纳入或排除结论；只有依据充分的规则进入 JSON 目录。
2. 每条纳入规则均有已核实出处，并可以区分原书依据、研究解释和工程阈值；来源不足的候选不会被当作规则暴露给下游。
3. 目标公式有数值测试，并规定何时可以计算；中继形态未突破前高/前低等解锁条件时不提前给出测幅目标。
4. 规则目录可由 Python 标准库读取，并提供明确的 schema/语义校验错误。
5. 既有形态算法在本模块实现期间不被悄然改变；规则版本可单独记录并供下游结果追溯。
6. 规则库专项测试与蔡森项目完整测试通过。

## 已确认的范围决策

- `patterns.v2.json` 作为当前机器规则规范来源，Markdown 通过规则 ID 关联；`patterns.v1.json` 保留首轮版本以便历史结果追溯。
- 现有 12 招清单作为核对起点，逐项按原书校正名称和内容。
- 原书依据不足的候选不保留为“人工复核”占位规则；只在审计记录排除原因，不推导自动规则。

## Open Questions

- 无。本模块规格已根据上述范围决策更新，等待批准后进入计划阶段。
