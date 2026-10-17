"""全局配置：路径与策略参数。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
STOCKS_DIR = DATA_DIR / "stocks"
ETFS_DIR = DATA_DIR / "etfs"
SIGNALS_DIR = DATA_DIR / "signals"
BACKTEST_DIR = DATA_DIR / "backtests"
CONFIG_FILE = REPO_ROOT / "config" / "strategy.yaml"


@dataclass
class ZigzagConfig:
    threshold: float = 0.05      # 固定反转阈值（adaptive=False时生效）
    depth: int = 15              # 确认后回溯窗口（防未来函数：拐点延迟depth根确认）
    zone_max_width: float = 0.05 # 支撑/阻力带全局最大宽度；支撑只向下合并
    adaptive: bool = True        # 波动率自适应阈值：低波ETF自动放细、高波股自动放粗
    atr_mult: float = 3.0        # threshold = atr_mult × 当日ATR占收盘价比
    min_threshold: float = 0.02
    max_threshold: float = 0.08


@dataclass
class VolumeConfig:
    ma_window: int = 20        # 成交量均线窗口
    high_vol_ratio: float = 1.5   # 放量阈值（相对均量）
    low_vol_ratio: float = 0.7    # 缩量阈值


@dataclass
class SpringConfig:
    lookback: int = 60         # 支撑位有效回溯窗口
    recover_window: int = 5    # 破位/触带后收复天数上限
    recover_frac: float = 0.6  # 收复线 = 带下沿 + frac×带宽（0.6=腰部偏上）
    recover_margin: float = 0.002  # 收盘还须高出收复线这一比例
    vol_confirm_ratio: float = 1.8  # 超过此量比视为"震仓"（高风险）
    test_window: int = 30      # 二次测试等待窗口


@dataclass
class SupplyDemandConfig:
    min_impulse_change: float = 0.03  # 离开基带的最小冲动幅度
    impulse_vol_ratio: float = 1.5    # 冲动K线量比要求
    base_max_body_atr: float = 1.0    # 基带K线实体上限（ATR倍数）
    zone_width_atr_max: float = 3.0   # 区域高度上限（ATR倍数）


@dataclass
class SignalConfig:
    buy_threshold: float = 2.0     # 买入信号分数门槛
    sell_threshold: float = 2.0    # 卖出信号分数门槛
    confirm_margin: float = 1.5    # 买/卖分差确认倍数
    bear_accum_mult: float = 0.6   # 空头背景+吸筹阶段权重
    bear_mult: float = 0.35        # 空头背景权重（书中熊市禁Spring）
    dist_mult: float = 0.6         # 派发阶段买入权重


@dataclass
class TradeRuleConfig:
    take_profit: float = 0.15
    stop_loss: float = 0.05
    max_hold_days: int = 20
    commission: float = 0.001
    initial_cash: float = 100000.0


@dataclass
class StrategyConfig:
    zigzag: ZigzagConfig = field(default_factory=ZigzagConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    spring: SpringConfig = field(default_factory=SpringConfig)
    supply_demand: SupplyDemandConfig = field(default_factory=SupplyDemandConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    trade: TradeRuleConfig = field(default_factory=TradeRuleConfig)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "StrategyConfig":
        path = path or CONFIG_FILE
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = cls()
        for group_name, group in asdict(cfg).items():
            values = raw.get(group_name) or {}
            target = getattr(cfg, group_name)
            for key, value in values.items():
                if hasattr(target, key):
                    setattr(target, key, value)
        return cfg

    def save(self, path: Optional[Path] = None) -> None:
        path = path or CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(asdict(self), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def with_overrides(self, **flat: float) -> "StrategyConfig":
        """按 `分组_参数` 或参数名覆盖，用于网格搜索。"""
        import copy
        cfg = copy.deepcopy(self)
        for key, value in flat.items():
            for group_name in ("zigzag", "volume", "spring", "supply_demand", "signal", "trade"):
                target = getattr(cfg, group_name)
                if hasattr(target, key):
                    setattr(target, key, value)
                    break
        return cfg
