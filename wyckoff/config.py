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
    peak_percent_delta: float = 5.0  # day0百分数单位：5.0=5%
    min_bars_between_peaks: int = 0
    peaks: str = "All"  # All / High / Low
    adaptive: bool = True        # 按 ATR / 价格动态调整阈值
    atr_mult: float = 3.0
    min_threshold: float = 0.02
    max_threshold: float = 0.08


    @property
    def threshold(self):
        """Legacy fractional alias; canonical YAML parameter is peak_percent_delta."""
        return self.peak_percent_delta / 100

    @threshold.setter
    def threshold(self, value):
        self.peak_percent_delta = float(value)*100


@dataclass
class SupportConfig:
    lookback_bars: int = 0  # 0=全样本；交易输入另按确认时点过滤
    merge_distance: Optional[float] = None  # 与merge_percent二选一
    merge_percent: Optional[float] = 0.5  # 百分数；0.5表示0.5%
    level_selector: str = "median"


@dataclass
class VolumeConfig:
    ma_window: int = 20        # 成交量均线窗口
    high_vol_ratio: float = 1.5   # 放量阈值（相对均量）
    low_vol_ratio: float = 0.7    # 缩量阈值


@dataclass
class SpringConfig:
    recover_window: int = 5    # 破位/触带后收复天数上限
    recover_margin_atr: float = 0.05  # 收盘收复水平线的ATR缓冲
    vol_confirm_ratio: float = 1.8  # 超过此量比视为"震仓"（高风险）
    test_window: int = 30      # 二次测试等待窗口
    spring_window: int = 3
    ideal_range_atr: float = 1.2
    absorption_range_atr: float = 1.8
    test_volume_ratio: float = 0.7
    test_range_ratio: float = 0.6
    test_range_atr: float = 1.2
    test_tolerance_atr: float = 0.5
    min_test_tolerance: float = 0.003
    max_test_tolerance: float = 0.02


@dataclass
class SupplyDemandConfig:
    min_impulse_change: float = 0.03  # 离开基带的最小冲动幅度
    impulse_vol_ratio: float = 1.5    # 冲动K线量比要求
    base_max_body_atr: float = 1.0    # 基带K线实体上限（ATR倍数）
    zone_width_atr_max: float = 3.0   # 区域高度上限（ATR倍数）


@dataclass
class ClimaxConfig:
    lookback: int = 20
    volume_ratio: float = 1.8
    range_atr: float = 1.5
    confirm_bars: int = 3
    stopping_close_position: float = 0.6
    cooldown: int = 20


@dataclass
class SignalConfig:
    diagnostic_scores: bool = True  # 分数只展示，不生成交易动作
    entry_sc: bool = True
    entry_spring: bool = True
    entry_tests: bool = True
    entry_lps: bool = True
    entry_breakouts: bool = False  # 原文默认等待SOS/JOC后成功测试；开启为积极研究模式


@dataclass
class TradeRuleConfig:
    take_profit: float = 0.15
    stop_loss: float = 0.05
    max_hold_days: int = 20
    commission: float = 0.001
    initial_cash: float = 100000.0
    slippage: float = 0.0005


@dataclass
class StrategyConfig:
    zigzag: ZigzagConfig = field(default_factory=ZigzagConfig)
    support: SupportConfig = field(default_factory=SupportConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    spring: SpringConfig = field(default_factory=SpringConfig)
    supply_demand: SupplyDemandConfig = field(default_factory=SupplyDemandConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    climax: ClimaxConfig = field(default_factory=ClimaxConfig)
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
            for group_name in ("zigzag", "support", "volume", "spring", "supply_demand", "signal", "climax", "trade"):
                target = getattr(cfg, group_name)
                group_key = key.split('_', 1)
                if hasattr(target, key) or (len(group_key) == 2
                                            and group_key[0] == group_name
                                            and hasattr(target, group_key[1])):
                    real = key if hasattr(target, key) else group_key[1]
                    setattr(target, real, value)
                    break
            else:
                raise ValueError(f"未知策略参数: {key}（可用：裸参数名或 分组_参数名）")
        return cfg
