#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
威科夫分析器 - 重构版（带事件验证机制）
基于《威科夫操盘法》理论，核心是"没有验证就没有确认"
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum

try:
    import akshare as ak
    import pandas as pd
    import numpy as np
except ImportError:
    print("请安装所需库: pip install akshare pandas numpy")
    sys.exit(1)


class EventStatus(Enum):
    """事件状态"""
    PENDING = "pending"    # 待验证
    CONFIRMED = "confirmed" # 已确认
    FAILED = "failed"       # 已失效


@dataclass
class WyckoffEvent:
    """威科夫事件数据结构"""
    event_type: str
    date: datetime
    price: float
    volume: float
    status: EventStatus = EventStatus.PENDING
    likelihood: float = 0.0  # 概率云计算的初始似然度
    verification_deadline: int = 0  # 验证截止索引
    verification_details: str = ""
    confirmed_by: str = ""  # 通过什么验证通过
    failed_reason: str = ""  # 失效原因
    
    def to_dict(self) -> dict:
        return {
            'type': self.event_type,
            'name': get_event_name(self.event_type),
            'date': self.date,
            'price': self.price,
            'volume': self.volume,
            'status': self.status.value,
            'likelihood': self.likelihood,
            'confidence': get_confidence_label(self.likelihood),
            'verification_details': self.verification_details,
            'confirmed_by': self.confirmed_by,
            'failed_reason': self.failed_reason
        }


def get_event_name(event_type: str) -> str:
    """获取事件中文名称"""
    names = {
        'SC': '恐慌抛售',
        'AR': '自动反弹',
        'SPRING': '弹簧效应',
        'SPRING_BEST': '最佳弹簧',
        'SPRING_STRONG': '强劲需求弹簧',
        'SPRING_CONFIRM': '需确认弹簧',
        'SPRING_SHAKEOUT': '震仓弹簧',
        'UT': '上冲回落',
        'BC': '抢购高潮',
        'SOW': '弱势信号',
        'LPS': '最后支撑点',
        'ST': '二次测试',
        'SOS': '强势出现',
        'JOC': '跳离震荡区'
    }
    return names.get(event_type, event_type)


def get_confidence_label(likelihood: float) -> str:
    """根据似然度返回置信等级"""
    if likelihood > 0.8:
        return "高置信"
    elif likelihood > 0.6:
        return "中置信"
    elif likelihood > 0.4:
        return "低置信"
    else:
        return "观察"


class EventVerifier:
    """事件验证器 - 状态机实现"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.pending_events: List[WyckoffEvent] = []
        self.confirmed_events: List[WyckoffEvent] = []
        self.failed_events: List[WyckoffEvent] = []
        
        # 计算基础指标
        self._prepare_indicators()
    
    def _prepare_indicators(self):
        """准备计算所需的指标"""
        df = self.df
        
        # 均线
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        
        # 成交量均线
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        
        # 20日高低点
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        
        # 60日高低点
        df['high_60'] = df['high'].rolling(60).max()
        df['low_60'] = df['low'].rolling(60).min()
        
        # 振幅
        df['amplitude'] = (df['high'] - df['low']) / df['close'] * 100
        
        # 涨跌幅
        df['pct_change'] = df['close'].pct_change() * 100
        
        # 量比
        df['vol_ratio'] = df['volume'] / df['vol_ma20']
        
        # ===== SASE动态支撑（布林带下轨） =====
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        
        # 计算ATR
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_100'] = df['tr'].rolling(100).mean()
        df['atr_ratio'] = df['atr_14'] / (df['atr_100'] + 1e-9)
        
        # 恢复原容差2.0
        atr_ratio = df['atr_ratio'].fillna(1.0)
        tolerance = 2.0 * (1.0 + (1 - atr_ratio.clip(0, 2)) * 0.5)
        df['bb_lower'] = df['bb_mid'] - tolerance * df['bb_std']
        
        # 支撑位：布林下轨（SASE动态支撑）
        df['support'] = df['bb_lower']
        
        self.df = df
    
    def _calculate_zigzag(self, df, threshold=0.02, depth=14, backstep=1):
        """
        计算Zigzag拐点（使用high/low高低点）
        返回: (pivots, supports, resistances)
        """
        high = df['high'].values
        low = df['low'].values
        n = len(df)
        
        pivots = [0]
        direction = 0
        last_pivot_direction = 0
        last_pivot_idx = 0
        
        for i in range(1, n):
            if i - last_pivot_idx < depth:
                continue
            
            if direction == 0:
                if high[i] >= high[0] * (1 + threshold):
                    direction = 1
                    pivots.append(i)
                    last_pivot_direction = 1
                    last_pivot_idx = i
                elif low[i] <= low[0] * (1 - threshold):
                    direction = -1
                    pivots.append(i)
                    last_pivot_direction = -1
                    last_pivot_idx = i
            elif direction == 1:
                if high[i] > high[pivots[-1]]:
                    pivots[-1] = i
                elif low[i] <= low[pivots[-1]] * (1 - threshold):
                    if i - pivots[-1] >= backstep:
                        direction = -1
                        pivots.append(i)
                        last_pivot_direction = -1
                        last_pivot_idx = i
            elif direction == -1:
                if low[i] < low[pivots[-1]]:
                    pivots[-1] = i
                elif high[i] >= high[pivots[-1]] * (1 + threshold):
                    if i - pivots[-1] >= backstep:
                        direction = 1
                        pivots.append(i)
                        last_pivot_direction = 1
                        last_pivot_idx = i
        
        # 提取支撑位（波谷）
        supports = []
        for i in range(1, len(pivots) - 1):
            idx = pivots[i]
            if (low[pivots[i-1]] > low[idx]) and (low[pivots[i+1]] > low[idx]):
                supports.append({'idx': idx, 'price': low[idx], 'type': 'support'})
        
        # 提取阻力位（波峰）
        resistances = []
        for i in range(1, len(pivots) - 1):
            idx = pivots[i]
            if (high[pivots[i-1]] < high[idx]) and (high[pivots[i+1]] < high[idx]):
                resistances.append({'idx': idx, 'price': high[idx], 'type': 'resistance'})
        
        return pivots, supports, resistances
    
    def add_pending_event(self, event: WyckoffEvent):
        """添加待验证事件"""
        self.pending_events.append(event)
    
    def verify_all(self):
        """验证所有待验证事件"""
        total_bars = len(self.df)
        
        for event in self.pending_events:
            if event.status != EventStatus.PENDING:
                continue
            
            # 根据事件类型选择验证器
            if event.event_type in ['SC']:
                self._verify_sc(event, total_bars)
            elif event.event_type in ['SPRING_ZIGZAG']:
                self._verify_spring_zigzag(event, total_bars)
            elif event.event_type in ['SPRING', 'SPRING_BEST', 'SPRING_STRONG', 'SPRING_CONFIRM', 'SPRING_SHAKEOUT']:
                self._verify_spring(event, total_bars)
            elif event.event_type in ['SOS', 'JOC']:
                self._verify_sos(event, total_bars)
            elif event.event_type in ['BC']:
                self._verify_bc(event, total_bars)
            elif event.event_type in ['UT']:
                self._verify_ut(event, total_bars)
            elif event.event_type in ['SOW']:
                self._verify_sow(event, total_bars)
            elif event.event_type in ['ST']:
                self._verify_st(event, total_bars)
        
        # 分类事件
        self.confirmed_events = [e for e in self.pending_events if e.status == EventStatus.CONFIRMED]
        self.failed_events = [e for e in self.pending_events if e.status == EventStatus.FAILED]
    
    def _verify_sc(self, event: WyckoffEvent, total_bars: int):
        """验证SC - 检测到直接确立，后续检查是否失效"""
        event_idx = self.df[self.df['date'] == event.date].index[0]
        sc_volume = event.volume
        sc_price = event.price
        
        # SC检测到直接确立
        event.status = EventStatus.CONFIRMED
        event.confirmed_by = "恐慌抛售已确立"
        event.verification_details = f"SC低点: ¥{sc_price:.2f}"
        
        # 检查是否失效：出现新低（放量跌破SC低点）
        check_start = event_idx + 1
        check_end = min(event_idx + 25, total_bars)
        
        for idx in range(check_start, check_end):
            if idx >= total_bars:
                break
            row = self.df.iloc[idx]
            # 放量跌破SC低点 = 失效
            if row['close'] < sc_price * 0.96 and row['vol_ratio'] >= 1.1:
                event.status = EventStatus.FAILED
                event.failed_reason = "放量跌破SC低点，熊市继续"
                return
        
        # 没有失效，保持CONFIRMED状态
        event.verification_details = f"SC已确立，待观察是否跌破支撑"


    
    def _verify_spring(self, event: WyckoffEvent, total_bars: int):
        """验证Spring - 需要立即反弹确认"""
        event_idx = self.df[self.df['date'] == event.date].index[0]
        
        # 验证窗口：3-5天（对于普通Spring）或10-30天（对于带量Spring）
        if event.event_type == 'SPRING_SHAKEOUT':
            verify_start = event_idx + 5
            verify_end = min(event_idx + 30, total_bars)
        else:
            verify_start = event_idx + 1
            verify_end = min(event_idx + 5, total_bars)
        
        if verify_start >= total_bars:
            event.status = EventStatus.FAILED
            event.failed_reason = "数据不足，无法验证"
            return
        
        # 支撑位 - 使用SASE动态支撑（布林带下轨）
        support = self.df.iloc[event_idx]['support']
        
        # 检查1：是否迅速收复支撑
        recovered = False
        for idx in range(verify_start, verify_end):
            if idx >= total_bars:
                break
            # 收盘价需要超过支撑位1%以上
            if self.df.iloc[idx]['close'] > support * 1.01:
                recovered = True
                break
        
        if not recovered:
            event.status = EventStatus.FAILED
            event.failed_reason = "未能在5日内收复支撑位1%"
            return
        
        # 检查2：是否有连续阴线（失效信号）
        consecutive_down = 0
        for idx in range(event_idx + 1, min(event_idx + 4, total_bars)):
            if idx >= total_bars:
                break
            if self.df.iloc[idx]['close'] < self.df.iloc[idx]['open']:
                consecutive_down += 1
            else:
                break
        
        if consecutive_down >= 2:
            event.status = EventStatus.FAILED
            event.failed_reason = "跌破后出现连续阴线"
            return
        
        # 对于带量Spring，需要二次测试
        if event.event_type in ['SPRING_SHAKEOUT', 'SPRING_STRONG']:
            # 寻找二次测试
            for idx in range(verify_end, min(event_idx + 30, total_bars)):
                if idx >= total_bars:
                    break
                row = self.df.iloc[idx]
                price_near = abs(row['close'] - support) / support < 0.03
                vol_shrink = row['vol_ratio'] < 0.5
                
                if price_near and vol_shrink:
                    event.status = EventStatus.CONFIRMED
                    event.confirmed_by = f"二次测试确认: {row['date'].strftime('%Y-%m-%d')}"
                    event.verification_details = f"缩量二次测试"
                    return
            
            event.status = EventStatus.FAILED
            event.failed_reason = "带量Spring未通过二次测试"
            return
        
        # 普通Spring确认
        event.status = EventStatus.CONFIRMED
        event.confirmed_by = "立即反弹确认"
        event.verification_details = "收盘站稳支撑1%，无连续阴线"
    
    def _verify_spring_zigzag(self, event: WyckoffEvent, total_bars: int):
        """验证SPRING_ZIGZAG - 使用Zigzag支撑位验证"""
        event_idx = self.df[self.df['date'] == event.date].index[0]
        
        # 验证窗口：5天内
        verify_start = event_idx + 1
        verify_end = min(event_idx + 5, total_bars)
        
        if verify_start >= total_bars:
            event.status = EventStatus.FAILED
            event.failed_reason = "数据不足，无法验证"
            return
        
        # 重新计算Zigzag支撑位（使用event_idx之前的数据，避免前视偏差）
        historical_df = self.df.iloc[:event_idx]
        if len(historical_df) < 30:
            event.status = EventStatus.FAILED
            event.failed_reason = "历史数据不足"
            return
        
        _, zigzag_supports, _ = self._calculate_zigzag(historical_df, threshold=0.05)
        
        # 获取最近的Zigzag支撑位
        support_price = None
        for s in zigzag_supports:
            if s['idx'] < event_idx - 1:
                support_price = s['price']
        
        if support_price is None:
            event.status = EventStatus.FAILED
            event.failed_reason = "无Zigzag支撑位"
            return
        
        # 检查1：是否迅速收复支撑
        recovered = False
        for idx in range(verify_start, verify_end):
            if idx >= total_bars:
                break
            # 收盘价需要超过支撑位
            if self.df.iloc[idx]['close'] > support_price:
                recovered = True
                break
        
        if not recovered:
            event.status = EventStatus.FAILED
            event.failed_reason = "未能在5日内收复Zigzag支撑位"
            return
        
        # 检查2：是否有连续阴线（失效信号）
        consecutive_down = 0
        for idx in range(event_idx + 1, min(event_idx + 4, total_bars)):
            if idx >= total_bars:
                break
            if self.df.iloc[idx]['close'] < self.df.iloc[idx]['open']:
                consecutive_down += 1
            else:
                break
        
        if consecutive_down >= 2:
            event.status = EventStatus.FAILED
            event.failed_reason = "跌破后出现连续阴线"
            return
        
        # 确认
        event.status = EventStatus.CONFIRMED
        event.confirmed_by = "Zigzag支撑反弹确认"
        event.verification_details = f"收盘站稳Zigzag支撑{support_price:.2f}，无连续阴线"
    
    def _verify_sos(self, event: WyckoffEvent, total_bars: int):
        """验证SOS - 需要回测不破，且回测振幅小于SOS到回测期间的均幅"""
        event_idx = self.df[self.df['date'] == event.date].index[0]
        sos_price = event.price
        
        # 验证窗口：3-20天
        verify_start = event_idx + 3
        verify_end = min(event_idx + 20, total_bars)
        
        if verify_start >= total_bars:
            event.verification_details = "数据不足，无法验证，继续观察"
            return
        
        # 计算SOS到回测期间的均幅
        amp_values = []
        for i in range(event_idx, verify_end):
            if i < total_bars:
                amp = (self.df.iloc[i]['high'] - self.df.iloc[i]['low']) / self.df.iloc[i]['close'] * 100
                amp_values.append(amp)
        avg_amp = sum(amp_values) / len(amp_values) if amp_values else 5
        
        # 寻找回测
        for idx in range(verify_start, verify_end):
            if idx >= total_bars:
                break
            
            row = self.df.iloc[idx]
            
            # 计算当前K线振幅和下影线
            total_range = row['high'] - row['low'] + 0.001
            amp = total_range / row['close'] * 100
            lower_shadow = (min(row['close'], row['open']) - row['low']) / total_range
            
            # 回测条件：
            # 1. 价格回到SOS附近（5%以内）
            price_near = abs(row['close'] - sos_price) / sos_price <= 0.05
            # 2. 成交量萎缩
            vol_shrink = row['vol_ratio'] < 0.8
            # 3. 不破SOS低点
            no_break = row['close'] >= sos_price * 0.98
            # 4. 振幅小于SOS到回测期间的均幅，或下影线长（>30%）
            small_amp = amp < avg_amp * 0.8
            long_lower_shadow = lower_shadow > 0.3
            candle_confirm = small_amp or long_lower_shadow
            
            if price_near and vol_shrink and no_break and candle_confirm:
                event.status = EventStatus.CONFIRMED
                event.confirmed_by = f"回测不破: {row['date'].strftime('%Y-%m-%d')}"
                detail = []
                if small_amp:
                    detail.append(f"振幅{amp:.1f}%<均幅{avg_amp:.1f}%")
                if long_lower_shadow:
                    detail.append(f"下影线{lower_shadow*100:.1f}%")
                event.verification_details = f"缩量回调+{'+'.join(detail)}"
                return
        
        # 检查是否放量跌破SOS（失效）
        for idx in range(event_idx + 1, min(event_idx + 15, total_bars)):
            if idx >= total_bars:
                break
            row = self.df.iloc[idx]
            if row['close'] < sos_price * 0.96 and row['vol_ratio'] >= 1.1:
                event.status = EventStatus.FAILED
                event.failed_reason = "放量跌破SOS低点"
                return
        
        # 验证窗口结束，未确认也未失效，保持PENDING（等待后续确认）
        event.verification_details = f"20日内未出现有效回测，继续观察"
    
    def _verify_bc(self, event: WyckoffEvent, total_bars: int):
        """验证BC - 检测到直接确立，后续检查是否失效"""
        event_idx = self.df[self.df['date'] == event.date].index[0]
        bc_volume = event.volume
        bc_price = event.price
        
        # BC检测到直接确立
        event.status = EventStatus.CONFIRMED
        event.confirmed_by = "抢购高潮已确立"
        event.verification_details = f"BC高点: ¥{bc_price:.2f}"
        
        # 检查是否失效：出现放量下跌
        check_start = event_idx + 1
        check_end = min(event_idx + 25, total_bars)
        
        for idx in range(check_start, check_end):
            if idx >= total_bars:
                break
            row = self.df.iloc[idx]
            # 放量下跌跌破BC支撑 = 失效
            if row['close'] < bc_price * 0.95 and row['vol_ratio'] >= 1.2:
                event.status = EventStatus.FAILED
                event.failed_reason = "放量下跌跌破BC，趋势反转"
                return
        
        # 没有失效，保持CONFIRMED状态
        event.verification_details = f"BC已确立，待观察是否跌破支撑"


    
    def _verify_ut(self, event: WyckoffEvent, total_bars: int):
        """验证UT - 需要放量下跌跟随"""
        event_idx = self.df[self.df['date'] == event.date].index[0]
        ut_low = event.price * 0.98  # UT的低点
        
        # 验证窗口：3-10天
        verify_start = event_idx + 1
        verify_end = min(event_idx + 10, total_bars)
        
        if verify_start >= total_bars:
            event.status = EventStatus.FAILED
            event.failed_reason = "数据不足"
            return
        
        # 寻找SOW（放量下跌）
        for idx in range(verify_start, verify_end):
            if idx >= total_bars:
                break
            
            row = self.df.iloc[idx]
            
            # SOW条件：跌破UT低点，放量下跌
            break_down = row['close'] < ut_low
            vol_increase = row['vol_ratio'] > 1.3
            big_drop = row['pct_change'] < -2
            
            if break_down and vol_increase and big_drop:
                event.status = EventStatus.CONFIRMED
                event.confirmed_by = f"SOW跟随确认: {row['date'].strftime('%Y-%m-%d')}"
                event.verification_details = "UT后出现放量下跌"
                return
        
        # 检查是否缩量反弹创新高（失效）
        for idx in range(event_idx + 1, min(event_idx + 10, total_bars)):
            if idx >= total_bars:
                break
            row = self.df.iloc[idx]
            if row['close'] > event.price and row['vol_ratio'] < 1.2:
                event.status = EventStatus.FAILED
                event.failed_reason = "UT后缩量反弹创新高"
                return
        
        event.status = EventStatus.FAILED
        event.failed_reason = "10日内未出现SOW跟随"
    
    def _verify_sow(self, event: WyckoffEvent, total_bars: int):
        """验证SOW - 需要无力反弹确认"""
        event_idx = self.df[self.df['date'] == event.date].index[0]
        sow_low = event.price
        
        # 验证窗口：5-15天
        verify_start = event_idx + 3
        verify_end = min(event_idx + 15, total_bars)
        
        if verify_start >= total_bars:
            event.status = EventStatus.FAILED
            event.failed_reason = "数据不足"
            return
        
        # 寻找无力反弹
        for idx in range(verify_start, verify_end):
            if idx >= total_bars:
                break
            
            row = self.df.iloc[idx]
            
            # 无力反弹条件：缩量反弹，高度不足
            vol_shrink = row['vol_ratio'] < 0.6
            weak_rally = row['close'] < sow_low * 1.03  # 反弹不到3%
            
            if vol_shrink and weak_rally:
                event.status = EventStatus.CONFIRMED
                event.confirmed_by = f"无力反弹确认: {row['date'].strftime('%Y-%m-%d')}"
                event.verification_details = "缩量反弹，高度不足SOW低点"
                return
        
        # 检查是否放量收复失地（失效）
        for idx in range(event_idx + 1, min(event_idx + 10, total_bars)):
            if idx >= total_bars:
                break
            row = self.df.iloc[idx]
            if row['close'] > event.price and row['vol_ratio'] > 1.0:
                event.status = EventStatus.FAILED
                event.failed_reason = "SOW后放量收复失地"
                return
        
        event.status = EventStatus.FAILED
        event.failed_reason = "15日内未出现无力反弹"
    
    def _verify_st(self, event: WyckoffEvent, total_bars: int):
        """验证ST - 检测时即确认，后续只检查是否失效"""
        event_idx = self.df[self.df['date'] == event.date].index[0]
        
        event.status = EventStatus.CONFIRMED
        event.confirmed_by = "ST检测确认"
        
        if event.volume < self.df['vol_ma20'].iloc[event_idx] * 0.7:
            event.verification_details = "成交量显著萎缩，确认有效"
        else:
            event.verification_details = "已确认，待观察是否失效"

    def get_confirmed_events(self) -> List[WyckoffEvent]:
        """获取已确认事件"""
        return self.confirmed_events
    
    def get_all_events(self) -> List[WyckoffEvent]:
        """获取所有事件"""
        return self.pending_events


class WyckoffPhaseDetector:
    """阶段检测器 - 基于状态机的威科夫阶段判断"""
    
    # 状态枚举
    class State(Enum):
        ACCUMULATION_A = "吸筹A-熊市结束"
        ACCUMULATION_B = "吸筹B-区间"
        ACCUMULATION_C = "吸筹C-确认"
        ACCUMULATION_D = "吸筹D-上涨"
        DISTRIBUTION_A = "派发A-牛市结束"
        DISTRIBUTION_B = "派发B-区间"
        DISTRIBUTION_C = "派发C-确认"
        DISTRIBUTION_D = "派发D-下跌"
        UNKNOWN = "未知"
    
    def __init__(self, events: List[WyckoffEvent], df: pd.DataFrame):
        self.events = events
        self.df = df
        self.phase = ""
        self.trend = ""
        self.current_state = self.State.UNKNOWN
        self.state_history = []  # 状态历史
    
    def detect_phase(self) -> Tuple[str, str]:
        """使用状态机检测当前阶段"""
        confirmed = [e for e in self.events if e.status == EventStatus.CONFIRMED]
        
        if not confirmed:
            return self._detect_by_price()
        
        # 按日期排序
        confirmed_sorted = sorted(confirmed, key=lambda x: x.date)
        
        # 从头开始遍历事件，构建状态序列
        self.state_history = []
        current_state = self.State.UNKNOWN
        
        for event in confirmed_sorted:
            new_state = self._get_next_state(current_state, event.event_type)
            if new_state != current_state:
                self.state_history.append((event.date, event.event_type, new_state.value))
                current_state = new_state
        
        # 当前状态
        self.current_state = current_state if current_state != self.State.UNKNOWN else self._infer_state(confirmed_sorted)
        
        # 转换为输出格式
        self.phase = self.current_state.value
        self.trend = self._get_trend_from_state(self.current_state)
        
        return self.phase, self.trend
    
    def _get_next_state(self, current_state, event_type: str) -> 'WyckoffPhaseDetector.State':
        """根据事件类型计算下一状态"""
        
        # 吸筹方向的事件序列
        if current_state == self.State.UNKNOWN or current_state == self.State.DISTRIBUTION_D:
            if event_type == 'SC':
                return self.State.ACCUMULATION_A
            return self.State.UNKNOWN
        
        elif current_state == self.State.ACCUMULATION_A:
            if event_type in ['ST', 'SPRING', 'SPRING_BEST', 'SPRING_STRONG']:
                return self.State.ACCUMULATION_B
            elif event_type == 'SC':
                return self.State.ACCUMULATION_A  # 重新开始
            return self.State.ACCUMULATION_A
        
        elif current_state == self.State.ACCUMULATION_B:
            if event_type in ['SPRING', 'SPRING_BEST', 'SPRING_STRONG', 'SPRING_SHAKEOUT']:
                return self.State.ACCUMULATION_C
            elif event_type in ['SOS', 'JOC']:
                return self.State.ACCUMULATION_D
            return self.State.ACCUMULATION_B
        
        elif current_state == self.State.ACCUMULATION_C:
            if event_type in ['SOS', 'JOC']:
                return self.State.ACCUMULATION_D
            return self.State.ACCUMULATION_C
        
        elif current_state == self.State.ACCUMULATION_D:
            if event_type in ['SOW', 'BC', 'UT']:
                return self.State.DISTRIBUTION_A
            elif event_type in ['SC']:
                return self.State.ACCUMULATION_A
            return self.State.ACCUMULATION_D
        
        # 派发方向的事件序列
        elif current_state == self.State.DISTRIBUTION_A:
            if event_type in ['ST', 'SOW']:
                return self.State.DISTRIBUTION_B
            elif event_type in ['BC', 'UT']:
                return self.State.DISTRIBUTION_A
            return self.State.DISTRIBUTION_A
        
        elif current_state == self.State.DISTRIBUTION_B:
            if event_type in ['UT', 'SOW']:
                return self.State.DISTRIBUTION_C
            return self.State.DISTRIBUTION_B
        
        elif current_state == self.State.DISTRIBUTION_C:
            if event_type in ['SOW', 'BC']:
                return self.State.DISTRIBUTION_D
            return self.State.DISTRIBUTION_C
        
        elif current_state == self.State.DISTRIBUTION_D:
            if event_type in ['SC']:
                return self.State.ACCUMULATION_A
            return self.State.DISTRIBUTION_D
        
        return current_state
    
    def _infer_state(self, confirmed_sorted) -> 'WyckoffPhaseDetector.State':
        """从事件列表推断当前状态"""
        if not confirmed_sorted:
            return self.State.UNKNOWN
        
        latest_event = confirmed_sorted[-1]
        event_type = latest_event.event_type
        
        # 根据最新事件推断
        if event_type in ['SC']:
            return self.State.ACCUMULATION_A
        elif event_type in ['ST', 'SPRING', 'SPRING_BEST', 'SPRING_STRONG']:
            return self.State.ACCUMULATION_B
        elif event_type in ['SOS', 'JOC']:
            return self.State.ACCUMULATION_D
        elif event_type in ['BC', 'UT']:
            return self.State.DISTRIBUTION_A
        elif event_type in ['SOW']:
            return self.State.DISTRIBUTION_D
        
        return self.State.UNKNOWN
    
    def _get_trend_from_state(self, state: 'WyckoffPhaseDetector.State') -> str:
        """根据状态获取趋势方向"""
        if state in [self.State.ACCUMULATION_A, self.State.ACCUMULATION_B]:
            return "筑底"
        elif state == self.State.ACCUMULATION_C:
            return "可能转涨"
        elif state == self.State.ACCUMULATION_D:
            return "上涨"
        elif state in [self.State.DISTRIBUTION_A, self.State.DISTRIBUTION_B]:
            return "可能见顶"
        elif state in [self.State.DISTRIBUTION_C, self.State.DISTRIBUTION_D]:
            return "下跌"
        return "震荡"
    
    def _detect_by_price(self) -> Tuple[str, str]:
        """基于价格位置辅助判断"""
        if len(self.df) < 60:
            return "数据不足", "未知"
        
        recent = self.df.tail(60)
        
        volatility = recent['close'].std() / recent['close'].mean()
        vol_ratio = recent['volume'].tail(20).mean() / recent['volume'].mean() if recent['volume'].mean() > 0 else 1
        
        ma20 = recent['ma20'].iloc[-1] if not pd.isna(recent['ma20'].iloc[-1]) else recent['close'].iloc[-1]
        ma60 = recent['ma60'].iloc[-1] if 'ma60' in recent.columns and not pd.isna(recent['ma60'].iloc[-1]) else ma20
        
        if volatility < 0.08 and vol_ratio < 0.6:
            return "阶段B-吸筹区间", "震荡筑底"
        elif volatility >= 0.12 and ma20 > ma60:
            return "阶段D-上涨趋势", "上涨"
        elif volatility >= 0.12 and ma20 < ma60:
            return "阶段D-下跌趋势", "下跌"
        else:
            return "阶段B-盘整", "震荡"


class ProbabilityCloud:
    """概率云分析器 - 用于初筛潜在事件"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.likelihood = pd.DataFrame(index=df.index)
        self._prepare_indicators()
    
    def sigmoid_score(self, value, min_threshold, max_threshold, mid_threshold=None, steepness=5):
        """
        平滑评分函数（Sigmoid改进）
        value: 特征实际值
        min_threshold: 最低阈值（得0分）
        max_threshold: 最高阈值（得1分）
        mid_threshold: 中间阈值（得0.5分），默认=(min+max)/2
        steepness: 曲线陡峭度
        """
        import numpy as np
        if mid_threshold is None:
            mid_threshold = (min_threshold + max_threshold) / 2
        
        normalized = (value - mid_threshold) / (max_threshold - min_threshold) * 2
        return 1 / (1 + np.exp(-steepness * normalized))
    
    def _prepare_indicators(self):
        df = self.df
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['amplitude'] = (df['high'] - df['low']) / df['close'] * 100
        df['pct_change'] = df['close'].pct_change() * 100
        df['vol_ratio'] = df['volume'] / df['vol_ma20']
        
        # ===== SASE支撑位预计算 =====
        # 计算ATR
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr_14'] = df['tr'].rolling(14).mean()
        df['atr_100'] = df['tr'].rolling(100).mean()
        df['atr_ratio'] = df['atr_14'] / (df['atr_100'] + 1e-9)
        
        # 计算ADX
        df['plus_dm'] = df['high'].diff()
        df['minus_dm'] = -df['low'].diff()
        df['plus_dm'] = df['plus_dm'].apply(lambda x: x if x > 0 else 0)
        df['minus_dm'] = df['minus_dm'].apply(lambda x: x if x > 0 else 0)
        df['plus_di'] = df['plus_dm'].rolling(14).mean() / (df['atr_14'] + 1e-9) * 100
        df['minus_di'] = df['minus_dm'].rolling(14).mean() / (df['atr_14'] + 1e-9) * 100
        dx = abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-9) * 100
        df['adx'] = dx.rolling(14).mean()
        
        # 布林带支撑 - 动态容差
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        
        # 恢复原容差2.0
        atr_ratio = df['atr_ratio'].fillna(1.0)
        tolerance = 2.0 * (1.0 + (1 - atr_ratio.clip(0, 2)) * 0.5)
        df['bb_lower'] = df['bb_mid'] - tolerance * df['bb_std']
        
        # ===== ATR自适应Zigzag支撑计算 =====
        df['atr_multiplier'] = 2.0
        df['atr_threshold'] = df['atr_14'] * df['atr_multiplier']
        
        self.df = df

    
    def calculate_all_likelihoods(self):
        self._calc_sc_likelihood()
        self._calc_spring_zigzag_likelihood()  # Zigzag支撑的Spring检测（使用最佳参数: threshold=2%, depth=14, backstep=1 + 趋势过滤）
        self._calc_sos_likelihood()  # SOS包含JOC标记
        self._calc_bc_likelihood()
        self._calc_ut_likelihood()
        self._calc_sow_likelihood()
        self._calc_st_likelihood()
        return self.likelihood
    
    def _calculate_zigzag(self, df, threshold=0.02, depth=14, backstep=1):
        """
        计算Zigzag拐点（使用high/low高低点）
        返回: (pivots, supports, resistances)
        """
        high = df['high'].values
        low = df['low'].values
        n = len(df)
        
        pivots = [0]
        direction = 0
        last_pivot_direction = 0
        last_pivot_idx = 0
        
        for i in range(1, n):
            if i - last_pivot_idx < depth:
                continue
            
            if direction == 0:
                if high[i] >= high[0] * (1 + threshold):
                    direction = 1
                    pivots.append(i)
                    last_pivot_direction = 1
                    last_pivot_idx = i
                elif low[i] <= low[0] * (1 - threshold):
                    direction = -1
                    pivots.append(i)
                    last_pivot_direction = -1
                    last_pivot_idx = i
            elif direction == 1:
                if high[i] > high[pivots[-1]]:
                    pivots[-1] = i
                elif low[i] <= low[pivots[-1]] * (1 - threshold):
                    if i - pivots[-1] >= backstep:
                        direction = -1
                        pivots.append(i)
                        last_pivot_direction = -1
                        last_pivot_idx = i
            elif direction == -1:
                if low[i] < low[pivots[-1]]:
                    pivots[-1] = i
                elif high[i] >= high[pivots[-1]] * (1 + threshold):
                    if i - pivots[-1] >= backstep:
                        direction = 1
                        pivots.append(i)
                        last_pivot_direction = 1
                        last_pivot_idx = i
        
        supports = []
        for i in range(1, len(pivots) - 1):
            idx = pivots[i]
            if (low[pivots[i-1]] > low[idx]) and (low[pivots[i+1]] > low[idx]):
                supports.append({'idx': idx, 'price': low[idx], 'type': 'support'})
        
        resistances = []
        for i in range(1, len(pivots) - 1):
            idx = pivots[i]
            if (high[pivots[i-1]] < high[idx]) and (high[pivots[i+1]] < high[idx]):
                resistances.append({'idx': idx, 'price': high[idx], 'type': 'resistance'})
        
        return pivots, supports, resistances
    
    def _is_uptrend_or_sideways(self, current_idx):
        """
        趋势过滤：只允许非下跌趋势
        条件：最近2个Zigzag低点不创新低（Low2 >= Low1）
        """
        if current_idx < 30:
            return True
        
        historical_df = self.df.iloc[:current_idx]
        
        if len(historical_df) < 30:
            return True
        
        _, supports, _ = self._calculate_zigzag(
            historical_df,
            threshold=0.05, depth=10, backstep=3
        )
        
        if len(supports) < 2:
            return True
        
        last1 = supports[-1]['price']
        last2 = supports[-2]['price']
        
        return last1 >= last2
    
    def _calc_sc_likelihood(self):
        """SC概率云：使用sigmoid平滑评分"""
        df = self.df
        
        # 15日内跌幅: 3%得0分，10%得1分（更宽松）
        price_decline = df['close'].pct_change(15)
        cond_decline = (-price_decline).apply(
            lambda x: self.sigmoid_score(x, min_threshold=0.03, max_threshold=0.10, steepness=4)
        )
        
        # 当日跌幅: -1.5%得0分，-6%得1分
        cond_drop = (-df['pct_change']).apply(
            lambda x: self.sigmoid_score(x, min_threshold=1.5, max_threshold=6, steepness=4)
        )
        
        # 量比: 0.8得0分，1.8得1分（恐慌抛售通常放量）
        cond_vol = df['vol_ratio'].apply(
            lambda x: self.sigmoid_score(x, min_threshold=0.8, max_threshold=1.8, steepness=4)
        )
        
        # 振幅: 2.5%得0分，7%得1分
        cond_amp = df['amplitude'].apply(
            lambda x: self.sigmoid_score(x, min_threshold=2.5, max_threshold=7, steepness=4)
        )
        
        self.likelihood['SC'] = (
            cond_decline * 0.15 + 
            cond_drop * 0.35 +
            cond_vol * 0.30 + 
            cond_amp * 0.20
        )
    
    def _is_uptrend_or_sideways(self, current_idx):
        """
        趋势过滤：只允许非下跌趋势
        条件：最近2个Zigzag低点不创新低（Low2 >= Low1）
        """
        if current_idx < 30:
            return True
        
        historical_df = self.df.iloc[:current_idx]
        
        if len(historical_df) < 30:
            return True
        
        _, supports, _ = self._calculate_zigzag(
            historical_df,
            threshold=0.05, depth=10, backstep=3
        )
        
        if len(supports) < 2:
            return True
        
        last1 = supports[-1]['price']
        last2 = supports[-2]['price']
        
        return last1 >= last2
    
    def _calc_spring_zigzag_likelihood(self):
        """
        Spring概率云 - 基于Zigzag支撑位（使用high/low高低点，无未来函数）
        
        检测条件：
        1. 只用i之前的历史数据计算Zigzag支撑位
        2. 趋势过滤：只允许非下跌趋势
        3. 价格（low）跌破Zigzag支撑位
        4. 5天内收盘价超过支撑位
        5. Spring确立日 = 价格翻回支撑线的那天
        
        参数（使用最佳参数: threshold=2%, depth=14, backstep=1）
        """
        df = self.df
        spring_strength = pd.Series([0.0] * len(df), index=df.index)
        
        threshold = 0.02
        depth = 14
        backstep = 1
        
        close = df['close'].values
        low = df['low'].values
        high = df['high'].values
        
        for i in range(50, len(df)):
            # 趋势过滤：只允许非下跌趋势
            if not self._is_uptrend_or_sideways(i):
                continue
            
            historical_df = df.iloc[:i][['high', 'low']].copy()
            
            if len(historical_df) < 30:
                continue
            
            _, zigzag_supports, _ = self._calculate_zigzag(
                historical_df,
                threshold=threshold, 
                depth=depth, 
                backstep=backstep
            )
            
            if not zigzag_supports:
                continue
            
            support_price = None
            for s in zigzag_supports:
                if s['idx'] < i - 1:
                    support_price = s['price']
            
            if support_price is None:
                continue
            
            break_price = low[i]
            
            if break_price >= support_price:
                continue
            
            for j in range(1, 6):
                if i + j >= len(df):
                    break
                if close[i + j] > support_price:
                    spring_strength.iloc[i + j] = 1.0
                    break
        
        self.likelihood['SPRING_ZIGZAG'] = spring_strength
    
    def _calc_sos_likelihood(self):
        """SOS概率云：大阳线型 + JOC震荡区识别"""
        df = self.df
        
        # K线实体
        df['body'] = abs(df['close'] - df['open'])
        df['body_ratio'] = df['body'] / (df['high'] - df['low'] + 0.001)
        
        # ===== 大阳线型（使用sigmoid平滑评分） =====
        # 实体占比: 90%满分，60%得0分
        cond_body = df['body_ratio'].apply(
            lambda x: self.sigmoid_score(x, min_threshold=0.6, max_threshold=0.9, steepness=8)
        )
        # 量比: 1.5满分，0.8得0分
        cond_vol = df['vol_ratio'].apply(
            lambda x: self.sigmoid_score(x, min_threshold=0.8, max_threshold=1.5, steepness=5)
        )
        # 涨幅: 9.5%以上满分，3%得0分
        cond_pct = df['pct_change'].apply(
            lambda x: self.sigmoid_score(x, min_threshold=3, max_threshold=9.5, steepness=5)
        )
        
        self.likelihood['SOS'] = cond_body * 0.2 + cond_vol * 0.3 + cond_pct * 0.5
        
        # ===== JOC震荡区识别 - 两种方案 =====
        df['high_60'] = df['high'].rolling(60).max()
        df['low_60'] = df['low'].rolling(60).min()
        
        # 方案1: 布林带带宽 (BBW)
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bbw'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        df['bbw_ma20'] = df['bbw'].rolling(20).mean()
        
        # 震荡区判定：BBW低于20日均值30%
        squeeze_bbw = df['bbw'] < df['bbw_ma20'] * 0.7
        
        # 方案2: 线性回归斜率 + R² (简化版)
        def calc_linear_regression(x):
            """计算斜率和R²"""
            import numpy as np
            if len(x) < 5:
                return 0, 0
            y = np.array(x)
            n = len(y)
            x_idx = np.arange(n)
            sum_x = np.sum(x_idx)
            sum_y = np.sum(y)
            sum_xy = np.sum(x_idx * y)
            sum_x2 = np.sum(x_idx ** 2)
            sum_y2 = np.sum(y ** 2)
            
            denominator = np.sqrt((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
            if denominator == 0:
                return 0, 0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            r_squared = ((n * sum_xy - sum_x * sum_y) ** 2) / ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2))
            return slope, r_squared
        
        # 使用简化方法：计算10日价格波动率
        df['price_std'] = df['close'].rolling(10).std()
        df['price_range'] = df['close'].rolling(10).max() - df['close'].rolling(10).min()
        df['price_range_ratio'] = df['price_range'] / df['close']
        
        # 震荡区判定：10日价格波动率低于历史30%分位
        df['price_std_ma60'] = df['price_std'].rolling(60).mean()
        squeeze_lr = df['price_std'] < df['price_std_ma60'] * 0.7
        
        # 震荡区判定：BBW低于20日均值30% 或 价格波动率低于历史均值30%
        in_range = (squeeze_bbw | squeeze_lr).shift(1).fillna(False)
        
        # 突破60日震荡区
        break_range = (df['close'] > df['high_60'].shift(1)) & (~in_range)
        
        # JOC = SOS + 突破加分（但必须是先满足SOS条件）
        # SOS达标阈值: 0.6（与事件检测阈值一致）
        sos_threshold = 0.6
        sos达标 = (self.likelihood['SOS'] >= sos_threshold).astype(float)
        
        # 只有SOS达标后，突破加分才有效
        break_score = break_range.astype(float) * 0.5 * sos达标
        self.likelihood['JOC'] = self.likelihood['SOS'] + break_score
    
    def _calc_bc_likelihood(self):
        df = self.df
        # 调整条件：20日涨幅>15%（更严格），量比>1.8（更宽松）
        prior_up = (df['close'] > df['close'].shift(20) * 1.15).astype(float)
        big_up = (df['pct_change'] > 5).astype(float)
        up_score = (df['pct_change'] / 10).clip(0, 1)
        huge_vol = (df['vol_ratio'] > 1.8).astype(float)  # 调整为1.8
        vol_score = ((df['vol_ratio'] - 1) / 1.5).clip(0, 1)
        upper_shadow = ((df['high'] - df['close']) / (df['high'] - df['low'] + 0.001))
        shadow_score = (upper_shadow / 0.5).clip(0, 1)
        
        self.likelihood['BC'] = (
            prior_up * 0.20 +
            big_up * 0.25 +
            up_score * 0.15 +
            huge_vol * 0.25 +
            vol_score * 0.15 +
            shadow_score * 0.15
        )
    
    def _calc_ut_likelihood(self):
        df = self.df
        break_high = (df['high'] > df['high_20'].shift(1)).astype(float)
        pullback = (df['close'] < df['high_20'].shift(1)).astype(float)
        upper_shadow = ((df['high'] - df['close']) / (df['high'] - df['low'] + 0.001))
        shadow_score = (upper_shadow / 0.7).clip(0, 1)
        vol_score = ((df['vol_ratio'] - 1) / 1.2).clip(0, 1)
        
        self.likelihood['UT'] = (
            break_high * 0.20 +
            pullback * 0.25 +
            shadow_score * 0.30 +
            vol_score * 0.25
        )
    
    def _calc_sow_likelihood(self):
        df = self.df
        break_support = (df['close'] < df['low_20'].shift(1)).astype(float)
        down_vol = ((df['pct_change'] < 0) & (df['vol_ratio'] > 1.5)).astype(float)
        vol_score = ((df['vol_ratio'] - 1) / 1.2).clip(0, 1)
        big_down = (df['pct_change'] < -2.5).astype(float)
        down_score = (-df['pct_change'] / 6).clip(0, 1)
        prior_up = (df['close'] > df['close'].shift(20) * 1.08).astype(float)
        
        self.likelihood['SOW'] = (
            break_support * 0.30 +
            down_vol * 0.25 +
            big_down * 0.25 +
            prior_up * 0.20
        )
    
    def _calc_st_likelihood(self):
        """ST概率云优化版：动态支撑 + 拒绝信号 + 量价配合"""
        df = self.df
        
        # 震荡区间下轨作为支撑位（使用布林带下轨）
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        
        # 计算收盘价相对于布林下轨的距离
        dist_to_support = (df['close'] - df['bb_lower']) / df['bb_lower']
        
        lower_shadow_ratio = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
        
        def calc_support_score(row):
            dist = row['dist']
            shadow = row['shadow']
            
            if 0 <= dist <= 0.05:
                base_score = 1.0 - (dist / 0.05) * 0.5
            elif dist > 0.05:
                base_score = max(0, 0.5 - (dist - 0.05) * 5)
            else:
                if shadow > 0.6:
                    base_score = 0.8 + (shadow - 0.6) * 1.0
                else:
                    base_score = max(0, 0.2 + dist * 10)
                    
            return min(max(base_score, 0), 1)

        temp_calc = pd.DataFrame({
            'dist': dist_to_support,
            'shadow': lower_shadow_ratio
        })
        
        cond_near = temp_calc.apply(calc_support_score, axis=1)

        # cond_vol: 量比越小分数越高 (缩量是ST的标志)
        # 量比 0.4满分，1.4零分
        cond_vol = df['vol_ratio'].apply(
            lambda x: self.sigmoid_score(x, min_threshold=1.4, max_threshold=0.4, steepness=4)
        )
        
        # cond_amp: 实体振幅越小分数越高 (窄幅震荡是ST的标志)
        # 实体振幅 0.5%满分，3.0%零分
        body_amplitude = abs(df['close'] - df['open']) / df['open'] * 100
        cond_amp = body_amplitude.apply(
            lambda x: self.sigmoid_score(x, min_threshold=3.0, max_threshold=0.5, steepness=4)
        )
        
        self.likelihood['ST'] = (
            cond_near * 0.50 +
            cond_vol * 0.20 +
            cond_amp * 0.30
        )
    
    def get_events_above_threshold(self, threshold: float = 0.4) -> list:
        """获取高于阈值的所有事件 - 不限制数量"""
        events = []
        for col in self.likelihood.columns:
            above_thresh = self.likelihood[self.likelihood[col] >= threshold]
            if len(above_thresh) > 0:
                # 取所有高于阈值的事件
                for idx in above_thresh.index:
                    events.append({
                        'type': col,
                        'likelihood': self.likelihood.loc[idx, col],
                        'date': self.df.loc[idx, 'date'],
                        'price': self.df.loc[idx, 'close'],
                        'volume': self.df.loc[idx, 'volume']
                    })
        # 修改排序：优先取最近的事件，同时保持高分优先
        return sorted(events, key=lambda x: (x['date'], x['likelihood']), reverse=True)


class WyckoffAnalyzer:
    """威科夫分析器 - 主类"""
    
    def __init__(self, symbol: str, days: int = 120):
        self.symbol = symbol
        self.days = days
        self.df = None
        self.stock_name = ""
        self.events: List[WyckoffEvent] = []
        self.verifier: Optional[EventVerifier] = None
        self.phase = ""
        self.trend = ""
    
    def fetch_data(self) -> bool:
        """获取股票/ETF历史数据"""
        try:
            print(f"正在获取 {self.symbol} 日线数据...")
            
            end_date = datetime.now().strftime('%Y%m%d')
            # 如果请求天数超过500天，获取全部历史数据
            if self.days >= 500:
                start_date = '20070101'  # 获取全部历史
            else:
                start_date = (datetime.now() - timedelta(days=self.days + 60)).strftime('%Y%m%d')
            
            # 判断是否为ETF
            is_etf = self.symbol.startswith('5') or self.symbol.startswith('15') or \
                     self.symbol.startswith('16') or self.symbol.startswith('159') or \
                     self.symbol.startswith('51') or self.symbol.startswith('58')
            
            if is_etf:
                print(f"检测为ETF，使用ETF数据接口...")
                try:
                    if self.symbol.startswith('15') or self.symbol.startswith('16') or self.symbol.startswith('159'):
                        etf_prefix = 'sz'
                    else:
                        etf_prefix = 'sh'
                    
                    self.df = ak.fund_etf_hist_sina(symbol=f"{etf_prefix}{self.symbol}")
                    
                    if self.df is not None and not self.df.empty:
                        self.df['date'] = pd.to_datetime(self.df['date'])
                        start_dt = datetime.strptime(start_date, '%Y%m%d')
                        end_dt = datetime.strptime(end_date, '%Y%m%d')
                        self.df = self.df[(self.df['date'] >= start_dt) & (self.df['date'] <= end_dt)]
                        self.stock_name = f"ETF{self.symbol}"
                    else:
                        raise Exception("ETF数据为空")
                except Exception as e:
                    print(f"ETF接口失败: {e}")
                    return False
            else:
                # 股票数据接口
                try:
                    symbol_with_prefix = f"sh{self.symbol}" if self.symbol.startswith('6') else f"sz{self.symbol}"
                    self.df = ak.stock_zh_a_daily(
                        symbol=symbol_with_prefix,
                        start_date=start_date,
                        end_date=end_date
                    )
                except Exception as e:
                    print(f"股票接口失败: {e}")
                    try:
                        self.df = ak.stock_zh_a_hist(
                            symbol=self.symbol,
                            period="daily",
                            start_date=start_date,
                            end_date=end_date,
                            adjust="qfq"
                        )
                    except:
                        self.df = None
                
                # 获取股票名称
                if self.df is not None and not self.df.empty:
                    try:
                        info = ak.stock_individual_info_em(symbol=self.symbol)
                        if info is not None and not info.empty:
                            name_row = info[info['item'] == '股票简称']
                            if not name_row.empty:
                                self.stock_name = name_row['value'].values[0]
                            else:
                                self.stock_name = "未知"
                        else:
                            self.stock_name = "未知"
                    except:
                        self.stock_name = "未知"
            
            if self.df is None or self.df.empty:
                print(f"未能获取 {self.symbol} 的数据")
                return False
            
            # 数据预处理 - 新接口返回英文列名
            # 尝试兼容中英文列名
            if '日期' in self.df.columns:
                column_mapping = {
                    '日期': 'date', '股票代码': 'symbol', '开盘': 'open',
                    '收盘': 'close', '最高': 'high', '最低': 'low',
                    '成交量': 'volume', '成交额': 'amount', '振幅': 'amplitude',
                    '涨跌幅': 'change_pct', '涨跌额': 'change', '换手率': 'turnover'
                }
                self.df = self.df.rename(columns=column_mapping)
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date').reset_index(drop=True)
            
            print(f"成功获取 {len(self.df)} 条日线数据")
            return True
            
        except Exception as e:
            print(f"获取数据失败: {e}")
            return False
    
    def detect_events(self):
        """检测并验证威科夫事件"""
        if self.df is None or len(self.df) < 30:
            return
        
        # 1. 使用概率云初筛潜在事件
        pc = ProbabilityCloud(self.df)
        pc.calculate_all_likelihoods()
        
        # 不同事件类型使用不同阈值
        threshold_map = {
            'SOS': 0.6,
            'JOC': 0.6,
            'SC': 0.5,
            'ST': 0.76,
            'BC': 0.5,
            'UT': 0.5,
            'SOW': 0.5,
            'SPRING_ZIGZAG': 0.5,   # Zigzag支撑的Spring（使用最佳参数: threshold=2%, depth=14, backstep=3）
            'SPRING_BEST': 0.5,
            'SPRING_STRONG': 0.5,
            'SPRING_CONFIRM': 0.5,
            'SPRING_SHAKEOUT': 0.5,
        }
        
        potential_events = []
        for event_type, threshold in threshold_map.items():
            if event_type in pc.likelihood.columns:
                above_thresh = pc.likelihood[pc.likelihood[event_type] >= threshold]
                for idx in above_thresh.index:
                    potential_events.append({
                        'type': event_type,
                        'likelihood': pc.likelihood.loc[idx, event_type],
                        'date': self.df.loc[idx, 'date'],
                        'price': self.df.loc[idx, 'close'],
                        'volume': self.df.loc[idx, 'volume']
                    })
        
        # 2. 创建验证器
        self.verifier = EventVerifier(self.df)
        
        # 3. 将潜在事件加入验证队列
        for pe in potential_events:
            event = WyckoffEvent(
                event_type=pe['type'],
                date=pe['date'],
                price=pe['price'],
                volume=pe['volume'],
                likelihood=pe['likelihood']
            )
            self.verifier.add_pending_event(event)
        
        # 4. 执行验证
        self.verifier.verify_all()
        
        # 5. 获取所有事件
        self.events = self.verifier.get_all_events()
    
    def get_phase(self) -> Tuple[str, str]:
        """获取当前阶段"""
        if not self.events:
            self.phase = "数据不足"
            self.trend = "未知"
            return self.phase, self.trend
        
        detector = WyckoffPhaseDetector(self.events, self.df)
        self.phase, self.trend = detector.detect_phase()
        return self.phase, self.trend
    
    def get_entry_signal(self) -> Tuple[bool, str]:
        """判断进场信号"""
        confirmed = [e for e in self.events if e.status == EventStatus.CONFIRMED]
        event_types = [e.event_type for e in confirmed]
        
        if 'SOS' in event_types or 'JOC' in event_types:
            lps_events = [e for e in confirmed if e.event_type == 'LPS']
            if lps_events:
                return True, "出现SOS/JOC后回调形成LPS，是较好的进场时机"
            else:
                return True, "出现强势突破信号(SOS/JOC)，但建议等待回调再进场"
        elif any(t.startswith('SPRING') for t in event_types):
            return True, "出现弹簧效应(Spring)，是吸筹确认信号，可考虑进场"
        elif 'ST' in event_types:
            return True, "二次测试成功，供应枯竭，可能进入上涨趋势"
        
        return False, "当前未出现明确的进场信号"
    
    def get_analysis_result(self) -> dict:
        """获取分析结果"""
        return {
            'symbol': self.symbol,
            'name': self.stock_name,
            'df': self.df,
            'events': [e.to_dict() for e in self.events],
            'phase': self.phase,
            'trend': self.trend,
            'entry_signal': self.get_entry_signal()
        }


def analyze_symbol(symbol: str, days: int = 120) -> dict:
    """分析单个股票/ETF"""
    analyzer = WyckoffAnalyzer(symbol, days)
    
    if not analyzer.fetch_data():
        return None
    
    # 检测事件（带验证）
    analyzer.detect_events()
    
    # 获取阶段
    analyzer.get_phase()
    
    return analyzer.get_analysis_result()


def print_analysis_report(result: dict):
    """打印分析报告"""
    if result is None:
        print("数据获取失败")
        return
    
    print("\n" + "=" * 70)
    print(f"威科夫分析报告 - {result['name']} ({result['symbol']})")
    print("=" * 70)
    
    # 阶段
    print(f"\n【当前阶段】{result['phase']}")
    print(f"【趋势方向】{result['trend']}")
    
    # 事件列表
    events = result['events']
    if events:
        print(f"\n【威科夫事件】(共{len(events)}个)")
        
        # 按状态分组
        pending = [e for e in events if e['status'] == 'pending']
        confirmed = [e for e in events if e['status'] == 'confirmed']
        failed = [e for e in events if e['status'] == 'failed']
        
        print(f"\n✅ 已确认事件 ({len(confirmed)}个):")
        if confirmed:
            for e in confirmed:
                print(f"  • {e['name']} @ {e['date'].strftime('%Y-%m-%d')} ¥{e['price']:.3f}")
                print(f"    验证: {e['confirmed_by']} | {e['verification_details']}")
        
        print(f"\n🟡 待确认事件 ({len(pending)}个):")
        if pending:
            for e in pending[:10]:
                print(f"  • {e['name']} @ {e['date'].strftime('%Y-%m-%d')} ¥{e['price']:.3f} ({e['confidence']})")
                if e.get('verification_details'):
                    print(f"    状态: {e['verification_details']}")
        
        print(f"\n❌ 已失效事件 ({len(failed)}个):")
        if failed:
            for e in failed[:3]:
                print(f"  • {e['name']} @ {e['date'].strftime('%Y-%m-%d')} - {e['failed_reason']}")
    else:
        print("\n【威科夫事件】未检测到显著事件")
    
    # 进场信号
    has_signal, reason = result['entry_signal']
    print(f"\n【进场建议】")
    if has_signal:
        print(f"  ✅ {reason}")
    else:
        print(f"  ⚠️ {reason}")
    
    print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='威科夫分析器（事件验证版）')
    parser.add_argument('symbol', nargs='?', help='股票代码')
    parser.add_argument('days', nargs='?', type=int, default=120, help='分析天数(默认120=6个月)')
    
    args = parser.parse_args()
    
    if not args.symbol:
        print("用法: python wyckoff_analysis.py <代码> [天数]")
        print("示例: python wyckoff_analysis.py 601600")
        sys.exit(1)
    
    result = analyze_symbol(args.symbol, args.days)
    print_analysis_report(result)


if __name__ == "__main__":
    main()
