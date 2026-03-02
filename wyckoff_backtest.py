#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
威科夫阶段与事件回测验证
包含:
1. 威科夫阶段规则判断
2. 威科夫事件程序化检测(含AR)
3. 概率云辅助事件判断
4. 2年周期回测验证
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class WyckoffBacktest:
    """威科夫回测分析器"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_indicators()
        
    def _prepare_indicators(self):
        """准备技术指标"""
        df = self.df
        
        # 均线
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        
        # 成交量均线
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['vol_ma60'] = df['volume'].rolling(60).mean()
        
        # 量比
        df['vol_ratio'] = df['volume'] / df['vol_ma20']
        
        # 20日/60日高低点
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['high_60'] = df['high'].rolling(60).max()
        df['low_60'] = df['low'].rolling(60).min()
        
        # 振幅
        df['amplitude'] = (df['high'] - df['low']) / df['close'] * 100
        
        # 涨跌幅
        df['pct_change'] = df['close'].pct_change() * 100
        
        # 10日/20日涨跌
        df['pct_10'] = df['close'].pct_change(10) * 100
        df['pct_20'] = df['close'].pct_change(20) * 100
        
        # 上影线/下影线比例
        df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low'] + 0.001)
        df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 0.001)
        
        self.df = df
    
    # ==================== 威科夫事件检测 ====================
    
    def detect_sc(self, idx: int) -> dict:
        """
        检测SC(恐慌抛售) - 平衡参数
        条件:
        ①近期下跌趋势
        ②当日下跌
        ③放量
        ④宽振幅或长下影
        """
        if idx < 15:
            return {'detected': False, 'score': 0}
        
        df = self.df
        row = df.iloc[idx]
        prev_15 = df.iloc[max(0, idx-15):idx]
        
        # 条件1: 近期有下跌趋势
        price_decline = (prev_15['close'].iloc[0] - prev_15['close'].iloc[-1]) / prev_15['close'].iloc[0] if len(prev_15) > 0 else 0
        cond1 = price_decline > 0.05
        
        # 条件2: 当日下跌
        cond2 = row['pct_change'] < -2
        
        # 条件3: 放量
        cond3 = row['vol_ratio'] > 1.3
        
        # 条件4: 宽振幅或长下影
        cond4 = (row['amplitude'] > 5) or (row['lower_shadow'] > 0.35)
        
        score = (cond1 * 0.20 + cond2 * 0.25 + cond3 * 0.25 + cond4 * 0.30)
        
        return {
            'detected': score >= 0.45,
            'score': score,
            'conditions': {'近期跌幅': cond1, '当日下跌': cond2, '放量': cond3, '波动大': cond4},
            'price': row['close'],
            'date': row['date']
        }
    
    def detect_ar(self, idx: int) -> dict:
        """
        检测AR(自动反弹) - 平衡参数
        条件(SC后1-10天内):
        ①反弹(上涨)
        ②成交量不过于萎缩
        ③收盘价高于开盘价
        """
        if idx < 3:
            return {'detected': False, 'score': 0}
        
        df = self.df
        row = df.iloc[idx]
        
        # 查找前10天内的SC
        sc_idx = None
        for i in range(max(0, idx-10), idx):
            sc_result = self.detect_sc(i)
            if sc_result['detected']:
                sc_idx = i
                break
        
        if sc_idx is None:
            return {'detected': False, 'score': 0, 'reason': '无前置SC'}
        
        # 条件1: 反弹(上涨即可)
        sc_price = df.iloc[sc_idx]['close']
        cond1 = row['close'] > sc_price
        
        # 条件2: 成交量不太低
        cond2 = row['vol_ratio'] > 0.6
        
        # 条件3: 阳线
        cond3 = row['close'] > row['open']
        
        score = (cond1 * 0.40 + cond2 * 0.25 + cond3 * 0.35)
        
        return {
            'detected': score >= 0.4,
            'score': score,
            'conditions': {'反弹': cond1, '有量': cond2, '收阳': cond3},
            'sc_date': df.iloc[sc_idx]['date'],
            'price': row['close'],
            'date': row['date']
        }
    
    def detect_st(self, idx: int) -> dict:
        """
        检测ST(二次测试) - 优化参数
        条件:
        ①在SC/AR后5-25天内
        ②价格接近SC低点±10%(更宽松)
        ③缩量(量比<0.8)
        ④小振幅(<6%)
        """
        if idx < 20:
            return {'detected': False, 'score': 0}
        
        df = self.df
        row = df.iloc[idx]
        
        # 查找35天内的SC
        sc_info = None
        for i in range(max(0, idx-35), idx):
            sc_result = self.detect_sc(i)
            if sc_result['detected']:
                sc_info = sc_result
                break
        
        # 如果没有SC，找AR
        if sc_info is None:
            for i in range(max(0, idx-35), idx):
                ar_result = self.detect_ar(i)
                if ar_result['detected']:
                    sc_info = ar_result
                    break
        
        if sc_info is None:
            return {'detected': False, 'score': 0, 'reason': '无前置SC/AR'}
        
        sc_price = sc_info['price']
        
        # 条件1: 在SC/AR后5-25天
        time_since_sc = idx - df[df['date'] == sc_info['date']].index[0] if 'date' in df.columns else idx
        cond1 = 5 <= time_since_sc <= 25
        
        # 条件2: 接近SC低点±10%
        cond2 = abs(row['close'] - sc_price) / sc_price < 0.10
        
        # 条件3: 缩量(更宽松)
        cond3 = row['vol_ratio'] < 0.8
        
        # 条件4: 小振幅(更宽松)
        cond4 = row['amplitude'] < 6
        
        score = (cond1 * 0.25 + cond2 * 0.30 + cond3 * 0.25 + cond4 * 0.20)
        
        return {
            'detected': score >= 0.5,
            'score': score,
            'conditions': {'时间窗口': cond1, '接近SC低': cond2, '缩量': cond3, '小振幅': cond4},
            'sc_price': sc_price,
            'price': row['close'],
            'date': row['date']
        }
    
    def detect_spring(self, idx: int, require_bull_market: bool = True) -> dict:
        """
        检测SPRING(弹簧效应) - 增强版
        包含两种成功Spring:
        1. 最佳Spring: 振幅<3%, 量比<0.5 -> 直接进场
        2. 强劲需求Spring: 成交量大 + 收盘在50%之上 -> 直接进场
        
        新增过滤:
        1. 市场背景过滤: 只在牛市/吸筹阶段报告Spring
        2. 位置判断: JOC回测/震荡区底部的Spring更可靠
        
        条件:
        ①跌破20日低点
        ②跌破幅度1-10%(更宽松)
        ③3日内收回
        ④缩量(最佳Spring) 或 放量收高(强劲需求Spring)
        """
        if idx < 5:
            return {'detected': False, 'score': 0, 'spring_type': None}
        
        df = self.df
        row = df.iloc[idx]
        
        # 条件1: 收盘价跌破20日低点或接近20日低点
        low_20 = df.iloc[idx]['low_20']
        cond1 = row['close'] < low_20 or (low_20 - row['close']) / low_20 < 0.02
        
        if not cond1:
            return {'detected': False, 'score': 0, 'spring_type': None}
        
        # 条件2: 跌破幅度1-10%(或日内跌破)
        break_pct = (low_20 - row['low']) / low_20 if row['low'] < low_20 else 0
        cond2 = break_pct > 0.005  # 允许更宽松的跌破
        
        # 条件3: 3日内收回(有收回即可)
        next_3 = df.iloc[idx:min(idx+4, len(df))]
        cond3 = (next_3['close'] > low_20).any()
        
        if not cond3:
            return {'detected': False, 'score': 0, 'spring_type': None}
        
        # 计算振幅和量比
        amplitude = row['amplitude']
        vol_ratio = row['vol_ratio']
        
        # 判断收盘位置(是否在50%之上)
        mid_price = (row['high'] + row['low']) / 2
        close_above_mid = row['close'] > mid_price
        
        # ==================== 市场背景过滤 ====================
        # 判断当前位置的市场背景
        market_context = self._get_market_context(idx)
        
        # 只有在牛市/吸筹阶段才报告Spring
        if require_bull_market and market_context not in ['BULL', 'ACCUMULATION', 'CONSOLIDATION']:
            return {'detected': False, 'score': 0, 'spring_type': None, 'reason': '非牛市背景'}
        
        # ==================== 位置判断 ====================
        # 判断Spring发生的位置
        spring_position = self._get_spring_position(idx, market_context)
        
        # ==================== 类型判断 ====================
        
        # 类型1: 最佳Spring (振幅<3%, 量比<0.5)
        best_spring = (amplitude < 3) and (vol_ratio < 0.5)
        
        # 类型2: 强劲需求Spring (成交量大 + 收盘在50%之上)
        # wyckoff.md: "成交量扩大, 但是收盘在中点之上" = 强劲需求
        strong_spring = (vol_ratio > 1.0) and close_above_mid
        
        # 计算得分 - 根据位置调整
        position_bonus = {
            'JOC_BACKTEST': 0.3,      # JOC回测 - 最高加分
            'ACCUMULATION_BOTTOM': 0.25,  # 吸筹区间底部
            'CONSOLIDATION_BOTTOM': 0.2,  # 震荡区底部
            'PULLBACK': 0.15,         # 回调中的Spring
            'ORDINARY': 0.0           # 普通位置
        }.get(spring_position, 0.0)
        
        if best_spring:
            # 最佳Spring: 缩量 + 小振幅
            cond4 = vol_ratio < 0.5
            score = 0.6 + position_bonus  # 基础分 + 位置加分
            spring_type = 'SPRING_BEST'
        elif strong_spring:
            # 强劲需求Spring: 放量 + 收盘高
            cond4 = True
            score = 0.6 + position_bonus  # 基础分 + 位置加分
            spring_type = 'SPRING_STRONG'
        else:
            # 普通Spring - 需要位置加分才能达到阈值
            cond4 = vol_ratio < 0.85
            base_score = 0.4 if cond4 else 0.2
            score = base_score + position_bonus
            spring_type = 'SPRING_CONFIRM' if cond4 else 'SPRING_WEAK'
        
        return {
            'detected': score >= 0.5,
            'score': score,
            'spring_type': spring_type,
            'position': spring_position,
            'market_context': market_context,
            'conditions': {'跌破20日低': cond1, '跌破': cond2, '3日收回': cond3, '缩量/收高': cond4},
            'amplitude': amplitude,
            'vol_ratio': vol_ratio,
            'close_above_mid': close_above_mid,
            'price': row['close'],
            'date': row['date']
        }
    
    def _get_market_context(self, idx: int) -> str:
        """
        判断市场背景
        返回: BULL(牛市), ACCUMULATION(吸筹), CONSOLIDATION(震荡), DISTRIBUTION(派发), BEAR(熊市)
        """
        if idx < 60:
            return 'UNKNOWN'
        
        df = self.df
        recent = df.iloc[max(0, idx-60):idx]
        
        # 计算趋势
        ma20 = recent['ma20'].iloc[-1] if not pd.isna(recent['ma20'].iloc[-1]) else recent['close'].iloc[-1]
        ma60 = recent['ma60'].iloc[-1] if 'ma60' in recent.columns and not pd.isna(recent['ma60'].iloc[-1]) else ma20
        
        # 计算波动率
        volatility = recent['close'].std() / recent['close'].mean()
        
        # 计算成交量变化
        vol_ratio = recent['volume'].iloc[-20:].mean() / recent['volume'].mean() if recent['volume'].mean() > 0 else 1
        
        # 判断背景
        if volatility < 0.08 and vol_ratio < 0.6:
            return 'CONSOLIDATION'  # 低波动低量 = 震荡/吸筹
        elif ma20 > ma60 and volatility >= 0.08:
            return 'BULL'  # 上涨趋势
        elif ma20 < ma60 and volatility >= 0.08:
            return 'BEAR'  # 下跌趋势
        else:
            return 'CONSOLIDATION'
    
    def _get_spring_position(self, idx: int, market_context: str) -> str:
        """
        判断Spring发生的位置
        返回: JOC_BACKTEST, ACCUMULATION_BOTTOM, CONSOLIDATION_BOTTOM, PULLBACK, ORDINARY
        """
        if idx < 20:
            return 'ORDINARY'
        
        df = self.df
        
        # 检查是否在JOC回测位置
        # JOC: 价格突破20日高点后回测
        for i in range(max(0, idx-30), idx):
            if df.iloc[i]['close'] > df.iloc[i]['high_20']:  # 曾经突破20日高点
                # 检查当前是否在回测
                if df.iloc[idx]['close'] < df.iloc[idx]['high_20']:
                    return 'JOC_BACKTEST'
        
        # 检查是否在吸筹区间底部
        if market_context == 'CONSOLIDATION':
            # 检查近期是否有低点抬升(吸筹特征)
            lows = df.iloc[max(0, idx-20):idx]['low']
            if len(lows) >= 10:
                recent_lows = lows.tail(5)
                early_lows = lows.head(5)
                if recent_lows.mean() > early_lows.mean() * 0.98:  # 低点抬升
                    return 'ACCUMULATION_BOTTOM'
        
        # 检查是否是回调中的Spring(上涨趋势中的回调)
        if market_context == 'BULL':
            # 检查近期是否有高点
            highs = df.iloc[max(0, idx-20):idx]['high']
            if len(highs) >= 10:
                recent_highs = highs.tail(5)
                early_highs = highs.head(5)
                if recent_highs.mean() < early_highs.mean():  # 高点降低 = 回调
                    return 'PULLBACK'
        
        return 'ORDINARY'
    
    def detect_sos(self, idx: int) -> dict:
        """
        检测SOS(强势出现) - 收紧参数
        条件:
        ①涨幅>4%
        ②明显放量(量比>1.8)
        ③突破20日高点并收于高点附近
        ④连续3日上涨
        """
        if idx < 5:
            return {'detected': False, 'score': 0}
        
        df = self.df
        row = df.iloc[idx]
        prev_3 = df.iloc[max(0, idx-2):idx+1]
        
        # 条件1: 涨幅>4%
        cond1 = row['pct_change'] > 4
        
        # 条件2: 明显放量(更严格)
        cond2 = row['vol_ratio'] > 1.8
        
        # 条件3: 突破20日高点并收于高点附近
        cond3 = row['close'] > df.iloc[idx]['high_20'] if idx < len(df) else False
        near_high = (row['close'] - row['low']) / (row['high'] - row['low']) if row['high'] > row['low'] else 0
        cond3 = cond3 and near_high > 0.7  # 收于高位
        
        # 条件4: 连续3日上涨(更严格)
        up_days = (prev_3['pct_change'] > 0).sum() if len(prev_3) >= 3 else 0
        cond4 = up_days >= 3
        
        # 增加: 突破时成交量放大
        prev_vol = df.iloc[max(0, idx-5):idx]['volume'].mean()
        vol_expand = row['volume'] > prev_vol * 1.3 if prev_vol > 0 else False
        
        score = (cond1 * 0.20 + cond2 * 0.25 + cond3 * 0.25 + cond4 * 0.20 + vol_expand * 0.10)
        
        return {
            'detected': score >= 0.5,
            'score': score,
            'conditions': {'涨幅>4%': cond1, '放量>1.8': cond2, '突破收高': cond3, '连续3日涨': cond4, '成交量放大': vol_expand},
            'price': row['close'],
            'date': row['date']
        }
    
    def detect_bc(self, idx: int) -> dict:
        """
        检测BC(抢购高潮) - 收紧参数
        条件:
        ①前期明显上涨(20日涨幅>12%)
        ②大阳线(涨幅>5%)
        ②巨量(量比>2.2)
        ④上影线或带量滞涨
        """
        if idx < 30:
            return {'detected': False, 'score': 0}
        
        df = self.df
        row = df.iloc[idx]
        prev_20 = df.iloc[max(0, idx-20):idx]
        
        # 条件1: 前期明显上涨
        prior_up = (prev_20['close'].iloc[-1] - prev_20['close'].iloc[0]) / prev_20['close'].iloc[0] if len(prev_20) > 0 else 0
        cond1 = prior_up > 0.12
        
        # 条件2: 大阳线
        
        # 条件2: 大阳线
        cond2 = row['pct_change'] > 5
        
        # 条件3: 巨量
        cond3 = row['vol_ratio'] > 2.2
        
        # 条件4: 上影线明显
        cond4 = row['upper_shadow'] > 0.35
        
        score = (cond1 * 0.20 + cond2 * 0.25 + cond3 * 0.30 + cond4 * 0.25)
        
        return {
            'detected': score >= 0.5,
            'score': score,
            'conditions': {'前期涨幅>12%': cond1, '涨幅>5%': cond2, '巨量>2.2': cond3, '上影线': cond4},
            'price': row['close'],
            'date': row['date']
        }
    
    def detect_ut(self, idx: int) -> dict:
        """
        检测UT(上冲回落) - 收紧参数
        条件:
        ①有效突破20日高点
        ②收盘在突破点下方(上影线明显)
        ③上影线>60%实体(更严格)
        ④明显放量(量比>1.5)
        """
        if idx < 2:
            return {'detected': False, 'score': 0}
        
        df = self.df
        row = df.iloc[idx]
        
        # 条件1: 突破20日高点
        high_20 = df.iloc[idx]['high_20'] if idx < len(df) else row['high']
        cond1 = row['high'] > high_20
        
        # 条件2: 收盘在突破点下方(上影线明显)
        cond2 = row['close'] < high_20
        
        # 条件3: 上影线>60%实体(更严格)
        cond3 = row['upper_shadow'] > 0.6
        
        # 条件4: 明显放量(更严格)
        cond4 = row['vol_ratio'] > 1.5
        
        score = (cond1 * 0.20 + cond2 * 0.25 + cond3 * 0.30 + cond4 * 0.25)
        
        return {
            'detected': score >= 0.55,  # 提高阈值
            'score': score,
            'conditions': {'突破20日高': cond1, '回落': cond2, '上影线>60%': cond3, '放量>1.5': cond4},
            'price': row['close'],
            'date': row['date']
        }
    
    def detect_sow(self, idx: int) -> dict:
        """
        检测SOW(弱势信号) - 收紧参数
        条件:
        ①跌破20日支撑(收盘价)
        ②放量下跌(量比>1.5)
        ③明显跌幅(>2.5%)
        ④前期已上涨一段时间
        """
        if idx < 30:
            return {'detected': False, 'score': 0}
        
        df = self.df
        row = df.iloc[idx]
        prev_20 = df.iloc[max(0, idx-20):idx]
        
        # 条件1: 收盘跌破20日支撑
        cond1 = row['close'] < df.iloc[idx]['low_20'] if idx < len(df) else False
        
        # 条件2: 放量下跌(更严格)
        cond2 = (row['pct_change'] < 0) and (row['vol_ratio'] > 1.5)
        
        # 条件3: 明显跌幅(更宽松)
        cond3 = row['pct_change'] < -2.5
        
        # 条件4: 前期已上涨
        prior_up = (prev_20['close'].iloc[-1] - prev_20['close'].iloc[0]) / prev_20['close'].iloc[0] if len(prev_20) > 0 else 0
        cond4 = prior_up > 0.08
        
        score = (cond1 * 0.30 + cond2 * 0.25 + cond3 * 0.25 + cond4 * 0.20)
        
        return {
            'detected': score >= 0.5,
            'score': score,
            'conditions': {'跌破20日低': cond1, '放量下跌': cond2, '跌幅>2.5%': cond3, '前期涨幅': cond4},
            'price': row['close'],
            'date': row['date']
        }
    
    def detect_lps(self, idx: int) -> dict:
        """
        检测LPS(最后支撑点) - 大幅收紧参数
        条件:
        ①在SOS/BC后5-20天内(更严格窗口)
        ②明显缩量(量比<0.6)
        ③小幅回调(3-8%)
        ④不破SOS启动低点
        """
        if idx < 15:
            return {'detected': False, 'score': 0}
        
        df = self.df
        
        # 查找20天内的SOS或BC
        sos_idx = None
        sos_type = None
        for i in range(max(0, idx-20), idx):
            sos_result = self.detect_sos(i)
            if sos_result['detected']:
                sos_idx = i
                sos_type = 'SOS'
                break
        
        # 如果没有SOS，找BC
        if sos_idx is None:
            for i in range(max(0, idx-20), idx):
                bc_result = self.detect_bc(i)
                if bc_result['detected']:
                    sos_idx = i
                    sos_type = 'BC'
                    break
        
        if sos_idx is None:
            return {'detected': False, 'score': 0, 'reason': '无前置SOS/BC'}
        
        row = df.iloc[idx]
        
        # 条件1: 在SOS/BC后5-20天
        cond1 = 5 <= (idx - sos_idx) <= 20
        
        # 条件2: 明显缩量(更严格)
        cond2 = row['vol_ratio'] < 0.6
        
        # 条件3: 小幅回调3-8%(更严格区间)
        sos_price = df.iloc[sos_idx]['close']
        pullback = (sos_price - row['close']) / sos_price
        cond3 = 0.03 < pullback < 0.08
        
        # 条件4: 不破SOS启动低点
        sos_start_low = df.iloc[sos_idx:idx+1]['low'].min()
        cond4 = row['close'] > sos_start_low
        
        # 增加: 回调时振幅小
        cond5 = row['amplitude'] < 4
        
        score = (cond1 * 0.20 + cond2 * 0.25 + cond3 * 0.25 + cond4 * 0.20 + cond5 * 0.10)
        
        return {
            'detected': score >= 0.55,  # 提高阈值
            'score': score,
            'conditions': {'5-20天窗口': cond1, '缩量<0.6': cond2, '回调3-8%': cond3, '不破低': cond4, '振幅<4%': cond5},
            'sos_date': df.iloc[sos_idx]['date'],
            'sos_type': sos_type,
            'price': row['close'],
            'date': row['date']
        }
    
    # ==================== 威科夫阶段判断 ====================
    
    def detect_phase(self, idx: int) -> str:
        """
        检测威科夫阶段(规则方法)
        阶段A: 下跌趋势终止
        阶段B: 吸筹区间/盘整
        阶段C: 吸筹确认(Spring/震仓)
        阶段D: 上涨趋势(SOS/JOC)
        阶段E: 派发/下跌
        """
        if idx < 60:
            return "未知"
        
        df = self.df
        recent = df.iloc[max(0, idx-60):idx]
        
        # 计算指标
        volatility = recent['close'].std() / recent['close'].mean()
        vol_ratio = recent['volume'].iloc[-20:].mean() / recent['volume'].mean() if recent['volume'].mean() > 0 else 1
        ma20 = recent['ma20'].iloc[-1] if not pd.isna(recent['ma20'].iloc[-1]) else recent['close'].iloc[-1]
        ma60 = recent['ma60'].iloc[-1] if 'ma60' in recent.columns and not pd.isna(recent['ma60'].iloc[-1]) else ma20
        price_trend = (ma20 - ma60) / ma60 * 100 if ma60 > 0 else 0
        recent_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0] * 100
        
        # 检测近期事件
        has_sc = False
        has_st = False
        has_spring = False
        has_sos = False
        has_bc = False
        has_sow = False
        
        for i in range(max(0, idx-30), idx):
            if self.detect_sc(i)['detected']:
                has_sc = True
            if self.detect_st(i)['detected']:
                has_st = True
            if self.detect_spring(i)['detected']:
                has_spring = True
            if self.detect_sos(i)['detected']:
                has_sos = True
            if self.detect_bc(i)['detected']:
                has_bc = True
            if self.detect_sow(i)['detected']:
                has_sow = True
        
        # 阶段判断逻辑
        # 阶段D: 上涨趋势
        if has_sos and volatility >= 0.10:
            return "阶段D-上涨趋势"
        
        # 阶段C: 吸筹确认 (有Spring或SC+ST成功)
        if has_spring or (has_sc and has_st):
            return "阶段C-吸筹确认"
        
        # 阶段B: 吸筹区间 (低波动)
        if volatility < 0.08 and vol_ratio < 0.6:
            return "阶段B-吸筹区间"
        
        # 阶段E: 派发/下跌
        if has_bc or has_sow:
            return "阶段E-派发/下跌"
        
        # 阶段A: 下跌趋势
        if volatility >= 0.12 and price_trend < -3 and recent_change < -5:
            return "阶段A-下跌趋势"
        
        # 阶段B: 盘整
        if volatility < 0.12:
            return "阶段B-盘整"
        
        return "阶段B-盘整"
    
    # ==================== 概率云辅助 ====================
    
    def get_probability_cloud_likelihood(self, idx: int, event_type: str) -> float:
        """
        获取概率云似然度(简化版)
        基于程序化条件的加权得分
        """
        event_detector = {
            'SC': self.detect_sc,
            'AR': self.detect_ar,
            'ST': self.detect_st,
            'SPRING': self.detect_spring,
            'SOS': self.detect_sos,
            'BC': self.detect_bc,
            'UT': self.detect_ut,
            'SOW': self.detect_sow,
            'LPS': self.detect_lps
        }
        
        if event_type not in event_detector:
            return 0
        
        result = event_detector[event_type](idx)
        return result.get('score', 0)
    
    def get_confidence_level(self, likelihood: float) -> str:
        """根据似然度返回置信等级"""
        if likelihood > 0.8:
            return "高置信"
        elif likelihood > 0.6:
            return "中置信"
        elif likelihood > 0.4:
            return "低置信"
        else:
            return "观察"
    
    # ==================== 扫描所有事件 ====================
    
    def scan_all_events(self, start_idx: int = 60, end_idx: int = None) -> list:
        """扫描期间内所有检测到的事件 - 增加间隔避免重复检测"""
        if end_idx is None:
            end_idx = len(self.df)
        
        events = []
        event_types = ['SC', 'AR', 'ST', 'SPRING', 'SOS', 'BC', 'UT', 'SOW', 'LPS']
        detectors = {
            'SC': self.detect_sc,
            'AR': self.detect_ar,
            'ST': self.detect_st,
            'SPRING': self.detect_spring,
            'SOS': self.detect_sos,
            'BC': self.detect_bc,
            'UT': self.detect_ut,
            'SOW': self.detect_sow,
            'LPS': self.detect_lps        # 记录
        }
        
        # 记录上次检测到各事件的位置，避免重复检测
        last_detected = {et: -10 for et in event_types}
        
        for idx in range(start_idx, end_idx):
            for event_type in event_types:
                # 间隔检测，减少重复
                if idx - last_detected[event_type] < 5:
                    continue
                    
                result = detectors[event_type](idx)
                if result['detected']:
                    last_detected[event_type] = idx
                    events.append({
                        'type': event_type,
                        'date': self.df.iloc[idx]['date'],
                        'price': self.df.iloc[idx]['close'],
                        'score': result['score'],
                        'confidence': self.get_confidence_level(result['score']),
                        'idx': idx
                    })
        
        return events


def load_data(filepath: str) -> pd.DataFrame:
    """加载数据"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def run_backtest(df: pd.DataFrame, symbol: str, period_days: int = 730) -> dict:
    """
    运行回测
    将数据分成多个2年周期进行分析
    """
    if len(df) < 120:
        return {'error': '数据不足'}
    
    # 计算可以分割的周期数
    results = {
        'symbol': symbol,
        'periods': [],
        'summary': {
            'total_events': 0,
            'phase_distribution': defaultdict(int),
            'event_distribution': defaultdict(int)
        }
    }
    
    # 2年周期分割
    num_periods = len(df) // (period_days // 2)  # 步长为1年
    
    for i in range(num_periods):
        start_idx = i * (period_days // 2)
        end_idx = min(start_idx + period_days, len(df))
        
        if end_idx - start_idx < 180:  # 至少半年数据
            continue
        
        period_df = df.iloc[start_idx:end_idx].copy()
        period_df = period_df.reset_index(drop=True)
        
        # 创建分析器
        analyzer = WyckoffBacktest(period_df)
        
        # 扫描事件
        events = analyzer.scan_all_events(start_idx=20)
        
        # 分析阶段分布
        phases = []
        for idx in range(60, len(period_df), 30):  # 每30天分析一次阶段
            phase = analyzer.detect_phase(idx)
            phases.append(phase)
        
        # 统计
        phase_counts = defaultdict(int)
        for p in phases:
            phase_counts[p] += 1
        
        event_counts = defaultdict(int)
        for e in events:
            event_counts[e['type']] += 1
        
        period_result = {
            'period_index': i + 1,
            'start_date': period_df['date'].iloc[0].strftime('%Y-%m-%d'),
            'end_date': period_df['date'].iloc[-1].strftime('%Y-%m-%d'),
            'days': len(period_df),
            'events': events,
            'event_counts': dict(event_counts),
            'phases': phases,
            'phase_counts': dict(phase_counts)
        }
        
        results['periods'].append(period_result)
        results['summary']['total_events'] += len(events)
        
        for k, v in phase_counts.items():
            results['summary']['phase_distribution'][k] += v
        for k, v in event_counts.items():
            results['summary']['event_distribution'][k] += v
    
    return results


def print_results(results: dict):
    """打印回测结果"""
    print("\n" + "="*60)
    print(f"回测结果: {results['symbol']}")
    print("="*60)
    
    for period in results['periods']:
        print(f"\n【周期 {period['period_index']}】{period['start_date']} ~ {period['end_date']} ({period['days']}天)")
        print("-" * 40)
        
        print("阶段分布:")
        for phase, count in sorted(period['phase_counts'].items(), key=lambda x: -x[1]):
            print(f"  {phase}: {count}次")
        
        print("事件检测:")
        for event, count in sorted(period['event_counts'].items(), key=lambda x: -x[1]):
            print(f"  {event}: {count}次")
    
    print("\n" + "="*60)
    print("【汇总统计】")
    print("-" * 40)
    print(f"总事件数: {results['summary']['total_events']}")
    print("\n阶段分布(总计):")
    for phase, count in sorted(results['summary']['phase_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {phase}: {count}次")
    print("\n事件分布(总计):")
    for event, count in sorted(results['summary']['event_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {event}: {count}次")


def manual_validation(df: pd.DataFrame, analyzer: WyckoffBacktest, event_type: str, idx: int) -> dict:
    """
    人工验证接口
    返回事件的详细信息供人工确认
    """
    detectors = {
        'SC': analyzer.detect_sc,
        'AR': analyzer.detect_ar,
        'ST': analyzer.detect_st,
        'SPRING': analyzer.detect_spring,
        'SOS': analyzer.detect_sos,
        'BC': analyzer.detect_bc,
        'UT': analyzer.detect_ut,
        'SOW': analyzer.detect_sow,
        'LPS': analyzer.detect_lps
    }
    
    if event_type not in detectors:
        return {'error': '未知事件类型'}
    
    result = detectors[event_type](idx)
    
    return {
        'date': df.iloc[idx]['date'],
        'price': df.iloc[idx]['close'],
        'open': df.iloc[idx]['open'],
        'high': df.iloc[idx]['high'],
        'low': df.iloc[idx]['low'],
        'volume': df.iloc[idx]['volume'],
        'pct_change': df.iloc[idx]['pct_change'],
        'amplitude': df.iloc[idx]['amplitude'],
        'vol_ratio': df.iloc[idx]['vol_ratio'],
        'detected': result['detected'],
        'score': result['score'],
        'conditions': result.get('conditions', {}),
        'confidence': analyzer.get_confidence_level(result.get('score', 0))
    }


def validate_accuracy(df: pd.DataFrame, analyzer: WyckoffBacktest, sample_size: int = 20) -> dict:
    """
    验证识别准确率
    随机抽样事件，人工/规则验证
    """
    events = analyzer.scan_all_events()
    
    if len(events) == 0:
        return {'error': '无事件可验证'}
    
    # 随机抽样
    np.random.seed(42)
    if len(events) > sample_size:
        samples = np.random.choice(len(events), sample_size, replace=False)
    else:
        samples = range(len(events))
    
    correct = 0
    total = 0
    
    for i in samples:
        event = events[i]
        idx = event['idx']
        
        # 规则验证: 检查事件是否满足基本条件
        event_type = event['type']
        detectors = {
            'SC': analyzer.detect_sc,
            'AR': analyzer.detect_ar,
            'ST': analyzer.detect_st,
            'SPRING': analyzer.detect_spring,
            'SOS': analyzer.detect_sos,
            'BC': analyzer.detect_bc,
            'UT': analyzer.detect_ut,
            'SOW': analyzer.detect_sow,
            'LPS': analyzer.detect_lps
        }
        
        result = detectors[event_type](idx)
        
        # 如果检测分数>0.5，认为识别正确
        if result['score'] >= 0.5:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'total_events': len(events),
        'sampled': total,
        'correct': correct,
        'accuracy': accuracy,
        'accuracy_pct': f"{accuracy*100:.1f}%"
    }


def backtest_spring_success_rate(df: pd.DataFrame, analyzer: WyckoffBacktest, hold_days: list = [1, 3, 5], require_bull_market: bool = True) -> dict:
    """
    回测Spring检测后的成功率
    验证Spring检测后价格是否上涨
    
    参数:
        df: 数据
        analyzer: WyckoffBacktest实例
        hold_days: 持有天数列表
        require_bull_market: 是否要求牛市背景
    
    返回:
        回测结果统计
    """
    results = {
        'total_springs': 0,
        'by_type': {},
        'by_position': {},
        'hold_days': {}
    }
    
    # 初始化持有天数统计
    for days in hold_days:
        results['hold_days'][days] = {
            'count': 0,
            'success': 0,
            'avg_return': 0,
            'returns': []
        }
    
    # 遍历所有日期检测Spring
    for idx in range(60, len(df) - max(hold_days)):
        result = analyzer.detect_spring(idx, require_bull_market=require_bull_market)
        
        if not result['detected']:
            continue
        
        spring_type = result.get('spring_type', 'SPRING')
        spring_position = result.get('position', 'ORDINARY')
        
        # 初始化类型统计
        if spring_type not in results['by_type']:
            results['by_type'][spring_type] = {
                'total': 0,
                'success_by_days': {days: {'count': 0, 'success': 0} for days in hold_days}
            }
        
        # 初始化位置统计
        if spring_position not in results['by_position']:
            results['by_position'][spring_position] = {
                'total': 0,
                'success_by_days': {days: {'count': 0, 'success': 0} for days in hold_days}
            }
        
        results['by_type'][spring_type]['total'] += 1
        results['by_position'][spring_position]['total'] += 1
        results['total_springs'] += 1
        
        entry_price = df.iloc[idx]['close']
        
        # 测试不同持有天数
        for days in hold_days:
            if idx + days < len(df):
                exit_price = df.iloc[idx + days]['close']
                return_pct = (exit_price - entry_price) / entry_price * 100
                is_success = return_pct > 0  # 上涨算成功
                
                results['hold_days'][days]['count'] += 1
                results['hold_days'][days]['success'] += int(is_success)
                results['hold_days'][days]['returns'].append(return_pct)
                
                results['by_type'][spring_type]['success_by_days'][days]['count'] += 1
                results['by_type'][spring_type]['success_by_days'][days]['success'] += int(is_success)
                
                results['by_position'][spring_position]['success_by_days'][days]['count'] += 1
                results['by_position'][spring_position]['success_by_days'][days]['success'] += int(is_success)
    
    # 计算成功率
    for days in hold_days:
        if results['hold_days'][days]['count'] > 0:
            results['hold_days'][days]['success_rate'] = (
                results['hold_days'][days]['success'] / results['hold_days'][days]['count'] * 100
            )
            results['hold_days'][days]['avg_return'] = np.mean(
                results['hold_days'][days]['returns']
            ) if results['hold_days'][days]['returns'] else 0
    
    for spring_type in results['by_type']:
        for days in hold_days:
            cnt = results['by_type'][spring_type]['success_by_days'][days]['count']
            suc = results['by_type'][spring_type]['success_by_days'][days]['success']
            if cnt > 0:
                results['by_type'][spring_type]['success_by_days'][days]['success_rate'] = suc / cnt * 100
    
    for position in results['by_position']:
        for days in hold_days:
            cnt = results['by_position'][position]['success_by_days'][days]['count']
            suc = results['by_position'][position]['success_by_days'][days]['success']
            if cnt > 0:
                results['by_position'][position]['success_by_days'][days]['success_rate'] = suc / cnt * 100
    
    return results


def print_spring_backtest_results(results: dict):
    """打印Spring回测结果"""
    print("\n" + "="*60)
    print("Spring回测结果统计")
    print("="*60)
    
    print(f"\n总Spring数量: {results['total_springs']}")
    
    # 按类型统计
    print("\n--- 按Spring类型统计 ---")
    for spring_type, data in results['by_type'].items():
        print(f"\n{spring_type}: 共{data['total']}个")
        for days, stats in data['success_by_days'].items():
            if stats['count'] > 0:
                print(f"  持有{days}天: 成功率{stats['success_rate']:.1f}% ({stats['success']}/{stats['count']})")
    
    # 按位置统计
    if 'by_position' in results and results['by_position']:
        print("\n--- 按位置统计 ---")
        for position, data in results['by_position'].items():
            print(f"\n{position}: 共{data['total']}个")
            for days, stats in data['success_by_days'].items():
                if stats['count'] > 0:
                    print(f"  持有{days}天: 成功率{stats['success_rate']:.1f}% ({stats['success']}/{stats['count']})")
    
    # 总体统计
    print("\n--- 总体统计 ---")
    for days, stats in results['hold_days'].items():
        if stats['count'] > 0:
            print(f"持有{stats['count']}天: 成功率{stats['success_rate']:.1f}% ({stats['success']}/{stats['count']}), 平均收益{stats['avg_return']:.2f}%")
    
    return results


if __name__ == '__main__':
    import sys
    
    # 回测ETF 515100
    print("加载515100数据...")
    df_515100 = load_data('etf_515100_data.csv')
    print(f"515100数据: {len(df_515100)}条")
    
    results_515100 = run_backtest(df_515100, '515100')
    print_results(results_515100)
    
    # 验证准确率
    analyzer_515100 = WyckoffBacktest(df_515100)
    accuracy_515100 = validate_accuracy(df_515100, analyzer_515100)
    print(f"\n准确率验证: {accuracy_515100}")
    
    print("\n" + "="*60)
    
    # 回测601600
    print("\n加载601600数据...")
    df_601600 = load_data('stock_601600_data.csv')
    print(f"601600数据: {len(df_601600)}条")
    
    results_601600 = run_backtest(df_601600, '601600')
    print_results(results_601600)
    
    # 验证准确率
    analyzer_601600 = WyckoffBacktest(df_601600)
    accuracy_601600 = validate_accuracy(df_601600, analyzer_601600)
    print(f"\n准确率验证: {accuracy_601600}")
    
    # Spring回测 - 三轮回测
    print("\n" + "="*60)
    print("第一轮: 回测ETF 515100 (有市场背景过滤)")
    print("="*60)
    spring_results_1 = backtest_spring_success_rate(df_515100, WyckoffBacktest(df_515100), [1, 3, 5], require_bull_market=True)
    print_spring_backtest_results(spring_results_1)
    
    print("\n" + "="*60)
    print("第一轮对比: 515100 (无市场背景过滤)")
    print("="*60)
    spring_results_1_no_filter = backtest_spring_success_rate(df_515100, WyckoffBacktest(df_515100), [1, 3, 5], require_bull_market=False)
    print_spring_backtest_results(spring_results_1_no_filter)
    
    print("\n" + "="*60)
    print("第二轮: 回测601600 (有市场背景过滤)")
    print("="*60)
    spring_results_2 = backtest_spring_success_rate(df_601600, WyckoffBacktest(df_601600), [1, 3, 5], require_bull_market=True)
    print_spring_backtest_results(spring_results_2)
    
    print("\n" + "="*60)
    print("第二轮对比: 601600 (无市场背景过滤)")
    print("="*60)
    spring_results_2_no_filter = backtest_spring_success_rate(df_601600, WyckoffBacktest(df_601600), [1, 3, 5], require_bull_market=False)
    print_spring_backtest_results(spring_results_2_no_filter)
    
    print("\n" + "="*60)
    print("第三轮: 合并回测(515100 + 601600) - 有市场背景过滤")
    print("="*60)
    df_combined = pd.concat([df_515100, df_601600], ignore_index=True)
    spring_results_3 = backtest_spring_success_rate(df_combined, WyckoffBacktest(df_combined), [1, 3, 5], require_bull_market=True)
    print_spring_backtest_results(spring_results_3)
    
    print("\n" + "="*60)
    print("第三轮对比: 合并数据 (无市场背景过滤)")
    print("="*60)
    spring_results_3_no_filter = backtest_spring_success_rate(df_combined, WyckoffBacktest(df_combined), [1, 3, 5], require_bull_market=False)
    print_spring_backtest_results(spring_results_3_no_filter)
