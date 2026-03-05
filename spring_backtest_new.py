#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
威科夫Spring回测 - 使用Backtrader
对比三种支撑位：
1. 需求区支撑 (Demand Zone)
2. 布林下轨支撑 (Bollinger Lower Band)
3. SASE动态支撑 (SASE Dynamic Support)
"""

import backtrader as bt
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class SpringSignalStrategy(bt.Strategy):
    """基于预计算Spring信号的策略"""
    
    params = (
        ('profit_target', 0.15),  # 止盈15%
        ('stop_loss', 0.05),      # 止损5%
        ('max_hold_days', 20),    # 最大持仓天数
        ('buy_amount', 100000),   # 每次买入金额10万元
    )
    
    def __init__(self):
        self.order = None
        self.buy_price = None
        self.buy_date = None
        self.spring_signals = None
        self.trades = []
        self.spring_count = 0
        self.buy_idx = None
        self.in_position = False
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.buy_date = self.data.datetime.date(0)
                self.buy_idx = len(self.data) - 1
                self.in_position = True
            else:
                exit_price = order.executed.price
                exit_date = self.data.datetime.date(0)
                ret = (exit_price - self.buy_price) / self.buy_price * 100
                current_idx = len(self.data) - 1
                
                if ret >= self.params.profit_target:
                    trade_type = 'profit'
                elif ret <= -self.params.stop_loss:
                    trade_type = 'loss'
                else:
                    trade_type = 'time_exit'
                
                self.trades.append({
                    'entry_date': self.buy_date,
                    'entry_price': self.buy_price,
                    'exit_date': exit_date,
                    'exit_price': exit_price,
                    'return': ret,
                    'type': trade_type,
                    'hold_days': current_idx - self.buy_idx
                })
                
                self.in_position = False
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.debug:
                print(f"  订单失败: status={order.status}")
            self.order = None
    
    def next(self):
        if self.order:
            return
        
        current_idx = len(self.data) - 1
        
        # 如果没有持仓，检查买入信号
        if not self.in_position:
            # 检查今天是否是Spring信号日
            if self.spring_signals and current_idx in self.spring_signals:
                self.spring_count += 1
                # 固定金额买入
                price = self.data.close[0]
                size = int(self.params.buy_amount / price)
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            # 检查出场信号
            self.check_exit()
    
    def check_exit(self):
        """检查出场信号"""
        current_price = self.data.close[0]
        current_idx = len(self.data) - 1
        
        # 止盈
        if current_price >= self.buy_price * (1 + self.params.profit_target):
            self.order = self.close()  # Close entire position
            return
        
        # 止损
        if current_price <= self.buy_price * (1 - self.params.stop_loss):
            self.order = self.close()  # Close entire position
            return
        
        # 时间止损 - 超过最大持仓天数
        if current_idx - self.buy_idx >= self.params.max_hold_days:
            self.order = self.close()  # Close entire position
            return
        
        # 止损
        if current_price <= self.buy_price * (1 - self.params.stop_loss):
            self.order = self.sell()
            self.trades.append({
                'entry_date': self.buy_date,
                'entry_price': self.buy_price,
                'exit_date': self.data.datetime.date(0),
                'exit_price': current_price,
                'return': (current_price - self.buy_price) / self.buy_price * 100,
                'type': 'loss',
                'hold_days': current_idx - self.buy_idx
            })
            return
        
        # 时间止损 - 超过最大持仓天数
        if current_idx - self.buy_idx >= self.params.max_hold_days:
            self.order = self.sell()
            ret = (current_price - self.buy_price) / self.buy_price * 100
            self.trades.append({
                'entry_date': self.buy_date,
                'entry_price': self.buy_price,
                'exit_date': self.data.datetime.date(0),
                'exit_price': current_price,
                'return': ret,
                'type': 'time_exit',
                'hold_days': current_idx - self.buy_idx
            })
    
    def stop(self):
        if self.in_position and self.buy_price is not None:
            current_price = self.data.close[0]
            current_idx = len(self.data) - 1
            ret = (current_price - self.buy_price) / self.buy_price * 100
            self.trades.append({
                'entry_date': self.buy_date,
                'entry_price': self.buy_price,
                'exit_date': self.data.datetime.date(0),
                'exit_price': current_price,
                'return': ret,
                'type': 'final_close',
                'hold_days': current_idx - self.buy_idx
            })


def calculate_spring_signals(df, method='demand_zone'):
    """
    预计算Spring信号
    method: 'demand_zone', 'bollinger', 'sase'
    返回: set of indices where Spring is detected
    """
    spring_indices = set()
    
    if len(df) < 50:
        return spring_indices
    
    # 计算基础指标
    df = df.copy()
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / (df['range'] + 0.001)
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma20_up'] = df['ma20'] > df['ma20'].shift(5)
    
    # 布林带
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    
    # ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['atr_100'] = df['tr'].rolling(100).mean()
    df['atr_ratio'] = df['atr_14'] / (df['atr_100'] + 1e-9)
    
    # SASE支撑
    tolerance = 2.0 * (1.0 + (1 - df['atr_ratio'].clip(0, 2)) * 0.5)
    df['sase_support'] = df['bb_mid'] - tolerance * df['bb_std']
    
    # 需求区检测
    demand_zones = []
    for i in range(30, len(df) - 5):
        if not df.iloc[i]['ma20_up']:
            continue
        
        # 寻找静态区间
        static_start = None
        for j in range(i - 1, max(i - 10, 10), -1):
            if df.iloc[j]['body_ratio'] < 0.5:
                if static_start is None:
                    static_start = j
            else:
                break
        
        if static_start is None:
            continue
        
        breakout = df.iloc[i]
        if breakout['body_ratio'] < 0.5 or breakout['close'] <= breakout['open']:
            continue
        
        demand_top = breakout['close']
        
        demand_zones.append({
            'idx': i,
            'demand_top': demand_top
        })
    
    min_break_depth = 0.01
    
    for i in range(50, len(df) - 10):
        # 获取支撑
        if method == 'demand_zone':
            # 找需求区支撑
            current_price = df.iloc[i]['close']
            valid_zones = []
            for zone in demand_zones:
                if zone['idx'] >= i:
                    continue
                if zone['demand_top'] < current_price:
                    distance = (current_price - zone['demand_top']) / current_price
                    days_since = i - zone['idx']
                    if days_since <= 20:
                        valid_zones.append((zone, distance))
            
            if not valid_zones:
                continue
            
            valid_zones.sort(key=lambda x: x[1])
            support = valid_zones[0][0]['demand_top']
            
        elif method == 'bollinger':
            support = df.iloc[i]['bb_lower']
            if pd.isna(support):
                continue
                
        elif method == 'sase':
            support = df.iloc[i]['sase_support']
            if pd.isna(support):
                continue
        else:
            continue
        
        break_price = df.iloc[i]['low']
        
        # 检查跌破支撑
        break_depth = (support - break_price) / support
        if break_depth < min_break_depth:
            continue
        
        # 检查5天内是否反弹
        for j in range(1, 6):
            if i + j >= len(df):
                break
            if df.iloc[i + j]['close'] > support * 1.01:
                spring_indices.add(i)
                break
    
    return spring_indices


def calculate_zigzag(close, threshold=0.05):
    """
    计算Zigzag拐点
    threshold: 波动阈值（默认5%）
    返回: 支撑位列表
    """
    n = len(close)
    pivots = [0]
    direction = 0
    
    for i in range(1, n):
        if direction == 0:
            if close[i] >= close[0] * (1 + threshold):
                direction = 1
                pivots.append(i)
            elif close[i] <= close[0] * (1 - threshold):
                direction = -1
                pivots.append(i)
        elif direction == 1:
            if close[i] > close[pivots[-1]]:
                pivots[-1] = i
            elif close[i] <= close[pivots[-1]] * (1 - threshold):
                direction = -1
                pivots.append(i)
        elif direction == -1:
            if close[i] < close[pivots[-1]]:
                pivots[-1] = i
            elif close[i] >= close[pivots[-1]] * (1 + threshold):
                direction = 1
                pivots.append(i)
    
    # 提取支撑位（波谷）
    supports = []
    for i in range(1, len(pivots) - 1):
        idx = pivots[i]
        if (close[pivots[i-1]] > close[idx]) and (close[pivots[i+1]] > close[idx]):
            supports.append({'idx': idx, 'price': close[idx]})
    
    return supports


def calculate_spring_signals_zigzag(df, threshold=0.05):
    """
    基于Zigzag支撑计算Spring信号（无前视偏差）
    """
    spring_indices = set()
    
    if len(df) < 50:
        return spring_indices
    
    close = df['close'].values
    min_break_depth = 0.01
    
    for i in range(50, len(df) - 10):
        # 只使用i之前的数据计算Zigzag（避免前视偏差）
        historical_close = close[:i]
        
        if len(historical_close) < 30:
            continue
        
        # 计算历史数据的Zigzag支撑位
        zigzag_supports = calculate_zigzag(historical_close, threshold=threshold)
        
        if not zigzag_supports:
            continue
        
        # 获取最近的支撑位（至少1天前形成）
        support_price = None
        for s in zigzag_supports:
            if s['idx'] < i - 1:
                support_price = s['price']
        
        if support_price is None:
            continue
        
        # 检查是否跌破支撑
        break_price = df.iloc[i]['low']
        break_depth = (support_price - break_price) / support_price
        
        if break_depth < min_break_depth:
            continue
        
        # 检查5天内是否反弹
        for j in range(1, 6):
            if i + j >= len(df):
                break
            if df.iloc[i + j]['close'] > support_price * 1.01:
                spring_indices.add(i)
                break
    
    return spring_indices


def run_backtest(symbol, start_date='20180101'):
    """运行回测"""
    print(f"\n{'='*60}")
    print(f"回测 {symbol}")
    print(f"{'='*60}")
    
    # 获取数据
    df = fetch_stock_data(symbol, start_date)
    if df is None or len(df) < 100:
        print(f"  数据不足，跳过")
        return None
    
    date_min = df.index.min().strftime('%Y-%m-%d')
    date_max = df.index.max().strftime('%Y-%m-%d')
    print(f"  数据量: {len(df)}条 ({date_min} ~ {date_max})")
    
    # 计算Spring信号
    print(f"\n  计算Spring信号...")
    
    print(f"    - 布林下轨支撑...")
    bollinger_signals = calculate_spring_signals(df, 'bollinger')
    print(f"      布林信号数: {len(bollinger_signals)}")
    
    print(f"    - Zigzag支撑...")
    zigzag_signals = calculate_spring_signals_zigzag(df, threshold=0.05)
    print(f"      Zigzag信号数: {len(zigzag_signals)}")
    
    results = {}
    initial_cash = 10000000  # 1000万初始资金
    
    # 定义策略类
    class BollingerStrategy(SpringSignalStrategy):
        def __init__(self):
            super().__init__()
            self.spring_signals = bollinger_signals.copy()
    
    class ZigzagStrategy(SpringSignalStrategy):
        def __init__(self):
            super().__init__()
            self.spring_signals = zigzag_signals.copy()
    
    # 1. 布林下轨支撑回测
    print(f"\n  [1] 布林下轨支撑策略:")
    cerebro1 = bt.Cerebro()
    cerebro1.addstrategy(BollingerStrategy)
    data1 = bt.feeds.PandasData(dataname=df, datetime=None, open='open', high='high', low='low', close='close', volume='volume', openinterest=-1)
    cerebro1.adddata(data1)
    cerebro1.broker.setcash(initial_cash)
    cerebro1.broker.setcommission(commission=0.001)
    
    strat1 = cerebro1.run()
    final_value1 = cerebro1.broker.getvalue()
    returns1 = (final_value1 - initial_cash) / initial_cash * 100
    
    trades1 = strat1[0].trades
    win_count = len([t for t in trades1 if t['return'] > 0])
    loss_count = len([t for t in trades1 if t['return'] <= 0])
    print(f"    交易次数: {len(trades1)} (盈利:{win_count}, 亏损:{loss_count})")
    if trades1:
        avg_return = sum([t['return'] for t in trades1]) / len(trades1)
        print(f"    平均收益率: {avg_return:.4f}%")
    print(f"    初始资金: {initial_cash:,} | 最终资金: {final_value1:,.2f} | 收益率: {returns1:.4f}%")
    
    results['bollinger'] = {
        'final_value': final_value1,
        'return': returns1,
        'trades': trades1,
        'signal_count': len(bollinger_signals)
    }
    
    # 2. Zigzag支撑回测
    print(f"\n  [2] Zigzag支撑策略:")
    cerebro2 = bt.Cerebro()
    cerebro2.addstrategy(ZigzagStrategy)
    data2 = bt.feeds.PandasData(dataname=df, datetime=None, open='open', high='high', low='low', close='close', volume='volume', openinterest=-1)
    cerebro2.adddata(data2)
    cerebro2.broker.setcash(initial_cash)
    cerebro2.broker.setcommission(commission=0.001)
    
    strat2 = cerebro2.run()
    final_value2 = cerebro2.broker.getvalue()
    returns2 = (final_value2 - initial_cash) / initial_cash * 100
    
    trades2 = strat2[0].trades
    win_count = len([t for t in trades2 if t['return'] > 0])
    loss_count = len([t for t in trades2 if t['return'] <= 0])
    print(f"    交易次数: {len(trades2)} (盈利:{win_count}, 亏损:{loss_count})")
    if trades2:
        avg_return = sum([t['return'] for t in trades2]) / len(trades2)
        print(f"    平均收益率: {avg_return:.4f}%")
    print(f"    初始资金: {initial_cash:,} | 最终资金: {final_value2:,.2f} | 收益率: {returns2:.4f}%")
    
    results['zigzag'] = {
        'final_value': final_value2,
        'return': returns2,
        'trades': trades2,
        'signal_count': len(zigzag_signals)
    }
    
    return results


def fetch_stock_data(symbol, start_date='20180101'):
    """获取股票数据"""
    try:
        # 判断是否为ETF
        is_etf = symbol.startswith('5') or symbol.startswith('15') or \
                 symbol.startswith('16') or symbol.startswith('159') or \
                 symbol.startswith('51') or symbol.startswith('58')
        
        if is_etf:
            print(f"  {symbol} 检测为ETF，使用ETF接口...")
            try:
                etf_prefix = 'sz' if symbol.startswith('15') or symbol.startswith('16') or symbol.startswith('159') else 'sh'
                df = ak.fund_etf_hist_sina(symbol=f"{etf_prefix}{symbol}")
                if df is not None and not df.empty:
                    # ETF有7列，需要处理
                    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)
                    df = df.set_index('date')
                    return df
            except Exception as e:
                print(f"  ETF接口失败: {e}")
                return None
            return None
        
        # 股票
        symbol_prefix = f"sh{symbol}" if symbol.startswith('6') else f"sz{symbol}"
        df = ak.stock_zh_a_daily(
            symbol=symbol_prefix,
            start_date=start_date,
            end_date=datetime.now().strftime('%Y%m%d')
        )
        
        if df is None or df.empty:
            return None
        
        # 列名已经是英文
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.set_index('date')
        
        return df
    
    except Exception as e:
        print(f"  获取{symbol}数据失败: {e}")
        return None


def print_summary(all_results):
    """打印汇总结果"""
    print(f"\n{'='*60}")
    print("回测结果汇总")
    print(f"{'='*60}")
    
    for symbol, results in all_results.items():
        if results is None:
            continue
        
        print(f"\n【{symbol}】")
        print(f"  布林下轨: 信号数={results['bollinger']['signal_count']}, 交易数={len(results['bollinger']['trades'])}, 收益率={results['bollinger']['return']:.4f}%, 最终资金={results['bollinger']['final_value']:,.2f}")
        print(f"  Zigzag支撑: 信号数={results['zigzag']['signal_count']}, 交易数={len(results['zigzag']['trades'])}, 收益率={results['zigzag']['return']:.4f}%, 最终资金={results['zigzag']['final_value']:,.2f}")
        
        # 最佳策略
        best = max([
            ('布林下轨', results['bollinger']['return']),
            ('Zigzag支撑', results['zigzag']['return'])
        ], key=lambda x: x[1])
        print(f"  → 最佳策略: {best[0]} ({best[1]:.2f}%)")


def main():
    """主函数"""
    stocks = ['600108', '515100']
    
    # 获取更早的数据（5年）
    start_date = '20200101'
    
    all_results = {}
    
    for symbol in stocks:
        results = run_backtest(symbol, start_date)
        if results:
            all_results[symbol] = results
    
    # 打印汇总
    print_summary(all_results)


if __name__ == "__main__":
    main()
