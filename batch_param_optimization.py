#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量参数优化 - 100个股票
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import os
import random
import warnings
warnings.filterwarnings('ignore')


def calculate_zigzag_high_low(high, low, threshold=0.02, depth=14, backstep=3):
    n = len(high)
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
            supports.append({'idx': idx, 'price': low[idx]})
    
    return supports


def find_spring_signals(df, threshold=0.02, depth=14, backstep=3):
    spring_signals = {}
    
    if df is None or len(df) < 100:
        return spring_signals
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    for i in range(50, len(df)):
        historical_high = high[:i]
        historical_low = low[:i]
        
        if len(historical_high) < 30:
            continue
        
        zigzag_supports = calculate_zigzag_high_low(
            historical_high, historical_low,
            threshold=threshold, depth=depth, backstep=backstep
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
                spring_signals[i + j] = {
                    'support_price': support_price,
                    'break_date': df.index[i],
                    'spring_date': df.index[i + j],
                    'buy_date': df.index[i + j],
                    'break_price': break_price
                }
                break
    
    return spring_signals


class SpringSignalStrategy(bt.Strategy):
    params = (
        ('spring_signals', None),
        ('profit_target', 0.15),
        ('stop_loss', 0.05),
        ('max_hold_days', 20),
        ('buy_amount', 100000),
    )
    
    def __init__(self):
        self.order = None
        self.buy_price = None
        self.buy_date = None
        self.spring_signals = self.params.spring_signals
        self.trades = []
        self.spring_count = 0
        self.buy_idx = None
        self.in_position = False
        self.current_support_price = None
        self.hold_days = 0
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.buy_date = self.data.datetime.date(0)
                self.buy_idx = len(self.data) - 1
                self.hold_days = 0
                self.in_position = True
            else:
                exit_price = order.executed.price
                exit_date = self.data.datetime.date(0)
                ret = (exit_price - self.buy_price) / self.buy_price * 100
                current_idx = len(self.data) - 1
                
                if ret >= self.params.profit_target:
                    trade_type = '止盈'
                elif ret <= -self.params.stop_loss:
                    trade_type = '止损'
                else:
                    trade_type = '到期卖出'
                
                self.trades.append({
                    '买入日期': str(self.buy_date),
                    '买入价格': self.buy_price,
                    '卖出日期': str(exit_date),
                    '卖出价格': exit_price,
                    '收益率%': round(ret, 2),
                    '持仓天数': current_idx - self.buy_idx,
                    '交易类型': trade_type,
                    '支撑位': self.current_support_price
                })
                
                self.in_position = False
            self.order = None
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.order = None
    
    def next(self):
        if self.order:
            return
        
        current_idx = len(self.data) - 1
        
        if not self.in_position:
            if self.spring_signals and current_idx in self.spring_signals:
                self.spring_count += 1
                signal_info = self.spring_signals[current_idx]
                self.current_support_price = signal_info['support_price']
                
                price = self.data.close[0]
                size = int(self.params.buy_amount / price)
                if size > 0:
                    self.order = self.buy(size=size)
        else:
            self.hold_days += 1
            current_price = self.data.close[0]
            
            if current_price >= self.buy_price * (1 + self.params.profit_target):
                self.order = self.close()
                return
            
            if current_price <= self.buy_price * (1 - self.params.stop_loss):
                self.order = self.close()
                return
            
            if self.hold_days >= self.params.max_hold_days:
                self.order = self.close()


def backtest_stock(filepath, threshold, depth, backstep):
    """对单个股票进行回测"""
    try:
        df = pd.read_csv(filepath)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)
        df = df.set_index('日期')
        
        # 重命名列为英文
        if '最高' in df.columns:
            df = df.rename(columns={
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '开盘': 'open',
                '成交量': 'volume'
            })
        
        cutoff_date = pd.to_datetime('2021-01-01')
        df = df[df.index >= cutoff_date]
        
        if len(df) < 100:
            return None
        
        spring_signals = find_spring_signals(df, threshold=threshold, depth=depth, backstep=backstep)
        
        if len(spring_signals) == 0:
            return None
        
        cerebro = bt.Cerebro()
        cerebro.addstrategy(
            SpringSignalStrategy,
            spring_signals=spring_signals,
            profit_target=0.15,
            stop_loss=0.05,
            max_hold_days=20,
            buy_amount=100000
        )
        
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)
        
        cerebro.broker.setcash(1000000.0)
        cerebro.broker.setcommission(commission=0.001)
        
        results = cerebro.run()
        strategy = results[0]
        
        if not strategy.trades:
            return None
        
        trades_df = pd.DataFrame(strategy.trades)
        wins = len(trades_df[trades_df['收益率%'] > 0])
        losses = len(trades_df[trades_df['收益率%'] <= 0])
        win_rate = wins / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_return = trades_df['收益率%'].mean()
        
        return {
            'trades': len(strategy.trades),
            'win_rate': win_rate,
            'avg_return': avg_return
        }
    except Exception as e:
        return None


def main():
    # 获取50个随机股票（减少数量加快速度）
    stock_dir = 'data/stocks'
    files = [f for f in os.listdir(stock_dir) if f.endswith('.csv')]
    selected = random.sample(files, 50)
    
    print(f"选取了 {len(selected)} 个股票进行参数优化")
    
    # 简化为9个参数组合
    thresholds = [0.02, 0.03, 0.05]
    depths = [10, 14]
    backsteps = [1, 3]
    
    results = []
    
    for threshold in thresholds:
        for depth in depths:
            for backstep in backsteps:
                print(f"\n测试: threshold={threshold}, depth={depth}, backstep={backstep}")
                
                total_trades = 0
                total_win_rate = 0
                total_avg_return = 0
                valid_stocks = 0
                
                for i, filename in enumerate(selected):
                    filepath = os.path.join(stock_dir, filename)
                    result = backtest_stock(filepath, threshold, depth, backstep)
                    
                    if result:
                        total_trades += result['trades']
                        total_win_rate += result['win_rate']
                        total_avg_return += result['avg_return']
                        valid_stocks += 1
                
                if valid_stocks > 0:
                    avg_win_rate = total_win_rate / valid_stocks
                    avg_return = total_avg_return / valid_stocks
                    
                    print(f"  有效股票: {valid_stocks}, 总交易: {total_trades}")
                    print(f"  平均胜率: {avg_win_rate:.2f}%")
                    print(f"  平均收益率: {avg_return:.2f}%")
                    
                    results.append({
                        'threshold': threshold,
                        'depth': depth,
                        'backstep': backstep,
                        'valid_stocks': valid_stocks,
                        'total_trades': total_trades,
                        'avg_win_rate': avg_win_rate,
                        'avg_return': avg_return
                    })
    
    if results:
        print("\n" + "="*70)
        print("100个股票参数优化结果")
        print("="*70)
        
        results_df = pd.DataFrame(results)
        
        # 按平均收益率排序
        results_df = results_df.sort_values('avg_return', ascending=False)
        
        print("\n【按平均收益率排序】")
        print(results_df.to_string(index=False))
        
        # 最佳参数
        best = results_df.iloc[0]
        print(f"\n【最佳参数】")
        print(f"  threshold: {best['threshold']}")
        print(f"  depth: {best['depth']}")
        print(f"  backstep: {best['backstep']}")
        print(f"  平均胜率: {best['avg_win_rate']:.2f}%")
        print(f"  平均收益率: {best['avg_return']:.2f}%")
        print(f"  有效股票数: {best['valid_stocks']}")
        print(f"  总交易次数: {best['total_trades']}")


if __name__ == '__main__':
    main()
