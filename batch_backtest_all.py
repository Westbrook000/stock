#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股全量Spring回测 - 多线程版本
使用固定参数：threshold=2%, depth=14, backstep=1 + 趋势过滤
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


def calculate_zigzag_high_low(high, low, threshold=0.02, depth=14, backstep=1):
    """计算Zigzag拐点"""
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


def is_uptrend_or_sideways(df, current_idx):
    """趋势过滤"""
    if current_idx < 30:
        return True
    
    high = df['high'].values[:current_idx]
    low = df['low'].values[:current_idx]
    
    if len(high) < 30:
        return True
    
    supports = calculate_zigzag_high_low(high, low, threshold=0.05, depth=10, backstep=3)
    
    if len(supports) < 2:
        return True
    
    return supports[-1]['price'] >= supports[-2]['price']


def find_spring_signals(df, threshold=0.02, depth=14, backstep=1):
    """找出所有Spring信号"""
    spring_signals = {}
    
    if df is None or len(df) < 100:
        return spring_signals
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    for i in range(50, len(df)):
        if not is_uptrend_or_sideways(df, i):
            continue
        
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
                    '股票代码': '',
                    '股票名称': '',
                    '买入日期': str(self.buy_date),
                    '买入价格': round(self.buy_price, 2),
                    '卖出日期': str(exit_date),
                    '卖出价格': round(exit_price, 2),
                    '收益率%': round(ret, 2),
                    '持仓天数': current_idx - self.buy_idx,
                    '交易类型': trade_type,
                    '支撑位': round(self.current_support_price, 2)
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


def backtest_single_stock(filepath, output_dir):
    """回测单只股票"""
    try:
        code = os.path.basename(filepath).replace('.csv', '')
        
        df = pd.read_csv(filepath)
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)
        df = df.set_index('日期')
        
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
        
        name = df.iloc[0]['名称'] if '名称' in df.columns else code
        
        spring_signals = find_spring_signals(df, threshold=0.02, depth=14, backstep=1)
        
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
        
        for trade in strategy.trades:
            trade['股票代码'] = code
            trade['股票名称'] = name
        
        trades_df = pd.DataFrame(strategy.trades)
        
        code_dir = os.path.join(output_dir, code)
        os.makedirs(code_dir, exist_ok=True)
        trades_df.to_csv(os.path.join(code_dir, 'trades.csv'), index=False, encoding='utf-8-sig')
        
        return {
            'code': code,
            'name': name,
            'trades': len(trades_df),
            'wins': len(trades_df[trades_df['收益率%'] > 0]),
            'losses': len(trades_df[trades_df['收益率%'] <= 0]),
            'win_rate': len(trades_df[trades_df['收益率%'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            'avg_return': trades_df['收益率%'].mean(),
            'total_return': (cerebro.broker.getvalue() / 1000000 - 1) * 100
        }
    except Exception as e:
        return None


def main():
    stock_dir = 'data/stocks'
    output_dir = 'data/trades'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(stock_dir, '*.csv'))
    
    existing = set(os.listdir(output_dir))
    
    csv_files = [f for f in csv_files if os.path.basename(f).replace('.csv', '') not in existing]
    
    print(f"找到 {len(csv_files)} 个股票数据文件需要处理")
    print(f"已有 {len(os.listdir(output_dir))} 个股票处理完成")
    print(f"参数: threshold=2%, depth=14, backstep=1, 趋势过滤=开启")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    results = []
    processed = 0
    found = 0
    
    max_workers = 20
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(backtest_single_stock, f, output_dir): f for f in csv_files}
        
        for future in as_completed(futures):
            processed += 1
            result = future.result()
            
            if result:
                found += 1
                results.append(result)
                
                print(f"[{processed}/{len(csv_files)}] {result['code']} {result['name']}: "
                      f"{result['trades']}笔, 胜率{result['win_rate']:.1f}%, 收益{result['total_return']:.2f}%")
            else:
                if processed % 100 == 0:
                    print(f"[{processed}/{len(csv_files)}] 处理中...")
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('total_return', ascending=False)
        
        results_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False, encoding='utf-8-sig')
        
        print("\n" + "=" * 60)
        print("回测完成!")
        print("=" * 60)
        print(f"\n统计信息:")
        print(f"  总股票数: {len(csv_files)}")
        print(f"  有交易股票: {found}")
        print(f"  总交易笔数: {results_df['trades'].sum()}")
        
        total_wins = results_df['wins'].sum()
        total_losses = results_df['losses'].sum()
        total_trades = total_wins + total_losses
        
        print(f"  总胜率: {total_wins / total_trades * 100:.2f}%" if total_trades > 0 else "  总胜率: N/A")
        print(f"  平均收益: {results_df['total_return'].mean():.2f}%")
        
        print(f"\nTop 10 收益股票:")
        print(results_df.head(10)[['code', 'name', 'trades', 'win_rate', 'total_return']].to_string(index=False))
        
        print(f"\n详细交易记录已保存到: {output_dir}/")


if __name__ == '__main__':
    main()
