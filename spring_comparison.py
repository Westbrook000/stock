#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spring策略优化对比回测
原策略 vs 新策略（跌破支撑时成交量萎缩 + 蜡烛实体小）

真正的Spring特征：
1. 价格跌破支撑位
2. 跌破时成交量萎缩（< 20日均量的某个比例）
3. 蜡烛实体小（< 20日平均实体大小的某个比例）
4. 价格快速恢复到支撑位上方
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
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


def find_spring_signals_original(df, threshold=0.02, depth=14, backstep=1):
    """原策略：找出所有Spring信号（无成交量和实体过滤）"""
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
                    'break_price': break_price,
                    'buy_price': close[i + j],
                    'break_idx': i
                }
                break
    
    return spring_signals


def find_spring_signals_with_filter(df, threshold=0.02, depth=14, backstep=1, 
                                    vol_ratio=0.7, body_ratio=0.7):
    """
    优化策略：跌破支撑时成交量萎缩 + 蜡烛实体小
    
    Args:
        df: 股票数据
        vol_ratio: 成交量比例阈值（如0.7表示跌破时成交量 < 20日均量的70%）
        body_ratio: 蜡烛实体比例阈值（如0.7表示实体 < 20日平均实体的70%）
    """
    spring_signals = {}
    
    if df is None or len(df) < 100:
        return spring_signals
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    open_price = df['open'].values
    
    if 'volume' not in df.columns:
        return spring_signals
    
    volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))
    
    if len(volume) < 50:
        return spring_signals
    
    vol_ma = pd.Series(volume).rolling(20).mean().values
    
    body_sizes = np.abs(close - open_price) / open_price
    body_ma = pd.Series(body_sizes).rolling(20).mean().values
    
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
        
        break_vol = volume[i]
        avg_vol = vol_ma[i]
        body_size = body_sizes[i]
        avg_body = body_ma[i]
        
        if avg_vol is None or avg_vol <= 0:
            continue
        if avg_body is None or avg_body <= 0:
            continue
        
        if break_vol / avg_vol > vol_ratio:
            continue
        
        if body_size / avg_body > body_ratio:
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
                    'break_price': break_price,
                    'buy_price': close[i + j],
                    'break_idx': i,
                    'break_volume': break_vol,
                    'avg_volume': avg_vol,
                    'vol_ratio': break_vol / avg_vol if avg_vol > 0 else 1,
                    'body_size': body_size,
                    'avg_body': avg_body,
                    'body_ratio': body_size / avg_body if avg_body > 0 else 1
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


def backtest_single_stock(filepath, strategy='original'):
    """回测单只股票
    
    Args:
        filepath: 股票数据文件路径
        strategy: 'original' 或 'filtered'
    """
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
        
        if 'volume' not in df.columns:
            df['volume'] = df['close'] * 0 + 1
        
        cutoff_date = pd.to_datetime('2021-01-01')
        df = df[df.index >= cutoff_date]
        
        if len(df) < 100:
            return None
        
        name = df.iloc[0]['名称'] if '名称' in df.columns else code
        
        if strategy == 'original':
            spring_signals = find_spring_signals_original(df, threshold=0.02, depth=14, backstep=1)
        else:
            spring_signals = find_spring_signals_with_filter(
                df, threshold=0.02, depth=14, backstep=1,
                vol_ratio=0.7, body_ratio=0.7
            )
        
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
        
        return {
            'code': code,
            'name': name,
            'trades': len(trades_df),
            'wins': len(trades_df[trades_df['收益率%'] > 0]),
            'losses': len(trades_df[trades_df['收益率%'] <= 0]),
            'win_rate': len(trades_df[trades_df['收益率%'] > 0]) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            'avg_return': trades_df['收益率%'].mean(),
            'total_return': (cerebro.broker.getvalue() / 1000000 - 1) * 100,
            'trades_df': trades_df
        }
    except Exception as e:
        return None


def get_sample_stocks(n=100):
    """获取抽样股票（排除ST和科创板）"""
    stock_dir = 'data/stocks'
    csv_files = glob.glob(os.path.join(stock_dir, '*.csv'))
    
    valid_stocks = []
    for f in csv_files:
        code = os.path.basename(f).replace('.csv', '')
        
        if code.startswith('688'):
            continue
        
        try:
            df = pd.read_csv(f)
            if '名称' in df.columns and len(df) > 0:
                name = df.iloc[0]['名称']
                if 'ST' in name or '*ST' in name or 'ST' in name:
                    continue
                valid_stocks.append(f)
        except:
            continue
    
    np.random.seed(42)
    if len(valid_stocks) > n:
        valid_stocks = np.random.choice(valid_stocks, n, replace=False).tolist()
    
    return valid_stocks


def main():
    print("=" * 70)
    print("Spring策略优化对比回测 - 成交量+蜡烛实体过滤")
    print("=" * 70)
    print("\n策略说明:")
    print("  原策略: Spring确认日直接买入（无成交量和实体过滤）")
    print("  优化策略: 跌破支撑时成交量<20日均量的70% 且 实体<20日均量的70%")
    print("\n抽样规则: 100只股票, 排除ST和科创板")
    print("=" * 70)
    
    sample_stocks = get_sample_stocks(n=100)
    print(f"\n抽样股票数量: {len(sample_stocks)}")
    
    results_original = []
    results_filtered = []
    
    for i, filepath in enumerate(sample_stocks):
        code = os.path.basename(filepath).replace('.csv', '')
        
        print(f"\n[{i+1}/100] 回测 {code}...")
        
        result_orig = backtest_single_stock(filepath, strategy='original')
        if result_orig:
            results_original.append(result_orig)
            print(f"  原策略: {result_orig['trades']}笔, 胜率{result_orig['win_rate']:.1f}%, 收益{result_orig['total_return']:.2f}%")
        
        result_filt = backtest_single_stock(filepath, strategy='filtered')
        if result_filt:
            results_filtered.append(result_filt)
            print(f"  优化策略: {result_filt['trades']}笔, 胜率{result_filt['win_rate']:.1f}%, 收益{result_filt['total_return']:.2f}%")
    
    print("\n" + "=" * 70)
    print("对比结果汇总")
    print("=" * 70)
    
    if results_original:
        orig_df = pd.DataFrame(results_original)
        print(f"\n【原策略】")
        print(f"  股票数量: {len(results_original)}")
        print(f"  总交易笔数: {orig_df['trades'].sum()}")
        print(f"  总胜率: {orig_df['wins'].sum() / orig_df['trades'].sum() * 100:.2f}%")
        print(f"  平均收益: {orig_df['total_return'].mean():.2f}%")
        print(f"  盈利股票占比: {(orig_df['total_return'] > 0).sum() / len(orig_df) * 100:.1f}%")
    
    if results_filtered:
        filt_df = pd.DataFrame(results_filtered)
        print(f"\n【优化策略（成交量+实体过滤）】")
        print(f"  股票数量: {len(results_filtered)}")
        print(f"  总交易笔数: {filt_df['trades'].sum()}")
        print(f"  总胜率: {filt_df['wins'].sum() / filt_df['trades'].sum() * 100:.2f}%")
        print(f"  平均收益: {filt_df['total_return'].mean():.2f}%")
        print(f"  盈利股票占比: {(filt_df['total_return'] > 0).sum() / len(filt_df) * 100:.1f}%")
    
    if results_original and results_filtered:
        print(f"\n【对比分析】")
        orig_total = orig_df['trades'].sum()
        filt_total = filt_df['trades'].sum()
        orig_wins = orig_df['wins'].sum()
        filt_wins = filt_df['wins'].sum()
        
        print(f"  交易次数变化: {orig_total} -> {filt_total} ({filt_total-orig_total:+d})")
        print(f"  胜率变化: {orig_wins/orig_total*100:.2f}% -> {filt_wins/filt_total*100:.2f}% ({filt_wins/filt_total*100 - orig_wins/orig_total*100:+.2f}%)")
        print(f"  平均收益变化: {orig_df['total_return'].mean():.2f}% -> {filt_df['total_return'].mean():.2f}% ({filt_df['total_return'].mean() - orig_df['total_return'].mean():+.2f}%)")
        
        orig_profitable = (orig_df['total_return'] > 0).sum() / len(orig_df) * 100
        filt_profitable = (filt_df['total_return'] > 0).sum() / len(filt_df) * 100
        print(f"  盈利股票占比变化: {orig_profitable:.1f}% -> {filt_profitable:.1f}% ({filt_profitable - orig_profitable:+.1f}%)")
    
    output_dir = 'data/trades_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    if results_original:
        orig_summary = orig_df[['code', 'name', 'trades', 'wins', 'losses', 'win_rate', 'avg_return', 'total_return']]
        orig_summary.to_csv(os.path.join(output_dir, 'original_strategy.csv'), index=False, encoding='utf-8-sig')
    
    if results_filtered:
        filt_summary = filt_df[['code', 'name', 'trades', 'wins', 'losses', 'win_rate', 'avg_return', 'total_return']]
        filt_summary.to_csv(os.path.join(output_dir, 'vol_body_filtered_strategy.csv'), index=False, encoding='utf-8-sig')
    
    print(f"\n详细结果已保存到: {output_dir}/")


if __name__ == '__main__':
    main()
