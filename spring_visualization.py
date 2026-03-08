#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spring交易可视化 - 使用backtrader回测
显示交易前后一年的K线图，并标注详细交易信息
包含Zigzag转折线（只画到Spring跌破前一天）
"""

import backtrader as bt
import pandas as pd
import numpy as np
from math import floor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def calculate_zigzag_static(prices, threshold=0.05, depth=15):
    """
    静态Zigzag算法 - 顺序确认，无未来数据
    threshold: 反转阈值 (如0.05代表5%)
    depth: 确认需要的K线数
    
    返回: [(idx, price, type), ...] type='high' or 'low'
    """
    if len(prices) < 2:
        return []
    
    pivots = []
    current_trend = 'up' if prices[1] > prices[0] else 'down'
    current_extreme_index = 0
    current_extreme_price = prices[0]
    
    for i in range(1, len(prices)):
        price = prices[i]
        
        if current_trend == 'down':
            if price < current_extreme_price:
                current_extreme_price = price
                current_extreme_index = i
            
            if current_extreme_price > 0:
                rise_pct = (price - current_extreme_price) / current_extreme_price
                if rise_pct >= threshold and (i - current_extreme_index) >= depth:
                    pivots.append((current_extreme_index, current_extreme_price, 'low'))
                    current_trend = 'up'
                    current_extreme_price = price
                    current_extreme_index = i
        
        elif current_trend == 'up':
            if price > current_extreme_price:
                current_extreme_price = price
                current_extreme_index = i
            
            if current_extreme_price > 0:
                drop_pct = (current_extreme_price - price) / current_extreme_price
                if drop_pct >= threshold and (i - current_extreme_index) >= depth:
                    pivots.append((current_extreme_index, current_extreme_price, 'high'))
                    current_trend = 'down'
                    current_extreme_price = price
                    current_extreme_index = i
    
    return pivots


def is_uptrend_or_sideways(df, current_idx):
    """趋势过滤"""
    if current_idx < 30:
        return True
    
    high = df['high'].values[:current_idx]
    low = df['low'].values[:current_idx]
    
    if len(high) < 30:
        return True
    
    pivots = calculate_zigzag_static(low, threshold=0.05, depth=15)
    lows = [p for p in pivots if p[2] == 'low']
    
    if len(lows) < 2:
        return True
    
    return lows[-1][1] >= lows[-2][1]


def find_spring_signals(df, threshold=0.05, depth=15):
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
        
        historical_low = low[:i]
        
        if len(historical_low) < 30:
            continue
        
        pivots = calculate_zigzag_static(historical_low, threshold=threshold, depth=depth)
        lows = [p for p in pivots if p[2] == 'low']
        
        if len(lows) < 2:
            continue
        
        support_price = lows[-1][1]
        
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
                    'break_idx': i,
                    'spring_date': df.index[i + j],
                    'buy_date': df.index[i + j],
                    'break_price': break_price,
                    'buy_price': close[i + j]
                }
                break
    
    return spring_signals


def calculate_daily_supports_and_highs(df, threshold=0.05, depth=15, backstep=2):
    """计算每日的支撑位和高点（基于i-1日的数据，先验）"""
    results = []
    
    high = df['high'].values
    low = df['low'].values
    
    for i in range(50, len(df)):
        end_idx = i - 1
        
        if end_idx < 30:
            results.append({
                '日期': df.index[i],
                'Index': i,
                '前一日支撑位': None,
                '前一日高点': None,
                '当日最低价': low[i],
                '当日最高价': high[i],
                '当日收盘价': df['close'].values[i]
            })
            continue
        
        historical_low = low[:end_idx + 1]
        
        pivots = calculate_zigzag_static(historical_low, threshold=threshold, depth=depth)
        lows = [p for p in pivots if p[2] == 'low']
        highs = [p for p in pivots if p[2] == 'high']
        
        support_price = lows[-1][1] if lows else None
        high_price = highs[-1][1] if highs else None
        
        results.append({
            '日期': df.index[i],
            'Index': i,
            '前一日支撑位': round(support_price, 4) if support_price else None,
            '前一日高点': round(high_price, 4) if high_price else None,
            '当日最低价': round(low[i], 4),
            '当日最高价': round(high[i], 4),
            '当日收盘价': round(df['close'].values[i], 4)
        })
    
    return pd.DataFrame(results)


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
                    '买入日期': str(self.buy_date),
                    '买入价格': round(self.buy_price, 2),
                    '卖出日期': str(exit_date),
                    '卖出价格': round(exit_price, 2),
                    '收益率%': round(ret, 2),
                    '持仓天数': current_idx - self.buy_idx,
                    '交易类型': trade_type,
                    '支撑位': round(self.current_support_price, 2),
                    'buy_idx': self.buy_idx,
                    'sell_idx': current_idx
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


def draw_trade_chart(df, trade, output_dir, stock_code='600108', stock_name='股票'):
    """绘制单笔交易的K线图，包含Zigzag转折线"""
    buy_idx = trade['buy_idx']
    sell_idx = trade['sell_idx']
    break_idx = trade['break_idx']
    
    lookback = 250
    start_idx = max(0, buy_idx - lookback)
    end_idx = min(len(df) - 1, sell_idx + lookback)
    
    plot_df = df.iloc[start_idx:end_idx + 1].copy()
    
    if len(plot_df) < 50:
        return
    
    high = df['high'].values
    low = df['low'].values
    
    zigzag_end_idx = break_idx
    
    pivots = calculate_zigzag_static(low[:zigzag_end_idx+1], threshold=0.05, depth=15)
    zigzag_points = [(p[0], p[1], p[2]) for p in pivots if p[0] < zigzag_end_idx]
    zigzag_points = sorted(zigzag_points, key=lambda x: x[0])
    
    lows = [p for p in pivots if p[2] == 'low' and p[0] < zigzag_end_idx]
    supports = [{'idx': p[0], 'price': p[1]} for p in lows]
    
    high_pivots = calculate_zigzag_static(high[:zigzag_end_idx+1], threshold=0.05, depth=15)
    highs_list = [p for p in high_pivots if p[2] == 'high' and p[0] < zigzag_end_idx]
    resistances = [{'idx': p[0], 'price': p[1]} for p in highs_list]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(28, 10))
    
    # 绘制K线
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        date = idx
        open_price = row['open']
        close_price = row['close']
        high_price = row['high']
        low_price = row['low']
        
        color = 'red' if close_price >= open_price else 'green'
        
        # 实体
        ax.plot([date, date], [open_price, close_price], color=color, linewidth=1.5)
        # 上影线
        ax.plot([date, date], [high_price, max(open_price, close_price)], color=color, linewidth=1)
        # 下影线
        ax.plot([date, date], [min(open_price, close_price), low_price], color=color, linewidth=1)
    
    # 绘制Zigzag线（只到break_idx前一天）
    if len(zigzag_points) > 1:
        zigzag_x = []
        zigzag_y = []
        directions = []
        
        for (p_idx, p_price, p_type) in zigzag_points:
            if p_idx >= break_idx:
                break
            if start_idx <= p_idx < end_idx:
                zigzag_x.append(df.index[p_idx])
                zigzag_y.append(p_price)
                directions.append(p_type)
        
        if len(zigzag_x) > 1:
            ax.plot(zigzag_x, zigzag_y, 'b--', linewidth=1, alpha=0.7, label='Zigzag折线')
            
            # 标注pivot点
            for i, (x, y, d) in enumerate(zip(zigzag_x, zigzag_y, directions)):
                marker = '^' if d == 'high' else 'v'
                color = 'red' if d == 'high' else 'green'
                ax.scatter([x], [y], marker=marker, s=50, c=color, zorder=5)
    
    # 绘制支撑位（只显示break_idx之前的支撑）
    for s in supports:
        if s['idx'] < break_idx and start_idx <= s['idx'] < end_idx:
            ax.axhline(y=s['price'], color='orange', linestyle='-.', alpha=0.5, linewidth=1)
            ax.annotate(f"支撑: {s['price']:.2f}", 
                       xy=(df.index[s['idx']], s['price']),
                       xytext=(df.index[s['idx']], s['price'] * 1.02),
                       fontsize=8, color='orange', alpha=0.8)
    
    # 绘制压力位
    for r in resistances:
        if r['idx'] < break_idx and start_idx <= r['idx'] < end_idx:
            ax.axhline(y=r['price'], color='red', linestyle='-.', alpha=0.5, linewidth=1)
            ax.annotate(f"压力: {r['price']:.2f}", 
                       xy=(df.index[r['idx']], r['price']),
                       xytext=(df.index[r['idx']], r['price'] * 0.98),
                       fontsize=8, color='red', alpha=0.8)
    
    # 标记买入点
    buy_date = pd.to_datetime(trade['买入日期'])
    if buy_date in plot_df.index:
        buy_price = trade['买入价格']
        ax.axvline(x=buy_date, color='blue', linestyle='--', alpha=0.7, linewidth=1)
        ax.annotate(f"买入\n{buy_price:.2f}", 
                   xy=(buy_date, buy_price),
                   xytext=(buy_date, buy_price * 1.05),
                   fontsize=10, color='blue', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='blue'))
    
    # 标记卖出点
    sell_date = pd.to_datetime(trade['卖出日期'])
    if sell_date in plot_df.index:
        sell_price = trade['卖出价格']
        ax.axvline(x=sell_date, color='purple', linestyle='--', alpha=0.7, linewidth=1)
        ax.annotate(f"卖出\n{sell_price:.2f}", 
                   xy=(sell_date, sell_price),
                   xytext=(sell_date, sell_price * 1.05),
                   fontsize=10, color='purple', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='purple'))
    
    # 标记Spring跌破点（break_idx）
    break_date = df.index[break_idx]
    if break_date in plot_df.index:
        break_price = trade['break_price']
        ax.axvline(x=break_date, color='red', linestyle=':', alpha=0.8, linewidth=2, label=f"Spring跌破: {break_date.strftime('%Y-%m-%d')}")
        ax.annotate(f"跌破\n{break_price:.2f}", 
                   xy=(break_date, break_price),
                   xytext=(break_date, break_price * 0.95),
                   fontsize=10, color='red', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red'))
    
    # 标记支撑位
    support_price = trade['支撑位']
    ax.axhline(y=support_price, color='orange', linestyle='-', alpha=0.9, linewidth=2, label=f"支撑位: {support_price:.2f}")
    
    # 添加交易信息文本框
    info_text = f"""
    股票: {stock_code} {stock_name}
    买入日期: {trade['买入日期']}
    买入价格: {trade['买入价格']:.2f}
    卖出日期: {trade['卖出日期']}
    卖出价格: {trade['卖出价格']:.2f}
    收益率: {trade['收益率%']:.2f}%
    持仓天数: {trade['持仓天数']}天
    交易类型: {trade['交易类型']}
    支撑位: {support_price:.2f}
    跌破价格: {trade.get('break_price', 'N/A')}
    """
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # 设置标题和标签
    ax.set_title(f"{stock_code} {stock_name} - Spring交易可视化 (第{trade['index']+1}笔交易)", fontsize=14, fontweight='bold')
    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("价格", fontsize=12)
    
    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 保存图片
    idx_num = trade['index'] + 1
    filename = f"trade_{idx_num:02d}_{trade['买入日期'].replace('-', '')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    
    return filepath


def main():
    print("=" * 70)
    print("Spring Trade Visualization - 600108 Yasheng Group")
    print("=" * 70)
    
    # 加载数据
    code = '600108'
    filepath = f'data/stocks/{code}.csv'
    
    df = pd.read_csv(filepath)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values('日期').reset_index(drop=True)
    df = df.set_index('日期')
    
    df = df.rename(columns={
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '收盘': 'close',
        '成交量': 'volume'
    })
    
    # 筛选2021年后的数据
    cutoff_date = pd.to_datetime('2021-01-01')
    df = df[df.index >= cutoff_date]
    
    print(f"\nData Range: {df.index[0]} ~ {df.index[-1]}")
    print(f"Total Days: {len(df)}")
    
    # 计算每日支撑和高点（先验数据）
    print("\n计算每日支撑和高点...")
    daily_df = calculate_daily_supports_and_highs(df, threshold=0.05, depth=15)
    output_dir = f'data/trades_visualization/{code}'
    os.makedirs(output_dir, exist_ok=True)
    daily_csv_path = os.path.join(output_dir, 'daily_supports_highs.csv')
    daily_df.to_csv(daily_csv_path, index=False, encoding='utf-8-sig')
    print(f"每日支撑和高点已保存到: {daily_csv_path}")
    
    # 查找Spring信号
    spring_signals = find_spring_signals(df, threshold=0.05, depth=15)
    print(f"\nFound Spring Signals: {len(spring_signals)}")
    
    # 回测
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
    
    print(f"\nTotal Trades: {len(strategy.trades)}")
    
    # 为每笔交易添加break_idx信息
    for trade in strategy.trades:
        buy_date = pd.to_datetime(trade['买入日期'])
        # 查找对应的spring signal
        for idx, sig in spring_signals.items():
            sig_date = df.index[idx]
            if sig_date == buy_date:
                trade['break_idx'] = sig['break_idx']
                trade['break_price'] = sig['break_price']
                break
        else:
            # 如果没找到，使用默认值
            trade['break_idx'] = trade['buy_idx'] - 1
            trade['break_price'] = trade['支撑位']
    
    # 绘制每笔交易的图
    print(f"\nGenerating trade charts...")
    
    for i, trade in enumerate(strategy.trades):
        trade['index'] = i
        filepath = draw_trade_chart(df, trade, output_dir, stock_code=code)
        if filepath:
            print(f"  [{i+1}/{len(strategy.trades)}] Saved: {os.path.basename(filepath)}")
    
    print(f"\nCharts saved to: {output_dir}/")
    
    # 保存所有Spring信号（包括未成交的）
    all_spring_signals = []
    for idx, sig in spring_signals.items():
        all_spring_signals.append({
            '信号序号': len(all_spring_signals) + 1,
            '股票代码': code,
            '股票名称': '亚盛集团',
            'Spring跌破日期': str(sig['break_date'])[:10],
            'Spring跌破价格': round(sig['break_price'], 2),
            'Spring跌破Index': sig['break_idx'],
            'Spring确认日期': str(sig['spring_date'])[:10],
            '买入日期': str(sig['buy_date'])[:10],
            '买入价格': round(sig['buy_price'], 2),
            '支撑位': round(sig['support_price'], 2),
            '是否成交': '是' if any(df.index[idx] == pd.to_datetime(t['买入日期']) for t in strategy.trades) else '否'
        })
    
    spring_signals_df = pd.DataFrame(all_spring_signals)
    spring_signals_path = os.path.join(output_dir, 'all_spring_signals.csv')
    spring_signals_df.to_csv(spring_signals_path, index=False, encoding='utf-8-sig')
    print(f"\n所有Spring信号已保存到: {spring_signals_path}")
    
    # 保存详细交易信息到CSV
    detailed_trades = []
    for trade in strategy.trades:
        buy_date = pd.to_datetime(trade['买入日期'])
        
        # 获取Spring跌破日期和价格
        spring_date = None
        break_date = None
        break_price = None
        for idx, sig in spring_signals.items():
            if df.index[idx] == buy_date:
                spring_date = sig['spring_date']
                break_date = sig['break_date']
                break_price = sig['break_price']
                break
        
        detailed_trades.append({
            '交易序号': trade['index'] + 1,
            '股票代码': code,
            '股票名称': '亚盛集团',
            '买入日期': trade['买入日期'],
            '买入价格': trade['买入价格'],
            '卖出日期': trade['卖出日期'],
            '卖出价格': trade['卖出价格'],
            '收益率%': trade['收益率%'],
            '持仓天数': trade['持仓天数'],
            '交易类型': trade['交易类型'],
            '支撑位': trade['支撑位'],
            'Spring跌破日期': str(break_date)[:10] if break_date else '',
            'Spring跌破价格': break_price if break_price else '',
            'Spring确认日期': str(spring_date)[:10] if spring_date else '',
            '买入Index': trade['buy_idx'],
            '卖出Index': trade['sell_idx']
        })
    
    detailed_df = pd.DataFrame(detailed_trades)
    detailed_csv_path = os.path.join(output_dir, 'detailed_trades.csv')
    detailed_df.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n详细交易信息已保存到: {detailed_csv_path}")
    
    # 输出交易汇总
    print("\n" + "=" * 70)
    print("Trade Summary")
    print("=" * 70)
    
    trades_df = pd.DataFrame(strategy.trades)
    if len(trades_df) > 0:
        print(f"\nTotal Trades: {len(trades_df)}")
        print(f"Wins: {len(trades_df[trades_df['收益率%'] > 0])}")
        print(f"Losses: {len(trades_df[trades_df['收益率%'] <= 0])}")
        print(f"Win Rate: {len(trades_df[trades_df['收益率%'] > 0]) / len(trades_df) * 100:.1f}%")
        print(f"Avg Return: {trades_df['收益率%'].mean():.2f}%")
        
        print("\nTrade Details:")
        print(trades_df[['买入日期', '买入价格', '卖出价格', '收益率%', '持仓天数', '交易类型', '支撑位']].to_string())


if __name__ == '__main__':
    main()
