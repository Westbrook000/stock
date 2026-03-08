#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量回测Spring策略 - 生成完整可视化输出
"""

import sys
import pandas as pd
import backtrader as bt
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from spring_visualization import (
    calculate_zigzag_static,
    find_spring_signals,
    calculate_daily_supports_and_highs,
    SpringSignalStrategy,
    draw_trade_chart
)


def run_backtest_for_stock(code, threshold=0.05, depth=15):
    """对单只股票运行回测并生成可视化"""
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
    
    cutoff_date = pd.to_datetime('2021-01-01')
    df = df[df.index >= cutoff_date]
    
    if len(df) < 100:
        return None
    
    output_dir = f'data/trades_visualization/{code}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'\n=== {code} ===')
    print(f'Data Range: {df.index[0]} ~ {df.index[-1]}, Days: {len(df)}')
    
    # 计算每日支撑和高点
    daily_df = calculate_daily_supports_and_highs(df, threshold=threshold, depth=depth)
    daily_csv_path = os.path.join(output_dir, 'daily_supports_highs.csv')
    daily_df.to_csv(daily_csv_path, index=False, encoding='utf-8-sig')
    
    # 查找Spring信号
    spring_signals = find_spring_signals(df, threshold=threshold, depth=depth)
    print(f'Spring Signals: {len(spring_signals)}')
    
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
    
    trades = strategy.trades
    print(f'Trades: {len(trades)}')
    
    # 为每笔交易添加break_idx信息
    for trade in trades:
        buy_date = pd.to_datetime(trade['买入日期'])
        for idx, sig in spring_signals.items():
            sig_date = df.index[idx]
            if sig_date == buy_date:
                trade['break_idx'] = sig['break_idx']
                trade['break_price'] = sig['break_price']
                break
        else:
            trade['break_idx'] = trade['buy_idx'] - 1
            trade['break_price'] = trade['支撑位']
    
    # 绘制每笔交易的图
    for i, trade in enumerate(trades):
        trade['index'] = i
        draw_trade_chart(df, trade, output_dir, stock_code=code)
    
    # 保存所有Spring信号
    all_spring_signals = []
    for idx, sig in spring_signals.items():
        all_spring_signals.append({
            '信号序号': len(all_spring_signals) + 1,
            '股票代码': code,
            'Spring跌破日期': str(sig['break_date'])[:10],
            'Spring跌破价格': round(sig['break_price'], 2),
            'Spring跌破Index': sig['break_idx'],
            'Spring确认日期': str(sig['spring_date'])[:10],
            '买入日期': str(sig['buy_date'])[:10],
            '买入价格': round(sig['buy_price'], 2),
            '支撑位': round(sig['support_price'], 2),
            '是否成交': '是' if any(df.index[idx] == pd.to_datetime(t['买入日期']) for t in trades) else '否'
        })
    
    spring_signals_df = pd.DataFrame(all_spring_signals)
    spring_signals_path = os.path.join(output_dir, 'all_spring_signals.csv')
    spring_signals_df.to_csv(spring_signals_path, index=False, encoding='utf-8-sig')
    
    # 保存详细交易信息
    detailed_trades = []
    for trade in trades:
        buy_date = pd.to_datetime(trade['买入日期'])
        
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
    
    # 计算统计
    if len(trades) > 0:
        wins = len([t for t in trades if t['收益率%'] > 0])
        losses = len([t for t in trades if t['收益率%'] <= 0])
        avg_return = sum([t['收益率%'] for t in trades]) / len(trades)
        win_rate = wins / len(trades) * 100
        
        print(f'Wins: {wins}, Losses: {losses}, Win Rate: {win_rate:.1f}%, Avg Return: {avg_return:.2f}%')
        print(f'Output: {output_dir}')
        
        return {
            'code': code,
            'trades': len(trades),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_return': avg_return
        }
    else:
        print('No trades')
        return {'code': code, 'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 'avg_return': 0}


if __name__ == '__main__':
    stocks = ['601012', '002329', '601816', '000505', '600138', '600713', '601600', '600515', '600309']
    all_stocks = ['600108'] + stocks
    
    all_results = []
    
    for code in all_stocks:
        result = run_backtest_for_stock(code)
        if result:
            all_results.append(result)
    
    # 汇总
    print('\n' + '='*70)
    print('汇总结果')
    print('='*70)
    
    total_trades = sum([r['trades'] for r in all_results])
    total_wins = sum([r['wins'] for r in all_results])
    total_losses = sum([r['losses'] for r in all_results])
    overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
    overall_avg_return = sum([r['trades'] * r['avg_return'] for r in all_results]) / total_trades if total_trades > 0 else 0
    
    print(f'\n股票数量: {len(all_results)}')
    print(f'总交易数: {total_trades}')
    print(f'盈利次数: {total_wins}')
    print(f'亏损次数: {total_losses}')
    print(f'总胜率: {overall_win_rate:.1f}%')
    print(f'加权平均收益: {overall_avg_return:.2f}%')
    
    print('\n各股票详情:')
    print('代码         交易数      盈利     亏损     胜率         平均收益')
    print('-'*60)
    for r in all_results:
        row = f'{r["code"]:<10} {r["trades"]:<8} {r["wins"]:<6} {r["losses"]:<6} {r["win_rate"]:.1f}%    {r["avg_return"]:.2f}%'
        print(row)
