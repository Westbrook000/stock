#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spring回测脚本 - 分析Spring检测效果
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, '.')
from wyckoff_analysis import ProbabilityCloud


def backtest_spring(symbol: str, threshold: float = 0.5, holding_days: list = [5, 10, 20]):
    """回测Spring检测效果"""
    
    print(f"\n{'='*60}")
    print(f"Spring回测 - 股票 {symbol}")
    print(f"{'='*60}")
    
    # 加载数据
    try:
        import akshare as ak
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = '20070101'
        
        prefix = f"sh{symbol}" if symbol.startswith('6') else f"sz{symbol}"
        df = ak.stock_zh_a_daily(
            symbol=prefix,
            start_date=start_date,
            end_date=end_date
        )
        
        df = df.rename(columns={
            '日期': 'date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume'
        })
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"加载数据: {len(df)} 条记录")
        print(f"时间范围: {df['date'].min()} ~ {df['date'].max()}")
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 分析 - 使用ProbabilityCloud
    pc = ProbabilityCloud(df)
    pc.calculate_all_likelihoods()
    
    # 获取Spring概率
    spring_prob = pc.likelihood.get('SPRING', pd.Series([0]*len(df)))
    spring_best = pc.likelihood.get('SPRING_BEST', pd.Series([0]*len(df)))
    spring_confirm = pc.likelihood.get('SPRING_CONFIRM', pd.Series([0]*len(df)))
    spring_shakeout = pc.likelihood.get('SPRING_SHAKEOUT', pd.Series([0]*len(df)))
    spring_1day = pc.likelihood.get('SPRING_1DAY', pd.Series([0]*len(df)))
    
    # 筛选Spring信号
    signals = []
    for i in range(len(df)):
        if spring_prob.iloc[i] >= threshold:
            signals.append({
                'date': df.iloc[i]['date'],
                'close': df.iloc[i]['close'],
                'prob': spring_prob.iloc[i],
                'best': spring_best.iloc[i],
                'confirm': spring_confirm.iloc[i],
                'shakeout': spring_shakeout.iloc[i],
                '1day': spring_1day.iloc[i],
                'idx': i
            })
    
    print(f"\n检测到 {len(signals)} 个Spring信号 (阈值={threshold})")
    
    if not signals:
        return
    
    # 分析每个信号之后的涨幅
    results = []
    for sig in signals:
        idx = sig['idx']
        result = {
            'date': sig['date'].strftime('%Y-%m-%d'),
            'close': sig['close'],
            'prob': sig['prob'],
            'best': sig['best'],
            'confirm': sig['confirm'],
            'shakeout': sig['shakeout'],
            '1day': sig['1day']
        }
        
        for days in holding_days:
            if idx + days < len(df):
                future_close = df.iloc[idx + days]['close']
                pct = (future_close - sig['close']) / sig['close'] * 100
                result[f'pct_{days}d'] = pct
                result[f'up_{days}d'] = 1 if pct > 0 else 0
            else:
                result[f'pct_{days}d'] = None
                result[f'up_{days}d'] = None
        
        results.append(result)
    
    # 统计
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("Spring信号统计")
    print("="*60)
    
    for days in holding_days:
        valid = df_results[f'pct_{days}d'].dropna()
        if len(valid) > 0:
            avg_pct = valid.mean()
            success_rate = (valid > 0).sum() / len(valid) * 100
            print(f"\n{days}天后:")
            print(f"  样本数: {len(valid)}")
            print(f"  平均涨幅: {avg_pct:.2f}%")
            print(f"  成功率: {success_rate:.1f}%")
            print(f"  最大涨幅: {valid.max():.2f}%")
            print(f"  最大跌幅: {valid.min():.2f}%")
    
    # 按Spring类型分析
    print("\n" + "="*60)
    print("按Spring类型分析")
    print("="*60)
    
    # 最佳Spring
    best_signals = df_results[df_results['best'] > 0.3]
    if len(best_signals) > 0:
        print(f"\n最佳Spring (SPRING_BEST > 0.3): {len(best_signals)}个")
        for days in holding_days:
            valid = best_signals[f'pct_{days}d'].dropna()
            if len(valid) > 0:
                print(f"  {days}天: 平均{valid.mean():.2f}%, 成功率{(valid>0).mean()*100:.1f}%")
    
    # 需确认Spring
    confirm_signals = df_results[df_results['confirm'] > 0.3]
    if len(confirm_signals) > 0:
        print(f"\n需确认Spring (SPRING_CONFIRM > 0.3): {len(confirm_signals)}个")
        for days in holding_days:
            valid = confirm_signals[f'pct_{days}d'].dropna()
            if len(valid) > 0:
                print(f"  {days}天: 平均{valid.mean():.2f}%, 成功率{(valid>0).mean()*100:.1f}%")
    
    # 震仓Spring
    shakeout_signals = df_results[df_results['shakeout'] > 0.3]
    if len(shakeout_signals) > 0:
        print(f"\n震仓Spring (SPRING_SHAKEOUT > 0.3): {len(shakeout_signals)}个")
        for days in holding_days:
            valid = shakeout_signals[f'pct_{days}d'].dropna()
            if len(valid) > 0:
                print(f"  {days}天: 平均{valid.mean():.2f}%, 成功率{(valid>0).mean()*100:.1f}%")
    
    # 打印所有信号详情
    print("\n" + "="*60)
    print("所有Spring信号详情")
    print("="*60)
    print(df_results[['date', 'close', 'prob', 'best', 'confirm', 'shakeout', 'pct_5d', 'pct_10d', 'pct_20d']].to_string())
    
    return df_results


if __name__ == '__main__':
    # 测试不同阈值 - 0.6以上普通Spring, 0.8以上高置信度
    thresholds = [0.3, 0.5, 0.6, 0.7, 0.8]
    
    for th in thresholds:
        print("\n" + "="*60)
        print(f"阈值测试: threshold={th}")
        print("="*60)
        results = backtest_spring('600108', threshold=th)
