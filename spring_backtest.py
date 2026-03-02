#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spring回测 - 基于wyckoff-analyzer脚本 (增强版)
使用ProbabilityCloud类的增强Spring检测
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.agents/skills/wyckoff-analyzer/scripts'))
from wyckoff_analysis import ProbabilityCloud


def load_data(filename: str) -> pd.DataFrame:
    """加载CSV数据"""
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def backtest_spring_enhanced(df: pd.DataFrame, hold_days: list = [1, 3, 5], 
                            require_bull_market: bool = True,
                            require_demand_followup: bool = True,
                            followup_days: int = 3) -> dict:
    """
    增强版Spring回测 - 包含市场背景过滤和需求跟随验证
    """
    print("正在计算似然度...")
    cloud = ProbabilityCloud(df)
    cloud.calculate_all_likelihoods()
    
    results = {
        'total_springs': 0,
        'by_position': {},
        'by_context': {},
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
    for idx in range(30, len(df) - max(hold_days)):
        result = cloud.detect_spring_enhanced(
            idx, 
            require_bull_market=require_bull_market,
            require_demand_followup=require_demand_followup,
            followup_days=followup_days
        )
        
        if not result.get('detected', False):
            continue
        
        position = result.get('position', 'ORDINARY')
        context = result.get('market_context', 'ANY')
        
        # 初始化统计
        if position not in results['by_position']:
            results['by_position'][position] = {
                'total': 0,
                'success_by_days': {days: {'count': 0, 'success': 0} for days in hold_days}
            }
        
        if context not in results['by_context']:
            results['by_context'][context] = {
                'total': 0,
                'success_by_days': {days: {'count': 0, 'success': 0} for days in hold_days}
            }
        
        results['by_position'][position]['total'] += 1
        results['by_context'][context]['total'] += 1
        results['total_springs'] += 1
        
        entry_price = df.iloc[idx]['close']
        
        # 测试不同持有天数
        for days in hold_days:
            if idx + days < len(df):
                exit_price = df.iloc[idx + days]['close']
                return_pct = (exit_price - entry_price) / entry_price * 100
                is_success = return_pct > 0
                
                results['hold_days'][days]['count'] += 1
                results['hold_days'][days]['success'] += int(is_success)
                results['hold_days'][days]['returns'].append(return_pct)
                
                results['by_position'][position]['success_by_days'][days]['count'] += 1
                results['by_position'][position]['success_by_days'][days]['success'] += int(is_success)
                
                results['by_context'][context]['success_by_days'][days]['count'] += 1
                results['by_context'][context]['success_by_days'][days]['success'] += int(is_success)
    
    # 计算成功率
    for days in hold_days:
        if results['hold_days'][days]['count'] > 0:
            results['hold_days'][days]['success_rate'] = (
                results['hold_days'][days]['success'] / results['hold_days'][days]['count'] * 100
            )
            results['hold_days'][days]['avg_return'] = np.mean(
                results['hold_days'][days]['returns']
            ) if results['hold_days'][days]['returns'] else 0
    
    for position in results['by_position']:
        for days in hold_days:
            cnt = results['by_position'][position]['success_by_days'][days]['count']
            suc = results['by_position'][position]['success_by_days'][days]['success']
            if cnt > 0:
                results['by_position'][position]['success_by_days'][days]['success_rate'] = suc / cnt * 100
    
    for context in results['by_context']:
        for days in hold_days:
            cnt = results['by_context'][context]['success_by_days'][days]['count']
            suc = results['by_context'][context]['success_by_days'][days]['success']
            if cnt > 0:
                results['by_context'][context]['success_by_days'][days]['success_rate'] = suc / cnt * 100
    
    return results


def print_results(results: dict, name: str):
    """打印回测结果"""
    print("\n" + "="*60)
    print(f"回测结果: {name}")
    print("="*60)
    
    print(f"\n总Spring数量: {results['total_springs']}")
    
    # 按位置统计
    if results['by_position']:
        print("\n--- 按位置统计 ---")
        for position, data in results['by_position'].items():
            print(f"\n{position}: 共{data['total']}个")
            for days, stats in data['success_by_days'].items():
                if stats['count'] > 0:
                    print(f"  持有{days}天: 成功率{stats['success_rate']:.1f}% ({stats['success']}/{stats['count']})")
    
    # 按背景统计
    if results['by_context']:
        print("\n--- 按市场背景统计 ---")
        for context, data in results['by_context'].items():
            print(f"\n{context}: 共{data['total']}个")
            for days, stats in data['success_by_days'].items():
                if stats['count'] > 0:
                    print(f"  持有{days}天: 成功率{stats['success_rate']:.1f}% ({stats['success']}/{stats['count']})")
    
    # 总体统计
    print("\n--- 总体统计 ---")
    for days, stats in results['hold_days'].items():
        if stats['count'] > 0:
            print(f"持有{days}天: 成功率{stats['success_rate']:.1f}% ({stats['success']}/{stats['count']}), 平均收益{stats['avg_return']:.2f}%")
    
    return results


if __name__ == '__main__':
    # 加载数据
    print("加载数据...")
    df_515100 = load_data('etf_515100_data.csv')
    df_601600 = load_data('stock_601600_data.csv')
    df_combined = pd.concat([df_515100, df_601600], ignore_index=True)
    
    print(f"515100: {len(df_515100)}条, 601600: {len(df_601600)}条, 合并: {len(df_combined)}条")
    
    # ==================== 第一轮: 基础对比 ====================
    print("\n" + "="*70)
    print("第一轮: 对比不同配置")
    print("="*70)
    
    configs = [
        {'name': '增强版(完整)', 'require_bull_market': True, 'require_demand_followup': True, 'followup_days': 3},
        {'name': '无背景+需求跟随', 'require_bull_market': False, 'require_demand_followup': True, 'followup_days': 3},
    ]
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        results = backtest_spring_enhanced(
            df_combined, 
            hold_days=[1, 3, 5],
            require_bull_market=config['require_bull_market'],
            require_demand_followup=config['require_demand_followup'],
            followup_days=config['followup_days']
        )
        print_results(results, config['name'])
    
    # ==================== 第二轮: 优化参数 ====================
    print("\n" + "="*70)
    print("第二轮: 优化参数组合")
    print("="*70)
    
    configs2 = [
        {'name': '需求跟随2天', 'require_bull_market': False, 'require_demand_followup': True, 'followup_days': 2},
        {'name': '需求跟随4天', 'require_bull_market': False, 'require_demand_followup': True, 'followup_days': 4},
        {'name': '仅背景过滤', 'require_bull_market': True, 'require_demand_followup': False, 'followup_days': 3},
    ]
    
    for config in configs2:
        print(f"\n--- {config['name']} ---")
        results = backtest_spring_enhanced(
            df_combined, 
            hold_days=[1, 3, 5],
            require_bull_market=config['require_bull_market'],
            require_demand_followup=config['require_demand_followup'],
            followup_days=config['followup_days']
        )
        print_results(results, config['name'])
    
    # ==================== 第三轮: 最终验证 ====================
    print("\n" + "="*70)
    print("第三轮: 最终验证 - 最佳配置")
    print("="*70)
    
    # 最佳配置: 无背景过滤 + 需求跟随(3天)
    print("\n--- 最终验证: 无背景+需求跟随(3天) ---")
    results_final = backtest_spring_enhanced(
        df_combined, 
        hold_days=[1, 3, 5, 7, 10],
        require_bull_market=False,
        require_demand_followup=True,
        followup_days=3
    )
    print_results(results_final, "最终配置")
