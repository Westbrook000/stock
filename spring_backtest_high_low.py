#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spring回测 - 使用Backtrader + high/low Zigzag
使用600108（亚盛集团）数据
"""

import backtrader as bt
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.sans-serif'] = ['Heiti SC', 'STHeiti', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def draw_trade_chart(df, trade, spring_date, break_date, support_price, output_dir, trade_num):
    """为单笔交易绘制K线图"""
    try:
        buy_date = pd.to_datetime(trade['买入日期'])
        sell_date = pd.to_datetime(trade['卖出日期'])
        
        hold_days = trade['持仓天数']
        
        # 买入点在中间，展示一年数据
        half_days = 120
        start_date = buy_date - timedelta(days=half_days)
        end_date = buy_date + timedelta(days=half_days)
        
        mask = (df.index >= start_date) & (df.index <= end_date)
        plot_df = df.loc[mask].copy()
        
        if len(plot_df) < 10:
            return None
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        ax.plot(plot_df.index, plot_df['close'], 'b-', linewidth=1, label='收盘价')
        ax.fill_between(plot_df.index, plot_df['low'], plot_df['high'], alpha=0.3, color='gray')
        
        ax.axhline(y=support_price, color='green', linestyle='--', linewidth=2, label=f'支撑位: {support_price:.2f}')
        
        # 标记跌破日（红色五角星）
        if break_date and break_date in plot_df.index:
            break_price = plot_df.loc[break_date, 'low']
            ax.scatter([break_date], [break_price], color='red', s=400, marker='*', zorder=5, label=f'跌破日: {break_date.strftime("%m-%d")}')
        
        # 标记Spring确立日（红色三角）
        if spring_date and spring_date in plot_df.index:
            spring_price = plot_df.loc[spring_date, 'close']
            ax.scatter([spring_date], [spring_price], color='red', s=300, marker='v', zorder=5, label=f'Spring确立日: {spring_date.strftime("%m-%d")}')
        
        # 标记买入点
        ax.scatter([buy_date], [trade['买入价格']], color='blue', s=300, marker='^', zorder=5, label=f'买入: {trade["买入价格"]:.2f}')
        
        # 标记卖出点
        ax.scatter([sell_date], [trade['卖出价格']], color='orange', s=300, marker='s', zorder=5, label=f'卖出: {trade["卖出价格"]:.2f}')
        
        return_pct = trade['收益率%']
        color = 'green' if return_pct > 0 else 'red'
        ax.set_title(f"股票交易 #{trade_num} | 收益率: {return_pct:.2f}% | 持仓: {hold_days}天", fontsize=14, color=color, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('价格')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        filename = f"trade_{trade_num:03d}_{buy_date.strftime('%Y%m%d')}_{sell_date.strftime('%Y%m%d')}_{return_pct:.1f}pct.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=100)
        plt.close()
        
        return filepath
        
    except Exception as e:
        print(f"    画图失败: {e}")
        return None


def calculate_zigzag_high_low(high, low, threshold=0.02, depth=14, backstep=3):
    """
    计算Zigzag拐点（使用high/low高低点）
    返回: (pivots, supports, resistances)
    """
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
    
    # 提取支撑位（波谷）
    supports = []
    for i in range(1, len(pivots) - 1):
        idx = pivots[i]
        if (low[pivots[i-1]] > low[idx]) and (low[pivots[i+1]] > low[idx]):
            supports.append({'idx': idx, 'price': low[idx]})
    
    # 提取阻力位（波峰）
    resistances = []
    for i in range(1, len(pivots) - 1):
        idx = pivots[i]
        if (high[pivots[i-1]] < high[idx]) and (high[pivots[i+1]] < high[idx]):
            resistances.append({'idx': idx, 'price': high[idx]})
    
    return pivots, supports, resistances


def is_uptrend_or_sideways(df, current_idx):
    """
    趋势过滤：只允许非下跌趋势
    条件：最近2个Zigzag低点不创新低（Low2 >= Low1）
    """
    if current_idx < 30:
        return True
    
    high = df['high'].values[:current_idx]
    low = df['low'].values[:current_idx]
    
    if len(high) < 30:
        return True
    
    _, supports, _ = calculate_zigzag_high_low(high, low, threshold=0.05, depth=10, backstep=3)
    
    if len(supports) < 2:
        return True
    
    last1 = supports[-1]['price']
    last2 = supports[-2]['price']
    
    return last1 >= last2


def find_spring_signals(df, threshold=0.02, depth=14, backstep=3, use_trend_filter=True):
    """
    找出所有Spring信号（使用high/low）
    - 跌破Zigzag支撑位
    - 5天内收盘价收复支撑位
    - Spring确立日 = 翻回支撑位那天
    - 趋势过滤：只允许非下跌趋势（可选）
    """
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
        
        # 趋势过滤
        if use_trend_filter:
            if not is_uptrend_or_sideways(df, i):
                continue
        
        _, zigzag_supports, _ = calculate_zigzag_high_low(
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
                # Spring确立当天买入
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
    """基于Spring信号的策略"""
    
    params = (
        ('spring_signals', None),
        ('profit_target', 0.15),  # 止盈15%
        ('stop_loss', 0.05),      # 止损5%
        ('max_hold_days', 20),    # 最大持仓天数
        ('buy_amount', 100000),   # 每次买入金额10万元
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


def run_backtest(symbol='600108', start_date='20180101'):
    """运行回测"""
    print(f"\n{'='*60}")
    print(f"Spring回测 - 股票 {symbol}")
    print(f"{'='*60}")
    
    # 获取数据
    print(f"\n获取数据中...")
    try:
        symbol_with_prefix = f"sh{symbol}" if symbol.startswith('6') else f"sz{symbol}"
        df = ak.stock_zh_a_daily(
            symbol=symbol_with_prefix,
            start_date=start_date,
            end_date=datetime.now().strftime('%Y%m%d')
        )
    except Exception as e:
        print(f"获取数据失败: {e}")
        return
    
    if df is None or len(df) < 100:
        print("数据不足")
        return
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df.set_index('date')
    
    print(f"数据量: {len(df)} 条")
    print(f"时间范围: {df.index[0]} ~ {df.index[-1]}")
    
    # 计算Spring信号
    print(f"\n计算Spring信号 (high/low Zigzag)...")
    print(f"  参数: threshold=2%, depth=14, backstep=1, 趋势过滤=开启")
    
    spring_signals = find_spring_signals(df, threshold=0.02, depth=14, backstep=1, use_trend_filter=True)
    print(f"  Spring信号数: {len(spring_signals)}")
    
    if len(spring_signals) == 0:
        print("没有找到Spring信号")
        return
    
    # 打印Spring信号
    print(f"\nSpring信号详情:")
    for idx, info in sorted(spring_signals.items()):
        print(f"  {df.index[idx].strftime('%Y-%m-%d')}: 支撑位={info['support_price']:.2f}, 跌破日={info['break_date'].strftime('%Y-%m-%d')}")
    
    # Backtrader回测
    print(f"\n开始回测...")
    
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
    
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='tradeanalyzer')
    
    results = cerebro.run()
    strategy = results[0]
    
    # 输出结果
    print(f"\n{'='*60}")
    print(f"回测结果")
    print(f"{'='*60}")
    
    print(f"\n【账户信息】")
    print(f"  初始资金: ¥1,000,000")
    print(f"  最终资金: ¥{cerebro.broker.getvalue():,.2f}")
    print(f"  总收益率: {(cerebro.broker.getvalue() / 1000000 - 1) * 100:.2f}%")
    
    print(f"\n【交易统计】")
    print(f"  Spring信号总数: {strategy.spring_count}")
    print(f"  实际成交次数: {len(strategy.trades)}")
    
    if strategy.trades:
        trades_df = pd.DataFrame(strategy.trades)
        
        wins = len(trades_df[trades_df['收益率%'] > 0])
        losses = len(trades_df[trades_df['收益率%'] <= 0])
        win_rate = wins / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_return = trades_df['收益率%'].mean()
        
        print(f"  盈利次数: {wins}")
        print(f"  亏损次数: {losses}")
        print(f"  胜率: {win_rate:.2f}%")
        print(f"  平均收益率: {avg_return:.2f}%")
        
        profit_trades = trades_df[trades_df['收益率%'] > 0]['收益率%']
        loss_trades = trades_df[trades_df['收益率%'] <= 0]['收益率%']
        
        if len(profit_trades) > 0:
            print(f"  平均盈利: {profit_trades.mean():.2f}%")
        if len(loss_trades) > 0:
            print(f"  平均亏损: {loss_trades.mean():.2f}%")
        
        print(f"\n【交易记录】")
        print(trades_df.to_string(index=False))
    
    # 风险指标
    print(f"\n【风险指标】")
    drawdown = strategy.analyzers.drawdown.get_analysis()
    if drawdown:
        print(f"  最大回撤: {drawdown.get('max', {}).get('drawdown', 0):.2f}%")
    
    returns = strategy.analyzers.returns.get_analysis()
    if returns:
        print(f"  年化收益率: {returns.get('rnorm100', 0):.2f}%")
    
    sharpe = strategy.analyzers.sharpe.get_analysis()
    if sharpe and sharpe.get('sharperatio'):
        print(f"  夏普比率: {sharpe['sharperatio']:.2f}")
    
    # 画图
    if strategy.trades:
        output_dir = 'data/spring_charts'
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n开始画图...")
        for i, trade in enumerate(strategy.trades, 1):
            spring_date = None
            break_date = None
            support_price = None
            buy_date_str = trade['买入日期']
            
            for idx, info in spring_signals.items():
                if pd.to_datetime(info['buy_date']).strftime('%Y-%m-%d') == buy_date_str:
                    spring_date = pd.to_datetime(info['spring_date'])
                    break_date = pd.to_datetime(info['break_date'])
                    support_price = info['support_price']
                    break
            
            if support_price is None:
                support_price = trade['支撑位']
            
            filepath = draw_trade_chart(df, trade, spring_date, break_date, support_price, output_dir, i)
            if filepath:
                print(f"  已保存: {os.path.basename(filepath)}")
        
        print(f"\n图片已保存到: {output_dir}/")


if __name__ == '__main__':
    run_backtest('600108')

                print(f"\n测试参数: threshold={threshold}, depth={depth}, backstep={backstep}")
                
                # 获取数据
                try:
                    symbol_with_prefix = "sh600108"
                    df = ak.stock_zh_a_daily(
                        symbol=symbol_with_prefix,
                        start_date='20180101',
                        end_date=datetime.now().strftime('%Y%m%d')
                    )
                except:
                    continue
                
                if df is None or len(df) < 100:
                    continue
                
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                df = df.set_index('date')
                
                # 计算Spring信号
                spring_signals = find_spring_signals(df, threshold=threshold, depth=depth, backstep=backstep)
                
                if len(spring_signals) == 0:
                    print(f"  无Spring信号")
                    continue
                
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
                
                try:
                    results_bt = cerebro.run()
                    strategy = results_bt[0]
                except:
                    continue
                
                if not strategy.trades:
                    continue
                
                trades_df = pd.DataFrame(strategy.trades)
                wins = len(trades_df[trades_df['收益率%'] > 0])
                losses = len(trades_df[trades_df['收益率%'] <= 0])
                win_rate = wins / len(trades_df) * 100 if len(trades_df) > 0 else 0
                avg_return = trades_df['收益率%'].mean()
                total_return = (cerebro.broker.getvalue() / 1000000 - 1) * 100
                
                print(f"  信号数: {len(spring_signals)}, 成交: {len(strategy.trades)}, 胜率: {win_rate:.1f}%, 收益: {total_return:.2f}%")
                
                results.append({
                    'threshold': threshold,
                    'depth': depth,
                    'backstep': backstep,
                    'signals': len(spring_signals),
                    'trades': len(strategy.trades),
                    'win_rate': win_rate,
                    'avg_return': avg_return,
                    'total_return': total_return
                })
    
    if results:
        print("\n" + "="*70)
        print("参数优化结果")
        print("="*70)
        
        results_df = pd.DataFrame(results)
        
        # 按收益率排序
        results_df = results_df.sort_values('total_return', ascending=False)
        
        print("\n【按总收益率排序】")
        print(results_df.to_string(index=False))
        
        # 最佳参数
        best = results_df.iloc[0]
        print(f"\n【最佳参数】")
        print(f"  threshold: {best['threshold']}")
        print(f"  depth: {best['depth']}")
        print(f"  backstep: {best['backstep']}")
        print(f"  胜率: {best['win_rate']:.2f}%")
        print(f"  总收益率: {best['total_return']:.2f}%")
