#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
威科夫分析器 - Wyckoff Analysis for A-Share Stocks & ETFs
基于《威科夫操盘法》理论进行股票分析支持短期（日
线）和长期（周线）分析
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

try:
    import akshare as ak
    import pandas as pd
    import numpy as np
except ImportError:
    print("请安装所需库: pip install akshare pandas numpy")
    sys.exit(1)


class WyckoffAnalyzer:
    """威科夫分析器"""
    
    def __init__(self, symbol: str, days: int = 120, is_weekly: bool = False):
        """
        初始化分析器
        
        Args:
            symbol: 股票代码 (如 000001, 600519)
            days: 分析天数
            is_weekly: 是否为周线分析
        """
        self.symbol = symbol
        self.days = days
        self.is_weekly = is_weekly
        self.df = None
        self.stock_name = ""
        self.signals = []
        self.current_phase = ""
        self.trend = ""
        self.risk_level = ""
        
    def fetch_data(self) -> bool:
        """获取股票/ETF历史数据"""
        try:
            period = "weekly" if self.is_weekly else "daily"
            print(f"正在获取 {self.symbol} {'周线' if self.is_weekly else '日线'}数据...")
            
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=self.days + 60)).strftime('%Y%m%d')
            
            # 判断是否为ETF (5开头或15/16/159开头的是ETF)
            is_etf = self.symbol.startswith('5') or self.symbol.startswith('15') or self.symbol.startswith('16') or self.symbol.startswith('159') or self.symbol.startswith('51') or self.symbol.startswith('58')
            
            if is_etf:
                # 使用ETF数据接口 - 新浪接口
                print(f"检测为ETF，使用ETF数据接口...")
                try:
                    # 确定市场前缀 - 159开头用sz，其他用sh
                    if self.symbol.startswith('15') or self.symbol.startswith('16') or self.symbol.startswith('159'):
                        etf_prefix = 'sz'
                    else:
                        etf_prefix = 'sh'
                    
                    self.df = ak.fund_etf_hist_sina(symbol=f"{etf_prefix}{self.symbol}")
                    
                    if self.df is not None and not self.df.empty:
                        # 筛选日期范围
                        self.df['date'] = pd.to_datetime(self.df['date'])
                        start_dt = datetime.strptime(start_date, '%Y%m%d')
                        end_dt = datetime.strptime(end_date, '%Y%m%d')
                        self.df = self.df[(self.df['date'] >= start_dt) & (self.df['date'] <= end_dt)]
                        self.stock_name = f"ETF{self.symbol}"
                    else:
                        raise Exception("ETF数据为空")
                except Exception as e:
                    print(f"ETF接口失败: {e}")
                    return False
            else:
                # 股票数据接口 - 使用东方财富接口
                try:
                    # 需要添加市场前缀: sh/sz
                    symbol_with_prefix = f"sh{self.symbol}" if self.symbol.startswith('6') else f"sz{self.symbol}"
                    self.df = ak.stock_zh_a_daily(
                        symbol=symbol_with_prefix,
                        start_date=start_date,
                        end_date=end_date
                    )
                except Exception as e:
                    print(f"股票接口失败，尝试备用接口: {e}")
                    try:
                        self.df = ak.stock_zh_a_hist(
                            symbol=self.symbol,
                            period=period,
                            start_date=start_date,
                            end_date=end_date,
                            adjust="qfq"
                        )
                    except:
                        self.df = None
                
                # 获取股票名称
                if self.df is not None and not self.df.empty:
                    try:
                        info = ak.stock_individual_info_em(symbol=self.symbol)
                        if info is not None and not info.empty:
                            name_row = info[info['item'] == '股票简称']
                            if not name_row.empty:
                                self.stock_name = name_row['value'].values[0]
                            else:
                                self.stock_name = "未知"
                        else:
                            self.stock_name = "未知"
                    except:
                        self.stock_name = "未知"
            
            if self.df is None or self.df.empty:
                print(f"未能获取 {self.symbol} 的数据")
                return False
            
            # 数据预处理
            print(f"原始数据列名: {list(self.df.columns)}")
            
            # 根据实际列名进行映射 - 新浪接口
            column_mapping = {
                '日期': 'date',
                '股票代码': 'symbol',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change',
                '换手率': 'turnover'
            }
            
            # 重命名列
            self.df = self.df.rename(columns=column_mapping)
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date').reset_index(drop=True)
            
            # 如果是周线分析，需要将日线转换为周线
            # 只有明确指定--weekly时才转换为周线，不根据天数自动转换
            if self.is_weekly and not self.df.empty:
                # 已经是周线模式，转换为周线
                self.df = self._convert_to_weekly(self.df)
            
            # 计算技术指标
            self._calculate_indicators()
            
            print(f"成功获取 {len(self.df)} 条{'周线' if self.is_weekly else '日线'}数据")
            return True
            
        except Exception as e:
            print(f"获取数据失败: {e}")
            return False
    
    def _convert_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """将日线数据转换为周线
        
        使用日历周（Calendar Week）而非ISO周
        每年1月1日所在的周为第1周，12月31日在第52/53周
        这样可以避免ISO周历年末的边界bug
        """
        df = df.copy()
        
        # 使用日历周：每年的第几周
        # isocalendar().week 有年末bug，用 dt.isocalendar().year 也会有问题
        # 使用基于1月1日的周数计算
        df['year'] = df['date'].dt.year
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # 计算calendar week: 第几周 = ceil(第几天 / 7)
        # 但这样不太准确，改用更简单的方法
        # 直接按 1月1日 = 第1周第1天 来计算
        df['week'] = (df['day_of_year'] - 1) // 7 + 1
        
        # 按年+周分组
        weekly = df.groupby(['year', 'week']).agg({
            'date': 'last',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'
        }).reset_index(drop=True)
        
        weekly = weekly.sort_values('date').reset_index(drop=True)
        
        # 计算周涨跌幅
        weekly['change_pct'] = weekly['close'].pct_change() * 100
        weekly['amplitude'] = (weekly['high'] - weekly['low']) / weekly['close'].shift(1) * 100
        
        return weekly
    
    def _calculate_indicators(self):
        """计算技术指标"""
        window = 20 if self.is_weekly else 20
        
        # 成交量均线
        self.df['vol_ma5'] = self.df['volume'].rolling(window=5).mean()
        self.df['vol_ma20'] = self.df['volume'].rolling(window=window).mean()
        
        # 价格均线
        self.df['ma5'] = self.df['close'].rolling(window=5).mean()
        self.df['ma20'] = self.df['close'].rolling(window=window).mean()
        
        if len(self.df) > 60:
            self.df['ma60'] = self.df['close'].rolling(window=60).mean()
        
        # 涨跌幅
        self.df['pct_change'] = self.df['close'].pct_change() * 100
        self.df['change_pct'] = self.df['pct_change']  # 兼容两种列名
        
        # 20日高低点
        self.df['high_20'] = self.df['high'].rolling(20).max()
        self.df['low_20'] = self.df['low'].rolling(20).min()
        
        # 60日高低点
        self.df['high_60'] = self.df['high'].rolling(60).max()
        self.df['low_60'] = self.df['low'].rolling(60).min()
        
        # 振幅
        self.df['amplitude'] = (self.df['high'] - self.df['low']) / self.df['close'] * 100
        
    def detect_signals(self, lookback: int = None):
        """检测威科夫关键信号 - 概率云版
        
        使用概率云方法计算各事件的似然度
        置信度分层: 高置信(>0.8), 中置信(0.6-0.8), 低置信(0.4-0.6)
        """
        if self.df is None or len(self.df) < 20:
            return
        
        # 默认回看周期
        if lookback is None:
            lookback = 104 if self.is_weekly else 90  # 周线2年，日线90天
        
        df = self.df.tail(min(lookback, len(self.df))).copy()
        df = df.reset_index(drop=True)
        
        # 使用概率云计算似然度
        pc = ProbabilityCloud(df)
        self.probability_cloud = pc
        self.likelihood = pc.calculate_all_likelihoods()
        
        # 获取高置信度事件(>0.4)
        events = pc.get_events_above_threshold(0.4)
        
        # 转换为信号格式
        self.signals = []
        for event in events:
            signal = {
                'type': event['type'],
                'name': self._get_event_name(event['type']),
                'date': event['date'],
                'price': event['price'],
                'likelihood': event['likelihood'],
                'confidence': event['confidence'],
                'pct': df.loc[df['date'] == event['date'], 'pct_change'].values[0] if len(df.loc[df['date'] == event['date']]) > 0 else 0,
                'description': f"概率云置信度: {event['confidence']}({event['likelihood']:.2f})"
            }
            self.signals.append(signal)
        
        # 按日期排序
        self.signals = sorted(self.signals, key=lambda x: x['date'], reverse=True)
    
    def _get_event_name(self, event_type: str) -> str:
        """获取事件中文名称"""
        names = {
            'AR': '自动反弹',
            'SPRING': '弹簧效应',
            'SPRING_BEST': '最佳弹簧',
            'SPRING_STRONG': '强劲需求弹簧',
            'SPRING_CONFIRM': '需确认弹簧',
            'SPRING_SHAKEOUT': '震仓弹簧',
            'UT': '上冲回落',
            'SC': '恐慌抛售',
            'ST': '二次测试',
            'SOS': '强势出现',
            'BC': '抢购高潮',
            'SOW': '弱势信号',
            'LPS': '最后支撑点'
        }
        return names.get(event_type, event_type)
    
    def get_phase_from_probability_cloud(self) -> str:
        """基于概率云事件判断当前阶段"""
        if not hasattr(self, 'signals') or not self.signals:
            return self._determine_phase_by_price()
        
        # 获取高置信事件
        high_conf = [s for s in self.signals if s['confidence'] == '高置信']
        event_types = [s['type'] for s in self.signals]
        high_conf_types = [s['type'] for s in high_conf]
        
        # 分类统计
        accumulation = ['SC', 'ST', 'SPRING']
        markup = ['SOS', 'LPS']
        distribution = ['BC', 'UT', 'SOW']
        
        # 计算加权分数（高置信事件权重更高）
        acc_score = sum(s['likelihood'] for s in self.signals if s['type'] in accumulation)
        mark_score = sum(s['likelihood'] for s in self.signals if s['type'] in markup)
        dist_score = sum(s['likelihood'] for s in self.signals if s['type'] in distribution)
        
        # 判断趋势方向
        trend = self.get_trend_direction()
        
        # 判断阶段
        max_score = max(acc_score, mark_score, dist_score)
        
        # 如果高置信事件是上涨/派发事件，优先考虑
        if high_conf_types:
            latest_high = high_conf[0]['type'] if high_conf else None
            if latest_high in markup:
                return "阶段D-上涨趋势"
            elif latest_high in distribution:
                return "阶段D-派发/下跌"
            elif latest_high in accumulation:
                return "阶段C-吸筹确认"
        
        # 基于分数判断
        if mark_score == max_score and mark_score > 0.3:
            return "阶段D-上涨趋势"
        elif acc_score == max_score and acc_score > 0.3:
            return "阶段C-吸筹确认"
        elif dist_score == max_score and dist_score > 0.3:
            return "阶段D-派发/下跌"
        else:
            return self._determine_phase_by_price()
    
    def _determine_phase_by_price(self) -> str:
        """基于价格位置判断阶段"""
        if len(self.df) < 60:
            return "阶段B-盘整"
        
        recent = self.df.tail(60)
        
        # 计算波动率
        volatility = recent['close'].std() / recent['close'].mean()
        
        # 计算成交量变化
        vol_ratio = recent['volume'].tail(20).mean() / recent['volume'].mean() if recent['volume'].mean() > 0 else 1
        
        # 计算趋势
        ma20 = recent['ma20'].iloc[-1] if not pd.isna(recent['ma20'].iloc[-1]) else recent['close'].iloc[-1]
        ma60 = recent['ma60'].iloc[-1] if 'ma60' in recent.columns and not pd.isna(recent['ma60'].iloc[-1]) else ma20
        
        if volatility < 0.08 and vol_ratio < 0.6:
            return "阶段B-吸筹区间"
        elif volatility >= 0.12 and ma20 > ma60:
            return "阶段D-上涨趋势"
        elif volatility >= 0.12 and ma20 < ma60:
            return "阶段D-下跌趋势"
        else:
            return "阶段B-盘整"
    
    def get_trend_from_probability_cloud(self) -> str:
        """基于概率云判断趋势方向"""
        if not hasattr(self, 'signals') or not self.signals:
            # 使用均线判断
            return self.get_trend_direction()
        
        event_types = [s['type'] for s in self.signals]
        
        # 上涨事件
        up_events = ['SOS', 'LPS', 'SPRING']
        # 下跌事件
        down_events = ['SOW', 'BC', 'UT']
        
        up_count = sum(1 for t in event_types if t in up_events)
        down_count = sum(1 for t in event_types if t in down_events)
        
        if up_count > down_count:
            return "上涨"
        elif down_count > up_count:
            return "下跌"
        else:
            return self.get_trend_direction()
    
    def determine_phase(self):
        """判断当前阶段 - 增强版
        
        基于威科夫理论综合判断：
        - 阶段A：下跌趋势结束，准备进入吸筹
        - 阶段B：吸筹区间，区间震荡，成交量萎缩
        - 阶段C：吸筹确认，Spring/终极震仓
        - 阶段D：上涨趋势，SOS/JOC突破
        """
        if not self.signals:
            if len(self.df) < 60:
                self.current_phase = "数据不足"
                return
            
            # 基于技术指标判断阶段
            recent = self.df.tail(60)
            
            # 计算波动率
            volatility = recent['close'].std() / recent['close'].mean()
            
            # 计算成交量变化
            vol_ratio = recent['volume'].iloc[-20:].mean() / recent['volume'].mean() if recent['volume'].mean() > 0 else 1
            
            # 计算价格趋势
            ma20 = recent['ma20'].iloc[-1] if not pd.isna(recent['ma20'].iloc[-1]) else recent['close'].iloc[-1]
            ma60 = recent['ma60'].iloc[-1] if not pd.isna(recent['ma60'].iloc[-1]) else recent['close'].iloc[-1]
            price_trend = (ma20 - ma60) / ma60 * 100 if ma60 > 0 else 0
            
            # 计算近期涨跌
            recent_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0] * 100
            
            # 阶段判断
            if volatility < 0.08 and vol_ratio < 0.6:
                self.current_phase = "阶段B-吸筹区间"
                self.trend = "震荡筑底"
            elif volatility < 0.12 and price_trend > 0 and recent_change > -5:
                self.current_phase = "阶段C-吸筹确认"
                self.trend = "可能转涨"
            elif volatility < 0.12 and price_trend < 0 and recent_change < -10:
                self.current_phase = "阶段A-下跌趋势"
                self.trend = "下跌"
            elif volatility >= 0.12 and price_trend > 3 and recent_change > 5:
                self.current_phase = "阶段D-上涨趋势"
                self.trend = "上涨"
            elif volatility >= 0.12 and price_trend < -3 and recent_change < -5:
                self.current_phase = "阶段D-下跌趋势"
                self.trend = "下跌"
            else:
                self.current_phase = "阶段B-盘整"
                self.trend = "盘整"
            return
        
        signal_types = [s['type'] for s in self.signals]
        
        # 派发信号优先
        if 'BC' in signal_types or 'UT' in signal_types:
            self.current_phase = "派发阶段-注意风险"
            self.trend = "可能见顶"
        elif 'SOW' in signal_types:
            self.current_phase = "下跌阶段"
            self.trend = "下跌"
        elif 'SC' in signal_types or 'ST' in signal_types:
            self.current_phase = "阶段A-熊市结束"
            self.trend = "筑底"
        elif 'Spring' in signal_types:
            self.current_phase = "阶段C-吸筹确认"
            self.trend = "可能转涨"
        elif 'SOS' in signal_types or 'JOC' in signal_types:
            self.current_phase = "阶段D-上涨趋势"
            self.trend = "上涨"
        elif 'LPS' in signal_types:
            self.current_phase = "阶段D-回调后上涨"
            self.trend = "上涨"
        else:
            self.current_phase = "阶段B-盘整"
            self.trend = "盘整"
    
    def analyze_supply_demand(self):
        """分析供需关系"""
        if len(self.df) < 10:
            return "数据不足"
            
        recent = self.df.tail(20)
        
        avg_up_volume = recent[recent['close'] > recent['open']]['volume'].mean()
        avg_down_volume = recent[recent['close'] < recent['open']]['volume'].mean()
        
        if pd.isna(avg_up_volume) or pd.isna(avg_down_volume):
            return "供需平衡"
            
        if avg_up_volume > avg_down_volume * 1.3:
            return "供不应求-需求主导"
        elif avg_down_volume > avg_up_volume * 1.3:
            return "供过于求-供应主导"
        else:
            return "供需平衡"
            
    def analyze_effort_result(self):
        """分析努力与结果"""
        if len(self.df) < 10:
            return "数据不足"
            
        recent = self.df.tail(10)
        
        high_vol_days = recent[recent['volume'] > recent['volume'].mean() * 1.3]
        
        if len(high_vol_days) > 0:
            for _, row in high_vol_days.iterrows():
                if abs(row['change_pct']) < 2:
                    return "背离-努力未产生结果"
                    
        return "正常-量价配合"
    
    def get_recent_signals(self, months: int = 3):
        """获取近N个月的信号"""
        if self.is_weekly:
            days = months * 30
        else:
            days = months * 30
            
        cutoff_date = datetime.now() - timedelta(days=days)
        return [s for s in self.signals if s['date'] >= cutoff_date]
    
    def get_entry_signal(self):
        """判断是否有进场信号"""
        recent = self.get_recent_signals(3)
        
        signal_types = [s['type'] for s in recent]
        
        if 'SOS' in signal_types or 'JOC' in signal_types:
            lps_signals = [s for s in recent if s['type'] == 'LPS']
            if lps_signals:
                return True, "出现SOS/JOC后回调形成LPS，是较好的进场时机"
            else:
                return True, "出现强势突破信号(SOS/JOC)，但建议等待回调再进场"
        elif 'Spring' in signal_types:
            return True, "出现弹簧效应(Spring)，是吸筹确认信号，可考虑进场"
        elif 'ST' in signal_types:
            return True, "二次测试成功，供应枯竭，可能进入上涨趋势"
        
        return False, "当前未出现明确的进场信号"

    def get_trend_direction(self) -> str:
        """获取趋势方向 - 量化判断
        
        上涨趋势: MA20 > MA60 且 MA20向上
        下跌趋势: MA20 < MA60 且 MA20向下  
        横盘震荡: 其他情况
        """
        if len(self.df) < 60:
            return "数据不足"
        
        ma20 = self.df['ma20'].iloc[-1]
        ma60 = self.df['ma60'].iloc[-1] if 'ma60' in self.df.columns and not pd.isna(self.df['ma60'].iloc[-1]) else ma20
        
        if pd.isna(ma20) or pd.isna(ma60):
            return "均线不足"
        
        # 判断MA20方向
        ma20_prev = self.df['ma20'].iloc[-5] if len(self.df) >= 5 else ma20
        ma20_rising = ma20 > ma20_prev if not pd.isna(ma20_prev) else True
        
        if ma20 > ma60 and ma20_rising:
            return "上涨"
        elif ma20 < ma60 and not ma20_rising:
            return "下跌"
        else:
            return "震荡"

    def get_trend_strength(self) -> float:
        """获取趋势强度 (-1 到 1)
        
        计算方法: 
        - 基于价格与均线的关系
        - 基于均线角度
        """
        if len(self.df) < 10:
            return 0.0
        
        latest = self.df.iloc[-1]
        ma5 = self.df['ma5'].iloc[-1]
        ma20 = self.df['ma20'].iloc[-1] if 'ma20' in self.df.columns else ma5
        
        if pd.isna(ma5):
            return 0.0
        
        # 价格与MA5的关系
        price_ma_ratio = (latest['close'] - ma5) / ma5
        
        # 计算N日价格变化（N为数据长度的一半，但不超过60）
        n = min(len(self.df) // 2, 60)
        if len(self.df) >= n and n >= 10:
            price_change = (self.df['close'].iloc[-1] - self.df['close'].iloc[-n]) / self.df['close'].iloc[-n]
        else:
            price_change = 0.0
        
        # 综合评分
        strength = (price_ma_ratio * 0.6 + price_change * 0.4)
        
        # 限制在 -1 到 1 之间
        return max(-1.0, min(1.0, strength))



class ProbabilityCloud:
    """威科夫概率云分析器
    
    为每个威科夫事件计算似然度(0-1)，而非简单二元判断
    置信度分层: 高(>0.8), 中(0.6-0.8), 低(0.4-0.6)
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.likelihood = pd.DataFrame(index=df.index)
        self._prepare_indicators()
        
    def _prepare_indicators(self):
        """准备计算所需的指标"""
        df = self.df
        
        # 均线
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        
        # 成交量均线
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        
        # 20日高低点
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        
        # 60日高低点
        df['high_60'] = df['high'].rolling(60).max()
        df['low_60'] = df['low'].rolling(60).min()
        
        # 振幅
        df['amplitude'] = (df['high'] - df['low']) / df['close'] * 100
        
        # 涨跌幅
        df['pct_change'] = df['close'].pct_change() * 100
        
        # 量比
        df['vol_ratio'] = df['volume'] / df['vol_ma20']
        
        self.df = df
        
    def calculate_all_likelihoods(self) -> pd.DataFrame:
        """计算所有事件的似然度"""
        df = self.df
        
        # SC似然度 - 恐慌抛售 (先计算)
        self._calc_sc_likelihood()
        
        # AR似然度 - 自动反弹(SC后)
        self._calc_ar_likelihood()
        
        # ST似然度 - 二次测试
        self._calc_st_likelihood()
        
        # SPRING似然度 - 跌破支撑后快速收回
        self._calc_spring_likelihood()
        
        # SOS似然度 - 强势出现
        self._calc_sos_likelihood()
        
        # UT似然度 - 上冲回落
        self._calc_ut_likelihood()
        
        # BC似然度 - 抢购高潮
        self._calc_bc_likelihood()
        
        # SOW似然度 - 弱势信号
        self._calc_sow_likelihood()
        
        # LPS似然度 - 最后支撑点
        self._calc_lps_likelihood()
        
        return self.likelihood
    
    def _calc_spring_likelihood(self):
        """计算Spring(弹簧)似然度 - 增强版
        根据wyckoff.md区分四种类型：
        1. 最佳Spring: 振幅<3%, 量比<0.5 -> 直接进场
        2. 强劲需求Spring: 成交量大 + 收盘在50%之上 -> 直接进场 (新增)
        3. 需确认Spring: 振幅3-5%, 量比0.5-1.0 -> 等待二次测试
        4. 震仓Spring: 振幅>5%, 量比>1.0 -> 熊市陷阱
        
        Spring定义：价格迅速突破支撑，然后迅速返回支撑上方
        参考wyckoff_backtest.py的条件 - 使用宽松条件因为A股很少跌破20日低点
        
        新增: 强劲需求Spring判断 (突破支撑时成交量大 + 收盘在50%之上)
        """
        df = self.df
        
        low_20 = df['low_20']
        
        # 宽松条件1: 收盘价接近或跌破20日低点(允许接近)
        close_near_low = ((df['close'] - low_20) / low_20).abs() < 0.02
        
        # 宽松条件2: 日内跌破20日低点(使用最低价)
        low_below = (df['low'] < low_20)
        
        # 条件组合: 接近低点或日内跌破
        near_support = (close_near_low | low_below).astype(float)
        
        # 3日内收回(收盘价回到20日低点上方的任意一天)
        def check_recovery(idx):
            if idx + 3 >= len(df):
                return 0.0
            next_3 = df.iloc[idx:min(idx+4, len(df))]
            return 1.0 if (next_3['close'] > low_20.iloc[idx]).any() else 0.0
        
        recovered = pd.Series([check_recovery(i) for i in range(len(df))], index=df.index)
        
        # 计算当日振幅
        amplitude = (df['high'] - df['low']) / df['close'] * 100
        
        # 计算量比
        vol_ratio = df['vol_ratio']
        
        # 缩量条件(参考wyckoff_backtest.py: vol_ratio < 0.85)
        vol_shrink = (vol_ratio < 0.85).astype(float)
        
        # 基础分数 - 接近支撑 + 收回 + 缩量
        # 参考wyckoff_backtest.py的权重分配
        base_score = (
            near_support * 0.30 +
            recovered * 0.30 +
            vol_shrink * 0.20
        )
        
        # === 四种Spring的似然度 ===
        
        # 类型1: 最佳Spring (振幅<3%, 量比<0.5)
        best_spring = (
            (amplitude < 3).astype(float) * 0.5 +
            (vol_ratio < 0.5).astype(float) * 0.5
        )
        self.likelihood['SPRING_BEST'] = base_score * best_spring
        
        # 类型2: 强劲需求Spring (成交量大 + 收盘在50%之上) - 新增
        # wyckoff.md: "成交量扩大, 但是收盘在中点之上" = 强劲需求, 可直接进场
        # 条件: 放量(量比>1.0) + 收盘在50%之上
        close_above_mid = (df['close'] > (df['high'] + df['low']) / 2).astype(float)
        vol_expand = (vol_ratio > 1.0).astype(float)
        
        strong_demand_spring = (
            vol_expand * 0.5 +
            close_above_mid * 0.5
        )
        # 强劲需求Spring不需要缩量条件, 反而需要放量
        strong_demand_base = (
            near_support * 0.30 +
            recovered * 0.30 +
            vol_expand * 0.20  # 放量作为基础条件
        )
        self.likelihood['SPRING_STRONG'] = strong_demand_base * strong_demand_spring
        
        # 类型3: 需确认Spring (振幅3-5%, 量比0.5-1.0)
        confirm_spring = (
            ((amplitude >= 3) & (amplitude < 5)).astype(float) * 0.4 +
            ((vol_ratio >= 0.5) & (vol_ratio < 1.0)).astype(float) * 0.6
        )
        self.likelihood['SPRING_CONFIRM'] = base_score * confirm_spring
        
        # 类型4: 震仓Spring (振幅>5%, 量比>1.0) - 危险信号
        shakeout_spring = (
            (amplitude > 5).astype(float) * 0.5 +
            (vol_ratio > 1.0).astype(float) * 0.5
        )
        self.likelihood['SPRING_SHAKEOUT'] = base_score * shakeout_spring
        
        # 综合Spring似然度(用于排序) - 包含强劲需求Spring
        self.likelihood['SPRING'] = (
            self.likelihood['SPRING_BEST'] * 1.0 +
            self.likelihood['SPRING_STRONG'] * 1.0 +  # 强劲需求Spring等权重
            self.likelihood['SPRING_CONFIRM'] * 0.7 +
            self.likelihood['SPRING_SHAKEOUT'] * 0.3
        )
        
    def _calc_ut_likelihood(self):
        """计算UT(上冲回落)似然度"""
        df = self.df
        
        # 条件1: 突破20日高点
        above_high = (df['close'] > df['high_20']).astype(float)
        
        # 条件2: 突破后回落
        pullback = (df['close'] < df['high_20'].shift(1)).astype(float)
        
        # 条件3: 上影线
        upper_shadow = ((df['high'] - df[['close', 'open']].max(axis=1)) / 
                       (df['high'] - df['low'] + 0.001))
        shadow_score = (upper_shadow / 0.5).clip(0, 1)
        
        # 条件4: 成交量放大
        vol_expand = ((df['vol_ratio'] - 1) / 1.5).clip(0, 1)
        
        self.likelihood['UT'] = (
            above_high * 0.30 + 
            pullback * 0.25 + 
            shadow_score * 0.25 + 
            vol_expand * 0.20
        )
        
    def _calc_sc_likelihood(self):
        """计算SC(恐慌抛售)似然度 - 优化参数
        条件: 近期下跌趋势, 当日下跌, 放量, 宽振幅
        """
        df = self.df
        
        # 条件1: 近期下跌趋势(15日内)
        price_decline = df['close'].pct_change(15)
        decline_score = (-price_decline / 0.10).clip(0, 1)
        
        # 条件2: 当日下跌
        down = (df['pct_change'] < -2).astype(float)
        drop_score = (-df['pct_change'] / 6).clip(0, 1)
        
        # 条件3: 放量
        vol_score = ((df['vol_ratio'] - 1) / 1.2).clip(0, 1)
        
        # 条件4: 宽振幅或长下影
        amp_score = (df['amplitude'] / 8).clip(0, 1)
        lower_shadow = (df['low'] - df[['close', 'open']].min(axis=1)) / (df['high'] - df['low'] + 0.001)
        shadow_score = (lower_shadow / 0.5).clip(0, 1)
        
        self.likelihood['SC'] = (
            decline_score * 0.20 + 
            down * 0.25 +
            drop_score * 0.20 + 
            vol_score * 0.20 + 
            (amp_score * 0.075 + shadow_score * 0.075)
        )
    
    def _calc_ar_likelihood(self):
        """计算AR(自动反弹)似然度 - 优化参数
        条件: SC后1-10天内反弹, 有量, 收阳
        """
        df = self.df
        
        # 条件1: 近期有SC
        sc_exists = (self.likelihood['SC'] > 0.4).rolling(10).max().fillna(0)
        
        # 条件2: 反弹(收盘价高于开盘价)
        bounce = (df['close'] > df['open']).astype(float)
        
        # 条件3: 成交量不太低
        vol_ok = (df['vol_ratio'] > 0.6).astype(float)
        
        # 条件4: 上涨
        up = (df['pct_change'] > 0).astype(float)
        
        self.likelihood['AR'] = (
            sc_exists * 0.35 +
            bounce * 0.25 +
            vol_ok * 0.20 +
            up * 0.20
        )
        
    def _calc_st_likelihood(self):
        """计算ST(二次测试)似然度"""
        df = self.df
        
        # 条件1: 接近前期低点
        near_low = ((df['close'] - df['low_20']) / df['low_20']).abs()
        low_score = (1 - (near_low / 0.08)).clip(0, 1)
        
        # 条件2: 成交量萎缩
        vol_shrink = (1 - df['vol_ratio']).clip(0, 1)
        
        # 条件3: 振幅小
        amp_score = (1 - df['amplitude'] / 8).clip(0, 1)
        
        # 条件4: 在SC/AR之后(更宽松)
        sc_exists = (self.likelihood['SC'] > 0.4).rolling(25).max().fillna(0)
        ar_exists = (self.likelihood['AR'] > 0.4).rolling(25).max().fillna(0)
        
        self.likelihood['ST'] = (
            low_score * 0.30 + 
            vol_shrink * 0.30 + 
            amp_score * 0.20 +
            (sc_exists + ar_exists) * 0.10
        )
        
    def _calc_sos_likelihood(self):
        """计算SOS(强势出现)似然度 - 优化参数
        条件: 涨幅>4%, 放量>1.8, 突破收高, 连续3日涨
        """
        df = self.df
        
        # 条件1: 涨幅>4%
        up_score = (df['pct_change'] / 8).clip(0, 1)
        
        # 条件2: 放量>1.8
        vol_score = ((df['vol_ratio'] - 1) / 1.5).clip(0, 1)
        
        # 条件3: 突破20日高点并收于高位
        break_high = (df['close'] > df['high_20'].shift(1)).astype(float)
        near_high = ((df['close'] - df['low']) / (df['high'] - df['low'] + 0.001)).clip(0, 1)
        
        # 条件4: 连续3日上涨
        consecutive_up = (df['pct_change'] > 0).rolling(3).sum() >= 3
        consec_score = consecutive_up.astype(float)
        
        self.likelihood['SOS'] = (
            up_score * 0.20 +
            vol_score * 0.25 + 
            break_high * 0.25 +
            near_high * 0.15 +
            consec_score * 0.15
        )
        
    def _calc_bc_likelihood(self):
        """计算BC(抢购高潮)似然度 - 优化参数
        条件: 前期涨幅>12%, 涨幅>5%, 巨量>2.2, 上影线
        """
        df = self.df
        
        # 条件1: 前期明显上涨
        prior_up = (df['close'] > df['close'].shift(20) * 1.12).astype(float)
        
        # 条件2: 大阳线>5%
        big_up = (df['pct_change'] > 5).astype(float)
        up_score = (df['pct_change'] / 10).clip(0, 1)
        
        # 条件3: 巨量>2.2
        huge_vol = (df['vol_ratio'] > 2.2).astype(float)
        vol_score = ((df['vol_ratio'] - 1) / 2).clip(0, 1)
        
        # 条件4: 上影线明显
        upper_shadow = ((df['high'] - df['close']) / (df['high'] - df['low'] + 0.001))
        shadow_score = (upper_shadow / 0.5).clip(0, 1)
        
        self.likelihood['BC'] = (
            prior_up * 0.20 +
            big_up * 0.25 +
            up_score * 0.15 +
            huge_vol * 0.25 +
            vol_score * 0.15 +
            shadow_score * 0.15
        )
    
    def _calc_ut_likelihood(self):
        """计算UT(上冲回落)似然度 - 优化参数
        条件: 突破20日高, 回落, 上影线>60%, 放量>1.5
        """
        df = self.df
        
        # 条件1: 突破20日高点
        break_high = (df['high'] > df['high_20'].shift(1)).astype(float)
        
        # 条件2: 收盘在高点下方
        pullback = (df['close'] < df['high_20'].shift(1)).astype(float)
        
        # 条件3: 上影线>60%
        upper_shadow = ((df['high'] - df['close']) / (df['high'] - df['low'] + 0.001))
        shadow_score = (upper_shadow / 0.7).clip(0, 1)
        
        # 条件4: 放量>1.5
        vol_score = ((df['vol_ratio'] - 1) / 1.2).clip(0, 1)
        
        self.likelihood['UT'] = (
            break_high * 0.20 +
            pullback * 0.25 +
            shadow_score * 0.30 +
            vol_score * 0.25
        )
        
    def _calc_sow_likelihood(self):
        """计算SOW(弱势信号)似然度 - 优化参数
        条件: 跌破20日低, 放量>1.5, 跌幅>2.5%
        """
        df = self.df
        
        # 条件1: 跌破20日支撑
        break_support = (df['close'] < df['low_20'].shift(1)).astype(float)
        
        # 条件2: 放量下跌>1.5
        down_vol = ((df['pct_change'] < 0) & (df['vol_ratio'] > 1.5)).astype(float)
        vol_score = ((df['vol_ratio'] - 1) / 1.2).clip(0, 1)
        
        # 条件3: 明显跌幅>2.5%
        big_down = (df['pct_change'] < -2.5).astype(float)
        down_score = (-df['pct_change'] / 6).clip(0, 1)
        
        # 条件4: 前期已上涨
        prior_up = (df['close'] > df['close'].shift(20) * 1.08).astype(float)
        
        self.likelihood['SOW'] = (
            break_support * 0.30 +
            down_vol * 0.25 +
            big_down * 0.25 +
            prior_up * 0.20
        )
        
    def _calc_lps_likelihood(self):
        """计算LPS(最后支撑点)似然度 - 优化参数
        条件: SOS/BC后5-20天, 缩量<0.6, 回调3-8%
        """
        df = self.df
        
        # 条件1: 在SOS或BC之后5-20天
        sos_after = (self.likelihood['SOS'].shift(range(5, 21)).max(axis=1) > 0.4).astype(float)
        bc_after = (self.likelihood['BC'].shift(range(5, 21)).max(axis=1) > 0.4).astype(float)
        
        # 条件2: 明显缩量<0.6
        vol_shrink = (df['vol_ratio'] < 0.6).astype(float)
        
        # 条件3: 小幅回调3-8%
        pullback = ((df['close'].shift(1) - df['close']) / df['close'].shift(1))
        pullback_score = ((pullback > 0.03) & (pullback < 0.08)).astype(float)
        
        # 条件4: 振幅小<4%
        small_amp = (df['amplitude'] < 4).astype(float)
        
        self.likelihood['LPS'] = (
            (sos_after + bc_after) * 0.20 +
            vol_shrink * 0.30 +
            pullback_score * 0.25 +
            small_amp * 0.25
        )
        
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
    
    def get_events_above_threshold(self, threshold: float = 0.4) -> list:
        """获取高于阈值的事件"""
        events = []
        for col in self.likelihood.columns:
            max_likelihood = self.likelihood[col].max()
            if max_likelihood >= threshold:
                idx = self.likelihood[col].idxmax()
                events.append({
                    'type': col,
                    'likelihood': max_likelihood,
                    'confidence': self.get_confidence_level(max_likelihood),
                    'date': self.df.loc[idx, 'date'],
                    'price': self.df.loc[idx, 'close']
                })
        return sorted(events, key=lambda x: x['likelihood'], reverse=True)
    
    # ==================== Spring增强检测方法 ====================
    
    def _get_market_context(self, idx: int) -> str:
        """判断市场背景 - 用于Spring过滤
        返回: BULL(牛市), ACCUMULATION(吸筹), CONSOLIDATION(震荡), BEAR(熊市)
        """
        if idx < 60:
            return 'UNKNOWN'
        
        df = self.df
        recent = df.iloc[max(0, idx-60):idx]
        
        ma20 = recent['ma20'].iloc[-1] if not pd.isna(recent['ma20'].iloc[-1]) else recent['close'].iloc[-1]
        ma60 = recent['ma60'].iloc[-1] if 'ma60' in recent.columns and not pd.isna(recent['ma60'].iloc[-1]) else ma20
        
        volatility = recent['close'].std() / recent['close'].mean()
        vol_ratio = recent['volume'].iloc[-20:].mean() / recent['volume'].mean() if recent['volume'].mean() > 0 else 1
        
        if ma20 > ma60 and volatility >= 0.06:
            return 'BULL'
        elif ma20 < ma60 and volatility >= 0.06:
            return 'BEAR'
        elif volatility < 0.08 and vol_ratio < 0.7:
            return 'CONSOLIDATION'
        else:
            return 'CONSOLIDATION'
    
    def _check_demand_followup(self, idx: int, days: int = 3) -> bool:
        """验证需求跟随 - Spring后N天内必须有需求跟随"""
        if idx + days >= len(self.df):
            return False
        
        df = self.df
        entry_price = df.iloc[idx]['close']
        
        future = df.iloc[idx+1:idx+days+1]
        price_up = future['close'].max() > entry_price * 1.01
        
        entry_vol = df.iloc[idx]['volume']
        avg_future_vol = future['volume'].mean()
        vol_expand = avg_future_vol > entry_vol * 0.8
        
        return price_up or vol_expand
    
    def _get_spring_position(self, idx: int) -> str:
        """判断Spring发生的位置"""
        if idx < 30:
            return 'ORDINARY'
        
        df = self.df
        
        for i in range(max(0, idx-30), idx):
            if df.iloc[i]['close'] > df.iloc[i]['high_20']:
                if df.iloc[idx]['close'] < df.iloc[idx]['high_20']:
                    return 'JOC_BACKTEST'
        
        if idx >= 20:
            lows = df.iloc[max(0, idx-20):idx]['low']
            if len(lows) >= 10:
                recent_lows = lows.tail(5).mean()
                early_lows = lows.head(5).mean()
                if recent_lows > early_lows * 0.98:
                    return 'ACCUMULATION_BOTTOM'
        
        ma20 = df.iloc[idx]['ma20']
        ma60 = df.iloc[idx]['ma60'] if 'ma60' in df.columns and not pd.isna(df.iloc[idx]['ma60']) else ma20
        
        if ma20 > ma60:
            highs = df.iloc[max(0, idx-20):idx]['high']
            if len(highs) >= 10:
                recent_highs = highs.tail(5).mean()
                early_highs = highs.head(5).mean()
                if recent_highs < early_highs:
                    return 'PULLBACK'
        
        return 'ORDINARY'
    
    def detect_spring_enhanced(self, idx: int, require_bull_market: bool = True, 
                               require_demand_followup: bool = True, 
                               followup_days: int = 3) -> dict:
        """增强版Spring检测"""
        if idx < 25:
            return {'detected': False, 'reason': '数据不足'}
        
        likelihood = self.likelihood.iloc[idx]
        
        spring_best = likelihood.get('SPRING_BEST', 0)
        spring_strong = likelihood.get('SPRING_STRONG', 0)
        
        if spring_strong < 0.3:
            return {'detected': False, 'reason': '似然度不足'}
        
        if require_bull_market:
            market_context = self._get_market_context(idx)
            if market_context == 'BEAR':
                return {'detected': False, 'reason': '熊市背景', 'market_context': market_context}
        
        position = self._get_spring_position(idx)
        
        if require_demand_followup:
            has_followup = self._check_demand_followup(idx, followup_days)
            if not has_followup:
                return {'detected': False, 'reason': '无需求跟随', 'position': position}
        
        return {
            'detected': True,
            'spring_type': 'SPRING_STRONG',
            'score': spring_strong,
            'position': position,
            'market_context': self._get_market_context(idx) if require_bull_market else 'ANY',
            'demand_followup': self._check_demand_followup(idx, followup_days) if require_demand_followup else True
        }


def analyze_symbol(symbol: str, days: int = 120, is_weekly: bool = False, signal_months: int = 6) -> dict:
    """分析单个股票/ETF - 概率云版
    
    参数:
        symbol: 股票代码
        days: 数据天数 (日线默认120天=6个月, 周线默认500天=2年)
        is_weekly: 是否为周线
        signal_months: 信号显示月数 (默认6个月)
    """
    analyzer = WyckoffAnalyzer(symbol, days, is_weekly)
    
    if not analyzer.fetch_data():
        return None
    
    # 使用概率云检测信号
    analyzer.detect_signals()
    
    # 使用概率云判断阶段和趋势
    phase = analyzer.get_phase_from_probability_cloud()
    trend = analyzer.get_trend_from_probability_cloud()
    
    return {
        'symbol': symbol,
        'name': analyzer.stock_name,
        'df': analyzer.df,
        'signals': analyzer.signals,
        'phase': phase,
        'trend': trend,
        'supply_demand': analyzer.analyze_supply_demand(),
        'effort_result': analyzer.analyze_effort_result(),
        'entry_signal': analyzer.get_entry_signal(),
        'is_weekly': analyzer.is_weekly,
        'likelihood': analyzer.likelihood if hasattr(analyzer, 'likelihood') else None,
        'signal_months': signal_months
    }


def generate_report(result: dict, period_name: str = "短期") -> str:
    """生成分析报告 - 用户模板格式
    
    显示:
    - 事件表格
    - 威科夫解读
    - 进场建议
    """
    if result is None:
        return "无法生成报告：数据获取失败"
    
    df = result['df']
    if df is None or df.empty:
        return "无法生成报告：数据获取失败"
    
    latest = df.iloc[-1]
    period = result['is_weekly']
    signal_months = result.get('signal_months', 6)
    
    # 获取近N个月的信号
    months_days = signal_months * 30
    recent_date = df.iloc[-1]['date'] if 'date' in df.columns else datetime.now()
    cutoff_date = recent_date - timedelta(days=months_days)
    recent_signals = [s for s in result['signals'] if s['date'] >= cutoff_date]
    
    entry_signal, entry_reason = result['entry_signal']
    
    # 计算技术指标
    ma5 = df['ma5'].iloc[-1] if 'ma5' in df.columns else 0
    ma20 = df['ma20'].iloc[-1] if 'ma20' in df.columns else 0
    ma60 = df['ma60'].iloc[-1] if 'ma60' in df.columns else 0
    
    # 事件名称映射
    event_names = {
        'SC': '恐慌抛售',
        'AR': '自动反弹',
        'ST': '二次测试',
        'SPRING': '弹簧',
        'SPRING_BEST': '最佳弹簧',
        'SPRING_STRONG': '强劲弹簧',
        'SPRING_CONFIRM': '需确认弹簧',
        'SOS': '强势出现',
        'BC': '抢购高潮',
        'UT': '上冲回落',
        'SOW': '弱势信号',
        'LPS': '最后支撑'
    }
    
    # 分类信号
    bullish_signals = [s for s in recent_signals if s['type'] in ['SPRING', 'SPRING_BEST', 'SPRING_STRONG', 'SOS', 'AR', 'ST', 'LPS']]
    bearish_signals = [s for s in recent_signals if s['type'] in ['BC', 'UT', 'SOW', 'SC']]
    
    # 按日期排序（正序）
    recent_signals_sorted = sorted(recent_signals, key=lambda x: x['date'])
    
    # 构建信号表格
    signal_table = ""
    if recent_signals_sorted:
        for sig in recent_signals_sorted:
            name = event_names.get(sig['type'], sig['name'])
            conf = "高" if sig['confidence'] == "高置信" else "中" if sig['confidence'] == "中置信" else "低"
            signal_table += f"| {name} | {sig['date'].strftime('%Y-%m-%d')} | ¥{sig['price']:.3f} | {conf} |\n"
    else:
        signal_table = "| - | - | - | - |\n"
    
    # 威科夫解读
    interpretation = ""
    if recent_signals_sorted:
        if 'SPRING' in [s['type'] for s in recent_signals_sorted] or 'SPRING_STRONG' in [s['type'] for s in recent_signals_sorted]:
            interpretation += "1. 前期出现弹簧效应(Spring)，主力测试支撑后快速收回，吸筹可能性增加\n"
        if 'SC' in [s['type'] for s in recent_signals_sorted]:
            interpretation += "1. 前期出现恐慌抛售(SC)，恐慌盘被主力接走\n"
        if 'AR' in [s['type'] for s in recent_signals_sorted]:
            interpretation += "2. 自动反弹(AR)建立交易区间\n"
        if 'ST' in [s['type'] for s in recent_signals_sorted]:
            interpretation += "2. 二次测试(ST)确认供应枯竭\n"
        if 'SOS' in [s['type'] for s in recent_signals_sorted]:
            interpretation += "3. 强势出现(SOS)突破阻力，需求主导市场\n"
        if 'LPS' in [s['type'] for s in recent_signals_sorted]:
            interpretation += "4. 形成最后支撑点(LPS)，回调后上涨动能仍在\n"
        if 'BC' in [s['type'] for s in recent_signals_sorted]:
            interpretation += "3. 出现抢购高潮(BC)，注意短期派发风险\n"
        if 'UT' in [s['type'] for s in recent_signals_sorted]:
            interpretation += "3. 出现上冲回落(UT)，可能形成短期顶部\n"
        if 'SOW' in [s['type'] for s in recent_signals_sorted]:
            interpretation += "4. 出现弱势信号(SOW)，供应增加\n"
    
    period_label = "周线" if period else "日线"
    
    # 处理解读文本
    if not interpretation:
        interpretation_text = "  暂无明显信号，等待进一步观察\n"
    else:
        interpretation_text = interpretation
    
    report = f"""
======================================================================
                    威科夫分析报告 - {result['name']}
======================================================================

【基本信息】
• 股票代码: {result['symbol']}
• 股票名称: {result['name']}
• 分析周期: {df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['date'].iloc[-1].strftime('%Y-%m-%d')}
• 当前价格: ¥{latest['close']:.3f}
• 涨跌幅: {latest['pct_change']:+.2f}%

======================================================================
                         短期分析（{period_label}）
======================================================================

【威科夫阶段】
┌──────────────────────────────────────────────────────────────────┐
│ 当前阶段: {result['phase']:<50}│
│ 趋势方向: {result['trend']:<50}│
└──────────────────────────────────────────────────────────────────┘

【威科夫事件序列】（近{signal_months}个月）

| 事件 | 日期 | 价格 | 置信度 |
|------|------|------|--------|
{signal_table}
【威科夫解读】

{interpretation_text}
【进场建议】
{entry_reason if entry_reason else "当前未出现明确的进场信号"}

======================================================================
"""
    
    return report


def generate_long_term_report(weekly_result: dict) -> str:
    """生成长期分析报告"""
    if weekly_result is None:
        return ""
    
    df = weekly_result['df']
    if df is None or df.empty:
        return ""
    
    latest = df.iloc[-1]
    
    # 事件名称映射
    event_names = {
        'SC': '恐慌抛售',
        'AR': '自动反弹',
        'ST': '二次测试',
        'SPRING': '弹簧',
        'SPRING_BEST': '最佳弹簧',
        'SPRING_STRONG': '强劲弹簧',
        'SPRING_CONFIRM': '需确认弹簧',
        'SOS': '强势出现',
        'BC': '抢购高潮',
        'UT': '上冲回落',
        'SOW': '弱势信号',
        'LPS': '最后支撑'
    }
    
    # 长期信号
    weekly_signals = weekly_result.get('signals', [])[:5]
    
    long_signals = ""
    if weekly_signals:
        for sig in weekly_signals:
            name = event_names.get(sig['type'], sig['name'])
            conf = "高" if sig['confidence'] == "高置信" else "中" if sig['confidence'] == "中置信" else "低"
            long_signals += f"  • {sig['date'].strftime('%Y-%m-%d')}: {name}({conf})\n"
    else:
        long_signals = "  暂无明显信号\n"
    
    report = f"""
======================================================================
                         长期分析（周线）
======================================================================

【长期阶段】
  当前阶段: {weekly_result['phase']}
  趋势方向: {weekly_result['trend']}

【长期信号】
{long_signals}
【趋势判断】
  主要趋势: {weekly_result['trend']}

"""
    
    return report


def generate_conclusion(daily_result: dict, weekly_result: dict) -> str:
    """生成综合结论"""
    daily_phase = daily_result['phase']
    weekly_phase = weekly_result['phase']
    daily_trend = daily_result['trend']
    weekly_trend = weekly_result['trend']
    
    # 综合判断
    if "上涨" in daily_trend and "上涨" in weekly_trend:
        conclusion = "日周线共振上涨，建议积极做多"
        risk = "中等"
    elif "下跌" in daily_trend and "下跌" in weekly_trend:
        conclusion = "日周线共振下跌，建议观望或做空"
        risk = "高"
    elif "上涨" in daily_trend and "派发" in weekly_phase:
        conclusion = "日线上涨但周线有派发风险，建议谨慎做多"
        risk = "中高"
    elif "下跌" in daily_trend and "吸筹" in weekly_phase:
        conclusion = "日线下跌但周线有吸筹迹象，关注支撑位"
        risk = "中"
    else:
        conclusion = "日周线趋势不明，建议观望等待"
        risk = "中"
    
    entry_signal, entry_reason = daily_result['entry_signal']
    
    # 统计
    daily_events = set(s['type'] for s in daily_result['signals'])
    accumulation = {'SC', 'ST', 'SPRING', 'AR'}
    markup = {'SOS', 'LPS'}
    distribution = {'BC', 'UT', 'SOW'}
    
    acc_day = len(daily_events & accumulation)
    up_day = len(daily_events & markup)
    down_day = len(daily_events & distribution)
    
    report = f"""
======================================================================
                         综合结论
======================================================================

  日线判断: {daily_phase} / {daily_trend}
  周线判断: {weekly_phase} / {weekly_trend}

【综合分析】
  {conclusion}
  日线事件: 吸筹{acc_day}个 / 上涨{up_day}个 / 派发{down_day}个

【操作建议】
  {'✅ 建议进场 / 持有 - ' + entry_reason if entry_signal else '⚠️ 建议观望'}

【风险提示】
  风险等级: {risk}
  • 短期涨幅过大时注意回调风险
  • 有效跌破20日均线应考虑止损
  • 关注成交量变化，缩量上涨需警惕

======================================================================
"""
    
    return report


def main():
    """主函数 - 威科夫概率云双周期分析
    
    默认:
    - 日线: 近6个月数据
    - 周线: 近2年数据
    - 信号显示: 6个月
    """
    parser = argparse.ArgumentParser(description='威科夫分析器（概率云版）')
    parser.add_argument('symbol', nargs='?', help='股票代码')
    parser.add_argument('days', nargs='?', type=int, default=120, help='日线分析天数(默认120=6个月)')
    parser.add_argument('--weekly-days', type=int, default=500, help='周线分析天数(默认500=2年)')
    parser.add_argument('--signal-months', type=int, default=6, help='信号显示月数(默认6个月)')
    
    args = parser.parse_args()
    
    if not args.symbol:
        print("用法: python wyckoff_analysis.py <代码> [天数]")
        print("示例: python wyckoff_analysis.py 600108")
        print("       python wyckoff_analysis.py 510100")
        sys.exit(1)
    
    symbol = args.symbol
    daily_days = args.days
    weekly_days = args.weekly_days
    signal_months = args.signal_months
    
    print("\n" + "="*70)
    print("              威科夫双周期分析报告")
    print("="*70)
    
    # 日线分析（小周期）
    print(f"\n[日线分析] 数据范围: 近{daily_days}天")
    daily_result = analyze_symbol(symbol, daily_days, is_weekly=False, signal_months=signal_months)
    
    # 周线分析（大周期）
    print(f"[周线分析] 数据范围: 近{weekly_days}天\n")
    weekly_result = analyze_symbol(symbol, weekly_days, is_weekly=True, signal_months=signal_months)
    
    if not daily_result or not weekly_result:
        print("数据获取失败")
        sys.exit(1)
    
    # 生成日线报告
    print(generate_report(daily_result, "日线"))
    
    # 生成周线报告
    print(generate_long_term_report(weekly_result))
    
    # 生成综合结论
    print(generate_conclusion(daily_result, weekly_result))


if __name__ == "__main__":
    main()
