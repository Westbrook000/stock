"""Wyckoff price/volume events against already-known day0 horizontal lines.

Candidates, confirmations, tests and failures are separate causal events. Scores are
annotations; entry eligibility is an explicit boolean based on the pattern rules.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Event:
    etype: str
    idx: int
    confirm_idx: int
    price: float
    quality: float = 1.0
    kind: str = ''
    failed_idx: Optional[int] = None  # legacy read compatibility; new events never backfill this
    meta: Dict = field(default_factory=dict)


def trend_state(close, ema20, ema60):
    result = np.zeros(len(close), dtype=int)
    result[(close > ema20) & (ema20 > ema60)] = 1
    result[(close < ema20) & (ema20 < ema60)] = -1
    return result


class EventScanner:
    def __init__(self, df, lines, cfg):
        self.n, self.cfg, self.lines = len(df), cfg, lines
        for attr, col in [('o','open'), ('h','high'), ('l','low'), ('c','close'), ('v','volume')]:
            setattr(self, attr, df[col].to_numpy(float))
        self.atr = df['atr'].to_numpy(float) if 'atr' in df else self.h-self.l
        self.ema20 = df['ema20'].to_numpy(float)
        self.ema60 = df['ema60'].to_numpy(float)
        self.pos60 = df['pos_in_60'].to_numpy(float)
        self.close_pos = df['close_pos'].to_numpy(float)
        self.trend = trend_state(self.c, self.ema20, self.ema60)
        self.support, self.res = lines.support, lines.resistance
        self.events: List[Event] = []
        self.sc_candidates, self.resolved_sc = [], set()
        self.failed_parents, self.tested_springs = set(), set()

    def avg_vol_before(self, t, k=None):
        k = self.cfg.volume.ma_window if k is None else k
        return float(np.mean(self.v[max(0,t-k):t])) if t else float(self.v[0])

    def _atr(self, t):
        return max(float(self.atr[max(0,t)]), 1e-9)

    def _volume_ratio(self, t):
        return float(self.v[t]/max(self.avg_vol_before(t),1e-9))

    def big_down(self, t):
        return self.c[t]<self.o[t] and self.h[t]-self.l[t]>=1.5*self._atr(t-1)

    def _last(self, types, t, only_active=False):
        types = (types,) if isinstance(types,str) else types
        for event in reversed(self.events):
            if event.etype not in types or event.confirm_idx>t: continue
            if event.failed_idx is not None and event.failed_idx<=t: continue
            if only_active and event.etype in ('SPRING','SC'):
                key='spring_idx' if event.etype=='SPRING' else 'sc_idx'
                failure=event.etype+'_FAIL'
                if any(e.confirm_idx<=t and e.meta.get(key)==event.idx
                       and (e.etype==failure or (event.etype=='SPRING' and e.etype=='TEST')) for e in self.events):
                    continue
            return event
        return None

    def _recent(self, types, within, t):
        event=self._last(types,t)
        return event is not None and t-event.confirm_idx<=within

    def _tolerance(self, ref, idx):
        c=self.cfg.spring
        return float(np.clip(c.test_tolerance_atr*self._atr(idx)/max(ref,1e-9),c.min_test_tolerance,c.max_test_tolerance))

    def _reference(self, idx, side='support'):
        value=float((self.support if side=='support' else self.res)[idx])
        ids=getattr(self.lines,side+'_id',np.full(self.n,-1))
        counts=getattr(self.lines,side+'_count',np.ones(self.n,dtype=int))
        return value,int(ids[idx]),int(counts[idx])

    def _context(self, t):
        return (self.trend[t]>=0 or self.c[t]>=self.ema60[t]
                or self._recent(('ST','TEST','SOS','JOC'),60,t))

    def _sc_events_at(self,t):
        cfg=self.cfg.climax
        output=[]
        previous=self.c[max(0,t-10):t]
        falling=(self.trend[t-1]<0 or (len(previous) and previous[0]-self.c[t-1]>=2*self._atr(t-1)))
        ref,line_id,count=self._reference(t-1)
        past_low=float(min(self.l[max(0,t-cfg.lookback):t]))
        new_low=self.l[t]<=past_low+0.1*self._atr(t-1)
        near_line=(np.isfinite(ref) and self.l[t]<=ref+0.5*self._atr(t-1)
                   and self.h[t]>=ref-0.5*self._atr(t-1))
        spread=self.h[t]-self.l[t]
        if (not self._recent('SC',cfg.cooldown,t) and falling and np.isfinite(ref) and near_line
                and self.c[t]<self.c[t-1] and spread>=cfg.range_atr*self._atr(t-1)
                and self._volume_ratio(t)>=cfg.volume_ratio):
            reference=ref
            meta={'reference_price':float(reference),'line_id':line_id,'support_count':count,
                  'support_source':'day0',
                  'reference_asof':t-1,'vol':float(self.v[t]),'range':float(spread),
                  'climax_low':float(self.l[t]),'climax_high':float(self.h[t]),
                  'midpoint':float((self.h[t]+self.l[t])/2),'atr':self._atr(t-1),'entry_allowed':False}
            candidate=Event('SC_CANDIDATE',t,t,float(self.l[t]),kind='panic',meta=meta)
            self.sc_candidates.append(candidate)
            output.append(candidate)
        for candidate in reversed(self.sc_candidates):
            if candidate.idx in self.resolved_sc: continue
            age=t-candidate.idx
            if age>cfg.confirm_bars:
                self.resolved_sc.add(candidate.idx)
                continue
            if age<0: continue
            m=candidate.meta
            if min(self.l[candidate.idx:t+1])<m['climax_low']-0.2*m['atr']:
                self.resolved_sc.add(candidate.idx)
                continue
            lower_wick=min(self.o[t],self.c[t])-self.l[t]
            stopping=(age==0 and self.close_pos[t]>=cfg.stopping_close_position
                      and (self.c[t]>self.o[t] or lower_wick>=.3*(self.h[t]-self.l[t])))
            recovery=(age>0 and self.c[t]>m['midpoint'] and self.c[t]>self.c[t-1]
                      and self.c[t]>self.o[t] and self._volume_ratio(t)>=.6)
            if stopping or recovery:
                meta={**m,'entry_allowed':True,'confirmation':'stopping' if stopping else 'demand_reversal',
                      'stop_price':float(min(self.l[candidate.idx:t+1])-.15*m['atr'])}
                output.append(Event('SC',candidate.idx,t,float(m['climax_low']),quality=.85,kind='confirmed',meta=meta))
                self.resolved_sc.update(e.idx for e in self.sc_candidates if e.idx<=candidate.idx)
                break
        return output

    def _spring_at(self,t):
        cfg=self.cfg.spring
        for start in range(max(61,t-cfg.spring_window+1),t+1):
            ref,line_id,count=self._reference(start-1)
            atr=self._atr(start-1)
            if not np.isfinite(ref) or self.c[start-1]<ref or self.l[start]>=ref-.05*atr: continue
            if any(e.etype=='SPRING' and e.idx==start for e in self.events): continue
            if any(e.etype=='SPRING' and e.meta.get('line_id')==line_id and t-e.confirm_idx<=cfg.recover_window for e in self.events): continue
            reclaim=ref+cfg.recover_margin_atr*atr
            if self.c[t]<reclaim: continue
            probes=[i for i in range(start,t+1) if self.l[i]<ref]
            volume=max(self.v[i] for i in probes)
            ratio=float(volume/max(self.avg_vol_before(start),1e-9))
            spread=max(self.h[i]-self.l[i] for i in probes)
            deepest=(ref-min(self.l[start:t+1]))/atr
            context=self._context(start-1)
            if ratio<=self.cfg.volume.low_vol_ratio and spread<=cfg.ideal_range_atr*atr and deepest<=1:
                kind='ideal'
            elif context and self.close_pos[t]>=.6 and self.c[t]>=self.o[t] and spread<=cfg.absorption_range_atr*atr and deepest<=1.5:
                kind='absorption'
            elif ratio>=cfg.vol_confirm_ratio and (spread>=1.5*atr or deepest>1.5):
                kind='shock'
            else:
                kind='needs_test'
            low=float(min(self.l[start:t+1]))
            return Event('SPRING',start,t,float(ref),quality=float(np.clip(1.1-.3*ratio,.1,1)),kind=kind,
                         meta={'reference_price':ref,'line_id':line_id,'support_count':count,'reference_asof':start-1,
                               'break_low':low,'break_vol':float(volume),'probe_range':float(spread),'atr':atr,
                               'recover_line':float(reclaim),'vol_ratio':ratio,'entry_allowed':context and kind in ('ideal','absorption'),
                               'stop_price':low-.15*atr})
        return None

    def _sc_test_at(self,t):
        cfg=self.cfg.spring
        sc=self._last('SC',t,only_active=True)
        if sc is None or not 4<=t-sc.idx<=cfg.test_window: return None
        if any(e.etype=='ST' and e.meta.get('sc_idx')==sc.idx for e in self.events): return None
        # A recovery cannot revive an already lost climax structure.
        if min(self.c[sc.confirm_idx+1:t+1]) < sc.price-self._tolerance(sc.price,sc.confirm_idx)*sc.price: return None
        for start in range(max(sc.confirm_idx+2,t-cfg.recover_window),t+1):
            if not any(e.etype=='AR' and e.meta.get('sc_idx')==sc.idx and e.confirm_idx<start for e in self.events): continue
            ref=sc.price
            tol=self._tolerance(ref,start)*ref
            spread=self.h[start]-self.l[start]
            if not (ref-tol<=self.l[start]<=ref+tol and self.v[start]<=cfg.test_volume_ratio*sc.meta['vol']
                    and self.v[start]<=self.avg_vol_before(start) and spread<=cfg.test_range_ratio*sc.meta['range']
                    and spread<=cfg.test_range_atr*self._atr(start-1)): continue
            if min(self.l[start:t+1])<ref-tol: continue
            if any(self.big_down(i) and self._volume_ratio(i)>1 for i in range(start,t+1)): continue
            if self.c[t]>self.o[t] and self.c[t]>=ref+cfg.recover_margin_atr*self._atr(start):
                low=float(min(self.l[start:t+1]))
                return Event('ST',start,t,float(ref),quality=.85,kind='sc_test',
                             meta={'sc_idx':sc.idx,'origin':'SC','reference_price':ref,'line_id':sc.meta.get('line_id',-1),
                                   'test_low':low,'stop_price':low-.15*self._atr(start),'entry_allowed':True})
        return None

    def _spring_test_at(self,t):
        cfg=self.cfg.spring
        parent=self._last('SPRING',t,only_active=True)
        if parent is None or not 2<=t-parent.confirm_idx<=cfg.test_window: return None
        ref,low=parent.meta['reference_price'],parent.meta['break_low']
        parent_atr=parent.meta.get('atr',self._atr(parent.confirm_idx))
        if min(self.c[parent.confirm_idx+1:t+1]) < low-.05*parent_atr: return None
        tol=self._tolerance(ref,t)*ref
        if max(self.c[parent.confirm_idx+1:t])<=ref+tol: return None
        if not (low-.2*parent_atr<=self.l[t]<=ref+tol and self.c[t]>=ref+cfg.recover_margin_atr*self._atr(t-1)
                and self.v[t]<=cfg.test_volume_ratio*parent.meta['break_vol'] and self.v[t]<=self.avg_vol_before(t)
                and self.h[t]-self.l[t]<=cfg.test_range_atr*self._atr(t-1) and self.c[t]>self.o[t]): return None
        return Event('TEST',t,t,float(ref),quality=.9,kind='spring_test',
                     meta={'spring_idx':parent.idx,'reference_price':ref,'line_id':parent.meta['line_id'],
                           'test_low':float(self.l[t]),'break_low':low,'stop_price':low-.15*parent_atr,'entry_allowed':True})

    def _breakout_at(self,t):
        cfg=self.cfg.spring
        chain=(self.c[t]>self.c[t-1]>self.c[t-2] and self.l[t]>self.l[t-1]>self.l[t-2]
               and self.v[t]>self.v[t-1]>=self.v[t-2])
        start=t-2 if chain else t
        atr=self._atr(start-1)
        spread=max(self.h[start:t+1])-min(self.l[start:t+1])
        strong=(self.close_pos[t]>=.7 and ((self.c[t]>self.o[t] and self.h[t]-self.l[t]>=1.2*atr and self._volume_ratio(t)>=1.5)
                                        or (chain and self.c[t]-self.c[start-1]>=2*atr)))
        if not strong: return None
        ref,line_id,count=self._reference(start-1,'resistance')
        if np.isfinite(ref) and self.c[t]>ref+.05*atr and self.c[start-1]<=ref+.2*atr:
            etype,kind='JOC','breakout'
        elif self.c[t]>max(self.h[max(0,start-5):start]) and self.c[t]-self.c[start-1]>=1.5*atr:
            etype,kind,ref,line_id,count='SOS','demand',float(self.l[start]),-1,0
        else: return None
        if self._recent(etype,5,t): return None
        low=float(min(self.l[start:t+1]))
        return Event(etype,start,t,float(ref),quality=.9,kind=kind,
                     meta={'ref':float(ref),'reference_price':float(ref),'line_id':line_id,'support_count':count,
                           'wave_start':start,'volume':float(np.mean(self.v[start:t+1])),'range':float(spread),
                           'wave_low':low,'wave_high':float(max(self.h[start:t+1])),
                           'stop_price':float(ref-.3*atr),'entry_allowed':True})

    def _detect_breakout_test(self,t,cfg):
        parent=self._last(('JOC','SOS'),t,only_active=True)
        if parent is None or not 2<=t-parent.confirm_idx<=cfg.test_window: return None
        if any(e.etype=='ST' and e.meta.get('breakout_idx')==parent.idx for e in self.events): return None
        ref,volume,spread=(parent.meta.get(k) for k in ('ref','volume','range'))
        if not ref or not volume or not spread: return None
        tol=self._tolerance(ref,parent.confirm_idx)*ref
        if any(self.c[i]<ref-tol or (self.big_down(i) and self.v[i]>volume) for i in range(parent.confirm_idx+1,t+1)): return None
        if not (self.c[t]>self.o[t] and self.c[t]>=ref+cfg.recover_margin_atr*self._atr(t-1)): return None
        for start in range(max(parent.confirm_idx+2,t-cfg.recover_window),t+1):
            if not (ref-tol<=self.l[start]<=ref+tol and self.v[start]<=cfg.test_volume_ratio*volume
                    and self.v[start]<=self.avg_vol_before(start) and self.h[start]-self.l[start]<=cfg.test_range_ratio*spread
                    and self.h[start]-self.l[start]<=cfg.test_range_atr*self._atr(start-1)): continue
            low=float(min(self.l[start:t+1]))
            return Event('ST',start,t,float(ref),quality=.85,kind='breakout_test',
                         meta={'origin':parent.etype,'breakout_idx':parent.idx,'reference_price':float(ref),
                               'line_id':parent.meta.get('line_id',-1),'test_low':low,'stop_price':low-.15*self._atr(start-1),'entry_allowed':True})
        return None

    def _lps_at(self,t):
        parent=self._last(('SOS','JOC'),t)
        if parent is None or not 2<=t-parent.confirm_idx<=20 or self._recent('LPS',5,t): return None
        ref=parent.meta['ref']
        peak=max(self.h[parent.idx:t+1])
        depth=(peak-self.l[t])/max(peak-ref,1e-9)
        if (0<=depth<=.5 and self.l[t]>=ref and self.c[t]>self.o[t] and self.v[t]<=.8*self.avg_vol_before(t)
                and self.h[t]-self.l[t]<=self._atr(t-1)
                and min(self.c[parent.confirm_idx+1:t+1])>=ref):
            return Event('LPS',t,t,float(ref),quality=.85,kind='held',
                         meta={'ref':ref,'reference_price':ref,'sos_idx':parent.idx,'test_low':float(self.l[t]),
                               'stop_price':float(self.l[t]-.15*self._atr(t-1)),'entry_allowed':True})
        return None

    def _failures_at(self,t):
        for parent in list(self.events):
            if parent.etype not in ('SPRING','SC'): continue
            key=(parent.etype,parent.idx)
            if key in self.failed_parents or not 1<=t-parent.confirm_idx<=12: continue
            low=parent.meta.get('break_low',parent.meta.get('climax_low',parent.price))
            atr=parent.meta.get('atr',self._atr(t-1))
            broke=self.c[t]<low-.05*atr
            supply=self.big_down(t) and self._volume_ratio(t)>1.3
            no_follow=(t-parent.confirm_idx==4 and max(self.c[parent.confirm_idx+1:t+1])<=self.c[parent.confirm_idx]+.25*atr)
            if broke or supply or no_follow:
                self.failed_parents.add(key)
                self.events.append(Event(parent.etype+'_FAIL',t,t,float(low),kind='lost_support' if broke else 'no_follow',
                                         meta={('spring_idx' if parent.etype=='SPRING' else 'sc_idx'):parent.idx}))

    def run(self):
        for t in range(61,self.n):
            self._failures_at(t)
            sc_events=self._sc_events_at(t)
            self.events.extend(sc_events)
            confirmed_sc=any(e.etype=='SC' for e in sc_events)
            spring=self._spring_at(t)
            if spring is not None and not confirmed_sc: self.events.append(spring)
            sc=self._last('SC',t)
            if sc is not None and 1<=t-sc.confirm_idx<=6 and self.c[t]>self.o[t] and self.c[t]>self.c[t-1]:
                if not any(e.etype=='AR' and e.meta.get('sc_idx')==sc.idx for e in self.events):
                    self.events.append(Event('AR',t,t,float(self.h[t]),meta={'sc_idx':sc.idx}))
            for detector in (self._sc_test_at,self._spring_test_at):
                event=detector(t)
                if event is not None: self.events.append(event)
            event=self._detect_breakout_test(t,self.cfg.spring)
            if event is not None: self.events.append(event)
            for detector in (self._breakout_at,self._lps_at):
                event=detector(t)
                if event is not None: self.events.append(event)
            if confirmed_sc or spring is not None: continue
            ref=self.support[t-1]
            if (np.isfinite(ref) and self.c[t]<ref-.1*self._atr(t-1) and self.l[t]<self.l[t-1]
                    and self.h[t]<self.h[t-1] and self.c[t]<self.c[t-1] and self.v[t]>=self.v[t-1]
                    and (self.big_down(t) or self._volume_ratio(t)>=1.5) and not self._recent('SOW',3,t)):
                self.events.append(Event('SOW',t,t,float(ref),quality=.85))
            if not self._recent('UT',3,t):
                for probe in range(max(61,t-3),t+1):
                    resistance=self.res[probe-1]
                    if np.isfinite(resistance) and self.h[probe]>resistance+.1*self._atr(probe-1) and self.c[t]<resistance:
                        self.events.append(Event('UT',probe,t,float(resistance),quality=.8))
                        break
            if (self.pos60[t]>=.8 and self._volume_ratio(t)>=2 and self.h[t]-self.l[t]>=1.5*self._atr(t-1)
                    and self.close_pos[t]<=.5 and not self._recent('BC',10,t)):
                self.events.append(Event('BC',t,t,float(self.h[t]),quality=.8))
            if self.trend[t]==-1 and self.c[t]>self.c[t-3] and self.v[t]<self.v[t-1]<self.v[t-2]:
                self.events.append(Event('NDR',t,t,float(self.c[t]),quality=.6))
        return self.events


def detect_events(df,lines,cfg):
    return EventScanner(df,lines,cfg).run()
