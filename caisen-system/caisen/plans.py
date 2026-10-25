import math

from .indicators import atr


def build_trade_plan(pattern, entry_price, cash=100000, lot_size=100, bars=None,
                     max_risk_pct=0.07, min_net_reward_risk=2.0, commission=0.0,
                     min_commission=0.0, slippage=0.0, adv_shares=None, tick_size=0.01):
    p = pattern.to_dict() if hasattr(pattern, "to_dict") else dict(pattern)
    reasons=[]; entry=float(entry_price); structure=float(p.get("invalidation") or 0); stop=structure
    if bars is not None and len(bars) and structure>0:
        current_atr=float(atr(bars).iloc[-1])
        if math.isfinite(current_atr): stop=structure-max(2*float(tick_size),0.5*current_atr)
    data_warnings=[]
    result={"pattern_id":p.get("pattern_id"),"decision":"WAIT","entry_price":entry,"stop":stop,"structure_level":structure,
            "targets":p.get("targets",[]),"reason_codes":reasons,"data_warnings":data_warnings,
            "planned_risk_pct":None,"net_reward_risk":None,"reward_risk":None,"shares":0,
            "cost_model":{"commission":commission,"minimum_commission":min_commission,"slippage":slippage}}
    if p.get("direction") != "bullish" or p.get("state") != "confirmed": reasons.append("pattern_not_confirmed_bullish")
    if not p.get("volume_verified"):
        data_warnings.append("volume unit or corporate-action adjustment is unverified")
        reasons.append("volume_data_unverified")
    if not p.get("execution_rules_verified"): data_warnings.append("instrument trading rules are unverified")
    if stop <= 0 or stop >= entry: reasons.append("invalid_structural_stop")
    if not p.get("targets"): reasons.append("no_verified_target")
    risk=(entry-stop)/entry if entry>0 and stop>0 else float("inf")
    result["planned_risk_pct"]=risk if math.isfinite(risk) else None
    if risk > max_risk_pct: reasons.append("structural_risk_over_7pct")
    target=float(p["targets"][0]) if p.get("targets") else None
    if target is None or target<=entry: reasons.append("first_target_already_reached_or_below_entry")
    if target is not None and stop>0:
        result["highest_allowed_entry"]=max(0.0,min(stop/(1-max_risk_pct),(target+2*stop)/3.0))
        if entry>result["highest_allowed_entry"]: reasons.append("entry_above_maximum_allowed_price")
    if bars is not None and len(bars)>0:
        latest_atr=float(atr(bars).iloc[-1])
        result["atr14"]=latest_atr if math.isfinite(latest_atr) else None
    if adv_shares is not None and adv_shares>0: result["adv_share_cap"]=int(adv_shares*0.01)//lot_size*lot_size
    shares=max(0,int(cash//(entry*lot_size))*lot_size) if entry>0 else 0
    if "adv_share_cap" in result: shares=min(shares,result["adv_share_cap"])
    result["shares"]=shares
    if shares==0: reasons.append("insufficient_cash_or_liquidity")
    if target is not None and target>entry and shares>0 and stop<entry:
        buy_fee=max(min_commission,entry*shares*commission)
        sell_fee=max(min_commission,target*shares*commission)
        buy_slip=entry*shares*slippage; sell_slip=target*shares*slippage
        reward=(target-entry)*shares-buy_fee-sell_fee-buy_slip-sell_slip
        risk_cash=(entry-stop)*shares+buy_fee+max(min_commission,stop*shares*commission)+buy_slip+stop*shares*slippage
        result["planned_risk_pct"]=risk_cash/(entry*shares)
        if result["planned_risk_pct"]>max_risk_pct and "structural_risk_over_7pct" not in reasons:
            reasons.append("structural_risk_over_7pct")
        rr=reward/risk_cash if risk_cash>0 else 0
        result["net_reward_risk"]=rr
        result["reward_risk"]=rr
        if rr<min_net_reward_risk: reasons.append("net_first_target_below_2R")
    if not reasons: result["decision"]="CONSIDER_BUY"
    result["explanation"]=";".join(reasons) if reasons else "所有首版门槛通过；只在下一有效交易日按计划条件复核。"
    return result
