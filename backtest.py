#!/usr/bin/env python3
"""
backtest_asia_SPOT_LONG_only.py
Asia ORB 00:00-04:00 UTC  →  LONG only  →  dynamic TP  →  spot prices
Risk 1 % equity per trade
"""

import pandas as pd
import numpy as np
import joblib

CSV_PATH     = "btc.csv"
MODEL_PATH   = "btc_asia_SPOT_LONG.joblib"
INIT_BALANCE = 10000.0
RISK_PCT     = 1.0
FEE          = 0.0004  # 0.04 % per side
ORB_START_H  = 0
ORB_END_H    = 4

def parse_time(s):
    for fmt in ["%d.%m.%Y %H:%M:%S.%f","%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M"]:
        try: return pd.to_datetime(s,format=fmt)
        except: continue
    return pd.to_datetime(s,dayfirst=True,errors='coerce')

def atr(df_,n): return (df_['high']-df_['low']).rolling(n).mean()

def build_feat(rh,rl,candle,momentum,atr20):
    orb_range=rh-rl; range_ratio=orb_range/atr20 if atr20 else 0; tod=0
    return np.array([[orb_range,candle,momentum,atr20,tod,range_ratio]])

def pos_size(balance,risk_dist):
    if risk_dist<=0: return 0.0
    risk_amt = balance*RISK_PCT/100.
    qty = risk_amt/risk_dist
    return max(0.001,round(qty,3))

# ---------- load ----------
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()
time_col = [c for c in df.columns if c in {'gmt time','gmt_time','time','timestamp'}][0]
df['gmt time'] = parse_time(df[time_col])
df = df.dropna(subset=['gmt time']).sort_values('gmt time').reset_index(drop=True)
df['date'] = df['gmt time'].dt.date
df['h']    = df['gmt time'].dt.hour
model = joblib.load(MODEL_PATH)

# ---------- backtest ----------
bal = INIT_BALANCE
trades=[]
for date,d in df.groupby('date'):
    d = d.copy()
    orb = d[(d['h']>=ORB_START_H)&(d['h']<ORB_END_H)]
    if len(orb)<30: continue
    rh,rl = orb['high'].max(), orb['low'].min()

    post = d[d['h']>=ORB_END_H].copy()
    if post.empty: continue

    # LONG breakout only
    entry = None
    for idx,r in post.iterrows():
        if r['close']>rh:
            entry,idx_entry = r['close'],idx
            break
    if entry is None: continue

    risk   = entry - rl
    if risk<=0: continue

    # dynamic TP  (DataFrame wrapper to suppress warning)
    k = post.index.get_loc(idx_entry)
    rem = post.iloc[k:]
    atr20 = atr(post.iloc[max(0,k-20):k],20).iloc[-1] if k>0 else (rh-rl)

    feat_array = build_feat(rh,rl,rem.iloc[0]['high']-rem.iloc[0]['low'],
                            rem.iloc[0]['close']-orb.iloc[-1]['close'],atr20)
    feat_df = pd.DataFrame(feat_array, columns=['orb_range','candle_range','momentum','atr20','tod','range_ratio'])
    pred_rr = float(model.predict(feat_df)[0])

    # position
    qty = pos_size(bal,risk)
    if qty<=0: continue

    sl = rl
    tp = entry + pred_rr*risk

    # simulate
    hit_sl=hit_tp=False; exit_p=None
    for _,r in rem.iterrows():
        if r['low']<=sl:
            hit_sl=True; exit_p=sl; break
        if r['high']>=tp:
            hit_tp=True; exit_p=tp; break
    if not(hit_sl or hit_tp): exit_p = rem.iloc[-1]['close']

    # pnl
    gross = (exit_p-entry)*qty
    net   = gross - (entry+exit_p)*qty*FEE
    bal  += net
    actual_rr = net/(risk*qty) if risk else 0
    trades.append({'date':date,'side':'long','entry':entry,'exit':exit_p,
                   'sl':sl,'tp':tp,'qty':qty,'pred_rr':pred_rr,
                   'actual_rr':actual_rr,'pnl':net,'bal':bal,
                   'hit_sl':hit_sl,'hit_tp':hit_tp})

# ---------- results ----------
if not trades:
    print('no trades')
    exit()
t = pd.DataFrame(trades)
output_csv = 'asia_spot_long_trades.csv'      # ← same folder as script
t.to_csv(output_csv, index=False)
print(f'💾 {output_csv} saved')

print('='*70)
print('ASIA-SPOT-LONG-RAW  RESULTS')
print('='*70)
ret = (t['bal'].iloc[-1]/INIT_BALANCE-1)*100
print(f'final ${t["bal"].iloc[-1]:,.2f}   ret {ret:+.2f}%')
print(f'trades {len(t)}  win {(t.pnl>0).sum()}/{len(t)}  PF {abs(t[t.pnl>0].pnl.sum()/t[t.pnl<0].pnl.sum()):.2f}')
print(f'avg pred RR {t.pred_rr.mean():.2f}   avg actual RR {t.actual_rr.mean():.2f}')
print('='*70)