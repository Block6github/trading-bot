#!/usr/bin/env python3
"""
train_asia_SPOT_LONG_only.py
Asia ORB 00:00-04:00 UTC  → predict optimal LONG TP (3-25R)
NO short trades  →  label = max favourable move above ORB high
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

CSV_PATH      = "btc.csv"
MODEL_OUT     = "btc_asia_SPOT_LONG.joblib"
ORB_START_H   = 0
ORB_END_H     = 4
CLIP_TAIL     = 25.0
TEST_SPLIT    = 0.2
RANDOM_STATE  = 42
EST,DEPTH     = 300,10

# ---------- helpers ----------
def parse_time(s):
    for fmt in ["%d.%m.%Y %H:%M:%S.%f","%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M"]:
        try: return pd.to_datetime(s,format=fmt)
        except: continue
    return pd.to_datetime(s,dayfirst=True,errors='coerce')

def atr(df_,n): return (df_['high']-df_['low']).rolling(n).mean()

# ---------- load ----------
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip().str.lower()
time_col = [c for c in df.columns if c in {'gmt time','gmt_time','time','timestamp'}][0]
df['gmt time'] = parse_time(df[time_col])
df = df.dropna(subset=['gmt time']).sort_values('gmt time').reset_index(drop=True)
df['date'] = df['gmt time'].dt.date
df['h']    = df['gmt time'].dt.hour
print(f'loaded {len(df):,} candles  {df.date.min()} → {df.date.max()}')

# ---------- daily loop ----------
samples=[]
for date,d in df.groupby('date'):
    d = d.copy()
    orb = d[(d['h']>=ORB_START_H)&(d['h']<ORB_END_H)]
    if len(orb)<30: continue          # ~30 1-min candles
    rh,rl = orb['high'].max(), orb['low'].min()

    post = d[d['h']>=ORB_END_H].copy()
    if post.empty: continue

    # LONG breakout only
    entry = None
    for idx,r in post.iterrows():
        if r['close']>rh:
            entry = r['close']
            break
    if entry is None: continue

    risk   = entry - rl                       # stop = ORB low
    k      = post.index.get_loc(post[post['close']==entry].index[0])
    rem    = post.iloc[k:]

    # label = max RR above entry
    max_rr = 0.0
    for _,r in rem.iterrows():
        max_rr = max(max_rr, (r['high']-entry)/risk)
    label = min(max_rr, CLIP_TAIL)

    # features
    prev20 = post.iloc[max(0,k-20):k]
    atr20  = (prev20['high']-prev20['low']).mean() if len(prev20)>1 else 0
    orb_range = rh-rl
    candle    = rem.iloc[0]['high'] - rem.iloc[0]['low']
    momentum  = rem.iloc[0]['close'] - rem.iloc[0]['open']
    tod       = 0
    range_ratio = orb_range/atr20 if atr20 else 0

    samples.append({'orb_range':orb_range,'candle_range':candle,'momentum':momentum,
                    'atr20':atr20,'tod':tod,'range_ratio':range_ratio,'label':label})

print(f'samples {len(samples)}')
if len(samples)<50: raise ValueError('too few samples')

# ---------- train ----------
ds = pd.DataFrame(samples)
X,y = ds[['orb_range','candle_range','momentum','atr20','tod','range_ratio']], ds['label']
Xtr,Xva,ytr,yva = train_test_split(X,y,test_size=TEST_SPLIT,random_state=RANDOM_STATE)
m = RandomForestRegressor(n_estimators=EST,max_depth=DEPTH,random_state=RANDOM_STATE,n_jobs=-1)
m.fit(Xtr,ytr)
print(f'train R² {m.score(Xtr,ytr):.3f}  val R² {m.score(Xva,yva):.3f}')
joblib.dump(m, MODEL_OUT)
print('model →',MODEL_OUT)