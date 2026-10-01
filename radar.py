import os,time,requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
BASE='https://fapi.binance.com'
MIN_SCORE=float(os.getenv('MIN_SCORE','80'))
TOP=int(os.getenv('TOP_SYMBOLS','80'))
INTERVAL=int(os.getenv('SCAN_INTERVAL_SECONDS','60'))
TOKEN=os.getenv('TELEGRAM_BOT_TOKEN','')
CHAT=os.getenv('TELEGRAM_CHAT_ID','')
last_alert={}

def get(path,params=None):
    r=requests.get(BASE+path,params=params,timeout=12); r.raise_for_status(); return r.json()

def symbols():
    tick=get('/fapi/v1/ticker/24hr')
    xs=[x for x in tick if x['symbol'].endswith('USDT') and float(x.get('quoteVolume',0))>0]
    xs.sort(key=lambda x:float(x['quoteVolume']),reverse=True)
    return [x['symbol'] for x in xs[:TOP] if x['symbol']!='USDCUSDT']

def klines(sym,iv,limit=220):
    a=get('/fapi/v1/klines',{'symbol':sym,'interval':iv,'limit':limit})
    d=pd.DataFrame(a,columns=['t','o','h','l','c','v','T','q','n','tb','tq','x'])
    for c in ['o','h','l','c','v','q','tb']: d[c]=pd.to_numeric(d[c])
    return d

def rsi(s,n=14):
    x=s.diff(); up=x.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-x.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-(100/(1+up/dn.replace(0,np.nan)))

def atr(d,n=14):
    pc=d.c.shift(); tr=pd.concat([(d.h-d.l),(d.h-pc).abs(),(d.l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def adx(d,n=14):
    up=d.h.diff(); down=-d.l.diff(); tr=atr(d,1)
    plus=100*(up.where((up>down)&(up>0),0).ewm(alpha=1/n,adjust=False).mean()/tr.ewm(alpha=1/n,adjust=False).mean())
    minus=100*(down.where((down>up)&(down>0),0).ewm(alpha=1/n,adjust=False).mean()/tr.ewm(alpha=1/n,adjust=False).mean())
    dx=100*(plus-minus).abs()/(plus+minus).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean(),plus,minus

def features(d):
    c=d.c; v=d.v
    e9=c.ewm(span=9,adjust=False).mean(); e21=c.ewm(span=21,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); e200=c.ewm(span=200,adjust=False).mean()
    rr=rsi(c); mac=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); sig=mac.ewm(span=9,adjust=False).mean()
    aa=atr(d); ax,di_p,di_m=adx(d)
    vm=v.rolling(20).mean(); vol_ratio=v/vm
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); bbpos=(c-(mid-2*sd))/((mid+2*sd)-(mid-2*sd))
    buy_ratio=d.tb/d.v.replace(0,np.nan)
    hi20=d.h.shift(1).rolling(20).max(); lo20=d.l.shift(1).rolling(20).min()
    return dict(price=c.iloc[-1],e9=e9.iloc[-1],e21=e21.iloc[-1],e50=e50.iloc[-1],e200=e200.iloc[-1],rsi=rr.iloc[-1],mac=mac.iloc[-1],sig=sig.iloc[-1],atr=aa.iloc[-1],adx=ax.iloc[-1],dip=di_p.iloc[-1],dim=di_m.iloc[-1],vr=vol_ratio.iloc[-1],bb=bbpos.iloc[-1],br=buy_ratio.iloc[-1],break_hi=c.iloc[-1]>hi20.iloc[-1],break_lo=c.iloc[-1]<lo20.iloc[-1])

def score(f,direction):
    long=direction=='LONG'; s=0; reasons=[]
    checks=[
      (f['e9']>f['e21']>f['e50'],18,'EMA trend'),
      (f['price']>f['e200'],8,'EMA200'),
      (f['mac']>f['sig'],12,'MACD'),
      (50<=f['rsi']<=70,10,'RSI'),
      (f['adx']>=22 and f['dip']>f['dim'],12,'ADX/DI'),
      (f['vr']>=1.5,14,'Volume spike'),
      (f['br']>=0.52,8,'Taker buy'),
      (f['break_hi'],10,'Breakout'),
      (0.45<=f['bb']<=1.15,8,'Bollinger position')]
    if not long:
      checks=[(not x,w,n) for x,w,n in checks]
      # replace range-based inversions with sensible short rules
      checks[3]=(30<=f['rsi']<=50,10,'RSI'); checks[4]=(f['adx']>=22 and f['dim']>f['dip'],12,'ADX/DI'); checks[5]=(f['vr']>=1.5,14,'Volume spike'); checks[6]=(f['br']<=0.48,8,'Taker sell'); checks[7]=(f['break_lo'],10,'Breakdown'); checks[8]=(-.15<=f['bb']<=.55,8,'Bollinger position')
    for ok,w,n in checks:
      if bool(ok): s+=w; reasons.append(n)
    return min(s,100),reasons

def analyze(sym):
    f15=features(klines(sym,'15m')); f1=features(klines(sym,'1h'))
    btc=features(klines('BTCUSDT','15m'))
    out=[]
    for dr in ['LONG','SHORT']:
      a,ra=score(f15,dr); b,rb=score(f1,dr)
      btc_ok=(btc['e9']>=btc['e21']) if dr=='LONG' else (btc['e9']<=btc['e21'])
      total=.58*a+.32*b+(10 if btc_ok else 0)
      total=min(round(total,1),100)
      if total>=MIN_SCORE:
        p=f15['price']; A=f15['atr']; mult=1 if dr=='LONG' else -1
        sl=p-mult*1.35*A; tp1=p+mult*1.5*A; tp2=p+mult*2.5*A
        out.append((total,dr,p,sl,tp1,tp2,sorted(set(ra+rb)),f15['rsi'],f15['vr'],f15['adx']))
    return out

def telegram(msg):
    if not TOKEN or not CHAT: print(msg); return
    requests.post(f'https://api.telegram.org/bot{TOKEN}/sendMessage',json={'chat_id':CHAT,'text':msg},timeout=12).raise_for_status()

def run():
    print('Coin Radar v1 started')
    while True:
      try:
        for sym in symbols():
          try:
            for sc,dr,p,sl,t1,t2,rs,rv,vr,ax in analyze(sym):
              key=(sym,dr); now=time.time()
              if now-last_alert.get(key,0)<3600: continue
              msg=(f'🔥 {sym} — {dr}\nRadar skoru: {sc}/100\nFiyat/Giriş bölgesi: {p:.8g}\nSL: {sl:.8g}\nTP1: {t1:.8g}\nTP2: {t2:.8g}\n15m RSI: {rv:.1f} | Hacim x{vr:.2f} | ADX {ax:.1f}\nOnaylar: '+', '.join(rs)+'\n\nNot: Radar skoru, gerçekleşmiş kazanma oranı değildir.')
              telegram(msg); last_alert[key]=now
          except Exception as e: print(sym,e)
        time.sleep(INTERVAL)
      except KeyboardInterrupt: break
      except Exception as e: print('scan error',e); time.sleep(20)
if __name__=='__main__': run()
