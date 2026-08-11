"""Validate the margin distribution used for P(cover).

Spread EV depends on the full predictive distribution, not just the mean:
  1. Is the residual roughly normal (or fat-tailed)?
  2. Is sigma constant, or does it grow with the projected margin?
  3. Does P(margin > x) match empirical frequency across the range?
  4. Push mass at whole-number margins (key numbers).
"""
import json,os
import numpy as np
from collections import defaultdict
from datetime import datetime,timedelta

HERE=os.path.dirname(__file__)
games=json.load(open(os.path.join(HERE,"wnba_games.json")))
box=json.load(open(os.path.join(HERE,"wnba_boxscores.json")))
TEAMS={'ATL','CHI','CON','DAL','GS','IND','LA','LV','MIN','NY','PHX','POR','SEA','TOR','WSH'}
games=[g for g in games if g["home"]["abbr"] in TEAMS and g["away"]["abbr"] in TEAMS and g.get("seasonType") in (2,3)]
games.sort(key=lambda g:g["date"])
def gdate(g): return datetime.fromisoformat(g["date"].replace("Z","+00:00"))
def eday(g): return (gdate(g)-timedelta(hours=5)).date()
played=defaultdict(set)
for g in games: played[eday(g)].add(g["home"]["abbr"]); played[eday(g)].add(g["away"]["abbr"])
def b2b(ab,g): return ab in played.get(eday(g)-timedelta(days=1),set())
LG=83.0;K=8;CAP=20.0

def build(half_life=25):
    lam=0.5**(1.0/half_life);prior={};rows=[]
    for season in sorted({g["seasonYear"] for g in games}):
        sg=[g for g in games if g["seasonYear"]==season]
        wPF=defaultdict(float);wPA=defaultdict(float);wN=defaultdict(float)
        pst=defaultdict(lambda: defaultdict(lambda:[0,0.0]));tgp=defaultdict(int)
        for ab,(pf,pa) in prior.items():
            wPF[ab]=(0.6*pf+0.4*LG)*4;wPA[ab]=(0.6*pa+0.4*LG)*4;wN[ab]=4
        pfS=0.0;nS=0
        for g in sg:
            h,a=g["home"],g["away"];hb,ab_=h["abbr"],a["abbr"]
            lg=(pfS/nS) if nS>=20 else LG
            bx=box.get(g["id"]) or {}
            def miss(abbr):
                if tgp[abbr]<4: return 0.0
                pl={p[0] for p in (bx.get(abbr) or []) if p[1] and p[1]>0}
                if not pl: return 0.0
                m=0.0
                for pid,(apps,pts) in pst[abbr].items():
                    if apps<3 or apps/tgp[abbr]<0.5: continue
                    ppg=pts/apps
                    if ppg>=6 and pid not in pl: m+=ppg
                return min(m,30.0)
            if wN[hb]>=3 and wN[ab_]>=3:
                def rate(w,n): return ((w/n)*n+lg*K)/(n+K)
                hPF=rate(wPF[hb],wN[hb]);hPA=rate(wPA[hb],wN[hb])
                aPF=rate(wPF[ab_],wN[ab_]);aPA=rate(wPA[ab_],wN[ab_])
                rows.append({"eH0":hPF*aPA/lg,"eA0":aPF*hPA/lg,
                    "b2bH":int(b2b(hb,g)),"b2bA":int(b2b(ab_,g)),
                    "missH":miss(hb),"missA":miss(ab_),
                    # FULL margin incl. OT, since spreads settle on final score
                    "marginFinal":h["score"]-a["score"],
                    "marginReg":sum(h["lines"][:4])-sum(a["lines"][:4])})
            hp=sum(h["lines"][:4]);ap=sum(a["lines"][:4])
            mid=(hp+ap)/2;d=max(-CAP/2,min(CAP/2,(hp-ap)/2));hp,ap=mid+d,mid-d
            for abbr,pf_g,pa_g in [(hb,hp,ap),(ab_,ap,hp)]:
                wPF[abbr]=wPF[abbr]*lam+pf_g;wPA[abbr]=wPA[abbr]*lam+pa_g;wN[abbr]=wN[abbr]*lam+1
            for abbr in (hb,ab_):
                for p in (bx.get(abbr) or []):
                    if p[1] and p[1]>0:
                        s=pst[abbr][p[0]];s[0]+=1;s[1]+=p[2]
                tgp[abbr]+=1
            pfS+=sum(h["lines"][:4])+sum(a["lines"][:4]);nS+=2
        prior={x:(wPF[x]/wN[x],wPA[x]/wN[x]) for x in wN if wN[x]>0}
    return rows

rows=build()
X=np.array([[r["eH0"]-r["eA0"],1.0,r["b2bH"]-r["b2bA"],r["missH"]-r["missA"]] for r in rows])
# IMPORTANT: fit against FINAL margin (what spreads settle on), not regulation
yF=np.array([r["marginFinal"] for r in rows],float)
yR=np.array([r["marginReg"] for r in rows],float)
cF,*_=np.linalg.lstsq(X,yF,rcond=None); muF=X@cF; resF=yF-muF
cR,*_=np.linalg.lstsq(X,yR,rcond=None); resR=yR-(X@cR)
print(f"n={len(rows)}")
print(f"margin(FINAL): coefs eff={cF[0]:.4f} home={cF[1]:.3f} b2b={cF[2]:.3f} inj={cF[3]:.4f} sd={resF.std():.3f}")
print(f"margin(REG)  : sd={resR.std():.3f}   <- v2 used this; spreads settle on FINAL")

print("\n== 1. normality of residuals ==")
from scipy import stats as st
print(f"  skew={st.skew(resF):.3f} kurtosis(excess)={st.kurtosis(resF):.3f}")
for q in [0.01,0.05,0.10,0.25,0.5,0.75,0.90,0.95,0.99]:
    emp=np.quantile(resF,q); theo=st.norm.ppf(q,0,resF.std())
    print(f"  q{q:>5.2f}: empirical={emp:+7.2f}  normal={theo:+7.2f}  diff={emp-theo:+5.2f}")

print("\n== 2. heteroscedasticity: sd by |projected margin| ==")
absmu=np.abs(muF)
for lo,hi in [(0,4),(4,8),(8,12),(12,50)]:
    m=(absmu>=lo)&(absmu<hi)
    if m.sum()>30: print(f"  |mu| {lo:>2}-{hi:<2}: n={m.sum():4d} resid sd={resF[m].std():.2f}")
# regression of |resid| on |mu|
A=np.stack([absmu,np.ones(len(absmu))],1)
ch,*_=np.linalg.lstsq(A,np.abs(resF),rcond=None)
print(f"  |resid| ~ {ch[0]:+.4f}*|mu| + {ch[1]:.2f}   (slope~0 => homoscedastic)")

print("\n== 3. calibration of P(margin > x) ==")
sd=resF.std()
for x in [-14,-10,-6,-3,0,3,6,10,14]:
    pred=np.mean(1-st.norm.cdf(x,muF,sd))
    emp=np.mean(yF>x)
    print(f"  x={x:+3d}: model={pred:.3f} empirical={emp:.3f} diff={pred-emp:+.3f}")

print("\n== 4. push mass at whole numbers (key numbers) ==")
tot=len(yF)
for k in range(1,13):
    c=np.sum(np.abs(yF)==k)
    if c: print(f"  |margin|={k:2d}: {c:3d} games ({c/tot*100:.1f}%)")
print(f"  P(exact tie in regulation-final)={np.mean(yF==0)*100:.2f}%")

print("\n== 5. ATS-style sanity: model vs a naive 'always home' spread bettor ==")
# Without historical lines we cannot backtest ATS directly. Proxy: how often
# does the model's projected margin land on the correct side of the ACTUAL margin?
print(f"  mean |projection error| = {np.mean(np.abs(resF)):.2f} pts")
print(f"  model beats a fixed +{cF[1]:.1f} home-edge-only baseline: "
      f"MAE {np.mean(np.abs(resF)):.2f} vs {np.mean(np.abs(yF-cF[1])):.2f}")
