"""Test remaining cheap feature candidates against the v2 baseline:
   (a) rest days / schedule congestion beyond simple B2B
   (b) minutes-weighted availability instead of PPG-weighted
   (c) 3-games-in-5-nights congestion
Same walk-forward harness; report logloss/acc deltas.
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
sched=defaultdict(list)
for g in games:
    sched[g["home"]["abbr"]].append(eday(g)); sched[g["away"]["abbr"]].append(eday(g))
for k in sched: sched[k]=sorted(sched[k])
def rest_feats(ab,g):
    d=eday(g); prior=[x for x in sched[ab] if x<d]
    if not prior: return 3,0
    rest=min((d-prior[-1]).days,4)
    in5=sum(1 for x in prior if 0<(d-x).days<=4)   # games in prior 4 days
    return rest,in5
LG=83.0;SHRINK_K=8;CAP=20.0
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
def fit_logistic(x,y):
    k=0.1;x=np.asarray(x);y=np.asarray(y,float)
    for _ in range(60):
        p=sigmoid(k*x);gg=np.mean((p-y)*x);h=np.mean(p*(1-p)*x*x)+1e-9;k-=gg/h
    return k

def build(half_life=25):
    lam=0.5**(1.0/half_life);prior={};rows=[]
    for season in sorted({g["seasonYear"] for g in games}):
        sg=[g for g in games if g["seasonYear"]==season]
        wPF=defaultdict(float);wPA=defaultdict(float);wN=defaultdict(float)
        pst=defaultdict(lambda: defaultdict(lambda:[0,0.0,0.0]));tgp=defaultdict(int)
        for ab,(pf,pa) in prior.items():
            wPF[ab]=(0.6*pf+0.4*LG)*4;wPA[ab]=(0.6*pa+0.4*LG)*4;wN[ab]=4
        pfS=0.0;nS=0
        for g in sg:
            h,a=g["home"],g["away"];hb,ab_=h["abbr"],a["abbr"]
            lg=(pfS/nS) if nS>=20 else LG
            bx=box.get(g["id"]) or {}
            def miss(abbr,mode):
                if tgp[abbr]<4: return 0.0
                played={p[0] for p in (bx.get(abbr) or []) if p[1] and p[1]>0}
                if not played: return 0.0
                m=0.0
                for pid,(apps,pts,mins) in pst[abbr].items():
                    if apps<3 or apps/tgp[abbr]<0.5: continue
                    ppg=pts/apps; mpg=mins/apps
                    if pid in played: continue
                    if mode=="ppg" and ppg>=6: m+=ppg
                    if mode=="min" and mpg>=15: m+=mpg
                return min(m,30.0 if mode=="ppg" else 90.0)
            rH,cH=rest_feats(hb,g); rA,cA=rest_feats(ab_,g)
            if wN[hb]>=3 and wN[ab_]>=3:
                def rate(w,n): return ((w/n)*n+lg*SHRINK_K)/(n+SHRINK_K)
                hPF=rate(wPF[hb],wN[hb]);hPA=rate(wPA[hb],wN[hb])
                aPF=rate(wPF[ab_],wN[ab_]);aPA=rate(wPA[ab_],wN[ab_])
                rows.append({"eH0":hPF*aPA/lg,"eA0":aPF*hPA/lg,
                  "b2bH":int(rH<=1),"b2bA":int(rA<=1),
                  "restH":rH,"restA":rA,"congH":cH,"congA":cA,
                  "missH":miss(hb,"ppg"),"missA":miss(ab_,"ppg"),
                  "minH":miss(hb,"min"),"minA":miss(ab_,"min"),
                  "hReg":sum(h["lines"][:4]),"aReg":sum(a["lines"][:4]),
                  "homeWon":int(h["score"]>a["score"])})
            hp=sum(h["lines"][:4]);ap=sum(a["lines"][:4])
            mid=(hp+ap)/2;d=max(-CAP/2,min(CAP/2,(hp-ap)/2));hp,ap=mid+d,mid-d
            for abbr,pf_g,pa_g in [(hb,hp,ap),(ab_,ap,hp)]:
                wPF[abbr]=wPF[abbr]*lam+pf_g;wPA[abbr]=wPA[abbr]*lam+pa_g;wN[abbr]=wN[abbr]*lam+1
            for abbr in (hb,ab_):
                for p in (bx.get(abbr) or []):
                    if p[1] and p[1]>0:
                        s=pst[abbr][p[0]];s[0]+=1;s[1]+=p[2];s[2]+=p[1]
                tgp[abbr]+=1
            pfS+=sum(h["lines"][:4])+sum(a["lines"][:4]);nS+=2
        prior={x:(wPF[x]/wN[x],wPA[x]/wN[x]) for x in wN if wN[x]>0}
    return rows

rows=build()
yM=np.array([r["hReg"]-r["aReg"] for r in rows],float)
yW=np.array([r["homeWon"] for r in rows])
def ev(cols,name):
    X=np.column_stack([np.array([r[c] if isinstance(c,str) else c(r) for r in rows],float) for c in cols]+[np.ones(len(rows))])
    c,*_=np.linalg.lstsq(X,yM,rcond=None)
    mu=X@c;k=fit_logistic(mu,yW);p=sigmoid(k*mu)
    ll=-np.mean(yW*np.log(p+1e-9)+(1-yW)*np.log(1-p+1e-9))
    print(f"  {name:38} logloss={ll:.4f} acc={np.mean((p>=0.5)==yW):.3f}")
    return ll

eff=lambda r:r["eH0"]-r["eA0"]
b2b=lambda r:r["b2bH"]-r["b2bA"]
mis=lambda r:r["missH"]-r["missA"]
rst=lambda r:r["restH"]-r["restA"]
cng=lambda r:r["congH"]-r["congA"]
mns=lambda r:r["minH"]-r["minA"]

print("== feature ablation (all walk-forward, n=%d) =="%len(rows))
base=ev([eff,b2b,mis],"v2 baseline (eff+b2b+missPPG)")
ev([eff,b2b,mis,rst],"+ rest-days diff")
ev([eff,b2b,mis,cng],"+ congestion (games in prior 4d)")
ev([eff,b2b,mis,rst,cng],"+ rest + congestion")
ev([eff,mis,rst,cng],"rest+cong INSTEAD of b2b")
ev([eff,b2b,mns],"minutes-weighted missing (vs PPG)")
ev([eff,b2b,mis,mns],"both PPG + minutes missing")
ev([eff,b2b],"no injury feature at all (control)")
