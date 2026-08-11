"""WNBA pace/efficiency model — phase 2 of the totals scope.

Decomposes scoring into tempo x efficiency:
    Poss  = FGA - OREB + TO + 0.44*FTA
    ORtg  = 100 * PTS / Poss          (points per 100 possessions)
    Pace  = Poss per game
Predicted possessions come from both teams' pace; points = poss * efficiency.

KILL CRITERIA (declared before running, per the scope):
  A. Totals MAE must beat the current PF/PA total regression by >= 1.0 point,
     else the pace model does NOT ship.
  B. Win-prob logloss must improve, else the moneyline path stays on v2
     regardless of what happens with totals.
"""
import json, os
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

HERE=os.path.dirname(__file__)
games=json.load(open(os.path.join(HERE,"wnba_games.json")))
box=json.load(open(os.path.join(HERE,"wnba_boxscores.json")))
ts=json.load(open(os.path.join(HERE,"wnba_teamstats.json")))

TEAMS={'ATL','CHI','CON','DAL','GS','IND','LA','LV','MIN','NY','PHX','POR','SEA','TOR','WSH'}
games=[g for g in games if g["home"]["abbr"] in TEAMS and g["away"]["abbr"] in TEAMS
       and g.get("seasonType") in (2,3)]
games.sort(key=lambda g:g["date"])
have=[g for g in games if len(ts.get(g["id"]) or {})==2]
print(f"games={len(games)}  with team stats={len(have)}")

def gd(g): return datetime.fromisoformat(g["date"].replace("Z","+00:00"))
def ed(g): return (gd(g)-timedelta(hours=5)).date()
pl=defaultdict(set)
for g in games: pl[ed(g)].add(g["home"]["abbr"]); pl[ed(g)].add(g["away"]["abbr"])
def b2b(a,g): return a in pl.get(ed(g)-timedelta(days=1),set())

def poss(s): return s["fga"]-s["oreb"]+s["to"]+0.44*s["fta"]
LG_PTS=83.0; K=8; HALF=25
LAM=0.5**(1.0/HALF)

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
def fit_logistic(x,y):
    k=0.1;x=np.asarray(x);y=np.asarray(y,float)
    for _ in range(60):
        p=sigmoid(k*x);g=np.mean((p-y)*x);h=np.mean(p*(1-p)*x*x)+1e-9;k-=g/h
    return k

rows=[]
prior_pf={}; prior_pc={}
for season in sorted({g["seasonYear"] for g in games}):
    sg=[g for g in games if g["seasonYear"]==season]
    # v2-style points ratings
    wPF=defaultdict(float);wPA=defaultdict(float);wN=defaultdict(float)
    # pace/efficiency ratings
    wPC=defaultdict(float)                      # possessions
    wORT=defaultdict(float);wDRT=defaultdict(float)   # pts scored/allowed per 100
    wNP=defaultdict(float)
    pst=defaultdict(lambda: defaultdict(lambda:[0,0.0]));tgp=defaultdict(int)
    for ab,v in prior_pf.items():
        wPF[ab]=(0.6*v[0]+0.4*LG_PTS)*4; wPA[ab]=(0.6*v[1]+0.4*LG_PTS)*4; wN[ab]=4
    for ab,v in prior_pc.items():
        wPC[ab]=v[0]*4; wORT[ab]=v[1]*4; wDRT[ab]=v[2]*4; wNP[ab]=4
    pfS=0.0;nS=0; pcS=0.0;pcN=0
    for g in sg:
        h,a=g["home"],g["away"];hb,ab_=h["abbr"],a["abbr"]
        lg=(pfS/nS) if nS>=20 else LG_PTS
        lgpc=(pcS/pcN) if pcN>=20 else 76.0
        bx=box.get(g["id"]) or {}; st=ts.get(g["id"]) or {}
        def miss(x):
            if tgp[x]<4: return 0.0
            p2={p[0] for p in (bx.get(x) or []) if p[1] and p[1]>0}
            if not p2: return 0.0
            m=0.0
            for pid,(apps,pts) in pst[x].items():
                if apps<3 or apps/tgp[x]<0.5: continue
                ppg=pts/apps
                if ppg>=6 and pid not in p2: m+=ppg
            return min(m,30.0)
        mH,mA=miss(hb),miss(ab_)
        if wN[hb]>=3 and wN[ab_]>=3 and wNP[hb]>=3 and wNP[ab_]>=3:
            def rt(w,n): return ((w/n)*n+lg*K)/(n+K)
            hPF=rt(wPF[hb],wN[hb]);hPA=rt(wPA[hb],wN[hb])
            aPF=rt(wPF[ab_],wN[ab_]);aPA=rt(wPA[ab_],wN[ab_])
            def rp(w,n,pr): return ((w/n)*n+pr*K)/(n+K)
            hPC=rp(wPC[hb],wNP[hb],lgpc); aPC=rp(wPC[ab_],wNP[ab_],lgpc)
            lgort=100*lg/lgpc
            hOR=rp(wORT[hb],wNP[hb],lgort); hDR=rp(wDRT[hb],wNP[hb],lgort)
            aOR=rp(wORT[ab_],wNP[ab_],lgort); aDR=rp(wDRT[ab_],wNP[ab_],lgort)
            rows.append({
              "eH0":hPF*aPA/lg,"eA0":aPF*hPA/lg,
              "pcExp":hPC*aPC/lgpc,                       # expected possessions
              "ortH":hOR*aDR/lgort,"ortA":aOR*hDR/lgort,  # opp-adj efficiency
              "b2bH":int(b2b(hb,g)),"b2bA":int(b2b(ab_,g)),
              "missH":mH,"missA":mA,
              "hReg":sum(h["lines"][:4]),"aReg":sum(a["lines"][:4]),
              "homeWon":int(h["score"]>a["score"]),
              "mFin":h["score"]-a["score"]})
        # update
        hp=sum(h["lines"][:4]);ap=sum(a["lines"][:4])
        for abbr,pf_g,pa_g in [(hb,hp,ap),(ab_,ap,hp)]:
            wPF[abbr]=wPF[abbr]*LAM+pf_g;wPA[abbr]=wPA[abbr]*LAM+pa_g;wN[abbr]=wN[abbr]*LAM+1
        if len(st)==2 and hb in st and ab_ in st:
            ph=poss(st[hb]); pa2=poss(st[ab_]); pavg=(ph+pa2)/2.0
            if pavg>40:
                for abbr,pts_f,pts_a in [(hb,h["score"],a["score"]),(ab_,a["score"],h["score"])]:
                    wPC[abbr]=wPC[abbr]*LAM+pavg
                    wORT[abbr]=wORT[abbr]*LAM+100*pts_f/pavg
                    wDRT[abbr]=wDRT[abbr]*LAM+100*pts_a/pavg
                    wNP[abbr]=wNP[abbr]*LAM+1
                pcS+=pavg;pcN+=1
        for abbr in (hb,ab_):
            for p in (bx.get(abbr) or []):
                if p[1] and p[1]>0:
                    s2=pst[abbr][p[0]];s2[0]+=1;s2[1]+=p[2]
            tgp[abbr]+=1
        pfS+=hp+ap;nS+=2
    prior_pf={x:(wPF[x]/wN[x],wPA[x]/wN[x]) for x in wN if wN[x]>0}
    prior_pc={x:(wPC[x]/wNP[x],wORT[x]/wNP[x],wDRT[x]/wNP[x]) for x in wNP if wNP[x]>0}

print(f"usable rows (both rating systems warm) = {len(rows)}")
yTot=np.array([r["hReg"]+r["aReg"] for r in rows],float)
yMar=np.array([r["hReg"]-r["aReg"] for r in rows],float)
yW=np.array([r["homeWon"] for r in rows])

def report(name,X,y):
    c,*_=np.linalg.lstsq(X,y,rcond=None)
    pred=X@c
    return c,pred,float(np.mean(np.abs(y-pred))),float(np.std(y-pred))

print("\n===== A. TOTALS =====")
Xa=np.column_stack([[r["eH0"]+r["eA0"] for r in rows],np.ones(len(rows)),
                    [r["b2bH"]+r["b2bA"] for r in rows],[r["missH"]+r["missA"] for r in rows]])
_,_,mae_v2,sd_v2=report("v2",Xa,yTot)
print(f"  v2  (PF/PA sums)      MAE={mae_v2:.3f}  sd={sd_v2:.3f}")
# pace model: total = poss * (ortH+ortA)/100
paceTot=np.array([r["pcExp"]*(r["ortH"]+r["ortA"])/100.0 for r in rows])
print(f"  pace raw (no fit)     MAE={np.mean(np.abs(yTot-paceTot)):.3f}")
Xb=np.column_stack([paceTot,np.ones(len(rows)),
                    [r["b2bH"]+r["b2bA"] for r in rows],[r["missH"]+r["missA"] for r in rows]])
_,_,mae_p,sd_p=report("pace",Xb,yTot)
print(f"  pace (calibrated)     MAE={mae_p:.3f}  sd={sd_p:.3f}")
Xc=np.column_stack([paceTot,[r["eH0"]+r["eA0"] for r in rows],np.ones(len(rows)),
                    [r["b2bH"]+r["b2bA"] for r in rows],[r["missH"]+r["missA"] for r in rows]])
_,_,mae_b,sd_b=report("blend",Xc,yTot)
print(f"  blend (pace + PF/PA)  MAE={mae_b:.3f}  sd={sd_b:.3f}")
best_mae=min(mae_p,mae_b); gain=mae_v2-best_mae
print(f"  --> best pace-based MAE gain vs v2: {gain:+.3f} pts   (KILL CRITERION A: need >= 1.0)")
print(f"  VERDICT A: {'PASS - ship totals' if gain>=1.0 else 'FAIL - do not ship pace totals'}")

print("\n===== B. WIN PROBABILITY =====")
def ll_of(X,y=yW):
    c,*_=np.linalg.lstsq(X,yMar,rcond=None); mu=X@c
    k=fit_logistic(mu,y); p=sigmoid(k*mu)
    return -np.mean(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9)), float(np.mean((p>=0.5)==y))
Xm_v2=np.column_stack([[r["eH0"]-r["eA0"] for r in rows],np.ones(len(rows)),
                       [r["b2bH"]-r["b2bA"] for r in rows],[r["missH"]-r["missA"] for r in rows]])
ll2,ac2=ll_of(Xm_v2)
print(f"  v2   logloss={ll2:.4f} acc={ac2:.3f}")
paceMar=np.array([r["pcExp"]*(r["ortH"]-r["ortA"])/100.0 for r in rows])
Xm_p=np.column_stack([paceMar,np.ones(len(rows)),
                      [r["b2bH"]-r["b2bA"] for r in rows],[r["missH"]-r["missA"] for r in rows]])
llp,acp=ll_of(Xm_p)
print(f"  pace logloss={llp:.4f} acc={acp:.3f}")
Xm_b=np.column_stack([paceMar,[r["eH0"]-r["eA0"] for r in rows],np.ones(len(rows)),
                      [r["b2bH"]-r["b2bA"] for r in rows],[r["missH"]-r["missA"] for r in rows]])
llb,acb=ll_of(Xm_b)
print(f"  blend logloss={llb:.4f} acc={acb:.3f}")
print(f"  VERDICT B: {'PASS - update ML path' if min(llp,llb)<ll2-0.002 else 'FAIL - keep ML on v2'}")

print("\n===== C. TOTALS BETTING VIABILITY =====")
cB,predB,maeB,sdB=report("best",Xc if mae_b<=mae_p else Xb,yTot)
print(f"  chosen total model: MAE={maeB:.3f} sd={sdB:.3f}")
from math import erf,sqrt
def ncdf(x,m,s): return 0.5*(1+erf((x-m)/(s*sqrt(2))))
# calibration of P(total > x)
print("  calibration of P(total > line):")
for off in [-10,-5,0,5,10]:
    lines=predB+off
    mp=np.array([1-ncdf(l,m,sdB) for l,m in zip(lines,predB)])
    emp=np.mean(yTot>lines)
    print(f"    line = proj{off:+3d}: model={mp.mean():.3f} empirical={emp:.3f} diff={mp.mean()-emp:+.3f}")
print(f"  NOTE: a totals edge needs |proj - line| big enough to clear ~4.5% vig;")
print(f"        with sd={sdB:.1f}, 1 pt of total ~= {(1-ncdf(0.5,0,sdB))-(1-ncdf(-0.5,0,sdB)):.3%} of probability")
