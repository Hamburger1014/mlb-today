"""WNBA model v2: recency-weighted ratings + prior-season carry + injury feature.

Walk-forward, point-in-time only. Injury feature = summed season PPG of
"regulars" (>=50% of team's prior games, >=3 apps, >=6 PPG) absent from the
game's boxscore — the live page mirrors this with ESPN's injury report.
"""
import json, os
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

HERE = os.path.dirname(__file__)
games = json.load(open(os.path.join(HERE, "wnba_games.json")))
box = json.load(open(os.path.join(HERE, "wnba_boxscores.json")))

TEAMS = {'ATL','CHI','CON','DAL','GS','IND','LA','LV','MIN','NY','PHX','POR','SEA','TOR','WSH'}
games = [g for g in games if g["home"]["abbr"] in TEAMS and g["away"]["abbr"] in TEAMS
         and g.get("seasonType") in (2,3)]
games.sort(key=lambda g: g["date"])
print("usable games:", len(games), "| boxscores:", sum(1 for g in games if box.get(g["id"])))

def gdate(g): return datetime.fromisoformat(g["date"].replace("Z","+00:00"))
played_on = defaultdict(set)
def eday(g): return (gdate(g)-timedelta(hours=5)).date()
for g in games:
    played_on[eday(g)].add(g["home"]["abbr"]); played_on[eday(g)].add(g["away"]["abbr"])
def is_b2b(ab,g): return ab in played_on.get(eday(g)-timedelta(days=1), set())

SHRINK_K = 8
LG_FALLBACK = 83.0

def build_rows(half_life, prior_w0, prior_r):
    lam = 0.5**(1.0/half_life)
    seasons = sorted({g["seasonYear"] for g in games})
    prior_rating = {}   # abbr -> (pf,pa) final decayed rating of previous season
    rows = []
    for season in seasons:
        sg = [g for g in games if g["seasonYear"]==season]
        wPF=defaultdict(float); wPA=defaultdict(float); wN=defaultdict(float)
        # per-season player tracking: pid -> [appearances, points]; team games count
        pstats=defaultdict(lambda: defaultdict(lambda: [0,0.0])); tgp=defaultdict(int)
        # seed with prior-season rating (regressed to league mean)
        for ab,(ppf,ppa) in prior_rating.items():
            wPF[ab]=(prior_r*ppf+(1-prior_r)*LG_FALLBACK)*prior_w0
            wPA[ab]=(prior_r*ppa+(1-prior_r)*LG_FALLBACK)*prior_w0
            wN[ab]=prior_w0
        pfS=0.0; nS=0
        for g in sg:
            h,a=g["home"],g["away"]; hb,ab_=h["abbr"],a["abbr"]
            lg=(pfS/nS) if nS>=20 else LG_FALLBACK
            bx=box.get(g["id"]) or {}
            def missing_ppg(abbr):
                if tgp[abbr] < 4: return 0.0
                played={p[0] for p in (bx.get(abbr) or []) if p[1] and p[1]>0}
                if not played: return 0.0     # no boxscore -> feature unavailable
                m=0.0
                for pid,(apps,pts) in pstats[abbr].items():
                    if apps<3 or apps/tgp[abbr] < 0.5: continue
                    ppg=pts/apps
                    if ppg<6: continue
                    if pid not in played: m+=ppg
                return min(m, 30.0)
            ok = wN[hb]>=3 and wN[ab_]>=3
            if ok:
                def rate(w,n): return ((w/n)*n + lg*SHRINK_K)/(n+SHRINK_K)
                hPF=rate(wPF[hb],wN[hb]); hPA=rate(wPA[hb],wN[hb])
                aPF=rate(wPF[ab_],wN[ab_]); aPA=rate(wPA[ab_],wN[ab_])
                rows.append({
                    "eH0":hPF*aPA/lg, "eA0":aPF*hPA/lg,
                    "b2bH":int(is_b2b(hb,g)), "b2bA":int(is_b2b(ab_,g)),
                    "missH":missing_ppg(hb), "missA":missing_ppg(ab_),
                    "hQ":h["lines"][:4], "aQ":a["lines"][:4],
                    "hReg":sum(h["lines"][:4]), "aReg":sum(a["lines"][:4]),
                    "homeWon":int(h["score"]>a["score"]),
                    "gpH":tgp[hb], "season":season,
                })
            # update state AFTER prediction
            hpts=sum(h["lines"][:4]); apts=sum(a["lines"][:4])
            for abbr,pf_g,pa_g in [(hb,hpts,apts),(ab_,apts,hpts)]:
                wPF[abbr]=wPF[abbr]*lam+pf_g; wPA[abbr]=wPA[abbr]*lam+pa_g; wN[abbr]=wN[abbr]*lam+1
            for abbr in (hb,ab_):
                for p in (bx.get(abbr) or []):
                    if p[1] and p[1]>0:
                        ps=pstats[abbr][p[0]]; ps[0]+=1; ps[1]+=p[2]
                tgp[abbr]+=1
            pfS+=hpts+apts; nS+=2
        prior_rating={ab2:(wPF[ab2]/wN[ab2], wPA[ab2]/wN[ab2]) for ab2 in wN if wN[ab2]>0}
    return rows

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
def fit_logistic(x,y):
    k=0.1; x=np.asarray(x); y=np.asarray(y,float)
    for _ in range(60):
        p=sigmoid(k*x); g=np.mean((p-y)*x); hh=np.mean(p*(1-p)*x*x)+1e-9
        k-=g/hh
    return k

def feats(rows):
    X=np.array([[r["eH0"]-r["eA0"],1.0,r["b2bH"]-r["b2bA"],r["missH"]-r["missA"]] for r in rows])
    yM=np.array([r["hReg"]-r["aReg"] for r in rows],float)
    yW=np.array([r["homeWon"] for r in rows])
    return X,yM,yW

def evaluate(rows):
    X,yM,yW=feats(rows)
    cM,*_=np.linalg.lstsq(X,yM,rcond=None)
    mu=X@cM; k=fit_logistic(mu,yW); p=sigmoid(k*mu)
    ll=-np.mean(yW*np.log(p+1e-9)+(1-yW)*np.log(1-p+1e-9))
    acc=float(np.mean((p>=0.5)==yW))
    return ll,acc,cM,k,mu

print("\n== grid: half-life x prior ==")
best=None
for h in [6,10,15,25,10000]:
    for (w0,r) in [(0,0.0),(4,0.6),(8,0.6),(8,0.75)]:
        rows=build_rows(h,w0,r)
        # comparable core: rows with real accumulated data
        core=[x for x in rows if x["gpH"]>=3]
        ll,acc,_,_,_=evaluate(core)
        tag=f"h={h:>5} W0={w0} r={r}"
        print(f"{tag}: n={len(rows)}(core {len(core)}) core-logloss={ll:.4f} acc={acc:.3f}")
        if best is None or ll<best["ll"]:
            best={"ll":ll,"h":h,"w0":w0,"r":r}
print("chosen:",best)

H,W0,R = best["h"],best["w0"],best["r"]
rows=build_rows(H,W0,R)
X,yM,yW=feats(rows)
cM,*_=np.linalg.lstsq(X,yM,rcond=None)
mu=X@cM; kcal=fit_logistic(mu,yW); p=sigmoid(kcal*mu)
resM=yM-mu
print(f"\nmargin: eff={cM[0]:.4f} homeEdge={cM[1]:.3f} b2b={cM[2]:.3f} injMiss={cM[3]:.4f} sd={np.std(resM):.2f}")
acc=float(np.mean((p>=0.5)==yW))
print(f"winprob k={kcal:.4f} walk-forward acc={acc:.3f} (home base {yW.mean():.3f}) n={len(rows)}")
for th in (0.60,0.65,0.70):
    m=np.maximum(p,1-p)>=th
    if m.sum(): print(f"  p>={th}: n={m.sum()} acc={np.mean((p[m]>=0.5)==yW[m]):.3f}")
# injury feature sanity
mm=np.array([r["missH"]-r["missA"] for r in rows])
print(f"inj feature: nonzero in {np.mean(mm!=0)*100:.0f}% of games, mean|missDiff|={np.mean(np.abs(mm)):.1f} ppg")

# total + per-team scores
XT=np.array([[r["eH0"]+r["eA0"],1.0,r["b2bH"]+r["b2bA"],r["missH"]+r["missA"]] for r in rows])
yT=np.array([r["hReg"]+r["aReg"] for r in rows],float)
cT,*_=np.linalg.lstsq(XT,yT,rcond=None)
print(f"total: eff={cT[0]:.4f} int={cT[1]:.2f} b2b={cT[2]:.3f} miss={cT[3]:.4f} sd={np.std(yT-XT@cT):.2f}")

# quarters vs predicted margin
hQ=np.array([r["hQ"] for r in rows],float); aQ=np.array([r["aQ"] for r in rows],float)
h_sh=hQ.mean(0)/hQ.mean(0).sum(); a_sh=aQ.mean(0)/aQ.mean(0).sum()
from math import erf,sqrt
def ncdf(x,m,s): return 0.5*(1+erf((x-m)/(s*sqrt(2))))
qout=[]
for q in range(4):
    qm=hQ[:,q]-aQ[:,q]
    A=np.stack([mu,np.ones(len(mu))],1)
    cf,*_=np.linalg.lstsq(A,qm,rcond=None)
    sd=float(np.std(qm-A@cf)); tie=float(np.mean(qm==0))
    lo,hi=0.1,3.0
    for _ in range(40):
        mid=(lo+hi)/2
        pred=np.mean([ncdf(mid,m,sd)-ncdf(-mid,m,sd) for m in cf[0]*mu+cf[1]])
        lo,hi=(mid,hi) if pred<tie else (lo,mid)
    qout.append({"share_h":float(h_sh[q]),"share_a":float(a_sh[q]),
                 "slope":float(cf[0]),"intercept":float(cf[1]),"sd":sd,"tieBand":(lo+hi)/2})
    mus=cf[0]*mu+cf[1]
    pH=1-np.array([ncdf(qout[q]["tieBand"],m,sd) for m in mus])
    pA=np.array([ncdf(-qout[q]["tieBand"],m,sd) for m in mus])
    pick=np.argmax(np.stack([pH,1-pH-pA,pA],1),1)
    actual=np.where(qm>0,0,np.where(qm==0,1,2))
    print(f"Q{q+1}: slope={cf[0]:.4f} int={cf[1]:+.3f} sd={sd:.2f} 3way-acc={np.mean(pick==actual):.3f}")

# 2025-final decayed priors for live early-season use
lam=0.5**(1.0/H)
prior={}
for season in [2025]:
    sg=[g for g in games if g["seasonYear"]==season]
    wPF=defaultdict(float);wPA=defaultdict(float);wN=defaultdict(float)
    for g in sg:
        h,a=g["home"],g["away"]
        hp=sum(h["lines"][:4]);ap=sum(a["lines"][:4])
        for abbr,pf_g,pa_g in [(h["abbr"],hp,ap),(a["abbr"],ap,hp)]:
            wPF[abbr]=wPF[abbr]*lam+pf_g;wPA[abbr]=wPA[abbr]*lam+pa_g;wN[abbr]=wN[abbr]*lam+1
    prior={ab:( round(wPF[ab]/wN[ab],2), round(wPA[ab]/wN[ab],2)) for ab in wN}

out={
 "halfLife":H,"priorW0":W0,"priorR":R,"shrinkK":SHRINK_K,"lgFallback":LG_FALLBACK,
 "margin":{"eff":float(cM[0]),"homeEdge":float(cM[1]),"b2b":float(cM[2]),"injMiss":float(cM[3]),"sd":float(np.std(resM))},
 "total":{"eff":float(cT[0]),"intercept":float(cT[1]),"b2b":float(cT[2]),"injMiss":float(cT[3])},
 "winProbK":float(kcal),
 "quarters":qout,
 "priors2025":prior,
 "metrics":{"games":len(rows),"acc":acc,"homeBase":float(yW.mean())},
}
json.dump(out,open(os.path.join(HERE,"wnba_fit_v2.json"),"w"),indent=1)
print("\nsaved wnba_fit_v2.json")
