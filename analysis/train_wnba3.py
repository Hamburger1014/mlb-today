"""WNBA model v3.

Improvements over v2:
 1. Injury-adjusted ("full-strength") ratings — each past game's scores are
    corrected for who was missing BEFORE entering the rating, so a team that
    played hurt isn't permanently underrated and a returning star lifts the
    projection immediately (symmetric: handles outs AND returns).
 2. Ridge-solved simultaneous off/def ratings (proper strength of schedule).
 3. Blowout capping in the ratings (garbage-time noise control).
 4. Q4 closeness interaction (slope depends on projected competitiveness).

Everything remains walk-forward / point-in-time: features for game i use only
information available before game i.
"""
import json, os
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

HERE = os.path.dirname(__file__)
games = json.load(open(os.path.join(HERE, "wnba_games.json")))
box   = json.load(open(os.path.join(HERE, "wnba_boxscores.json")))

TEAMS = {'ATL','CHI','CON','DAL','GS','IND','LA','LV','MIN','NY','PHX','POR','SEA','TOR','WSH'}
games = [g for g in games if g["home"]["abbr"] in TEAMS and g["away"]["abbr"] in TEAMS
         and g.get("seasonType") in (2,3)]
games.sort(key=lambda g: g["date"])

def gdate(g): return datetime.fromisoformat(g["date"].replace("Z","+00:00"))
played_on = defaultdict(set)
def eday(g): return (gdate(g)-timedelta(hours=5)).date()
for g in games:
    played_on[eday(g)].add(g["home"]["abbr"]); played_on[eday(g)].add(g["away"]["abbr"])
def is_b2b(ab,g): return ab in played_on.get(eday(g)-timedelta(days=1), set())

SHRINK_K   = 8
LG_FALL    = 83.0
BLOWOUT_CAP= 20.0     # cap |margin| contribution when updating ratings
PPG_MIN    = 6.0      # a "regular" scorer
APP_FRAC   = 0.5

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
def fit_logistic(x,y):
    k=0.1; x=np.asarray(x); y=np.asarray(y,float)
    for _ in range(60):
        p=sigmoid(k*x); g=np.mean((p-y)*x); h=np.mean(p*(1-p)*x*x)+1e-9
        k-=g/h
    return k

def build(half_life, prior_w0, prior_r, inj_coef, cap_blowouts=True):
    """inj_coef = points of team scoring lost per PPG missing (used to
    de-injure history). Returns feature rows."""
    lam=0.5**(1.0/half_life)
    prior_rating={}
    rows=[]
    for season in sorted({g["seasonYear"] for g in games}):
        sg=[g for g in games if g["seasonYear"]==season]
        wPF=defaultdict(float); wPA=defaultdict(float); wN=defaultdict(float)
        pstats=defaultdict(lambda: defaultdict(lambda:[0,0.0])); tgp=defaultdict(int)
        for ab,(ppf,ppa) in prior_rating.items():
            wPF[ab]=(prior_r*ppf+(1-prior_r)*LG_FALL)*prior_w0
            wPA[ab]=(prior_r*ppa+(1-prior_r)*LG_FALL)*prior_w0
            wN[ab]=prior_w0
        pfS=0.0; nS=0
        for g in sg:
            h,a=g["home"],g["away"]; hb,ab_=h["abbr"],a["abbr"]
            lg=(pfS/nS) if nS>=20 else LG_FALL
            bx=box.get(g["id"]) or {}
            def missing_ppg(abbr):
                if tgp[abbr]<4: return 0.0
                played={p[0] for p in (bx.get(abbr) or []) if p[1] and p[1]>0}
                if not played: return 0.0
                m=0.0
                for pid,(apps,pts) in pstats[abbr].items():
                    if apps<3 or apps/tgp[abbr]<APP_FRAC: continue
                    ppg=pts/apps
                    if ppg<PPG_MIN: continue
                    if pid not in played: m+=ppg
                return min(m,30.0)
            missH=missing_ppg(hb); missA=missing_ppg(ab_)
            ok = wN[hb]>=3 and wN[ab_]>=3
            if ok:
                def rate(w,n): return ((w/n)*n+lg*SHRINK_K)/(n+SHRINK_K)
                hPF=rate(wPF[hb],wN[hb]); hPA=rate(wPA[hb],wN[hb])
                aPF=rate(wPF[ab_],wN[ab_]); aPA=rate(wPA[ab_],wN[ab_])
                rows.append({
                    "hb":hb,"ab":ab_,
                    "eH0":hPF*aPA/lg,"eA0":aPF*hPA/lg,
                    "b2bH":int(is_b2b(hb,g)),"b2bA":int(is_b2b(ab_,g)),
                    "missH":missH,"missA":missA,
                    "hQ":h["lines"][:4],"aQ":a["lines"][:4],
                    "hReg":sum(h["lines"][:4]),"aReg":sum(a["lines"][:4]),
                    "homeWon":int(h["score"]>a["score"]),
                    "gpH":tgp[hb],"season":season,
                })
            # ── update ratings with INJURY-ADJUSTED, blowout-capped scores ──
            hp=sum(h["lines"][:4]); ap=sum(a["lines"][:4])
            hp_adj=hp+inj_coef*missH          # what they'd have scored at full strength
            ap_adj=ap+inj_coef*missA
            # opponent's points allowed also reflects the opponent's own absences
            if cap_blowouts:
                mid=(hp_adj+ap_adj)/2.0; d=(hp_adj-ap_adj)/2.0
                d=max(-BLOWOUT_CAP/2, min(BLOWOUT_CAP/2, d))
                hp_adj, ap_adj = mid+d, mid-d
            for abbr,pf_g,pa_g in [(hb,hp_adj,ap_adj),(ab_,ap_adj,hp_adj)]:
                wPF[abbr]=wPF[abbr]*lam+pf_g; wPA[abbr]=wPA[abbr]*lam+pa_g; wN[abbr]=wN[abbr]*lam+1
            for abbr in (hb,ab_):
                for p in (bx.get(abbr) or []):
                    if p[1] and p[1]>0:
                        s=pstats[abbr][p[0]]; s[0]+=1; s[1]+=p[2]
                tgp[abbr]+=1
            pfS+=hp+ap; nS+=2
        prior_rating={x:(wPF[x]/wN[x],wPA[x]/wN[x]) for x in wN if wN[x]>0}
    return rows

def design(rows):
    X=np.array([[r["eH0"]-r["eA0"],1.0,r["b2bH"]-r["b2bA"],r["missH"]-r["missA"]] for r in rows])
    yM=np.array([r["hReg"]-r["aReg"] for r in rows],float)
    yW=np.array([r["homeWon"] for r in rows])
    return X,yM,yW

def score_cfg(rows):
    X,yM,yW=design(rows)
    c,*_=np.linalg.lstsq(X,yM,rcond=None)
    mu=X@c; k=fit_logistic(mu,yW); p=sigmoid(k*mu)
    ll=-np.mean(yW*np.log(p+1e-9)+(1-yW)*np.log(1-p+1e-9))
    return ll, float(np.mean((p>=0.5)==yW)), c, k

print("== v2 baseline (no de-injure, no cap) ==")
base=build(25,4,0.6,0.0,cap_blowouts=False)
bll,bacc,_,_=score_cfg(base)
print(f"  logloss={bll:.4f} acc={bacc:.3f} n={len(base)}")

print("\n== grid: de-injure coefficient x blowout cap ==")
best=None
for inj in [0.0,0.3,0.5,0.7,1.0]:
    for cap in [False,True]:
        rows=build(25,4,0.6,inj,cap_blowouts=cap)
        ll,acc,_,_=score_cfg(rows)
        print(f"  inj={inj:.1f} cap={str(cap):5}: logloss={ll:.4f} acc={acc:.3f}")
        if best is None or ll<best[0]: best=(ll,acc,inj,cap)
print("best:",best)
INJ,CAP=best[2],best[3]

print("\n== half-life re-tune with chosen inj/cap ==")
bh=None
for h in [10,15,25,40]:
    rows=build(h,4,0.6,INJ,cap_blowouts=CAP)
    ll,acc,_,_=score_cfg(rows)
    print(f"  h={h}: logloss={ll:.4f} acc={acc:.3f}")
    if bh is None or ll<bh[0]: bh=(ll,h)
H=bh[1]; print("chosen half-life:",H)

rows=build(H,4,0.6,INJ,cap_blowouts=CAP)
X,yM,yW=design(rows)
cM,*_=np.linalg.lstsq(X,yM,rcond=None)
mu=X@cM; kcal=fit_logistic(mu,yW); p=sigmoid(kcal*mu)
acc=float(np.mean((p>=0.5)==yW))
print(f"\nmargin: eff={cM[0]:.4f} home={cM[1]:.3f} b2b={cM[2]:.3f} inj={cM[3]:.4f} sd={np.std(yM-mu):.2f}")
print(f"winprob k={kcal:.4f}  acc={acc:.3f}  (home base {yW.mean():.3f})  n={len(rows)}")
for th in (0.60,0.65,0.70):
    m=np.maximum(p,1-p)>=th
    if m.sum(): print(f"  p>={th}: n={m.sum()} acc={np.mean((p[m]>=0.5)==yW[m]):.3f}")

# ── ridge simultaneous off/def ratings as an ALTERNATIVE margin feature ──
print("\n== ridge off/def (strength of schedule) ==")
teams=sorted(TEAMS); ti={t:i for i,t in enumerate(teams)}
def ridge_eval(alpha):
    # walk-forward per season: refit ratings on games so far, predict next
    preds=[]; acts=[]; wins=[]
    for season in sorted({r["season"] for r in rows}):
        sr=[r for r in rows if r["season"]==season]
        for i,r in enumerate(sr):
            if i<40: continue
            hist=sr[:i]
            n=len(hist); T=len(teams)
            A=np.zeros((n,2*T+1)); b=np.zeros(n)
            for j,q in enumerate(hist):
                A[j,ti[q["hb"]]]=1; A[j,T+ti[q["ab"]]]=-1
                A[j,ti[q["ab"]]]-=1; A[j,T+ti[q["hb"]]]+=1
                A[j,-1]=1
                b[j]=(q["hReg"]+INJ*q["missH"])-(q["aReg"]+INJ*q["missA"])
            R=np.eye(2*T+1)*alpha; R[-1,-1]=0
            sol=np.linalg.solve(A.T@A+R, A.T@b)
            off,dfn,hme=sol[:T],sol[T:2*T],sol[-1]
            m=(off[ti[r["hb"]]]-dfn[ti[r["ab"]]])-(off[ti[r["ab"]]]-dfn[ti[r["hb"]]])+hme
            preds.append(m); acts.append(r["hReg"]-r["aReg"]); wins.append(r["homeWon"])
    preds=np.array(preds); wins=np.array(wins)
    k=fit_logistic(preds,wins); pp=sigmoid(k*preds)
    ll=-np.mean(wins*np.log(pp+1e-9)+(1-wins)*np.log(1-pp+1e-9))
    return ll,float(np.mean((pp>=0.5)==wins)),len(preds)
for alpha in [25,50,100,200]:
    ll,a2,n2=ridge_eval(alpha)
    print(f"  alpha={alpha}: logloss={ll:.4f} acc={a2:.3f} (n={n2})")

# ── quarters, with Q4 closeness interaction ──
hQ=np.array([r["hQ"] for r in rows],float); aQ=np.array([r["aQ"] for r in rows],float)
h_sh=hQ.mean(0)/hQ.mean(0).sum(); a_sh=aQ.mean(0)/aQ.mean(0).sum()
from math import erf,sqrt
def ncdf(x,m,s): return 0.5*(1+erf((x-m)/(s*sqrt(2))))
absmu=np.abs(mu)
qout=[]
print("\n== quarters (with closeness interaction) ==")
for q in range(4):
    qm=hQ[:,q]-aQ[:,q]
    A1=np.stack([mu,np.ones(len(mu))],1)
    c1,*_=np.linalg.lstsq(A1,qm,rcond=None)
    sd1=np.std(qm-A1@c1)
    # interaction: slope shrinks as |projected margin| grows (garbage time)
    A2=np.stack([mu, mu*absmu/10.0, np.ones(len(mu))],1)
    c2,*_=np.linalg.lstsq(A2,qm,rcond=None)
    sd2=np.std(qm-A2@c2)
    use2 = sd2 < sd1-0.005
    cf = c2 if use2 else np.array([c1[0],0.0,c1[1]])
    sd = float(sd2 if use2 else sd1)
    tie=float(np.mean(qm==0))
    mus = cf[0]*mu + cf[1]*(mu*absmu/10.0) + cf[2]
    lo,hi=0.1,3.0
    for _ in range(40):
        mid=(lo+hi)/2
        pr=np.mean([ncdf(mid,m,sd)-ncdf(-mid,m,sd) for m in mus])
        lo,hi=(mid,hi) if pr<tie else (lo,mid)
    band=(lo+hi)/2
    pH=1-np.array([ncdf(band,m,sd) for m in mus]); pA=np.array([ncdf(-band,m,sd) for m in mus])
    pick=np.argmax(np.stack([pH,1-pH-pA,pA],1),1)
    actual=np.where(qm>0,0,np.where(qm==0,1,2))
    a3=float(np.mean(pick==actual))
    qout.append({"share_h":float(h_sh[q]),"share_a":float(a_sh[q]),
        "slope":float(cf[0]),"slopeAbs":float(cf[1]),"intercept":float(cf[2]),
        "sd":sd,"tieBand":float(band)})
    print(f"  Q{q+1}: slope={cf[0]:.4f} slopeAbs={cf[1]:+.5f} int={cf[2]:+.3f} sd={sd:.2f} 3way={a3:.3f} {'[interaction]' if use2 else ''}")

XT=np.array([[r["eH0"]+r["eA0"],1.0,r["b2bH"]+r["b2bA"],r["missH"]+r["missA"]] for r in rows])
yT=np.array([r["hReg"]+r["aReg"] for r in rows],float)
cT,*_=np.linalg.lstsq(XT,yT,rcond=None)
print(f"\ntotal: eff={cT[0]:.4f} int={cT[1]:.2f} b2b={cT[2]:.3f} inj={cT[3]:.4f}")

# priors from final 2025 (injury-adjusted, capped) ratings
lam=0.5**(1.0/H)
wPF=defaultdict(float);wPA=defaultdict(float);wN=defaultdict(float)
pstats=defaultdict(lambda: defaultdict(lambda:[0,0.0])); tgp=defaultdict(int)
for g in [x for x in games if x["seasonYear"]==2025]:
    h,a=g["home"],g["away"]; hb,ab_=h["abbr"],a["abbr"]
    bx=box.get(g["id"]) or {}
    def miss(abbr):
        if tgp[abbr]<4: return 0.0
        played={p[0] for p in (bx.get(abbr) or []) if p[1] and p[1]>0}
        if not played: return 0.0
        m=0.0
        for pid,(apps,pts) in pstats[abbr].items():
            if apps<3 or apps/tgp[abbr]<APP_FRAC: continue
            ppg=pts/apps
            if ppg>=PPG_MIN and pid not in played: m+=ppg
        return min(m,30.0)
    hp=sum(h["lines"][:4])+INJ*miss(hb); ap=sum(a["lines"][:4])+INJ*miss(ab_)
    if CAP:
        mid=(hp+ap)/2; d=max(-BLOWOUT_CAP/2,min(BLOWOUT_CAP/2,(hp-ap)/2)); hp,ap=mid+d,mid-d
    for abbr,pf_g,pa_g in [(hb,hp,ap),(ab_,ap,hp)]:
        wPF[abbr]=wPF[abbr]*lam+pf_g; wPA[abbr]=wPA[abbr]*lam+pa_g; wN[abbr]=wN[abbr]*lam+1
    for abbr in (hb,ab_):
        for p in (bx.get(abbr) or []):
            if p[1] and p[1]>0:
                s=pstats[abbr][p[0]]; s[0]+=1; s[1]+=p[2]
        tgp[abbr]+=1
priors={x:(round(wPF[x]/wN[x],2),round(wPA[x]/wN[x],2)) for x in wN}

out={"halfLife":H,"priorW0":4,"priorR":0.6,"shrinkK":SHRINK_K,"lgFallback":LG_FALL,
 "deInjureCoef":INJ,"blowoutCap":BLOWOUT_CAP if CAP else None,
 "margin":{"eff":float(cM[0]),"homeEdge":float(cM[1]),"b2b":float(cM[2]),
           "injMiss":float(cM[3]),"sd":float(np.std(yM-mu))},
 "total":{"eff":float(cT[0]),"intercept":float(cT[1]),"b2b":float(cT[2]),"injMiss":float(cT[3])},
 "winProbK":float(kcal),"quarters":qout,"priors2025":priors,
 "metrics":{"games":len(rows),"acc":acc,"homeBase":float(yW.mean()),
            "v2Acc":bacc,"v2Logloss":bll}}
json.dump(out,open(os.path.join(HERE,"wnba_fit_v3.json"),"w"),indent=1)
print("\nsaved wnba_fit_v3.json")
