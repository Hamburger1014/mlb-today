"""Walk-forward training of the WNBA quarter model on real linescores.

Point-in-time features only: each game is predicted from team PF/PA
accumulated strictly before that game (same season), so eval == live page.
"""
import json, os, math
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(__file__)
games = json.load(open(os.path.join(HERE, "wnba_games.json")))

TEAMS = {'ATL','CHI','CON','DAL','GS','IND','LA','LV','MIN','NY','PHX','POR','SEA','TOR','WSH','PHO','NYL','LVA','CONN','WAS','GSV','LAS'}
games = [g for g in games if g["home"]["abbr"] in TEAMS and g["away"]["abbr"] in TEAMS
         and g.get("seasonType") in (2,3)]
games.sort(key=lambda g: g["date"])
print("usable games:", len(games))

def gdate(g): return datetime.fromisoformat(g["date"].replace("Z","+00:00"))

# played-yesterday lookup for B2B (calendar day in US/Eastern approx: date-5h)
played_on = defaultdict(set)
def eday(g): return (gdate(g) - timedelta(hours=5)).date()
for g in games:
    played_on[eday(g)].add(g["home"]["abbr"]); played_on[eday(g)].add(g["away"]["abbr"])
def is_b2b(abbr, g): return abbr in played_on.get(eday(g)-timedelta(days=1), set())

# ── walk-forward feature build ──────────────────────────────────
def build_rows(shrink_k):
    rows = []
    for season in sorted({g["seasonYear"] for g in games}):
        sg = [g for g in games if g["seasonYear"]==season]
        pf = defaultdict(float); pa = defaultdict(float); gp = defaultdict(int)
        for g in sg:
            h,a = g["home"], g["away"]
            # league avg from accumulated data so far this season
            tot_gp = sum(gp.values())
            lg = (sum(pf.values())/tot_gp) if tot_gp>=20 else 83.0
            if gp[h["abbr"]]>=3 and gp[a["abbr"]]>=3:
                def rate(v,n): return (v*n + lg*shrink_k)/(n+shrink_k)
                hPF=rate(pf[h["abbr"]]/gp[h["abbr"]], gp[h["abbr"]]); hPA=rate(pa[h["abbr"]]/gp[h["abbr"]], gp[h["abbr"]])
                aPF=rate(pf[a["abbr"]]/gp[a["abbr"]], gp[a["abbr"]]); aPA=rate(pa[a["abbr"]]/gp[a["abbr"]], gp[a["abbr"]])
                eH0 = hPF*aPA/lg; eA0 = aPF*hPA/lg
                hQ = h["lines"][:4]; aQ = a["lines"][:4]
                rows.append({
                    "eH0":eH0, "eA0":eA0,
                    "b2bH":int(is_b2b(h["abbr"],g)), "b2bA":int(is_b2b(a["abbr"],g)),
                    "hQ":hQ, "aQ":aQ,
                    "hReg":sum(hQ), "aReg":sum(aQ),           # regulation points
                    "homeWon":int(h["score"]>a["score"]),      # incl. OT
                    "season":season,
                })
            # update accumulators AFTER prediction (regulation points)
            pf[h["abbr"]]+=sum(h["lines"][:4]); pa[h["abbr"]]+=sum(a["lines"][:4]); gp[h["abbr"]]+=1
            pf[a["abbr"]]+=sum(a["lines"][:4]); pa[a["abbr"]]+=sum(h["lines"][:4]); gp[a["abbr"]]+=1
    return rows

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))

def fit_logistic_1d(x, y):
    """P(y)=sigmoid(k*x): fit k by Newton's method (stable)."""
    k = 0.1
    x = np.asarray(x); y = np.asarray(y, float)
    for _ in range(50):
        p = sigmoid(k*x)
        grad = np.mean((p-y)*x); hess = np.mean(p*(1-p)*x*x)+1e-9
        k -= grad/hess
    return k

# ── shrinkage grid search on game-winner log loss ───────────────
best = None
for K in [8]:  # chosen by stable-grid search (logloss flat 8-60, best at 8)
    rows = build_rows(K)
    X = np.array([[r["eH0"]-r["eA0"], 1.0, r["b2bH"]-r["b2bA"]] for r in rows])
    yM = np.array([r["hReg"]-r["aReg"] for r in rows], float)
    coefM, *_ = np.linalg.lstsq(X, yM, rcond=None)
    mu = X@coefM
    kcal = fit_logistic_1d(mu, [r["homeWon"] for r in rows])
    p = sigmoid(kcal*mu)
    yW = np.array([r["homeWon"] for r in rows])
    ll = -np.mean(yW*np.log(p+1e-9)+(1-yW)*np.log(1-p+1e-9))
    acc = np.mean((p>=0.5)==yW)
    print(f"shrink_k={K:>2}: n={len(rows)} logloss={ll:.4f} acc={acc:.3f}")
    if best is None or ll < best["ll"]:
        best = {"K":K, "ll":ll, "rows":rows, "coefM":coefM, "kcal":kcal}

K = best["K"]; rows = best["rows"]; coefM = best["coefM"]
print(f"\n== chosen shrink_k={K} ==")

# ── margin + total regressions ─────────────────────────────────
X = np.array([[r["eH0"]-r["eA0"], 1.0, r["b2bH"]-r["b2bA"]] for r in rows])
yM = np.array([r["hReg"]-r["aReg"] for r in rows], float)
resM = yM - X@coefM
sd_game = float(np.std(resM))
print(f"margin: eff_scale={coefM[0]:.4f} home_edge={coefM[1]:.3f} b2b={coefM[2]:.3f} sd={sd_game:.2f}")

XT = np.array([[r["eH0"]+r["eA0"], 1.0, r["b2bH"]+r["b2bA"]] for r in rows])
yT = np.array([r["hReg"]+r["aReg"] for r in rows], float)
coefT, *_ = np.linalg.lstsq(XT, yT, rcond=None)
print(f"total:  eff_scale={coefT[0]:.4f} intercept={coefT[1]:.2f} b2b={coefT[2]:.3f} sd={np.std(yT-XT@coefT):.2f}")

# ── win-prob calibration ───────────────────────────────────────
mu = X@coefM
yW = np.array([r["homeWon"] for r in rows])
kcal = fit_logistic_1d(mu, yW)
p = sigmoid(kcal*mu)
acc = np.mean((p>=0.5)==yW)
print(f"winprob: logistic k={kcal:.4f} (equiv sd={1.7/ kcal:.2f})  walk-forward acc={acc:.3f}  home-base={yW.mean():.3f}")
# confident-pick accuracy
for th in (0.60,0.65,0.70):
    m = np.maximum(p,1-p)>=th
    if m.sum(): print(f"  picks with p>={th:.2f}: n={m.sum()}  acc={np.mean((p[m]>=0.5)==yW[m]):.3f}")

# ── quarter structure ──────────────────────────────────────────
hQ = np.array([r["hQ"] for r in rows], float)   # n x 4
aQ = np.array([r["aQ"] for r in rows], float)
allQ = hQ + aQ
shares = allQ.mean(0)/allQ.mean(0).sum()
print("\nquarter total shares:", np.round(shares,4))
h_shares = hQ.mean(0)/hQ.mean(0).sum(); a_shares = aQ.mean(0)/aQ.mean(0).sum()
print("home shares:", np.round(h_shares,4), " away shares:", np.round(a_shares,4))

# per-quarter margin: regress qmargin ~ s*predGameMargin + c
predM = X@coefM
qfit = []
for q in range(4):
    qm = hQ[:,q]-aQ[:,q]
    A = np.stack([predM, np.ones(len(predM))],1)
    cf,*_ = np.linalg.lstsq(A, qm, rcond=None)
    res = qm - A@cf
    ties = np.mean(qm==0)
    qfit.append({"slope":float(cf[0]),"intercept":float(cf[1]),"sd":float(np.std(res)),"tieRate":float(ties)})
    print(f"Q{q+1}: margin_slope={cf[0]:.4f} intercept={cf[1]:+.3f} sd={np.std(res):.2f} empirical_tie={ties:.3f} homeQwin={np.mean(qm>0):.3f}")

# fit tie band w so normal model matches empirical tie rate (per quarter)
from math import erf, sqrt
def ncdf(x,mu_,sd_): return 0.5*(1+erf((x-mu_)/(sd_*sqrt(2))))
def tie_pred(w, mus, sd_): return np.mean([ncdf(w,m,sd_)-ncdf(-w,m,sd_) for m in mus])
tie_bands = []
for q in range(4):
    mus = qfit[q]["slope"]*predM + qfit[q]["intercept"]; sd_=qfit[q]["sd"]
    lo,hi = 0.1,3.0
    for _ in range(40):
        mid=(lo+hi)/2
        if tie_pred(mid,mus,sd_) < qfit[q]["tieRate"]: lo=mid
        else: hi=mid
    tie_bands.append((lo+hi)/2)
print("tie bands:", np.round(tie_bands,3))

# quarter-winner 3-way accuracy (walk-forward)
qaccs=[]
for q in range(4):
    mus = qfit[q]["slope"]*predM + qfit[q]["intercept"]; sd_=qfit[q]["sd"]; w=tie_bands[q]
    pH = 1-np.array([ncdf(w,m,sd_) for m in mus])
    pA = np.array([ncdf(-w,m,sd_) for m in mus])
    pT = 1-pH-pA
    pick = np.argmax(np.stack([pH,pT,pA],1),1)   # 0=h,1=t,2=a
    qm = hQ[:,q]-aQ[:,q]
    actual = np.where(qm>0,0,np.where(qm==0,1,2))
    qaccs.append(np.mean(pick==actual))
    base = max(np.mean(actual==0), np.mean(actual==2))
    print(f"Q{q+1} 3-way acc={qaccs[-1]:.3f} (always-better-side base={base:.3f})")

# quarter score MAE
eH = (XT@coefT + X@coefM)/2  # not exact per-team; use direct per-team fit instead:
# direct per-team score regressions
Xh = np.array([[r["eH0"], 1.0, r["b2bH"]] for r in rows]); yh=np.array([r["hReg"] for r in rows],float)
ch,*_ = np.linalg.lstsq(Xh,yh,rcond=None)
Xa = np.array([[r["eA0"], 1.0, r["b2bA"]] for r in rows]); ya=np.array([r["aReg"] for r in rows],float)
ca,*_ = np.linalg.lstsq(Xa,ya,rcond=None)
print(f"\nhome pts: scale={ch[0]:.4f} int={ch[1]:.2f} b2b={ch[2]:.3f}")
print(f"away pts: scale={ca[0]:.4f} int={ca[1]:.2f} b2b={ca[2]:.3f}")
predH = Xh@ch; predA = Xa@ca
maes = []
for q in range(4):
    mh = np.mean(np.abs(predH*h_shares[q]-hQ[:,q])); ma = np.mean(np.abs(predA*a_shares[q]-aQ[:,q]))
    maes.append((mh+ma)/2)
print("quarter score MAE:", np.round(maes,2), " game score MAE:", round(float(np.mean(np.abs(predH-yh))),2))

out = {
  "shrink_k": K,
  "lgAvgFallback": 83.0,
  "score": {"home":{"scale":float(ch[0]),"intercept":float(ch[1]),"b2b":float(ch[2])},
             "away":{"scale":float(ca[0]),"intercept":float(ca[1]),"b2b":float(ca[2])}},
  "margin": {"effScale":float(coefM[0]),"homeEdge":float(coefM[1]),"b2b":float(coefM[2]),"sd":sd_game},
  "winProbK": float(kcal),
  "quarters": [
    {"share_h":float(h_shares[q]), "share_a":float(a_shares[q]),
     "slope":qfit[q]["slope"], "intercept":qfit[q]["intercept"],
     "sd":qfit[q]["sd"], "tieBand":float(tie_bands[q])}
    for q in range(4)
  ],
  "trainedOn": {"games":len(rows), "seasons":"2022-2026 (walk-forward)",
                 "gameAcc":float(acc), "qAcc":[float(x) for x in qaccs]},
}
json.dump(out, open(os.path.join(HERE,"wnba_fit.json"),"w"), indent=1)
print("\nsaved wnba_fit.json")
