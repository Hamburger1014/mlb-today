"""Live in-game win probability, fitted on real quarter-by-quarter states.

At each checkpoint (end of Q1/Q2/Q3) we know: the current lead, and the
pregame projected margin. The question is how to weight them — a pure
"extrapolate the lead" model overreacts to noise; a pure "trust the pregame
number" ignores real information. Fit the blend empirically.

  P(home wins) = sigmoid(alpha_k*lead_k + beta_k*mu_pregame + gamma_k)

Validation is an honest out-of-time split: fit on 2022-2024, test on
2025-2026, and compare against three baselines.
"""
import json, os
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

HERE=os.path.dirname(__file__)
games=json.load(open(os.path.join(HERE,"wnba_games.json")))
box=json.load(open(os.path.join(HERE,"wnba_boxscores.json")))
TEAMS={'ATL','CHI','CON','DAL','GS','IND','LA','LV','MIN','NY','PHX','POR','SEA','TOR','WSH'}
games=[g for g in games if g["home"]["abbr"] in TEAMS and g["away"]["abbr"] in TEAMS
       and g.get("seasonType") in (2,3)]
games.sort(key=lambda g:g["date"])
def gd(g): return datetime.fromisoformat(g["date"].replace("Z","+00:00"))
def ed(g): return (gd(g)-timedelta(hours=5)).date()
pl=defaultdict(set)
for g in games: pl[ed(g)].add(g["home"]["abbr"]); pl[ed(g)].add(g["away"]["abbr"])
def b2b(a,g): return a in pl.get(ed(g)-timedelta(days=1),set())
LG=83.0;K=8;LAM=0.5**(1/25)

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
def fit_logistic_multi(X,y,iters=200,l2=1e-3):
    """Newton-IRLS on a small design matrix."""
    w=np.zeros(X.shape[1])
    for _ in range(iters):
        p=sigmoid(X@w)
        g=X.T@(p-y)/len(y)+l2*w
        W=p*(1-p)
        H=(X*W[:,None]).T@X/len(y)+l2*np.eye(X.shape[1])
        try: step=np.linalg.solve(H,g)
        except np.linalg.LinAlgError: break
        w-=step
        if np.max(np.abs(step))<1e-9: break
    return w

# ── build rows with point-in-time pregame mu + quarter states ──
rows=[]
prior={}
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
        if wN[hb]>=3 and wN[ab_]>=3 and len(h["lines"])>=4 and len(a["lines"])>=4:
            def rt(w,n): return ((w/n)*n+lg*K)/(n+K)
            hPF=rt(wPF[hb],wN[hb]);hPA=rt(wPA[hb],wN[hb])
            aPF=rt(wPF[ab_],wN[ab_]);aPA=rt(wPA[ab_],wN[ab_])
            rows.append({"season":season,
              "eDiff":hPF*aPA/lg-aPF*hPA/lg,
              "b2b":int(b2b(hb,g))-int(b2b(ab_,g)),
              "miss":miss(hb)-miss(ab_),
              "hQ":h["lines"][:4],"aQ":a["lines"][:4],
              "homeWon":int(h["score"]>a["score"]),
              "mFin":h["score"]-a["score"]})
        hp=sum(h["lines"][:4]);ap=sum(a["lines"][:4])
        for abbr,pf_g,pa_g in [(hb,hp,ap),(ab_,ap,hp)]:
            wPF[abbr]=wPF[abbr]*LAM+pf_g;wPA[abbr]=wPA[abbr]*LAM+pa_g;wN[abbr]=wN[abbr]*LAM+1
        for abbr in (hb,ab_):
            for p in (bx.get(abbr) or []):
                if p[1] and p[1]>0:
                    s=pst[abbr][p[0]];s[0]+=1;s[1]+=p[2]
            tgp[abbr]+=1
        pfS+=hp+ap;nS+=2
    prior={x:(wPF[x]/wN[x],wPA[x]/wN[x]) for x in wN if wN[x]>0}

# pregame margin model (same features as deployed v2)
Xpre=np.array([[r["eDiff"],1.0,r["b2b"],r["miss"]] for r in rows])
yMarReg=np.array([sum(r["hQ"])-sum(r["aQ"]) for r in rows],float)
cPre,*_=np.linalg.lstsq(Xpre,yMarReg,rcond=None)
mu=Xpre@cPre
for i,r in enumerate(rows): r["mu"]=mu[i]
print(f"rows={len(rows)}  pregame margin coefs={np.round(cPre,4)}")

train=[r for r in rows if r["season"]<=2024]
test =[r for r in rows if r["season"]>=2025]
print(f"train(2022-24)={len(train)}  test(2025-26)={len(test)}")

def leads(r):
    """cumulative home-away lead after each quarter"""
    out=[];h=0;a=0
    for q in range(4):
        h+=r["hQ"][q];a+=r["aQ"][q];out.append(h-a)
    return out

print("\n===== in-game win probability by checkpoint =====")
FIT={}
for k in [1,2,3]:                      # after Q1, Q2, Q3
    def design(rs):
        X=np.array([[leads(r)[k-1], r["mu"], 1.0] for r in rs])
        y=np.array([r["homeWon"] for r in rs],float)
        return X,y
    Xtr,ytr=design(train); Xte,yte=design(test)
    w=fit_logistic_multi(Xtr,ytr)
    def metrics(X,y,w):
        p=sigmoid(X@w)
        ll=-np.mean(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9))
        return ll,float(np.mean((p>=0.5)==y))
    ll_te,acc_te=metrics(Xte,yte,w)
    # baselines on the SAME test rows
    lead_te=Xte[:,0]
    # (a) pregame only
    wpre=fit_logistic_multi(np.array([[r["mu"],1.0] for r in train]),ytr)
    ppre=sigmoid(np.array([[r["mu"],1.0] for r in test])@wpre)
    ll_pre=-np.mean(yte*np.log(ppre+1e-9)+(1-yte)*np.log(1-ppre+1e-9))
    acc_pre=float(np.mean((ppre>=0.5)==yte))
    # (b) lead only
    wl=fit_logistic_multi(np.array([[leads(r)[k-1],1.0] for r in train]),ytr)
    pl_=sigmoid(np.array([[leads(r)[k-1],1.0] for r in test])@wl)
    ll_lead=-np.mean(yte*np.log(pl_+1e-9)+(1-yte)*np.log(1-pl_+1e-9))
    acc_lead=float(np.mean((pl_>=0.5)==yte))
    # (c) naive: whoever leads
    acc_naive=float(np.mean((lead_te>0)==(yte==1)))
    print(f"\n  after Q{k}:  alpha(lead)={w[0]:.4f}  beta(mu)={w[1]:.4f}  gamma={w[2]:+.4f}")
    print(f"    combined   logloss={ll_te:.4f} acc={acc_te:.3f}   <-- fitted blend")
    print(f"    pregame    logloss={ll_pre:.4f} acc={acc_pre:.3f}")
    print(f"    lead only  logloss={ll_lead:.4f} acc={acc_lead:.3f}")
    print(f"    naive lead                 acc={acc_naive:.3f}")
    # implied weighting: 1 pt of lead is worth this much pregame margin
    print(f"    => 1 pt of current lead == {w[0]/w[1]:.2f} pts of pregame margin")
    FIT[k]={"lead":float(w[0]),"mu":float(w[1]),"int":float(w[2]),
            "testLogloss":ll_te,"testAcc":acc_te}

# ── remaining-quarter scoring, given state ──
print("\n===== remaining quarter margins (for the live grid) =====")
QFIT={}
for k in [1,2,3]:
    for j in range(k,4):               # predict quarter j+1 (0-indexed j)
        pass
for k in [1,2,3]:
    rem=[]
    for r in rows:
        L=leads(r)[k-1]
        for j in range(k,4):
            qm=r["hQ"][j]-r["aQ"][j]
            rem.append((k,j,L,r["mu"],qm))
    for j in range(k,4):
        sub=[x for x in rem if x[1]==j]
        A=np.array([[x[2],x[3],1.0] for x in sub])
        b=np.array([x[4] for x in sub],float)
        c,*_=np.linalg.lstsq(A,b,rcond=None)
        sd=float(np.std(b-A@c))
        QFIT[f"{k}_{j}"]={"lead":float(c[0]),"mu":float(c[1]),"int":float(c[2]),"sd":sd}
        if k==2 and j>=2:
            print(f"  after Q{k}, predicting Q{j+1}: lead_w={c[0]:+.4f} mu_w={c[1]:+.4f} int={c[2]:+.3f} sd={sd:.2f}")

out={"winprob":FIT,"quarters":QFIT,
     "note":"P(home)=sigmoid(lead*a + mu*b + c) at end of Q1/Q2/Q3"}
json.dump(out,open(os.path.join(HERE,"wnba_fit_live.json"),"w"),indent=1)
print("\nsaved wnba_fit_live.json")
