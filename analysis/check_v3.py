"""Fairness checks on the v3 negative results.

1. Ridge vs baseline on the IDENTICAL row subset (the first run compared
   n=1046 ridge rows against n=1246 baseline rows — not a fair test).
2. Finer de-injure grid, plus the specific subset that motivated it:
   games where a key player RETURNS after missing time.
3. Alternative Q4 interaction forms.
"""
import json, os, importlib.util
import numpy as np
from collections import defaultdict

HERE=os.path.dirname(__file__)
spec=importlib.util.spec_from_file_location("t3", os.path.join(HERE,"train_wnba3.py"))
# don't exec (it runs the whole training); re-implement the pieces we need
games=json.load(open(os.path.join(HERE,"wnba_games.json")))
box=json.load(open(os.path.join(HERE,"wnba_boxscores.json")))
TEAMS={'ATL','CHI','CON','DAL','GS','IND','LA','LV','MIN','NY','PHX','POR','SEA','TOR','WSH'}
games=[g for g in games if g["home"]["abbr"] in TEAMS and g["away"]["abbr"] in TEAMS and g.get("seasonType") in (2,3)]
games.sort(key=lambda g:g["date"])
from datetime import datetime,timedelta
def gdate(g): return datetime.fromisoformat(g["date"].replace("Z","+00:00"))
played_on=defaultdict(set)
def eday(g): return (gdate(g)-timedelta(hours=5)).date()
for g in games:
    played_on[eday(g)].add(g["home"]["abbr"]); played_on[eday(g)].add(g["away"]["abbr"])
def is_b2b(ab,g): return ab in played_on.get(eday(g)-timedelta(days=1),set())
SHRINK_K=8; LG=83.0; CAPV=20.0
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
def fit_logistic(x,y):
    k=0.1;x=np.asarray(x);y=np.asarray(y,float)
    for _ in range(60):
        p=sigmoid(k*x);g=np.mean((p-y)*x);h=np.mean(p*(1-p)*x*x)+1e-9;k-=g/h
    return k

def build(half_life,inj,cap=True,track_return=True):
    lam=0.5**(1.0/half_life); prior={}; rows=[]
    for season in sorted({g["seasonYear"] for g in games}):
        sg=[g for g in games if g["seasonYear"]==season]
        wPF=defaultdict(float);wPA=defaultdict(float);wN=defaultdict(float)
        pstats=defaultdict(lambda: defaultdict(lambda:[0,0.0])); tgp=defaultdict(int)
        lastMiss=defaultdict(float)
        for ab,(pf,pa) in prior.items():
            wPF[ab]=(0.6*pf+0.4*LG)*4; wPA[ab]=(0.6*pa+0.4*LG)*4; wN[ab]=4
        pfS=0.0;nS=0
        for g in sg:
            h,a=g["home"],g["away"]; hb,ab_=h["abbr"],a["abbr"]
            lg=(pfS/nS) if nS>=20 else LG
            bx=box.get(g["id"]) or {}
            def miss(abbr):
                if tgp[abbr]<4: return 0.0
                played={p[0] for p in (bx.get(abbr) or []) if p[1] and p[1]>0}
                if not played: return 0.0
                m=0.0
                for pid,(apps,pts) in pstats[abbr].items():
                    if apps<3 or apps/tgp[abbr]<0.5: continue
                    ppg=pts/apps
                    if ppg>=6 and pid not in played: m+=ppg
                return min(m,30.0)
            mH,mA=miss(hb),miss(ab_)
            if wN[hb]>=3 and wN[ab_]>=3:
                def rate(w,n): return ((w/n)*n+lg*SHRINK_K)/(n+SHRINK_K)
                hPF=rate(wPF[hb],wN[hb]);hPA=rate(wPA[hb],wN[hb])
                aPF=rate(wPF[ab_],wN[ab_]);aPA=rate(wPA[ab_],wN[ab_])
                rows.append({"hb":hb,"ab":ab_,"season":season,
                  "eH0":hPF*aPA/lg,"eA0":aPF*hPA/lg,
                  "b2bH":int(is_b2b(hb,g)),"b2bA":int(is_b2b(ab_,g)),
                  "missH":mH,"missA":mA,
                  "hReg":sum(h["lines"][:4]),"aReg":sum(a["lines"][:4]),
                  "homeWon":int(h["score"]>a["score"]),
                  # "return" = team had someone missing recently, fewer missing now
                  "retH":max(0.0,lastMiss[hb]-mH),"retA":max(0.0,lastMiss[ab_]-mA)})
            lastMiss[hb]=mH; lastMiss[ab_]=mA
            hp=sum(h["lines"][:4])+inj*mH; ap=sum(a["lines"][:4])+inj*mA
            if cap:
                mid=(hp+ap)/2; d=max(-CAPV/2,min(CAPV/2,(hp-ap)/2)); hp,ap=mid+d,mid-d
            for abbr,pf_g,pa_g in [(hb,hp,ap),(ab_,ap,hp)]:
                wPF[abbr]=wPF[abbr]*lam+pf_g;wPA[abbr]=wPA[abbr]*lam+pa_g;wN[abbr]=wN[abbr]*lam+1
            for abbr in (hb,ab_):
                for p in (bx.get(abbr) or []):
                    if p[1] and p[1]>0:
                        s=pstats[abbr][p[0]];s[0]+=1;s[1]+=p[2]
                tgp[abbr]+=1
            pfS+=sum(h["lines"][:4])+sum(a["lines"][:4]);nS+=2
        prior={x:(wPF[x]/wN[x],wPA[x]/wN[x]) for x in wN if wN[x]>0}
    return rows

def evalrows(rows, idx=None):
    X=np.array([[r["eH0"]-r["eA0"],1.0,r["b2bH"]-r["b2bA"],r["missH"]-r["missA"]] for r in rows])
    yM=np.array([r["hReg"]-r["aReg"] for r in rows],float)
    yW=np.array([r["homeWon"] for r in rows])
    c,*_=np.linalg.lstsq(X,yM,rcond=None)
    mu=X@c;k=fit_logistic(mu,yW);p=sigmoid(k*mu)
    if idx is not None: p=p[idx];yW=yW[idx]
    ll=-np.mean(yW*np.log(p+1e-9)+(1-yW)*np.log(1-p+1e-9))
    return ll,float(np.mean((p>=0.5)==yW)),len(yW)

print("== 1. FAIR ridge comparison (identical rows) ==")
rows=build(25,0.0)
teams=sorted(TEAMS); ti={t:i for i,t in enumerate(teams)}
# build the ridge prediction set and record which row indices it covers
ridge_pred={}
for season in sorted({r["season"] for r in rows}):
    sr=[(i,r) for i,r in enumerate(rows) if r["season"]==season]
    for pos,(gi,r) in enumerate(sr):
        if pos<40: continue
        hist=[q for _,q in sr[:pos]]
        n=len(hist);T=len(teams)
        A=np.zeros((n,2*T+1));b=np.zeros(n)
        for j,q in enumerate(hist):
            A[j,ti[q["hb"]]]+=1; A[j,T+ti[q["ab"]]]-=1
            A[j,ti[q["ab"]]]-=1; A[j,T+ti[q["hb"]]]+=1
            A[j,-1]=1; b[j]=q["hReg"]-q["aReg"]
        R=np.eye(2*T+1)*50; R[-1,-1]=0
        sol=np.linalg.solve(A.T@A+R,A.T@b)
        off,dfn,hme=sol[:T],sol[T:2*T],sol[-1]
        ridge_pred[gi]=(off[ti[r["hb"]]]-dfn[ti[r["ab"]]])-(off[ti[r["ab"]]]-dfn[ti[r["hb"]]])+hme
idx=sorted(ridge_pred)
yW=np.array([rows[i]["homeWon"] for i in idx])
rp=np.array([ridge_pred[i] for i in idx])
kr=fit_logistic(rp,yW);pr=sigmoid(kr*rp)
llr=-np.mean(yW*np.log(pr+1e-9)+(1-yW)*np.log(1-pr+1e-9))
print(f"  ridge      : logloss={llr:.4f} acc={np.mean((pr>=0.5)==yW):.3f} n={len(idx)}")
llb,accb,nb=evalrows(rows,idx)
print(f"  baseline   : logloss={llb:.4f} acc={accb:.3f} n={nb}  <-- same rows")
# blend
Xb=np.stack([rp,np.array([ (rows[i]['eH0']-rows[i]['eA0']) for i in idx])],1)
yM=np.array([rows[i]["hReg"]-rows[i]["aReg"] for i in idx],float)
cb,*_=np.linalg.lstsq(np.column_stack([Xb,np.ones(len(idx))]),yM,rcond=None)
mub=np.column_stack([Xb,np.ones(len(idx))])@cb
kb=fit_logistic(mub,yW);pb=sigmoid(kb*mub)
llbl=-np.mean(yW*np.log(pb+1e-9)+(1-yW)*np.log(1-pb+1e-9))
print(f"  blend both : logloss={llbl:.4f} acc={np.mean((pb>=0.5)==yW):.3f}")

print("\n== 2. finer de-injure grid ==")
for inj in [0.0,0.05,0.10,0.15,0.20,0.25]:
    r2=build(25,inj)
    ll,acc,_=evalrows(r2)
    print(f"  inj={inj:.2f}: logloss={ll:.4f} acc={acc:.3f}")

print("\n== 2b. RETURN subset: games where a key scorer came back ==")
r0=build(25,0.0)
ret=[i for i,r in enumerate(r0) if max(r["retH"],r["retA"])>=6]
print(f"  n(return games)={len(ret)} of {len(r0)}")
for inj in [0.0,0.15,0.3]:
    r2=build(25,inj)
    ll,acc,n=evalrows(r2,ret)
    print(f"  inj={inj:.2f} on return-subset: logloss={ll:.4f} acc={acc:.3f} n={n}")

print("\n== 3. Q4 alternative interactions ==")
X=np.array([[r["eH0"]-r["eA0"],1.0,r["b2bH"]-r["b2bA"],r["missH"]-r["missA"]] for r in r0])
yM=np.array([r["hReg"]-r["aReg"] for r in r0],float)
c,*_=np.linalg.lstsq(X,yM,rcond=None); mu=X@c
gm=[g for g in games if True]
# rebuild quarter arrays aligned to r0 order
qs=[]
seen=0
for season in sorted({r["season"] for r in r0}):
    pass
# simpler: recompute from games list in same construction order
idx2=0; hQ=[];aQ=[]
prior={}
# reuse build() ordering by matching on hReg/aReg sequence
gi=0
for g in games:
    pass
# fallback: use stored regulation sums to locate; instead just re-derive quarters
# by rebuilding with quarter capture
def build_q(half_life):
    lam=0.5**(1.0/half_life); prior={}; out=[]
    for season in sorted({g["seasonYear"] for g in games}):
        sg=[g for g in games if g["seasonYear"]==season]
        wPF=defaultdict(float);wPA=defaultdict(float);wN=defaultdict(float)
        for ab,(pf,pa) in prior.items():
            wPF[ab]=(0.6*pf+0.4*LG)*4;wPA[ab]=(0.6*pa+0.4*LG)*4;wN[ab]=4
        for g in sg:
            h,a=g["home"],g["away"];hb,ab_=h["abbr"],a["abbr"]
            if wN[hb]>=3 and wN[ab_]>=3: out.append((h["lines"][:4],a["lines"][:4]))
            hp=sum(h["lines"][:4]);ap=sum(a["lines"][:4])
            mid=(hp+ap)/2;d=max(-CAPV/2,min(CAPV/2,(hp-ap)/2));hp,ap=mid+d,mid-d
            for abbr,pf_g,pa_g in [(hb,hp,ap),(ab_,ap,hp)]:
                wPF[abbr]=wPF[abbr]*lam+pf_g;wPA[abbr]=wPA[abbr]*lam+pa_g;wN[abbr]=wN[abbr]*lam+1
        prior={x:(wPF[x]/wN[x],wPA[x]/wN[x]) for x in wN if wN[x]>0}
    return out
qq=build_q(25)
hQ=np.array([x[0] for x in qq],float);aQ=np.array([x[1] for x in qq],float)
n=min(len(hQ),len(mu)); hQ=hQ[:n];aQ=aQ[:n];mu4=mu[:n]
qm=hQ[:,3]-aQ[:,3]; am=np.abs(mu4)
forms={
 "linear":       np.stack([mu4,np.ones(n)],1),
 "mu*|mu|":      np.stack([mu4,mu4*am/10,np.ones(n)],1),
 "close-indic":  np.stack([mu4,mu4*(am<6).astype(float),np.ones(n)],1),
 "mu/(1+|mu|/8)":np.stack([mu4/(1+am/8),np.ones(n)],1),
}
for name,A in forms.items():
    cf,*_=np.linalg.lstsq(A,qm,rcond=None)
    print(f"  Q4 {name:15}: resid sd={np.std(qm-A@cf):.4f}")
