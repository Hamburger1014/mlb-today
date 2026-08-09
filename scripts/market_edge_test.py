#!/usr/bin/env python3
"""
Two-logit market test: does the model add information the market price doesn't have?

For each market segment, fits a log-opinion pool over the model's probabilities
and the de-vigged market probabilities:

    P(outcome k)  proportional to  p_model[k]^Bm  *  p_market[k]^Bk

    Bm ~ 0, Bk ~ 1  -> model adds nothing; do not bet this market
    Bm > 0 signif.  -> real incremental information; blend weight w = Bm/(Bm+Bk)
    Bk < 1          -> market itself is biased here

This is the multi-outcome form of
    logit(P) = a + Bm*logit(p_model) + Bk*logit(p_market)
and reduces to exactly that for 2-way markets.

Also reports:
  - log-loss of model / market / fitted pool / equal pool
  - the winner's-curse slope: realized edge vs claimed edge on flagged plays

Usage:  python scripts/market_edge_test.py [--data DIR] [--out FILE]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

EPS = 1e-9


# ---------------------------------------------------------------- de-vigging

def devig(raw, method="power"):
    """raw: list of gross implied probabilities (e.g. Kalshi asks) summing to >1."""
    q = np.asarray(raw, dtype=float)
    q = np.clip(q, 1e-4, 1 - 1e-4)
    s = q.sum()
    if s <= 1.0:                      # already fair or better; just normalise
        return q / s
    if method == "multiplicative":
        return q / s
    if method == "additive":
        p = q - (s - 1.0) / len(q)
        return np.clip(p, 1e-4, None) / np.clip(p, 1e-4, None).sum()
    if method == "power":
        lo, hi = 0.5, 5.0
        for _ in range(200):
            k = 0.5 * (lo + hi)
            if (q ** k).sum() > 1.0:
                lo = k
            else:
                hi = k
        return q ** (0.5 * (lo + hi))
    if method == "shin":              # n-way Shin, bisection on z
        def pz(z):
            r = (np.sqrt(z * z + 4 * (1 - z) * q * q / s) - z) / (2 * (1 - z))
            return r
        lo, hi = 1e-6, 0.6
        for _ in range(200):
            z = 0.5 * (lo + hi)
            if pz(z).sum() > 1.0:
                lo = z
            else:
                hi = z
        p = pz(0.5 * (lo + hi))
        return p / p.sum()
    raise ValueError(method)


def american_to_prob(odds):
    o = float(odds)
    return 100.0 / (o + 100.0) if o > 0 else (-o) / (-o + 100.0)


# ------------------------------------------------------- log-opinion pool fit

def fit_pool(P_model, P_market, y, free_intercepts=False):
    """
    P_model, P_market : (n, K) probability matrices (rows sum ~1)
    y                 : (n,) index of realised outcome
    Returns dict with betas, standard errors, log-losses.
    """
    n, K = P_model.shape
    Lm = np.log(np.clip(P_model, EPS, 1.0))
    Lk = np.log(np.clip(P_market, EPS, 1.0))
    rows = np.arange(n)

    def negll(theta):
        bm, bk = theta[0], theta[1]
        U = bm * Lm + bk * Lk
        if free_intercepts and K > 1:
            U = U + np.concatenate([[0.0], theta[2:]])[None, :]
        U = U - U.max(axis=1, keepdims=True)
        ll = U[rows, y] - np.log(np.exp(U).sum(axis=1))
        return -ll.sum()

    x0 = np.array([0.3, 0.7] + ([0.0] * (K - 1) if free_intercepts and K > 1 else []))
    res = minimize(negll, x0, method="BFGS")
    theta = res.x

    # numerical Hessian -> standard errors
    h = 1e-4
    d = len(theta)
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            tpp, tpm, tmp, tmm = theta.copy(), theta.copy(), theta.copy(), theta.copy()
            tpp[i] += h; tpp[j] += h
            tpm[i] += h; tpm[j] -= h
            tmp[i] -= h; tmp[j] += h
            tmm[i] -= h; tmm[j] -= h
            H[i, j] = (negll(tpp) - negll(tpm) - negll(tmp) + negll(tmm)) / (4 * h * h)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.full(d, np.nan)

    def logloss(P):
        return float(-np.log(np.clip(P[rows, y], EPS, 1.0)).mean())

    bm, bk = theta[0], theta[1]
    U = bm * Lm + bk * Lk
    if free_intercepts and K > 1:
        U = U + np.concatenate([[0.0], theta[2:]])[None, :]
    U = U - U.max(axis=1, keepdims=True)
    P_pool = np.exp(U); P_pool /= P_pool.sum(axis=1, keepdims=True)

    Ue = 0.5 * Lm + 0.5 * Lk
    Ue -= Ue.max(axis=1, keepdims=True)
    P_eq = np.exp(Ue); P_eq /= P_eq.sum(axis=1, keepdims=True)

    base = np.bincount(y, minlength=K) / n
    P_base = np.tile(base, (n, 1))

    return {
        "n": n, "K": K,
        "b_model": float(bm), "se_model": float(se[0]),
        "b_market": float(bk), "se_market": float(se[1]),
        "w_model": float(bm / (bm + bk)) if abs(bm + bk) > 1e-8 else float("nan"),
        "z_model": float(bm / se[0]) if se[0] == se[0] and se[0] > 0 else float("nan"),
        "z_market_vs_1": float((bk - 1.0) / se[1]) if se[1] == se[1] and se[1] > 0 else float("nan"),
        "ll_model": logloss(P_model),
        "ll_market": logloss(P_market),
        "ll_pool": logloss(P_pool),
        "ll_equal": logloss(P_eq),
        "ll_base": logloss(P_base),
        "intercepts": [float(v) for v in theta[2:]] if free_intercepts else None,
    }


def report(title, r, outcome_names=None):
    lines = []
    lines.append("")
    lines.append("=" * 74)
    lines.append(title)
    lines.append("=" * 74)
    lines.append("  n = %d games, %d outcomes" % (r["n"], r["K"]))
    lines.append("")
    lines.append("  B_model  = %+.3f  (SE %.3f, z = %+.2f)" % (r["b_model"], r["se_model"], r["z_model"]))
    lines.append("  B_market = %+.3f  (SE %.3f, z vs 1.0 = %+.2f)" % (r["b_market"], r["se_market"], r["z_market_vs_1"]))
    lines.append("  implied blend weight on model  w = %.3f" % r["w_model"])
    lines.append("")
    lines.append("  log-loss   base rate     %.4f" % r["ll_base"])
    lines.append("             model alone   %.4f" % r["ll_model"])
    lines.append("             market alone  %.4f   <-- the bar" % r["ll_market"])
    lines.append("             50/50 pool    %.4f" % r["ll_equal"])
    lines.append("             fitted pool   %.4f" % r["ll_pool"])
    lines.append("")
    gain = r["ll_market"] - r["ll_pool"]
    lines.append("  in-sample gain of fitted pool over market alone: %+.4f nats" % gain)
    if r["z_model"] == r["z_model"] and abs(r["z_model"]) < 1.96:
        lines.append("  VERDICT: B_model is not distinguishable from zero.")
        lines.append("           No evidence the model adds information over the price.")
    elif r["b_model"] > 0:
        lines.append("  VERDICT: B_model > 0 and significant -> model carries incremental info.")
        lines.append("           Bet the blend at w = %.2f, not the raw model." % r["w_model"])
    else:
        lines.append("  VERDICT: B_model is significantly NEGATIVE -> the model is")
        lines.append("           anti-informative conditional on the price. Fade or stop.")
    return "\n".join(lines)


# ------------------------------------------------------------------ segments

def mlb_f5(path, method, use_closing=False):
    E = json.load(open(path, encoding="utf-8"))["entries"]
    Pm, Pk, y, meta = [], [], [], []
    key = "closing" if use_closing else "kalshi"
    for e in E:
        res = e.get("result") or {}
        mk = e.get(key) or {}
        mod = e.get("model") or {}
        if not res.get("outcome"):
            continue
        asks = [mk.get("askH"), mk.get("askT"), mk.get("askA")]
        if any(a is None for a in asks):
            continue
        if None in (mod.get("h"), mod.get("t"), mod.get("a")):
            continue
        m = np.array([mod["h"], mod["t"], mod["a"]], dtype=float)
        m = np.clip(m, 1e-4, None); m /= m.sum()
        Pm.append(m)
        Pk.append(devig(asks, method))
        y.append({"home": 0, "tie": 1, "away": 2}[res["outcome"]])
        meta.append(e)
    return np.array(Pm), np.array(Pk), np.array(y), meta


def wnba_game(path, method, source="vegas", use_closing=False):
    E = json.load(open(path, encoding="utf-8"))["entries"]
    Pm, Pk, y, meta = [], [], [], []
    for e in E:
        res = e.get("result") or {}
        mod = e.get("model") or {}
        if res.get("homeWon") is None or mod.get("game") is None:
            continue
        if use_closing:
            blk = e.get("closing") or {}
            veg = blk.get("vegas") or {}
            kal = blk.get("kalshi")
        else:
            veg = e.get("vegas") or {}
            kal = (e.get("kalshi") or {}).get("game")
        if source == "vegas":
            if veg.get("homeML") is None or veg.get("awayML") is None:
                continue
            raw = [american_to_prob(veg["homeML"]), american_to_prob(veg["awayML"])]
            pk = devig(raw, method)
        else:
            if kal is None:
                continue
            pk = np.array([float(kal), 1.0 - float(kal)])
        ph = float(mod["game"])
        Pm.append([ph, 1 - ph])
        Pk.append(pk)
        y.append(0 if res["homeWon"] else 1)
        meta.append(e)
    return np.array(Pm), np.array(Pk), np.array(y), meta


def wnba_quarters(path, method):
    """Quarter 3-way vs Kalshi quarter markets, if any are populated."""
    E = json.load(open(path, encoding="utf-8"))["entries"]
    Pm, Pk, y = [], [], []
    for e in E:
        res = e.get("result") or {}
        mod = e.get("model") or {}
        qk = ((e.get("kalshi") or {}).get("q")) or []
        qh, qa = res.get("qHome"), res.get("qAway")
        if not qh or not qa or not mod.get("q"):
            continue
        for i in range(4):
            if i >= len(qk) or not qk[i]:
                continue
            mq = mod["q"][i]
            m = np.array([mq["h"], mq["t"], mq["a"]], float); m /= m.sum()
            k = qk[i]
            asks = [k.get("h"), k.get("t"), k.get("a")]
            if any(a is None for a in asks):
                continue
            Pm.append(m); Pk.append(devig(asks, method))
            y.append(0 if qh[i] > qa[i] else (1 if qh[i] == qa[i] else 2))
    return np.array(Pm), np.array(Pk), np.array(y)


# --------------------------------------------------------- winner's-curse fit

def winners_curse(meta, Pk, y):
    """
    On flagged plays: regress the realised outcome on the CLAIMED edge
    (model fair minus de-vigged market prob). Slope < 1 means the flags
    over-state edge; every threshold in the app should be divided by it.
    """
    idx = {"home": 0, "tie": 1, "away": 2}
    x, z = [], []
    for i, e in enumerate(meta):
        play = e.get("play")
        if not play or play.get("key") not in idx:
            continue
        k = idx[play["key"]]
        claimed = float(play["fair"]) - float(Pk[i][k])
        x.append(claimed)
        z.append(1.0 if y[i] == k else 0.0)
    if len(x) < 10:
        return None
    x = np.array(x); z = np.array(z)
    pk = np.array([Pk[i][idx[e["play"]["key"]]]
                   for i, e in enumerate(meta)
                   if e.get("play") and e["play"].get("key") in idx])
    realised = z - pk                      # realised excess over market prob
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, realised, rcond=None)
    resid = realised - A @ coef
    dof = max(len(x) - 2, 1)
    s2 = (resid ** 2).sum() / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return {
        "n": int(len(x)),
        "intercept": float(coef[0]), "se_intercept": float(np.sqrt(cov[0, 0])),
        "slope": float(coef[1]), "se_slope": float(np.sqrt(cov[1, 1])),
        "mean_claimed_edge": float(x.mean()),
        "mean_realised_edge": float(realised.mean()),
        "hit": int(z.sum()), "shrink_factor": float(coef[1]),
    }


# ------------------------------------------------------------------- CLV calc

def f5_clv(path, method):
    E = json.load(open(path, encoding="utf-8"))["entries"]
    idx = {"home": 0, "tie": 1, "away": 2}
    out = []
    for e in E:
        play = e.get("play"); cl = e.get("closing") or {}
        if not play or play.get("key") not in idx:
            continue
        asks = [cl.get("askH"), cl.get("askT"), cl.get("askA")]
        if any(a is None for a in asks):
            continue
        pclose = devig(asks, method)[idx[play["key"]]]
        paid = float(play["price"])
        out.append(pclose - paid)          # probability points of CLV
    if not out:
        return None
    a = np.array(out)
    return {"n": len(a), "mean_pp": float(a.mean() * 100),
            "se_pp": float(a.std(ddof=1) / math.sqrt(len(a)) * 100),
            "beat_rate": float((a > 0).mean())}


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here, "..", "data"))
    ap.add_argument("--devig", default="power",
                    choices=["power", "multiplicative", "additive", "shin"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    f5p = os.path.join(a.data, "mlb_f5_predictions.json")
    wnp = os.path.join(a.data, "wnba_predictions.json")
    L = []
    L.append("TWO-LOGIT MARKET TEST   (de-vig = %s)" % a.devig)

    # ---- MLB F5 vs Kalshi, price at log time
    if os.path.exists(f5p):
        Pm, Pk, y, meta = mlb_f5(f5p, a.devig, use_closing=False)
        if len(y) >= 20:
            L.append(report("MLB F5 3-way  vs  Kalshi asks at log time", fit_pool(Pm, Pk, y)))
            wc = winners_curse(meta, Pk, y)
            if wc:
                L.append("")
                L.append("  WINNER'S CURSE (flagged plays, n=%d, %d hit)" % (wc["n"], wc["hit"]))
                L.append("    mean claimed edge  %+.4f" % wc["mean_claimed_edge"])
                L.append("    mean realised edge %+.4f" % wc["mean_realised_edge"])
                L.append("    slope of realised on claimed = %+.3f (SE %.3f)"
                         % (wc["slope"], wc["se_slope"]))
                L.append("    -> divide edge thresholds by this slope if it is < 1")
        Pm2, Pk2, y2, _ = mlb_f5(f5p, a.devig, use_closing=True)
        if len(y2) >= 20:
            L.append(report("MLB F5 3-way  vs  Kalshi CLOSING asks", fit_pool(Pm2, Pk2, y2)))
        clv = f5_clv(f5p, a.devig)
        if clv:
            L.append("")
            L.append("  F5 CLV on flagged plays: n=%d  mean %+.2f pp (SE %.2f)  beat-rate %.1f%%"
                     % (clv["n"], clv["mean_pp"], clv["se_pp"], clv["beat_rate"] * 100))

    # ---- WNBA game winner
    if os.path.exists(wnp):
        for src in ("vegas", "kalshi"):
            Pm, Pk, y, meta = wnba_game(wnp, a.devig, source=src)
            if len(y) >= 8:
                L.append(report("WNBA game winner  vs  %s (log time)" % src.upper(),
                                fit_pool(Pm, Pk, y)))
                if len(y) < 60:
                    L.append("  ** n=%d is far too small; treat as a pipeline check, "
                             "not a verdict. **" % len(y))
        Pm, Pk, y = wnba_quarters(wnp, a.devig)
        if len(y) >= 20:
            L.append(report("WNBA quarter 3-way  vs  Kalshi", fit_pool(Pm, Pk, y)))
        else:
            L.append("")
            L.append("WNBA quarters: %d usable rows (Kalshi quarter prices are null in the "
                     "log) - cannot test yet." % len(y))

    text = "\n".join(L)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)
    sys.stdout.write(text.encode("ascii", "replace").decode("ascii") + "\n")


if __name__ == "__main__":
    main()
