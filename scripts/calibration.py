#!/usr/bin/env python3
"""Calibration: is a 70% call right 70% of the time?

Accuracy answers "did the favourite win". Calibration answers "was the number
right", and for betting it is the one that matters: an edge is computed by
subtracting a price from a probability, so a model that is 65% accurate and
systematically overconfident produces edges that do not exist. That is the exact
failure mode behind the F5 model, whose ROI got WORSE as its claimed edge grew.

Reported here:

  ECE   expected calibration error — the average gap between what was said and
        what happened, weighted by how often each was said. Directly in
        probability points, so 4pp means the numbers are off by 4pp on average.

  BRIER DECOMPOSITION  brier = reliability - resolution + uncertainty.
        reliability is the calibration penalty (lower is better, 0 is perfect).
        resolution is how much the model separates games from the base rate
        (higher is better). uncertainty is the base rate's own variance and is a
        property of the sport, not the model. Splitting them matters because a
        model can improve its Brier by being timid, and resolution catches that.

  LOG LOSS  the scoring rule that punishes confident errors hardest.

Nothing here refits ratings. It takes predictions the model already made
walk-forward and asks whether the probabilities attached to them were honest.
"""
import math


def logit(p, eps=1e-9):
    p = min(1 - eps, max(eps, p))
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def log_loss(preds, ys, eps=1e-12):
    n = len(preds)
    if not n:
        return None
    s = 0.0
    for p, y in zip(preds, ys):
        p = min(1 - eps, max(eps, p))
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / n


def brier(preds, ys):
    return sum((p - y) ** 2 for p, y in zip(preds, ys)) / len(preds) if preds else None


def reliability_table(preds, ys, edges=(0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01)):
    """Bucket by CONFIDENCE, folding both sides onto the favourite.

    Folding matters: a two-sided model is symmetric, so 30% and 70% are the same
    claim seen from two ends. Bucketing raw probabilities would split one
    statement across two rows and halve every sample.
    """
    rows = []
    lo = 0.5
    for hi in edges[1:]:
        sel = [(max(p, 1 - p), (y if p >= 0.5 else 1 - y)) for p, y in zip(preds, ys)
               if lo <= max(p, 1 - p) < hi]
        if sel:
            said = sum(c for c, _ in sel) / len(sel)
            got = sum(h for _, h in sel) / len(sel)
            # Standard error of the observed rate. Without it a bucket gap is
            # unreadable: on 59 games at 57% the noise band is +-6.4pp, so an
            # 8pp miss is barely one and a third SE and means nothing.
            se = math.sqrt(max(got * (1 - got), 1e-9) / len(sel))
            rows.append({"lo": lo, "hi": hi, "n": len(sel), "said": said, "got": got,
                         "gap": got - said, "se": se, "z": (got - said) / se if se else 0.0})
        lo = hi
    return rows


def ece(preds, ys, edges=(0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01)):
    rows = reliability_table(preds, ys, edges)
    n = sum(r["n"] for r in rows)
    return sum(r["n"] * abs(r["gap"]) for r in rows) / n if n else None


def brier_decomp(preds, ys, edges=(0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01)):
    """reliability - resolution + uncertainty, on the folded buckets."""
    n = len(preds)
    if not n:
        return None
    base = sum(ys) / n
    unc = base * (1 - base)
    rel = res = 0.0
    lo = 0.5
    for hi in edges[1:]:
        sel = [(max(p, 1 - p), (y if p >= 0.5 else 1 - y)) for p, y in zip(preds, ys)
               if lo <= max(p, 1 - p) < hi]
        lo = hi
        if not sel:
            continue
        k = len(sel)
        said = sum(c for c, _ in sel) / k
        got = sum(h for _, h in sel) / k
        rel += k * (said - got) ** 2
        res += k * (got - base) ** 2
    return {"reliability": rel / n, "resolution": res / n, "uncertainty": unc, "base": base}


def fit_scale(margins, ys, lo=3.0, hi=25.0, iters=200):
    """The logistic scale that minimises log loss on already-made predictions.

    p = sigmoid(margin / scale). The scale is the only thing controlling how
    confident the model is, and in both football models it was hand-picked rather
    than fitted. Golden-section on a 1-D convex-ish objective; cheap because the
    expensive part (the margins) is already done.
    """
    def loss(s):
        return log_loss([sigmoid(m / s) for m in margins], ys)
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if loss(c) < loss(d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return (a + b) / 2


def fit_platt(preds, ys, iters=400, lr=0.5):
    """a,b for sigmoid(a*logit(p)+b). a<1 means the model was overconfident."""
    a, b = 1.0, 0.0
    n = len(preds)
    if not n:
        return 1.0, 0.0
    zs = [logit(p) for p in preds]
    for _ in range(iters):
        ga = gb = 0.0
        for z, y in zip(zs, ys):
            e = sigmoid(a * z + b) - y
            ga += e * z
            gb += e
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


def report(name, preds, ys, market=None):
    print(f"\n=== {name}  n={len(preds)} ===")
    acc = sum(1 for p, y in zip(preds, ys) if (p >= 0.5) == (y == 1)) / len(preds)
    d = brier_decomp(preds, ys)
    print(f"  accuracy {acc*100:.1f}%   brier {brier(preds,ys):.4f}   "
          f"logloss {log_loss(preds,ys):.4f}   ECE {ece(preds,ys)*100:.2f}pp")
    print(f"  reliability {d['reliability']:.4f} (0 = perfectly calibrated)   "
          f"resolution {d['resolution']:.4f}   uncertainty {d['uncertainty']:.4f}")
    if market:
        print(f"  market logloss {log_loss(market,ys):.4f}   "
              f"model minus market {log_loss(preds,ys)-log_loss(market,ys):+.4f} "
              f"(negative = model better)")
    print(f"  {'said':>8} {'happened':>9} {'gap':>8} {'+-1SE':>7} {'z':>6} {'n':>5}")
    for r in reliability_table(preds, ys):
        # Only call a bucket miscalibrated when it is outside 2 SE. Below that
        # the gap is sampling noise and chasing it is how a model gets fitted to
        # its own test set.
        flag = ""
        if abs(r["z"]) >= 2:
            flag = "  <-- overconfident" if r["gap"] < 0 else "  <-- underconfident"
        print(f"  {r['said']*100:7.1f}% {r['got']*100:8.1f}% {r['gap']*100:+7.1f}pp "
              f"{r['se']*100:6.1f}pp {r['z']:+6.2f} {r['n']:5d}{flag}")
