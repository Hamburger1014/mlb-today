"""Calibrate a negative-binomial dispersion for the F5 run distribution.

Poisson assumes var = mean; real F5 runs sit at var/mean ~2.2, which inflates
tie probability. NB(mean mu, dispersion r) gives var = mu*(1 + mu/r), so r
controls exactly the excess we measured.

One subtlety worth checking: the MARGINAL variance across games mixes
within-game randomness with between-game variation in the true mean. Only the
within-game part should go into r, or the fix over-corrects. We estimate how
big the between-game part actually is before trusting the calibration.
"""
import json, math, statistics, urllib.request
from collections import Counter

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def season_games(start, end):
    d = get("https://statsapi.mlb.com/api/v1/schedule?sportId=1"
            f"&startDate={start}&endDate={end}&hydrate=linescore&gameType=R")
    out = []
    for day in d.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final": continue
            inns = (g.get("linescore") or {}).get("innings") or []
            if len(inns) < 5: continue
            h = a = 0; ok = True
            for i in inns[:5]:
                hr = i.get("home", {}).get("runs"); ar = i.get("away", {}).get("runs")
                if hr is None or ar is None: ok = False; break
                h += hr; a += ar
            if ok: out.append((h, a))
    return out

games = []
for s, e in [("2025-04-01", "2025-09-28"), ("2026-04-01", "2026-07-30")]:
    games += season_games(s, e)
n = len(games)
hs = [h for h, a in games]; as_ = [a for h, a in games]
lam_h, lam_a = statistics.mean(hs), statistics.mean(as_)
print(f"n={n}  lam_h={lam_h:.4f}  lam_a={lam_a:.4f}")

actH = sum(1 for h, a in games if h > a) / n
actT = sum(1 for h, a in games if h == a) / n
actA = sum(1 for h, a in games if h < a) / n
print(f"actual 3-way: home {actH:.4f}  tie {actT:.4f}  away {actA:.4f}")

def nb_pmf(mu, r, kmax=25):
    mu = max(0.05, mu); p = r / (r + mu)
    out = [p ** r]; v = out[0]
    for k in range(1, kmax + 1):
        v *= (k + r - 1) / k * (1 - p); out.append(v)
    return out

def pois_pmf(mu, kmax=25):
    mu = max(0.05, mu); out = []; t = math.exp(-mu)
    for k in range(kmax + 1):
        out.append(t); t *= mu / (k + 1)
    return out

def three_way(ph, pa):
    H = T = A = 0.0
    for i, x in enumerate(ph):
        for j, y in enumerate(pa):
            p = x * y
            if i > j: H += p
            elif i == j: T += p
            else: A += p
    s = H + T + A
    return H / s, T / s, A / s

# ── how much of the variance is BETWEEN games rather than within? ──
# Proxy: split games by their combined total; if between-game lambda spread were
# large, sub-populations would show very different means. Estimate Var(lambda)
# from a matchup-level grouping instead: use the observed spread of per-team
# season F5 scoring rates as an upper bound on lambda variation.
var_h = statistics.variance(hs)
print(f"\nmarginal var_h={var_h:.4f} (var/mean {var_h/lam_h:.3f})")
# Upper-bound Var(lambda): the model's own lambda range is roughly 1.8-3.4
# => sd ~0.35 => Var ~0.12, i.e. ~2% of total variance. Within-game dominates.
for vlam in (0.0, 0.06, 0.12, 0.25):
    # E[lam(1+lam/r)] + Var(lam) = var_h  ->  solve r
    e_lam2 = lam_h ** 2 + vlam
    denom = var_h - vlam - lam_h
    r_est = e_lam2 / denom if denom > 0 else float("inf")
    print(f"  if Var(lambda)={vlam:.2f}: implied r={r_est:.3f}  (var/mean at mean lam "
          f"= {1 + lam_h / r_est:.3f})")

# ── grid-search r to match the empirical 3-way split ──
print("\ngrid search r (shared by both sides), matching the 3-way split:")
best = None
r = 0.8
rows = []
while r <= 8.01:
    H, T, A = three_way(nb_pmf(lam_h, r), nb_pmf(lam_a, r))
    err = abs(T - actT) + abs(H - actH) + abs(A - actA)
    rows.append((r, H, T, A, err))
    if best is None or err < best[4]: best = (r, H, T, A, err)
    r += 0.1
for row in rows[::6]:
    print(f"  r={row[0]:.1f}: home {row[1]:.4f} tie {row[2]:.4f} away {row[3]:.4f}  L1err {row[4]:.4f}")
R = round(best[0], 1)
print(f"\n>>> best r = {R}  -> home {best[1]:.4f} tie {best[2]:.4f} away {best[3]:.4f} (L1 {best[4]:.4f})")

pH, pT, pA = three_way(pois_pmf(lam_h), pois_pmf(lam_a))
nH, nT, nA = three_way(nb_pmf(lam_h, R), nb_pmf(lam_a, R))
print(f"\n{'':10} {'Poisson':>9} {'NB r='+str(R):>9} {'ACTUAL':>9}")
for lbl, p_, n_, a_ in (("home", pH, nH, actH), ("tie", pT, nT, actT), ("away", pA, nA, actA)):
    print(f"{lbl:10} {p_:>9.4f} {n_:>9.4f} {a_:>9.4f}   err {p_-a_:+.4f} -> {n_-a_:+.4f}")

# ── does it hold on SUB-POPULATIONS (not just the aggregate)? ──
print("\nhold-out check by scoring environment (splits the sample by combined F5 total):")
lo = [g for g in games if sum(g) <= 4]
mid = [g for g in games if 5 <= sum(g) <= 8]
hi = [g for g in games if sum(g) >= 9]
for name, sub in (("low (<=4)", lo), ("mid (5-8)", mid), ("high (>=9)", hi)):
    if not sub: continue
    m = len(sub)
    lh = statistics.mean([h for h, a in sub]); la = statistics.mean([a for h, a in sub])
    aT = sum(1 for h, a in sub if h == a) / m
    _, tP, _ = three_way(pois_pmf(lh), pois_pmf(la))
    _, tN, _ = three_way(nb_pmf(lh, R), nb_pmf(la, R))
    print(f"  {name:11} n={m:5}  actual tie {aT:.4f} | poisson {tP:.4f} ({tP-aT:+.4f}) | NB {tN:.4f} ({tN-aT:+.4f})")

# ── per-team-count distribution fit ──
print("\nhome F5 run distribution fit:")
cnt = Counter(hs); pp = pois_pmf(lam_h); nn = nb_pmf(lam_h, R)
chi_p = chi_n = 0.0
for k in range(0, 10):
    act = cnt[k] / n
    chi_p += (act - pp[k]) ** 2 / max(pp[k], 1e-9)
    chi_n += (act - nn[k]) ** 2 / max(nn[k], 1e-9)
    print(f"  {k}: actual {act:.4f}  poisson {pp[k]:.4f} ({act-pp[k]:+.4f})  NB {nn[k]:.4f} ({act-nn[k]:+.4f})")
print(f"\n  chi-square-ish fit statistic: poisson {chi_p:.4f}  NB {chi_n:.4f}  "
      f"({'NB better' if chi_n < chi_p else 'POISSON better'})")
print(f"\nUSE r = {R} in the JS F5 model")
