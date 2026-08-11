"""Audit the MLB F5 model's distributional assumption against real games.

The model computes P(home leads / tie / away leads after 5 innings) by
convolving two INDEPENDENT POISSON run distributions. Baseball runs are
famously overdispersed (big innings), and the same file already uses a
NEGATIVE BINOMIAL (r=4.5) for its full-game score model — so plain Poisson
in the F5 path is worth testing. If Poisson understates the spread of runs,
it will systematically OVERSTATE the probability of a tie, and every
"Tie F5" flag would be structurally -EV.
"""
import json, urllib.request, math, statistics
from collections import Counter

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def season_games(start, end):
    url = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1"
           f"&startDate={start}&endDate={end}&hydrate=linescore&gameType=R")
    d = get(url)
    out = []
    for day in d.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            inns = (g.get("linescore") or {}).get("innings") or []
            if len(inns) < 5:
                continue
            h = a = 0
            ok = True
            for i in inns[:5]:
                hr = i.get("home", {}).get("runs"); ar = i.get("away", {}).get("runs")
                if hr is None or ar is None: ok = False; break
                h += hr; a += ar
            if ok: out.append((h, a))
    return out

print("fetching real MLB games (2025 + 2026 to date)…")
games = []
for s, e in [("2025-04-01", "2025-09-28"), ("2026-04-01", "2026-07-30")]:
    try:
        got = season_games(s, e)
        print(f"  {s}..{e}: {len(got)} games")
        games += got
    except Exception as ex:
        print(f"  ! {s}: {ex}")
print(f"total F5 samples: {len(games)}")

hs = [h for h, a in games]; as_ = [a for h, a in games]
lam_h = statistics.mean(hs); lam_a = statistics.mean(as_)
var_h = statistics.variance(hs); var_a = statistics.variance(as_)
print(f"\nF5 runs — home mean {lam_h:.3f} var {var_h:.3f}  (var/mean {var_h/lam_h:.3f})")
print(f"F5 runs — away mean {lam_a:.3f} var {var_a:.3f}  (var/mean {var_a/lam_a:.3f})")
print("  Poisson requires var/mean = 1.0; >1 means overdispersed (Poisson too narrow)")

def pois(lam, kmax=15):
    out = []; t = math.exp(-lam)
    for k in range(kmax + 1):
        out.append(t); t *= lam / (k + 1)
    return out

# Poisson-predicted 3-way split using the TRUE observed means
ph, pa = pois(lam_h), pois(lam_a)
H = T = A = 0.0
for i, x in enumerate(ph):
    for j, y in enumerate(pa):
        p = x * y
        if i > j: H += p
        elif i == j: T += p
        else: A += p
s = H + T + A
H, T, A = H/s, T/s, A/s

n = len(games)
actH = sum(1 for h, a in games if h > a) / n
actT = sum(1 for h, a in games if h == a) / n
actA = sum(1 for h, a in games if h < a) / n

print(f"\n{'':10} {'Poisson model':>14} {'ACTUAL':>10} {'error':>9}")
print(f"{'home leads':10} {H:>14.4f} {actH:>10.4f} {H-actH:>+9.4f}")
print(f"{'TIE':10} {T:>14.4f} {actT:>10.4f} {T-actT:>+9.4f}")
print(f"{'away leads':10} {A:>14.4f} {actA:>10.4f} {A-actA:>+9.4f}")

rel = (T - actT) / actT * 100
print(f"\n>>> Poisson overstates the TIE probability by {T-actT:+.4f} absolute "
      f"({rel:+.1f}% relative)")

# What that bias is worth in EV terms on a typical tie contract
for price in (0.20, 0.22, 0.25):
    fee = 0.07 * price * (1 - price)
    ev_believed = T - price - fee
    ev_true = actT - price - fee
    print(f"    tie contract at {price:.2f}: model thinks EV {ev_believed:+.4f} "
          f"(ROI {ev_believed/price:+.1%}), TRUE EV {ev_true:+.4f} (ROI {ev_true/price:+.1%})")

# distribution detail
print("\nF5 run distribution (home), actual vs Poisson:")
cnt = Counter(hs)
for k in range(0, 8):
    print(f"  {k} runs: actual {cnt[k]/n:.4f}   poisson {ph[k]:.4f}   {cnt[k]/n-ph[k]:+.4f}")
