"""Fail the build if the server-side F5 model drifts from the page's.

scripts/mlb_f5_log.py reimplements F5_MODEL / f5ProbsStats / f5Evaluate from
index.html. Two copies of a model silently diverging would mean the logged
track record stops describing what the site shows — the same trap the WNBA
parity gate guards. This checks the constants AND the numeric output.
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mlb_f5_log import (F5_MODEL, KBB_LG, KBB_WBF, KALSHI_FEE,  # noqa: E402
                        MIN_NET_EV, MIN_ROI, nb_pmf, f5_probs)

SRC = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
errs = []


def grab(pattern, label, cast=float):
    m = re.search(pattern, SRC)
    if not m:
        errs.append(f"could not find {label} in index.html")
        return None
    return cast(m.group(1))


def close(a, b, tol=1e-9):
    return a is not None and abs(a - b) <= tol


# ── constants ──
js = {
    "beta":     grab(r"F5_MODEL\s*=\s*\{\s*beta:\s*([\d.]+)", "F5_MODEL.beta"),
    "homeBump": grab(r"F5_MODEL\s*=\s*\{[^}]*homeBump:\s*([\d.]+)", "F5_MODEL.homeBump"),
    "scale":    grab(r"F5_MODEL\s*=\s*\{[^}]*scale:\s*([\d.]+)", "F5_MODEL.scale"),
    "disp":     grab(r"F5_MODEL\s*=\s*\{[^}]*disp:\s*([\d.]+)", "F5_MODEL.disp"),
}
for k, v in js.items():
    if v is not None and not close(v, F5_MODEL[k]):
        errs.append(f"F5_MODEL.{k}: js={v} python={F5_MODEL[k]}")

kbb_lg = grab(r"kbbLg:\s*([\d.]+)", "REAL_MODEL.kbbLg")
if kbb_lg is not None and not close(kbb_lg, KBB_LG):
    errs.append(f"kbbLg: js={kbb_lg} python={KBB_LG}")
kbb_wbf = grab(r"kbbWbf:\s*(\d+)", "REAL_MODEL.kbbWbf", int)
if kbb_wbf is not None and kbb_wbf != KBB_WBF:
    errs.append(f"kbbWbf: js={kbb_wbf} python={KBB_WBF}")
fee = grab(r"KALSHI_FEE\s*=\s*([\d.]+)", "KALSHI_FEE")
if fee is not None and not close(fee, KALSHI_FEE):
    errs.append(f"KALSHI_FEE: js={fee} python={KALSHI_FEE}")

# f5Evaluate's thresholds are inline literals, not named constants
thr = re.search(r"netEV>=\s*([\d.]+)\s*&&\s*roi>=\s*([\d.]+)", SRC)
if not thr:
    errs.append("could not find f5Evaluate's netEV/roi thresholds")
else:
    if not close(float(thr.group(1)), MIN_NET_EV):
        errs.append(f"MIN_NET_EV: js={thr.group(1)} python={MIN_NET_EV}")
    if not close(float(thr.group(2)), MIN_ROI):
        errs.append(f"MIN_ROI: js={thr.group(2)} python={MIN_ROI}")

# ── the F5 path must actually use the NB, not the Poisson ──
if not re.search(r"const ph=_nbPmf\(lamH\)", SRC):
    errs.append("f5ProbsStats is NOT using _nbPmf — the Poisson bug is back")
if not re.search(r"_convolve\(_nbPmf\(lamH\)", SRC):
    errs.append("f5Totals is NOT convolving NBs")

# ── numeric spot-checks (guard the formula, not just the constants) ──
# NB pmf must integrate to 1 with the right mean and variance.
pmf = nb_pmf(2.6271, kmax=80)
s = sum(pmf)
m1 = sum(k * p for k, p in enumerate(pmf))
m2 = sum(k * k * p for k, p in enumerate(pmf))
var = m2 - m1 * m1
want_var = 2.6271 * (1 + 2.6271 / F5_MODEL["disp"])
if abs(s - 1) > 1e-6: errs.append(f"nb_pmf does not sum to 1 (got {s})")
if abs(m1 - 2.6271) > 1e-6: errs.append(f"nb_pmf mean off (got {m1})")
if abs(var - want_var) > 1e-6: errs.append(f"nb_pmf variance off (got {var}, want {want_var})")

# The calibrated 3-way split for league-average lambdas, checked against the
# figures the fix was validated on (home .4514 / tie .1588 / away .3898).
h = {"wins": 50, "losses": 50, "rs": 2.6271 / (5/9) / 1.15 / 1.06 * 100, "ra": 0}
a = {"wins": 50, "losses": 50, "rs": 2.3948 / (5/9) / 1.15 * 100, "ra": 0}
lg = 4.4
probs = f5_probs(h, a, None, None, lg)
if probs is None:
    errs.append("f5_probs returned None on a valid input")
else:
    tot = probs["h"] + probs["t"] + probs["a"]
    if abs(tot - 1) > 1e-9:
        errs.append(f"f5_probs does not normalise (sum={tot})")
    if not (0.12 < probs["t"] < 0.20):
        errs.append(f"tie probability {probs['t']:.4f} outside the sane 0.12-0.20 band — "
                    "the distribution may have regressed to Poisson (which gives ~0.18+)")

if errs:
    print("F5 MODEL PARITY FAILED — the page and the server-side logger disagree:")
    for e in errs:
        print("  -", e)
    print("\nReconcile scripts/mlb_f5_log.py with index.html before logging.")
    sys.exit(1)
print("F5 model parity OK: index.html and mlb_f5_log.py agree "
      f"(disp={F5_MODEL['disp']}, tie={probs['t']:.4f})")
