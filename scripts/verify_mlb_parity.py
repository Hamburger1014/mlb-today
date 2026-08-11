#!/usr/bin/env python3
"""Fail the build if the MLB full-game model in scripts/mlb_log.py has drifted
from the one in index.html.

The logger reimplements REAL_MODEL / realModelRawStats so predictions can be
recorded server-side. Two copies of a model silently diverging would mean the
logged record — and the CLV computed from it — stops describing what the site
actually showed. That is exactly the blind spot that let a refuted model run
unnoticed for two months, so it is gated rather than trusted.

Replaces verify_f5_parity.py, which went with the F5 model.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
LOG = open(os.path.join(HERE, "mlb_log.py"), encoding="utf-8").read()

errs = []


def grab(pattern, label, src=SRC):
    m = re.search(pattern, src)
    if not m:
        errs.append(f"could not find {label} in {'index.html' if src is SRC else 'mlb_log.py'}")
        return None
    return m.group(1)


# ── coefficients ──
html_coef = grab(r"coef:\s*\[([^\]]+)\]", "REAL_MODEL.coef")
py_coef = grab(r"COEF\s*=\s*\[([^\]]+)\]", "COEF", LOG)
if html_coef and py_coef:
    a = [round(float(x), 6) for x in html_coef.split(",")]
    b = [round(float(x), 6) for x in py_coef.split(",")]
    if a != b:
        errs.append(f"coef drift: index.html {a} vs mlb_log.py {b}")

# ── scalars ──
for label, html_pat, py_pat in [
    ("intercept", r"intercept:\s*([0-9.\-]+)", r"INTERCEPT\s*=\s*([0-9.\-]+)"),
    ("kbbLg",     r"kbbLg:\s*([0-9.\-]+)",     r"KBB_LG\s*=\s*([0-9.\-]+)"),
    ("kbbWbf",    r"kbbWbf:\s*([0-9]+)",       r"KBB_WBF\s*=\s*([0-9]+)"),
]:
    h, p = grab(html_pat, label), grab(py_pat, label, LOG)
    if h and p and round(float(h), 6) != round(float(p), 6):
        errs.append(f"{label} drift: index.html {h} vs mlb_log.py {p}")

# ── shape guards ──
# shrink() default k and the Pythagorean exponent are baked into both copies
# as literals, so a change on one side has to be mirrored deliberately.
h_k = grab(r"function shrink\(val,n,prior=0\.5,k=(\d+)\)", "shrink k")
p_k = grab(r"SHRINK_K\s*=\s*(\d+)", "SHRINK_K", LOG)
if h_k and p_k and h_k != p_k:
    errs.append(f"shrink k drift: index.html {h_k} vs mlb_log.py {p_k}")

h_e = grab(r"const e=([0-9.]+);\s*return rs\*\*e", "pythagorean exponent")
p_e = grab(r"PYTH_EXP\s*=\s*([0-9.]+)", "PYTH_EXP", LOG)
if h_e and p_e and round(float(h_e), 4) != round(float(p_e), 4):
    errs.append(f"pythagorean exponent drift: index.html {h_e} vs mlb_log.py {p_e}")

# The feature vector order must match; a silent reorder would be invisible in
# the numbers until it produced nonsense.
if "hPs-aPs, hWs-aWs, hRd-aRd, kbbDiff, spKnown, 1.0" not in SRC.replace(" ", " "):
    errs.append("index.html feature order changed (expected pyth, win%, run-diff, kbb, spKnown, home)")
if "[hPs - aPs, hWs - aWs, hRd - aRd, kbb_diff, sp_known, 1.0]" not in LOG:
    errs.append("mlb_log.py feature order changed")

if errs:
    print("MLB model parity FAILED:")
    for e in errs:
        print("  -", e)
    sys.exit(1)
n_coef = len(html_coef.split(",")) if html_coef else "?"
print(f"MLB model parity OK: index.html and mlb_log.py agree "
      f"({n_coef} coefficients, intercept={grab(r'intercept:\s*([0-9.\-]+)', 'intercept')})")
