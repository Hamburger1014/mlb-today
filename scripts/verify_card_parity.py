#!/usr/bin/env python3
"""Fail the build if Today's Card in scripts/card_log.py has drifted from the
card in index.html.

card_log.py reimplements cardCandidates() so the record is written on a schedule
instead of depending on a browser being open. That makes it a second copy of the
sizing and threshold rules, and a second copy that silently diverges would mean
the logged record stops describing the card anyone actually saw — the same class
of failure the WNBA and MLB parity gates exist to prevent.
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = open(os.path.join(ROOT, "index.html"), encoding="utf-8").read()
LOG = open(os.path.join(HERE, "card_log.py"), encoding="utf-8").read()

errs = []


def grab(pat, label, src, where):
    m = re.search(pat, src)
    if not m:
        errs.append(f"could not find {label} in {where}")
        return None
    return m.group(1)


CHECKS = [
    ("CARD_UNIT",          r"const CARD_UNIT\s*=\s*([0-9.]+)",          r"CARD_UNIT\s*=\s*([0-9.]+)"),
    ("CARD_MODEL_SHRINK",  r"const CARD_MODEL_SHRINK\s*=\s*([0-9.]+)",  r"CARD_MODEL_SHRINK\s*=\s*([0-9.]+)"),
    ("CARD_CROSS_SHRINK",  r"const CARD_CROSS_SHRINK\s*=\s*([0-9.]+)",  r"CARD_CROSS_SHRINK\s*=\s*([0-9.]+)"),
    ("CARD_MIN_EDGE",      r"const CARD_MIN_EDGE\s*=\s*([0-9.]+)",      r"CARD_MIN_EDGE\s*=\s*([0-9.]+)"),
    ("CARD_MAX_STALE_MIN", r"const CARD_MAX_STALE_MIN\s*=\s*([0-9.]+)", r"CARD_MAX_STALE_MIN\s*=\s*([0-9.]+)"),
    ("KELLY_FRACTION",     r"const KELLY_FRACTION\s*=\s*([0-9.]+)",     r"KELLY_FRACTION\s*=\s*([0-9.]+)"),
    ("MAX_STAKE_GAME",     r"MAX_STAKE_GAME\s*=\s*([0-9.]+)",           r"MAX_STAKE_GAME\s*=\s*([0-9.]+)"),
    ("MAX_STAKE_DAY",      r"MAX_STAKE_DAY\s*=\s*([0-9.]+)",            r"MAX_STAKE_DAY\s*=\s*([0-9.]+)"),
    ("KALSHI_FEE",         r"const KALSHI_FEE\s*=\s*([0-9.]+)",         r"KALSHI_FEE\s*=\s*([0-9.]+)"),
    ("MIN_PROB_EDGE",      r"const MIN_PROB_EDGE\s*=\s*([0-9.]+)",      r"MIN_PROB_EDGE\s*=\s*([0-9.]+)"),
]
for label, hp, pp in CHECKS:
    h = grab(hp, label, SRC, "index.html")
    p = grab(pp, label, LOG, "card_log.py")
    if h is not None and p is not None and round(float(h), 6) != round(float(p), 6):
        errs.append(f"{label} drift: index.html {h} vs card_log.py {p}")

# The WNBA blend weights live in MARKET_EVIDENCE on the page and as plain
# constants in the logger, so they are matched by value rather than by name.
for label, hp, pp in [
    ("WNBA moneyline w", r"WNBAML:\s*\{bet:true,\s*w:([0-9.]+)", r"W_WNBA_ML\s*=\s*([0-9.]+)"),
    ("WNBA spread w",    r"WNBASP:\s*\{bet:true,\s*w:([0-9.]+)", r"W_WNBA_SP\s*=\s*([0-9.]+)"),
]:
    h = grab(hp, label, SRC, "index.html")
    p = grab(pp, label, LOG, "card_log.py")
    if h is not None and p is not None and round(float(h), 6) != round(float(p), 6):
        errs.append(f"{label} drift: index.html {h} vs card_log.py {p}")

# Shape guard: both sides must apply the shrink per KIND. Collapsing them back
# to one factor is the specific mistake that made cross-book dead on arrival.
if "kind==='CROSS'?CARD_CROSS_SHRINK:CARD_MODEL_SHRINK" not in SRC.replace(" ", ""):
    errs.append("index.html no longer picks the shrink by candidate kind")
if 'CARD_CROSS_SHRINK if kind == "CROSS" else CARD_MODEL_SHRINK' not in LOG:
    errs.append("card_log.py no longer picks the shrink by candidate kind")

if errs:
    print("Card parity FAILED:")
    for e in errs:
        print("  -", e)
    sys.exit(1)
print("Card parity OK: index.html and card_log.py agree on all "
      f"{len(CHECKS) + 2} thresholds and both shrink by kind")
