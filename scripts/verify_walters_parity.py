#!/usr/bin/env python3
"""Fail if index.html's spread-number maths has drifted from scripts/walters.py.

The half-point valuation now exists twice: in Python, where it was developed and
validated, and in JavaScript, because the page has to price a number without a
round trip. This repo already carries three parity verifiers because duplicated
model code drifts — silently, and in the direction that flatters whoever edited
last. This is the fourth.

It does not compare source. It RUNS both over the same cases and compares the
numbers, which is the only check that catches a rewrite that looks equivalent and
is not. The JS is sliced out of index.html by its comment markers and executed in
node, so what gets tested is the code the browser actually loads.

Exit 1 on any disagreement past 1e-6, or if the JS block cannot be found.
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAGE = os.path.join(ROOT, "index.html")

START = "// ── SPREAD NUMBERS (the Walters half-point valuation)"
END = "// Scans one league and returns BOTH the qualifying plays"
TOL = 1e-6

# Both sides of a key number, a non-key number for contrast, a push boundary,
# a price-only difference, and both leagues' sigmas.
CASES = [
    # dkLine, dkHome, dkAway, fieldLine, fieldHome, fieldAway, sigma
    (-2.5, -110, -110, -3.0, -110, -110, 13.2),
    (-3.5, -110, -110, -3.0, -110, -110, 13.2),
    (-3.0, -110, -110, -3.0, -110, -110, 13.2),
    (-4.5, -110, -110, -5.0, -110, -110, 13.2),
    (-7.0, -115, -105, -7.5, -110, -110, 13.2),
    (10.5, -108, -112, 10.0, -110, -110, 13.2),
    (-2.5, -118, -102, -3.0, -105, -115, 13.2),
    (-6.0, -110, -110, -6.5, -110, -110, 16.0),
    (14.5, -110, -110, 14.0, -120, +100, 16.0),
    (0.0, -110, -110, -1.0, -110, -110, 16.0),
]


def js_results():
    src = io.open(PAGE, encoding="utf-8").read()
    i = src.find(START)
    j = src.find(END, i + 1) if i >= 0 else -1
    if i < 0 or j < 0:
        print("FAIL: could not find the Walters JS block in index.html "
              "(markers moved or the block was deleted)")
        sys.exit(1)
    block = src[i:j]
    driver = "\nconst CASES=%s;\nconst out=CASES.map(c=>{const r=wNumberEdge(c[0],c[1],c[2],c[3],c[4],c[5],c[6]);return [r.home,r.away];});\nconsole.log(JSON.stringify(out));\n" % json.dumps(CASES)
    fd, path = tempfile.mkstemp(suffix=".js")
    os.close(fd)
    try:
        io.open(path, "w", encoding="utf-8").write(block + driver)
        r = subprocess.run(["node", path], capture_output=True, text=True)
        if r.returncode:
            print("FAIL: the JS block did not run in node:\n" + r.stderr[:600])
            sys.exit(1)
        return json.loads(r.stdout.strip().splitlines()[-1])
    finally:
        os.unlink(path)


def py_results():
    spec = importlib.util.spec_from_file_location("walters", os.path.join(HERE, "walters.py"))
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    out = []
    for dkL, dkH, dkA, fL, fH, fA, sig in CASES:
        e = W.number_edge(dkL, dkH, dkA, fL, fH, fA, sig)
        out.append([e["home"], e["away"]])
    return out


def main():
    js, py = js_results(), py_results()
    if len(js) != len(py):
        print(f"FAIL: {len(js)} JS results vs {len(py)} Python")
        sys.exit(1)
    bad = 0
    for n, (a, b) in enumerate(zip(js, py)):
        for side, x, y in (("home", a[0], b[0]), ("away", a[1], b[1])):
            if x is None or y is None:
                if x is not y:
                    print(f"FAIL case {n} {side}: js={x} py={y}")
                    bad += 1
                continue
            if abs(x - y) > TOL:
                c = CASES[n]
                print(f"FAIL case {n} {side}: dk {c[0]:+} vs field {c[3]:+} "
                      f"-> js {x*100:+.4f}pp  py {y*100:+.4f}pp  (diff {abs(x-y)*100:.6f}pp)")
                bad += 1
    if bad:
        print(f"\nWalters parity FAILED on {bad} comparison(s) — index.html and "
              f"scripts/walters.py disagree.")
        sys.exit(1)
    print(f"Walters parity OK: index.html and walters.py agree on "
          f"{len(CASES)} cases x 2 sides (tol {TOL})")


if __name__ == "__main__":
    main()
