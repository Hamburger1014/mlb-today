"""Fail the build if the server-side model drifts from the one in the page.

The prediction logger (scripts/wnba_log.py) reimplements the model that lives
in the FIT block of index.html and wnba_today.html. Two copies of a model is a
standing invitation to silent divergence — the logged track record would stop
describing what the site actually shows. This extracts the JS constants and
diffs them against the Python ones.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from wnba_log import FIT  # noqa: E402

FILES = ["index.html", "wnba_today.html"]


def js_number(tok):
    return float(tok)


def fit_block(src):
    """Isolate the WNBA `const FIT={...}` literal.

    index.html also contains MLB models with their own shrinkK etc., so
    searching the whole file matches the wrong numbers.
    """
    i = src.find("const FIT=")
    if i < 0: i = src.find("const FIT =")
    if i < 0: return None
    j = src.find("{", i)
    if j < 0: return None
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{": depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0: return src[j:k + 1]
    return None


def extract_fit(src):
    """Pull the numbers we care about out of the JS FIT literal."""
    src = fit_block(src) or ""
    out = {}
    m = re.search(r"halfLife:\s*(\d+)", src)
    if m: out["halfLife"] = int(m.group(1))
    m = re.search(r"priorW0:\s*([\d.]+)", src)
    if m: out["priorW0"] = js_number(m.group(1))
    m = re.search(r"priorR:\s*([\d.]+)", src)
    if m: out["priorR"] = js_number(m.group(1))
    m = re.search(r"shrinkK:\s*(\d+)", src)
    if m: out["shrinkK"] = int(m.group(1))
    m = re.search(r"winProbK:\s*([\d.]+)", src)
    if m: out["winProbK"] = js_number(m.group(1))
    m = re.search(r"spreadSd:\s*([\d.]+)", src)
    if m: out["spreadSd"] = js_number(m.group(1))
    m = re.search(r"margin:\s*\{eff:\s*([-\d.]+),\s*homeEdge:\s*([-\d.]+),\s*b2b:\s*([-\d.]+),\s*injMiss:\s*([-\d.]+)", src)
    if m: out["margin"] = {"eff": js_number(m.group(1)), "homeEdge": js_number(m.group(2)),
                           "b2b": js_number(m.group(3)), "injMiss": js_number(m.group(4))}
    m = re.search(r"total:\s*\{eff:\s*([-\d.]+),\s*intercept:\s*([-\d.]+),\s*b2b:\s*([-\d.]+),\s*injMiss:\s*([-\d.]+)", src)
    if m: out["total"] = {"eff": js_number(m.group(1)), "intercept": js_number(m.group(2)),
                          "b2b": js_number(m.group(3)), "injMiss": js_number(m.group(4))}
    qs = re.findall(
        r"\{shareH:\s*([\d.]+),\s*shareA:\s*([\d.]+),\s*slope:\s*([\d.]+),\s*intercept:\s*([-\s\d.]+),\s*sd:\s*([\d.]+),\s*tieBand:\s*([\d.]+)\}",
        src)
    if qs:
        out["quarters"] = [{"shareH": js_number(a), "shareA": js_number(b), "slope": js_number(c),
                            "intercept": js_number(d.strip()), "sd": js_number(e), "tieBand": js_number(f)}
                           for a, b, c, d, e, f in qs]
    return out


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def compare(name, js):
    errs = []
    for k in ("halfLife", "priorW0", "priorR", "shrinkK", "winProbK", "spreadSd"):
        if k not in js:
            errs.append(f"{name}: could not find `{k}` in the JS FIT block"); continue
        if not close(float(js[k]), float(FIT[k])):
            errs.append(f"{name}: {k} js={js[k]} python={FIT[k]}")
    for grp in ("margin", "total"):
        if grp not in js:
            errs.append(f"{name}: could not find `{grp}` in the JS FIT block"); continue
        for k, v in js[grp].items():
            if not close(v, FIT[grp][k]):
                errs.append(f"{name}: {grp}.{k} js={v} python={FIT[grp][k]}")
    if "quarters" not in js or len(js["quarters"]) != 4:
        errs.append(f"{name}: expected 4 quarter blocks, found {len(js.get('quarters', []))}")
    else:
        for i, (jq, pq) in enumerate(zip(js["quarters"], FIT["quarters"])):
            for k in ("shareH", "shareA", "slope", "intercept", "sd", "tieBand"):
                if not close(jq[k], pq[k]):
                    errs.append(f"{name}: quarters[{i}].{k} js={jq[k]} python={pq[k]}")
    return errs


def main():
    all_errs = []
    for fn in FILES:
        path = os.path.join(ROOT, fn)
        if not os.path.exists(path):
            all_errs.append(f"missing {fn}"); continue
        all_errs += compare(fn, extract_fit(open(path, encoding="utf-8").read()))
    if all_errs:
        print("MODEL PARITY FAILED — the page and the server-side logger disagree:")
        for e in all_errs: print("  -", e)
        print("\nUpdate scripts/wnba_log.py FIT (or the HTML) so they match.")
        sys.exit(1)
    print("model parity OK: index.html, wnba_today.html and wnba_log.py agree")


if __name__ == "__main__":
    main()
