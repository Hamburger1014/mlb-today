"""WNBA backfill: point-in-time model predictions vs DraftKings CLOSING lines.

Answers the same question market_edge_test.py asked of MLB F5, but on ~250 games
instead of the 12 in the live log.

Honesty rules baked in:
  * Ratings are walk-forward within the season (state updated only AFTER the
    prediction), exactly as train_wnba2.build_rows does.
  * Coefficients are fit on seasons < TEST_SEASON and frozen. The shipped
    wnba_fit_v2.json was fit including 2026, so using it here would be lookahead.
  * The injury feature reads the game's own boxscore, i.e. who ACTUALLY played.
    That is unavailable pre-tip, so it is lookahead in the model's favour. The
    script reports both `--inj on` (optimistic, matches the 67.3% headline) and
    `--inj off` (conservative). If the model loses to the market even with the
    lookahead on, that is decisive.

Usage:
    python wnba_backfill.py                 # both injury variants, moneyline + spread
    python wnba_backfill.py --devig shin
"""
import argparse, json, os, sys
from collections import defaultdict
from datetime import datetime, timedelta
from math import erf, sqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mlb-today-repo", "scripts"))
import market_edge_test as MET          # devig, fit_pool, report, american_to_prob

TEAMS = {'ATL','CHI','CON','DAL','GS','IND','LA','LV','MIN','NY','PHX','POR','SEA','TOR','WSH'}
SHRINK_K = 8
LG_FALLBACK = 83.0
TEST_SEASON = 2026


# ----------------------------------------------------------------- model replay

def load():
    games = json.load(open(os.path.join(HERE, "wnba_games.json")))
    box = json.load(open(os.path.join(HERE, "wnba_boxscores.json")))
    odds = json.load(open(os.path.join(HERE, "wnba_odds.json")))
    games = [g for g in games
             if g["home"]["abbr"] in TEAMS and g["away"]["abbr"] in TEAMS
             and g.get("seasonType") in (2, 3)]
    games.sort(key=lambda g: g["date"])
    return games, box, odds


def build_rows(games, box, half_life, prior_w0, prior_r, use_inj=True):
    """Walk-forward point-in-time feature rows. Mirrors train_wnba2.build_rows,
    plus the game id / date so rows can be joined to odds."""
    lam = 0.5 ** (1.0 / half_life)

    def gdate(g):
        return datetime.fromisoformat(g["date"].replace("Z", "+00:00"))

    def eday(g):
        return (gdate(g) - timedelta(hours=5)).date()

    played_on = defaultdict(set)
    for g in games:
        played_on[eday(g)].add(g["home"]["abbr"])
        played_on[eday(g)].add(g["away"]["abbr"])

    def is_b2b(ab, g):
        return ab in played_on.get(eday(g) - timedelta(days=1), set())

    rows = []
    prior_rating = {}
    for season in sorted({g["seasonYear"] for g in games}):
        sg = [g for g in games if g["seasonYear"] == season]
        wPF, wPA, wN = defaultdict(float), defaultdict(float), defaultdict(float)
        pstats = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
        tgp = defaultdict(int)
        for ab, (ppf, ppa) in prior_rating.items():
            wPF[ab] = (prior_r * ppf + (1 - prior_r) * LG_FALLBACK) * prior_w0
            wPA[ab] = (prior_r * ppa + (1 - prior_r) * LG_FALLBACK) * prior_w0
            wN[ab] = prior_w0
        pfS, nS = 0.0, 0
        for g in sg:
            h, a = g["home"], g["away"]
            hb, ab_ = h["abbr"], a["abbr"]
            lg = (pfS / nS) if nS >= 20 else LG_FALLBACK
            bx = box.get(g["id"]) or {}

            def missing_ppg(abbr):
                if not use_inj or tgp[abbr] < 4:
                    return 0.0
                played = {p[0] for p in (bx.get(abbr) or []) if p[1] and p[1] > 0}
                if not played:
                    return 0.0
                m = 0.0
                for pid, (apps, pts) in pstats[abbr].items():
                    if apps < 3 or apps / tgp[abbr] < 0.5:
                        continue
                    ppg = pts / apps
                    if ppg < 6:
                        continue
                    if pid not in played:
                        m += ppg
                return min(m, 30.0)

            if wN[hb] >= 3 and wN[ab_] >= 3:
                def rate(w, n):
                    return ((w / n) * n + lg * SHRINK_K) / (n + SHRINK_K)
                hPF, hPA = rate(wPF[hb], wN[hb]), rate(wPA[hb], wN[hb])
                aPF, aPA = rate(wPF[ab_], wN[ab_]), rate(wPA[ab_], wN[ab_])
                rows.append({
                    "id": g["id"], "date": g["date"], "season": season,
                    "eH0": hPF * aPA / lg, "eA0": aPF * hPA / lg,
                    "b2bH": int(is_b2b(hb, g)), "b2bA": int(is_b2b(ab_, g)),
                    "missH": missing_ppg(hb), "missA": missing_ppg(ab_),
                    "hReg": sum(h["lines"][:4]), "aReg": sum(a["lines"][:4]),
                    "hFinal": h["score"], "aFinal": a["score"],
                    "homeWon": int(h["score"] > a["score"]),
                    "gpH": tgp[hb], "gpA": tgp[ab_],
                })
            hpts, apts = sum(h["lines"][:4]), sum(a["lines"][:4])
            for abbr, pf_g, pa_g in [(hb, hpts, apts), (ab_, apts, hpts)]:
                wPF[abbr] = wPF[abbr] * lam + pf_g
                wPA[abbr] = wPA[abbr] * lam + pa_g
                wN[abbr] = wN[abbr] * lam + 1
            for abbr in (hb, ab_):
                for p in (bx.get(abbr) or []):
                    if p[1] and p[1] > 0:
                        ps = pstats[abbr][p[0]]
                        ps[0] += 1
                        ps[1] += p[2]
                tgp[abbr] += 1
            pfS += hpts + apts
            nS += 2
        prior_rating = {ab2: (wPF[ab2] / wN[ab2], wPA[ab2] / wN[ab2])
                        for ab2 in wN if wN[ab2] > 0}
    return rows


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(x, y):
    k = 0.1
    x, y = np.asarray(x, float), np.asarray(y, float)
    for _ in range(80):
        p = sigmoid(k * x)
        gr = np.mean((p - y) * x)
        hh = np.mean(p * (1 - p) * x * x) + 1e-9
        k -= gr / hh
    return float(k)


def design(rows):
    return np.array([[r["eH0"] - r["eA0"], 1.0,
                      r["b2bH"] - r["b2bA"], r["missH"] - r["missA"]] for r in rows])


def fit_and_predict(rows, test_season):
    """Fit margin coefficients + win-prob calibration on prior seasons only."""
    tr = [r for r in rows if r["season"] < test_season and r["gpH"] >= 3]
    te = [r for r in rows if r["season"] == test_season and r["gpH"] >= 3]
    if not tr or not te:
        return None
    Xtr, Xte = design(tr), design(te)
    yM = np.array([r["hReg"] - r["aReg"] for r in tr], float)
    cM, *_ = np.linalg.lstsq(Xtr, yM, rcond=None)
    mu_tr = Xtr @ cM
    k = fit_logistic(mu_tr, np.array([r["homeWon"] for r in tr]))
    # final-margin sd, for the spread model (spreads settle on the FINAL score)
    finM_tr = np.array([r["hFinal"] - r["aFinal"] for r in tr], float)
    sd_fin = float(np.std(finM_tr - mu_tr))
    mu_te = Xte @ cM
    for r, m in zip(te, mu_te):
        r["mu"] = float(m)
        r["p_home"] = float(sigmoid(k * m))
    return {"test": te, "coef": cM, "k": k, "sd_final": sd_fin, "n_train": len(tr)}


def ncdf(x, m, s):
    return 0.5 * (1 + erf((x - m) / (s * sqrt(2))))


# ------------------------------------------------------------------- segments

def moneyline(te, odds, devig):
    Pm, Pk, y, keep = [], [], [], []
    for r in te:
        o = odds.get(r["id"]) or {}
        if o.get("homeML") is None or o.get("awayML") is None:
            continue
        raw = [MET.american_to_prob(o["homeML"]), MET.american_to_prob(o["awayML"])]
        Pm.append([r["p_home"], 1 - r["p_home"]])
        Pk.append(MET.devig(raw, devig))
        y.append(0 if r["homeWon"] else 1)
        keep.append(r)
    return np.array(Pm), np.array(Pk), np.array(y), keep


def spread(te, odds, devig, sd):
    """P(home covers) vs DK closing spread. Pushes dropped."""
    Pm, Pk, y = [], [], []
    for r in te:
        o = odds.get(r["id"]) or {}
        line, ho, ao = o.get("spread"), o.get("homeSpreadOdds"), o.get("awaySpreadOdds")
        if line is None or ho is None or ao is None:
            continue
        margin = r["hFinal"] - r["aFinal"]
        if margin + line == 0:                       # push
            continue
        pc = 1 - ncdf(-line, r["mu"], sd)            # P(margin > -line)
        raw = [MET.american_to_prob(ho), MET.american_to_prob(ao)]
        Pm.append([pc, 1 - pc])
        Pk.append(MET.devig(raw, devig))
        y.append(0 if margin + line > 0 else 1)
    return np.array(Pm), np.array(Pk), np.array(y)


def clv_vs_close(te, odds, devig, log_path):
    """For games that ALSO appear in the live log, CLV of the logged price
    against the DK close from ESPN."""
    if not os.path.exists(log_path):
        return None
    E = json.load(open(log_path, encoding="utf-8"))["entries"]
    out = []
    byid = {r["id"]: r for r in te}
    for e in E:
        r = byid.get(e["id"])
        o = odds.get(e["id"]) or {}
        v = e.get("vegas") or {}
        if not r or o.get("homeML") is None or v.get("homeML") is None:
            continue
        pc = MET.devig([MET.american_to_prob(o["homeML"]),
                        MET.american_to_prob(o["awayML"])], devig)[0]
        po = MET.devig([MET.american_to_prob(v["homeML"]),
                        MET.american_to_prob(v["awayML"])], devig)[0]
        out.append(pc - po)
    return np.array(out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devig", default="power",
                    choices=["power", "multiplicative", "additive", "shin"])
    ap.add_argument("--half-life", type=float, default=25)
    ap.add_argument("--prior-w0", type=float, default=4)
    ap.add_argument("--prior-r", type=float, default=0.6)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "wnba_backfill_results.txt"))
    a = ap.parse_args()

    games, box, odds = load()
    have_odds = sum(1 for g in games if (odds.get(g["id"]) or {}).get("homeML") is not None)
    L = ["WNBA BACKFILL - point-in-time model vs DraftKings CLOSING lines",
         "de-vig = %s | half-life %g | prior W0=%g r=%g" % (a.devig, a.half_life, a.prior_w0, a.prior_r),
         "%d games loaded, %d with a closing moneyline" % (len(games), have_odds)]

    for inj in (True, False):
        rows = build_rows(games, box, a.half_life, a.prior_w0, a.prior_r, use_inj=inj)
        fp = fit_and_predict(rows, TEST_SEASON)
        if not fp:
            continue
        te = fp["test"]
        tag = "injury feature ON (lookahead: uses who actually played)" if inj \
              else "injury feature OFF (no lookahead)"
        L.append("")
        L.append("#" * 74)
        L.append("# " + tag)
        L.append("# trained on %d pre-%d games, testing %d games in %d"
                 % (fp["n_train"], TEST_SEASON, len(te), TEST_SEASON))
        L.append("# margin coef: eff %.4f  home %+.3f  b2b %+.3f  inj %+.4f | winprob k %.4f | sd_final %.2f"
                 % (fp["coef"][0], fp["coef"][1], fp["coef"][2], fp["coef"][3], fp["k"], fp["sd_final"]))
        L.append("#" * 74)

        Pm, Pk, y, keep = moneyline(te, odds, a.devig)
        if len(y) >= 30:
            acc = float(((Pm[:, 0] >= 0.5) == (y == 0)).mean())
            base = float((y == 0).mean())
            mkacc = float(((Pk[:, 0] >= 0.5) == (y == 0)).mean())
            L.append(MET.report("WNBA moneyline  vs  DK closing", MET.fit_pool(Pm, Pk, y)))
            L.append("  straight-up accuracy: model %.3f | market %.3f | home base %.3f"
                     % (acc, mkacc, base))
        else:
            L.append("  moneyline: only %d joinable games" % len(y))

        Pms, Pks, ys = spread(te, odds, a.devig, fp["sd_final"])
        if len(ys) >= 30:
            L.append(MET.report("WNBA spread (home cover)  vs  DK closing", MET.fit_pool(Pms, Pks, ys)))
        else:
            L.append("  spread: only %d joinable non-push games" % len(ys))

    # CLV of the live log's captured prices vs the true DK close
    rows = build_rows(games, box, a.half_life, a.prior_w0, a.prior_r, use_inj=True)
    fp = fit_and_predict(rows, TEST_SEASON)
    logp = os.path.join(HERE, "..", "mlb-today-repo", "data", "wnba_predictions.json")
    c = clv_vs_close(fp["test"], odds, a.devig, logp) if fp else None
    if c is not None and len(c):
        L.append("")
        L.append("LIVE-LOG SANITY: logged open vs true DK close, n=%d" % len(c))
        L.append("  mean home-prob drift %+.4f  (how far the line moved after logging)" % c.mean())

    text = "\n".join(L)
    open(a.out, "w", encoding="utf-8").write(text)
    sys.stdout.write(text.encode("ascii", "replace").decode("ascii") + "\n")


if __name__ == "__main__":
    main()
