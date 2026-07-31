"""Server-side MLB First-5-Innings prediction log.

Same treatment the WNBA log got: predictions are recorded by a scheduled job
rather than by whichever browser happened to be open, so the track record
survives independently of anyone's localStorage.

  LOG   - every PREGAME game gets the F5 model's 3-way probabilities plus the
          Kalshi KXMLBF5 ask prices at that moment, and any value play the
          model would flag under the live thresholds.
  GRADE - once final, the first five innings are summed from the real
          linescore and each logged play is settled fee-aware.

The model here MUST match F5_MODEL / f5ProbsStats in index.html.
scripts/verify_f5_parity.py diffs the two and the workflow fails on drift.

NOTE ON THE DISTRIBUTION: runs are negative-binomial, not Poisson. Measured
on 3,929 real F5 linescores, var/mean is ~2.2; assuming Poisson overstated
ties by 2.44pp and made every "Tie F5" flag structurally -EV. disp=4.0 is
fit to the observed 3-way split.
"""
import json, os, math, urllib.request
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT  = os.path.join(ROOT, "data", "mlb_f5_predictions.json")
MLB  = "https://statsapi.mlb.com/api/v1"

# ── constants — keep identical to index.html ──────────────────────────
F5_MODEL   = {"beta": 1.5, "homeBump": 1.06, "scale": 1.15, "disp": 4.0}
KBB_LG     = 0.14138      # REAL_MODEL.kbbLg
KBB_WBF    = 250          # REAL_MODEL.kbbWbf
KALSHI_FEE = 0.07
MIN_NET_EV = 0.025        # f5Evaluate's absolute floor
MIN_ROI    = 0.05         # f5Evaluate's ROI floor
LG_RPG_FALLBACK = 4.4


def get(url, timeout=45):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def us_date(dt=None):
    dt = dt or datetime.now(timezone.utc)
    return (dt - timedelta(hours=5)).strftime("%Y-%m-%d")


def nb_pmf(mean, r=None, kmax=20):
    """P(runs=k). var = mean*(1+mean/r); r->inf recovers Poisson."""
    r = F5_MODEL["disp"] if r is None else r
    mean = max(0.05, mean)
    p = r / (r + mean)
    out = [p ** r]
    v = out[0]
    for k in range(1, kmax + 1):
        v *= ((k + r - 1) / k) * (1 - p)
        out.append(v)
    return out


def kbb_quality(kbb, bf):
    if kbb is None or bf is None or bf < 40:
        return None
    wc = min(1.0, bf / KBB_WBF)
    return wc * kbb + (1 - wc) * KBB_LG


def f5_probs(h, a, home_kbb, away_kbb, lg_rpg):
    """Mirror of f5ProbsStats() in index.html."""
    if not h or not a:
        return None
    hg = h["wins"] + h["losses"]; ag = a["wins"] + a["losses"]
    if hg < 1 or ag < 1:
        return None
    hoff = (h["rs"] + lg_rpg * 15) / (hg + 15)
    aoff = (a["rs"] + lg_rpg * 15) / (ag + 15)
    def spf(q):
        if q is None: return 1.0
        return max(0.55, min(1.5, 1 - F5_MODEL["beta"] * (q - KBB_LG)))
    lam_h = max(0.15, hoff * (5/9) * spf(away_kbb) * F5_MODEL["homeBump"] * F5_MODEL["scale"])
    lam_a = max(0.15, aoff * (5/9) * spf(home_kbb) * F5_MODEL["scale"])
    ph, pa = nb_pmf(lam_h), nb_pmf(lam_a)
    H = T = A = 0.0
    for i, x in enumerate(ph):
        for j, y in enumerate(pa):
            p = x * y
            if i > j: H += p
            elif i == j: T += p
            else: A += p
    s = H + T + A or 1.0
    return {"h": H/s, "t": T/s, "a": A/s, "lamH": lam_h, "lamA": lam_a}


def evaluate(model, mkt, home_abbr, away_abbr):
    """Mirror of f5Evaluate(): best side clearing both the absolute and ROI floors."""
    sides = [
        ("home", f"{home_abbr} F5", model["h"], mkt.get("askH")),
        ("away", f"{away_abbr} F5", model["a"], mkt.get("askA")),
        ("tie",  "Tie F5",          model["t"], mkt.get("askT")),
    ]
    best = None
    for key, pick, fair, price in sides:
        if price is None or price <= 0.02 or price >= 0.98:
            continue
        fee = KALSHI_FEE * price * (1 - price)
        net = fair - price - fee
        roi = net / price
        if net >= MIN_NET_EV and roi >= MIN_ROI:
            if best is None or net > best["netEV"]:
                best = {"key": key, "pick": pick, "fair": round(fair, 4),
                        "price": round(price, 4), "fee": round(fee, 4),
                        "netEV": round(net, 4), "roi": round(roi, 4)}
    return best


# ── data loading ──────────────────────────────────────────────────────
def load_team_stats(season):
    d = get(f"{MLB}/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason")
    stats = {}
    for rec in d.get("records", []):
        for t in rec.get("teamRecords", []):
            tid = t["team"]["id"]
            stats[tid] = {"name": t["team"]["name"],
                          "wins": t.get("wins", 0), "losses": t.get("losses", 0),
                          "rs": float(t.get("runsScored") or 0),
                          "ra": float(t.get("runsAllowed") or 0)}
    return stats


def league_rpg(stats):
    r = g = 0.0
    for t in stats.values():
        gp = t["wins"] + t["losses"]
        if gp > 0:
            r += t["rs"]; g += gp
    return (r / g) if g > 0 else LG_RPG_FALLBACK


def load_starter_kbb(pids, season):
    out = {}
    ids = [str(p) for p in dict.fromkeys([p for p in pids if p])]
    for i in range(0, len(ids), 100):
        chunk = ",".join(ids[i:i+100])
        try:
            pj = get(f"{MLB}/people?personIds={chunk}"
                     f"&hydrate=stats(group=[pitching],type=[season],season={season})")
        except Exception as e:
            print(f"  ! kbb chunk: {e}"); continue
        for p in pj.get("people", []):
            k = bb = bf = 0.0; have = False
            for st in p.get("stats", []):
                for sp in st.get("splits", []):
                    s = sp.get("stat", {})
                    k += float(s.get("strikeOuts") or 0)
                    bb += float(s.get("baseOnBalls") or 0)
                    bf += float(s.get("battersFaced") or 0)
                    have = True
            if have and bf > 0:
                q = kbb_quality((k - bb) / bf, bf)
                if q is not None:
                    out[p["id"]] = q
    return out


def schedule(date):
    d = get(f"{MLB}/schedule?sportId=1&date={date}&gameType=R"
            "&hydrate=team,probablePitcher,linescore")
    games = []
    for day in d.get("dates", []):
        for g in day.get("games", []):
            st = (g.get("status") or {}).get("abstractGameState", "")
            ls = g.get("linescore") or {}
            innings = ls.get("innings") or []
            h5 = a5 = None
            if len(innings) >= 5:
                hh = aa = 0; ok = True
                for i in innings[:5]:
                    hr = (i.get("home") or {}).get("runs")
                    ar = (i.get("away") or {}).get("runs")
                    if hr is None or ar is None: ok = False; break
                    hh += hr; aa += ar
                if ok: h5, a5 = hh, aa
            games.append({
                "gamePk": g["gamePk"],
                "date": (g.get("gameDate") or "")[:10],
                "gameDate": g.get("gameDate"),
                "homeId": g["teams"]["home"]["team"]["id"],
                "awayId": g["teams"]["away"]["team"]["id"],
                "homeAbbr": (g["teams"]["home"]["team"].get("abbreviation")
                             or g["teams"]["home"]["team"]["name"][:3].upper()),
                "awayAbbr": (g["teams"]["away"]["team"].get("abbreviation")
                             or g["teams"]["away"]["team"]["name"][:3].upper()),
                "homeSpId": ((g["teams"]["home"].get("probablePitcher") or {}).get("id")),
                "awaySpId": ((g["teams"]["away"].get("probablePitcher") or {}).get("id")),
                "state": st, "isFinal": st == "Final",
                "isLive": st == "Live",
                "h5": h5, "a5": a5,
            })
    return games


def kalshi_f5():
    """Kalshi F5 asks from the snapshot this workflow already writes, keyed by
    the sorted team pair. Market tickers end in the team abbreviation or TIE,
    and Kalshi's abbreviations match statsapi's exactly for all 30 clubs
    (verified), so no mapping table is needed."""
    path = os.path.join(ROOT, "data", "kalshi_mlb.json")
    if not os.path.exists(path):
        return {}
    try:
        snap = json.load(open(path))
    except Exception:
        return {}
    by_event = {}
    for series, markets in (snap.get("markets") or {}).items():
        if "F5" not in series.upper():
            continue
        for m in markets or []:
            by_event.setdefault(m.get("event_ticker"), []).append(m)

    def ask(m):
        try:
            v = float(m.get("yes_ask_dollars"))
            return v if 0 < v < 1 else None
        except (TypeError, ValueError):
            return None

    out = {}
    for ticker, mkts in by_event.items():
        sides = {}
        for m in mkts:
            suf = (m.get("ticker") or "").rsplit("-", 1)[-1].upper()
            a = ask(m)
            if a is not None:
                sides[suf] = a
        teams = [s for s in sides if s != "TIE"]
        if len(teams) != 2:
            continue
        out["|".join(sorted(teams))] = {"ticker": ticker, "sides": sides}
    return out


def main():
    season = datetime.now(timezone.utc).year
    store = {"updated_at": None, "entries": []}
    if os.path.exists(OUT):
        try: store = json.load(open(OUT))
        except Exception: pass
    entries = store.get("entries", [])
    by_id = {str(e["gamePk"]): e for e in entries}

    today = us_date()
    games = schedule(today)
    print(f"MLB {today}: {len(games)} games")

    pregame = [g for g in games
               if not g["isFinal"] and not g["isLive"] and str(g["gamePk"]) not in by_id]

    if pregame:
        stats = load_team_stats(season)
        lg = league_rpg(stats)
        kbb = load_starter_kbb([g["homeSpId"] for g in pregame] +
                               [g["awaySpId"] for g in pregame], season)
        kmap = kalshi_f5()
        print(f"  league RPG {lg:.4f}, {len(kbb)} starter K-BB% loaded, "
              f"{len(kmap)} Kalshi F5 events")
        for g in pregame:
            h, a = stats.get(g["homeId"]), stats.get(g["awayId"])
            m = f5_probs(h, a, kbb.get(g["homeSpId"]), kbb.get(g["awaySpId"]), lg)
            if not m:
                print(f"  skip {g['awayAbbr']}@{g['homeAbbr']} (no team stats)")
                continue
            e = {
                "gamePk": g["gamePk"], "date": today,
                "away": g["awayAbbr"], "home": g["homeAbbr"],
                "gameDate": g["gameDate"],
                "model": {"h": round(m["h"], 4), "t": round(m["t"], 4), "a": round(m["a"], 4),
                          "lamH": round(m["lamH"], 3), "lamA": round(m["lamA"], 3)},
                "starters": {"home": g["homeSpId"], "away": g["awaySpId"],
                             "homeKBB": round(kbb[g["homeSpId"]], 4) if kbb.get(g["homeSpId"]) else None,
                             "awayKBB": round(kbb[g["awaySpId"]], 4) if kbb.get(g["awaySpId"]) else None},
                "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "play": None, "result": None,
            }
            key = "|".join(sorted([g["homeAbbr"], g["awayAbbr"]]))
            km = kmap.get(key)
            if km:
                e["kalshi"] = {"ticker": km["ticker"],
                               "askH": km["sides"].get(g["homeAbbr"]),
                               "askA": km["sides"].get(g["awayAbbr"]),
                               "askT": km["sides"].get("TIE")}
                e["play"] = evaluate(m, e["kalshi"], g["homeAbbr"], g["awayAbbr"])
            entries.append(e); by_id[str(e["gamePk"])] = e
            flag = f" PLAY {e['play']['pick']} @{e['play']['price']} roi {e['play']['roi']:+.1%}" if e.get("play") else ""
            print(f"  logged {g['awayAbbr']}@{g['homeAbbr']}  "
                  f"h={m['h']:.3f} t={m['t']:.3f} a={m['a']:.3f}{flag}")

    # ── CLOSING LINE ──
    # Overwritten every run while a game is still pregame, so the final value
    # is the closing ask. CLV converges far faster than realized ROI: it asks
    # whether the market moved toward our side, not whether the ball bounced
    # our way, so a few weeks of it beats a season of P&L for detecting edge.
    still_pre = [g for g in games if not g["isFinal"] and not g["isLive"]]
    if still_pre:
        kmap2 = kalshi_f5()
        n = 0
        for g in still_pre:
            e = by_id.get(str(g["gamePk"]))
            if not e:
                continue
            km = kmap2.get("|".join(sorted([g["homeAbbr"], g["awayAbbr"]])))
            if not km:
                continue
            e["closing"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "askH": km["sides"].get(g["homeAbbr"]),
                            "askA": km["sides"].get(g["awayAbbr"]),
                            "askT": km["sides"].get("TIE")}
            n += 1
        print(f"  closing-line refreshed for {n} pregame games")

    # ── GRADE ──
    pending = [e for e in entries if not e.get("result")]
    if pending:
        dates = sorted({d for e in pending for d in (
            (datetime.strptime(e["date"], "%Y-%m-%d") + timedelta(days=off)).strftime("%Y-%m-%d")
            for off in (-1, 0, 1))})
        finals = {}
        for d in dates[-10:]:
            try:
                for g in schedule(d):
                    if g["isFinal"] and g["h5"] is not None:
                        finals[str(g["gamePk"])] = g
            except Exception as ex:
                print(f"  ! grade fetch {d}: {ex}")
        for e in pending:
            g = finals.get(str(e["gamePk"]))
            if not g: continue
            outcome = "home" if g["h5"] > g["a5"] else ("away" if g["a5"] > g["h5"] else "tie")
            e["result"] = {"h5": g["h5"], "a5": g["a5"], "outcome": outcome,
                           "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if e.get("play"):
                p = e["play"]
                won = (p["key"] == outcome)
                p["won"] = won
                p["profit"] = round((1 - p["price"] - p["fee"]) if won
                                    else -(p["price"] + p["fee"]), 4)
                print(f"  graded {e['away']}@{e['home']} F5 {g['a5']}-{g['h5']} "
                      f"({outcome}) play {p['pick']} {'WIN' if won else 'LOSS'} {p['profit']:+.4f}")
            else:
                print(f"  graded {e['away']}@{e['home']} F5 {g['a5']}-{g['h5']} ({outcome}), no play")

    settled = [e for e in entries if e.get("result") and e.get("play") and "won" in e["play"]]
    if settled:
        wins = sum(1 for e in settled if e["play"]["won"])
        pnl = sum(e["play"]["profit"] for e in settled)
        risked = sum(e["play"]["price"] for e in settled)
        store["summary"] = {"n": len(settled), "w": wins, "l": len(settled) - wins,
                            "profit": round(pnl, 4), "risked": round(risked, 4),
                            "roi": round(pnl / risked, 4) if risked else None}
        print(f"  F5 plays: {wins}-{len(settled)-wins}  {pnl:+.3f} on {risked:.3f} risked "
              f"(ROI {store['summary']['roi']:+.1%})")
    # accuracy of the 3-way call, independent of whether a bet was flagged
    graded = [e for e in entries if e.get("result")]
    if graded:
        hit = sum(1 for e in graded
                  if max(("h", e["model"]["h"]), ("t", e["model"]["t"]), ("a", e["model"]["a"]),
                         key=lambda x: x[1])[0][0] ==
                     {"home": "h", "tie": "t", "away": "a"}[e["result"]["outcome"]][0])
        store["accuracy"] = {"n": len(graded), "hit": hit, "rate": round(hit / len(graded), 4)}
        print(f"  3-way call accuracy: {hit}/{len(graded)} = {hit/len(graded):.1%}")

    # ── CLV on flagged plays ──
    # Buying a contract, we want the ask to RISE after we're in: the market
    # coming toward our side. Measured only on plays, since that's where the
    # model actually committed to a view.
    clv = []
    for e in entries:
        p, c = e.get("play"), e.get("closing")
        if not p or not c:
            continue
        close_ask = {"home": c.get("askH"), "away": c.get("askA"), "tie": c.get("askT")}.get(p["key"])
        if close_ask is None:
            continue
        clv.append(close_ask - p["price"])
    if clv:
        avg = sum(clv) / len(clv)
        beat = sum(1 for d in clv if d > 0)
        store["clv"] = {"n": len(clv), "avgCents": round(avg * 100, 2),
                        "beatRate": round(beat / len(clv), 4)}
        print(f"  CLV on plays: {len(clv)}, avg {avg*100:+.2f}c, "
              f"market came to us {beat}/{len(clv)} ({beat/len(clv):.0%})")

    entries.sort(key=lambda e: (e["date"], e["gamePk"]))
    store["entries"] = entries[-1500:]
    store["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(store, open(OUT, "w"), separators=(",", ":"))
    print(f"total {len(entries)} entries -> {OUT}")


if __name__ == "__main__":
    main()
