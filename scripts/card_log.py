#!/usr/bin/env python3
"""Server-side log for Today's Card.

The card was logging in the browser, which meant a day the page was never opened
was a day not recorded — the exact failure the model logs exist to avoid. This
recomputes the card on the schedule instead, so the record is written whether or
not anyone is looking.

  LOG    - every candidate the card would advise PREGAME, with the unit size.
  GRADE  - settle from the real result once final.
  REPORT - record, units staked, units won/lost, ROI.

Constants MUST match Today's Card in index.html; scripts/verify_card_parity.py
diffs them and the workflow fails on drift.

CROSS-BOOK IS NOW A NO-OP. It compared two independent de-vigged prices on the
same game and bought whichever venue was cheap — the only model-free edge this
site could detect automatically. It needed TWO price sources and Kalshi was the
second one. Kalshi is no longer used anywhere in this project, so one source
remains and there is nothing to compare it against. The machinery below is
source-agnostic and deliberately left in place: it yields nothing while a single
source exists, and revives untouched if a second feed is ever added.

Sources:
  DK MLB       ESPN scoreboard (one call — it carries structured moneylines now,
               which it did not in June, so no per-game summary fan-out)
  WNBA         data/wnba_predictions.json (model prob + vegas + spread)
"""
import importlib.util
import json, math, os, sys, urllib.request
from datetime import datetime, timedelta, timezone

_ob = importlib.util.spec_from_file_location(
    "odds_budget", os.path.join(os.path.dirname(os.path.abspath(__file__)), "odds_budget.py"))
OB = importlib.util.module_from_spec(_ob); _ob.loader.exec_module(OB)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT  = os.path.join(ROOT, "data", "card_log.json")
sys.path.insert(0, HERE)
from wnba_log import FIT, norm_cdf, sigmoid, ml_to_raw          # one model copy, reused

# ── constants — keep identical to index.html ──────────────────────────
CARD_UNIT          = 0.01
CARD_MODEL_SHRINK  = 0.37
CARD_CROSS_SHRINK  = 0.85
CARD_MIN_EDGE      = 0.02
CARD_MAX_STALE_MIN = 35
KALSHI_FEE         = 0.07
KELLY_FRACTION     = 0.25
MAX_STAKE_GAME     = 0.02
MAX_STAKE_DAY      = 0.06
W_WNBA_ML          = 0.15
W_WNBA_SP          = 0.40
MIN_PROB_EDGE      = 0.04


def get(url, tries=3):
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            last = e
    raise last


def devig_power(raw):
    """Solve sum(q^k)=1. Mirrors devigPower() in the page."""
    if any(x is None or x <= 0 for x in raw):
        return None
    s = sum(raw)
    if s <= 1:
        return [x / s for x in raw]
    lo, hi = 0.5, 5.0
    for _ in range(100):
        k = (lo + hi) / 2
        if sum(x ** k for x in raw) > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    return [x ** k for x in raw]


def logit(p):
    p = min(1 - 1e-9, max(1e-9, p))
    return math.log(p / (1 - p))


def blend(pm, pk, w):
    if pm is None:
        return pk
    if pk is None:
        return pm
    return sigmoid(w * logit(pm) + (1 - w) * logit(pk))


def payout(ml):
    return ml / 100.0 if ml > 0 else 100.0 / abs(ml)


def cost(p):
    return KALSHI_FEE * p * (1 - p)


def breakeven_american(p):
    if p is None or p <= 0 or p >= 1:
        return None
    return -round(100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


# ── price sources ─────────────────────────────────────────────────────

ODDS_PROXY = "https://mlb-kalshi.gabrielhiginio2005.workers.dev"


def sharp_reference(sport="mlb", store=None, starts=()):
    """The field's price, DraftKings EXCLUDED, plus DraftKings on its own.

    Read through the same Cloudflare Worker the page uses, so the API key stays
    a server-side secret and this script needs no credentials. Keyed by the
    normalised team pair.

    DraftKings is deliberately kept out of the consensus it is judged against —
    including it drags every gap toward zero in proportion to its weight.
    """
    out = {}
    if store is not None and not OB.spend_ok(store, sport, starts):
        return out, None      # inside the budget window
    try:
        d = OB.fetch_json(f"{ODDS_PROXY}/?odds={sport}", store)
    except Exception as e:
        print(f"  ! sharp odds ({sport}): {e}")
        return out, None
    for g in d if isinstance(d, list) else []:
        pin = dk = None
        others = []
        for bk in g.get("bookmakers", []) or []:
            mk = next((m for m in bk.get("markets", []) if m.get("key") == "h2h"), None)
            if not mk:
                continue
            ho = next((o for o in mk.get("outcomes", []) if o.get("name") == g.get("home_team")), None)
            ao = next((o for o in mk.get("outcomes", []) if o.get("name") == g.get("away_team")), None)
            if not ho or not ao or not ho.get("price") or not ao.get("price"):
                continue
            dv = devig_power([1 / ho["price"], 1 / ao["price"]])
            if not dv:
                continue
            if bk.get("key") == "pinnacle":
                pin = dv[0]
            if bk.get("key") == "draftkings":
                dk = dv[0]
                dk_home_ml, dk_away_ml = ho["price"], ao["price"]
                continue
            others.append(dv[0])
        if not others or dk is None:
            continue
        others.sort()
        n = len(others)
        med = others[n // 2] if n % 2 else (others[n // 2 - 1] + others[n // 2]) / 2
        # Teams play multi-game series and this feed spans several days, so a
        # team-pair key COLLIDES across dates. Keying by pair alone silently
        # priced tonight's DraftKings line against tomorrow's game and produced
        # a 7.6pp edge that did not exist. Store every occurrence; the caller
        # picks by kickoff time.
        key = norm_name(g.get("home_team")) + "|" + norm_name(g.get("away_team"))
        out.setdefault(key, []).append({
            "fairHome": pin if pin is not None else med,
            "dkHome": dk, "nBooks": n, "pinnacle": pin is not None,
            "commence": g.get("commence_time"),
        })
    return out, None


def pick_by_time(entries, start_iso):
    """The occurrence closest in time to the game we are actually pricing."""
    if not entries:
        return None
    if len(entries) == 1 or not start_iso:
        return entries[0]
    def ts(x):
        try:
            return datetime.fromisoformat((x or "").replace("Z", "+00:00")).timestamp()
        except Exception:
            return None
    t0 = ts(start_iso)
    if t0 is None:
        return entries[0]
    best, bd = None, None
    for e in entries:
        t = ts(e.get("commence"))
        if t is None:
            continue
        d = abs(t - t0)
        if bd is None or d < bd:
            best, bd = e, d
    # More than 12h away is a different game, not a line move.
    return best if (bd is not None and bd <= 12 * 3600) else None


def norm_name(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def dec_to_prob(dec):
    return 1.0 / dec if dec else None


def dk_mlb():
    """ESPN's MLB scoreboard now carries structured moneylines, so one call
    covers the slate. Keyed by sorted ESPN abbreviation pair."""
    out = {}
    try:
        d = get("https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard")
    except Exception as e:
        print(f"  ! dk_mlb: {e}")
        return out
    for ev in d.get("events", []):
        c = (ev.get("competitions") or [{}])[0]
        if ((c.get("status") or {}).get("type") or {}).get("name") != "STATUS_SCHEDULED":
            continue
        o = (c.get("odds") or [None])[0]
        if not o or not o.get("moneyline"):
            continue
        h = next((t for t in c.get("competitors", []) if t.get("homeAway") == "home"), None)
        a = next((t for t in c.get("competitors", []) if t.get("homeAway") == "away"), None)
        if not h or not a:
            continue
        try:
            hml = float(str(((o["moneyline"].get("home") or {}).get("close") or {})["odds"]).replace("+", ""))
            aml = float(str(((o["moneyline"].get("away") or {}).get("close") or {})["odds"]).replace("+", ""))
        except (KeyError, TypeError, ValueError):
            continue
        dv = devig_power([ml_to_raw(hml), ml_to_raw(aml)])
        if not dv:
            continue
        hab = (h.get("team") or {}).get("abbreviation")
        aab = (a.get("team") or {}).get("abbreviation")
        out["|".join(sorted([hab, aab]))] = {
            "home": hab, "away": aab, "fairHome": dv[0],
            "homeName": (h.get("team") or {}).get("displayName"),
            "awayName": (a.get("team") or {}).get("displayName"),
            "homeML": hml, "awayML": aml,
            "gameId": ev["id"], "start": ev.get("date"),
        }
    return out


# Kalshi uses statsapi abbreviations; ESPN differs on exactly four clubs.
# Verified against a live snapshot rather than assumed — Kalshi keys read
# HOU|SF and KC|LAD, so mapping SF->SFG and KC->KCR (which an earlier draft did)
# breaks those two joins and silently drops the games from the card.
ESPN_TO_STATS = {"ARI": "AZ", "OAK": "ATH", "TBR": "TB", "CHW": "CWS"}


def norm_pair(a, b):
    f = lambda x: ESPN_TO_STATS.get(x, x)
    return "|".join(sorted([f(a), f(b)]))


# ── candidate generation (mirrors cardCandidates() in the page) ───────

def candidates(store=None):
    out = []

    def add(league, gid, id_src, home, away, start, side_home, p_fair, p_price,
            kind, note, price_amer=None, market="ML", line=None):
        if p_fair is None or p_price is None or not (0.02 < p_price < 0.98):
            return
        raw = p_fair - p_price
        # Shrink by kind; subtract an exchange fee only where one is charged. A
        # sportsbook's cost is the vig, which is already inside p_price.
        shrink = CARD_MODEL_SHRINK if kind == "MODEL" else CARD_CROSS_SHRINK
        edge = raw * shrink - (cost(p_price) if kind == "CROSS" else 0.0)
        if edge < CARD_MIN_EDGE:
            return
        b = (1 - p_price) / p_price
        kelly = max(0.0, (p_fair * (b + 1) - 1) / b) if b > 0 else 0.0
        out.append({
            "league": league, "kind": kind, "note": note,
            "matchup": f"{away}@{home}", "gameId": str(gid), "idSrc": id_src,
            "start": start, "market": market, "side": home if side_home else away,
            "sideIsHome": side_home, "line": line,
            "priceAmer": price_amer if price_amer is not None else breakeven_american(p_price),
            "fair": round(p_fair, 4), "price": round(p_price, 4),
            "edge": round(edge, 4), "kelly": min(0.05, KELLY_FRACTION * kelly),
        })

    # ── BOOK: the field's price vs DraftKings, the venue actually bet ──
    # Fair is the consensus with DraftKings EXCLUDED; price is the VIGGED
    # DraftKings number, because that is what gets charged. No exchange fee is
    # subtracted — a sportsbook's cost is the vig, already inside the price.
    # Pass the store so the throttle can persist its timestamp, and today's
    # start times so it tightens up near first pitch.
    sharp, _ = sharp_reference("mlb", store,
                               [g.get("start") for g in (dk_mlb() or {}).values()])
    if sharp:
        for _key, dkg in dk_mlb().items():
            ref = pick_by_time(
                sharp.get(norm_name(dkg.get("homeName")) + "|" + norm_name(dkg.get("awayName"))),
                dkg.get("start"))
            if not ref:
                continue
            f = ref["fairHome"]
            note = ("Pinnacle" if ref["pinnacle"] else f"{ref['nBooks']}-book") + " vs DraftKings"
            raw_h, raw_a = ml_to_raw(dkg["homeML"]), ml_to_raw(dkg["awayML"])
            if f > raw_h:
                add("MLB", dkg["gameId"], "espn", dkg["home"], dkg["away"], dkg["start"],
                    True, f, raw_h, "BOOK", note, dkg["homeML"])
            if (1 - f) > raw_a:
                add("MLB", dkg["gameId"], "espn", dkg["home"], dkg["away"], dkg["start"],
                    False, 1 - f, raw_a, "BOOK", note, dkg["awayML"])

    # ── WNBA: cross-book AND model, both from the log this job already wrote ──
    wp = os.path.join(ROOT, "data", "wnba_predictions.json")
    if os.path.exists(wp):
        try:
            entries = json.load(open(wp)).get("entries", [])
        except Exception:
            entries = []
        for e in entries:
            if e.get("result"):
                continue
            v, sp = e.get("vegas"), e.get("spread")
            kg = None                      # no second source; see second_source()
            gid, home, away = e["id"], e["home"], e["away"]
            start = e.get("gameDate")
            dv = devig_power([ml_to_raw(v["homeML"]), ml_to_raw(v["awayML"])]) if v else None
            bk = dv[0] if dv else None
            # cross-book
            if False and bk is not None and kg is not None:   # WNBA sharp feed pending
                for side_home in (True, False):
                    kp = kg if side_home else 1 - kg
                    bp = bk if side_home else 1 - bk
                    amer = (v["homeML"] if side_home else v["awayML"])
                    if bp > kp:
                        add("WNBA", gid, "espn", home, away, start, side_home, bp, kp,
                            "CROSS", "Kalshi cheap vs book")
                    elif kp > bp:
                        add("WNBA", gid, "espn", home, away, start, side_home, kp, bp,
                            "CROSS", "Book cheap vs Kalshi", amer)
            # model — moneyline
            pm = (e.get("model") or {}).get("game")
            if pm is not None and bk is not None:
                fair = blend(pm, bk, W_WNBA_ML)
                for side_home in (True, False):
                    p = fair if side_home else 1 - fair
                    mkt = bk if side_home else 1 - bk
                    if p - mkt < MIN_PROB_EDGE:
                        continue
                    amer = v["homeML"] if side_home else v["awayML"]
                    if p * payout(amer) - (1 - p) < 0.03:
                        continue
                    add("WNBA", gid, "espn", home, away, start, side_home, p, mkt,
                        "MODEL", "moneyline model (w=0.15)", amer)
            # model — spread, BOTH SIDES.
            #
            # This used to evaluate the home side only, with the note "as the
            # page offers". The page has offered whichever side has better EV for
            # some time, so the independent record was silently dropping every
            # away-side play the card advised. That is worse than a missing
            # feature: the WNBA spread is the only market with earned weight, it
            # needs roughly 583 bets to reach 2 sigma, and a home-only sample
            # measures a biased subset of the thing being decided.
            mu = (e.get("model") or {}).get("mu")
            if mu is not None and sp and sp.get("homeOdds") is not None:
                sd = FIT["spreadSd"]
                p_cover = 1 - norm_cdf(-sp["homeLine"], mu, sd)
                dvs = devig_power([ml_to_raw(sp["homeOdds"]), ml_to_raw(sp["awayOdds"])])
                if dvs:
                    fair_h = sigmoid(W_WNBA_SP * logit(p_cover) + (1 - W_WNBA_SP) * logit(dvs[0]))
                    # No push mass under a continuous normal, so the away side is
                    # the complement. Each side carries ITS OWN line and price.
                    for side_home in (True, False):
                        fair = fair_h if side_home else 1 - fair_h
                        price = dvs[0] if side_home else dvs[1]
                        amer = sp["homeOdds"] if side_home else sp["awayOdds"]
                        line = sp["homeLine"] if side_home else sp.get(
                            "awayLine", -sp["homeLine"])
                        if fair * payout(amer) - (1 - fair) < 0.04:
                            continue
                        add("WNBA", gid, "espn", home, away, start, side_home, fair, price,
                            "MODEL", "spread model (w=0.40)", amer,
                            market="SPREAD", line=line)

    out.sort(key=lambda c: -c["edge"])
    # correlated exposure: one game is one position
    day = 0.0
    per_game = {}
    for c in out:
        g = c["gameId"]
        room = min(MAX_STAKE_GAME - per_game.get(g, 0.0), MAX_STAKE_DAY - day, c["kelly"])
        c["stakeCapped"] = max(0.0, room)
        per_game[g] = per_game.get(g, 0.0) + c["stakeCapped"]
        day += c["stakeCapped"]
    return out


# ── grading ───────────────────────────────────────────────────────────

def grade(entries):
    pend = [e for e in entries if not e.get("result") and e.get("start")]
    if not pend:
        return 0
    changed = 0
    by_day = {}
    for e in pend:
        try:
            t = datetime.fromisoformat(e["start"].replace("Z", "+00:00"))
        except Exception:
            continue
        if datetime.now(timezone.utc) < t + timedelta(hours=4):
            continue
        by_day.setdefault((e["league"], t.strftime("%Y%m%d")), []).append(e)
    # Every league the card can advise. NFL and CFB were missing entirely, so a
    # football play would have sat pending forever — silently, since an
    # ungradeable entry looks identical to one whose game has not finished.
    paths = {"MLB": "baseball/mlb", "WNBA": "basketball/wnba",
             "NFL": "football/nfl", "CFB": "football/college-football"}
    for (league, ymd), group in by_day.items():
        path = paths.get(league)
        if not path:
            continue
        # site.api.espn.com returns 403 to the GitHub runner for the basketball
        # scoreboard — proven by wnba_log.py's probe on 2026-08-14, which recorded
        # 403 on every attempt while the same call succeeded from a desktop. This
        # grader used that host alone, so WNBA card bets could be logged and then
        # never graded. site.web.api.espn.com answers the runner.
        # groups=80 is the FBS filter; without it ESPN returns a top-25 subset and
        # most college games would be missing from the lookup.
        qs = f"/scoreboard?dates={ymd}&limit=200" + ("&groups=80" if league == "CFB" else "")
        d = None
        for host in ("https://site.api.espn.com/apis/site/v2/sports/",
                     "https://site.web.api.espn.com/apis/site/v2/sports/"):
            try:
                d = get(host + path + qs)
                break
            except Exception as e:
                print(f"  ! grade {league} {ymd} via {host.split('//')[1][:16]}: {e}")
        if d is None:
            continue
        finals = {}
        for ev in d.get("events", []):
            c = (ev.get("competitions") or [{}])[0]
            if ((c.get("status") or {}).get("type") or {}).get("name") != "STATUS_FINAL":
                continue
            h = next((t for t in c.get("competitors", []) if t.get("homeAway") == "home"), None)
            a = next((t for t in c.get("competitors", []) if t.get("homeAway") == "away"), None)
            if not h or not a:
                continue
            try:
                finals[str(ev["id"])] = (int(float(h["score"])), int(float(a["score"])))
            except (TypeError, ValueError):
                pass
        for e in group:
            f = finals.get(str(e["gameId"]))
            if not f:
                continue
            hs, as_ = f
            won = push = None
            if e["market"] == "ML":
                if hs == as_:
                    push = True
                else:
                    won = (hs > as_) if e["sideIsHome"] else (as_ > hs)
            elif e["market"] == "SPREAD":
                # `line` is the BET SIDE's own line, so the margin has to be
                # measured from that side. Grading every spread from the home
                # team's point of view marks an away-side winner as a loss.
                if e.get("sideIsHome") is None:
                    continue
                margin = (hs - as_) if e["sideIsHome"] else (as_ - hs)
                m = margin + (e.get("line") or 0)
                push = abs(m) < 1e-9
                won = None if push else m > 0
            if won is None and not push:
                continue
            e["result"] = {"won": bool(won), "push": bool(push), "hs": hs, "as": as_,
                           "graded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            changed += 1
    return changed


def main():
    store = {"entries": []}
    if os.path.exists(OUT):
        try:
            store = json.load(open(OUT))
        except Exception:
            pass
    entries = store.get("entries", [])
    have = {e["id"] for e in entries}

    now = datetime.now(timezone.utc)
    added = 0
    for c in candidates(store):
        units = c["stakeCapped"] / CARD_UNIT
        if units < 0.05:
            continue                      # capped out — the card shows "no room", not advice
        try:
            st = datetime.fromisoformat((c["start"] or "").replace("Z", "+00:00"))
        except Exception:
            continue
        if st <= now:
            continue                      # advice after first pitch is not advice
        pid = "|".join([st.strftime("%Y-%m-%d"), c["gameId"], c["kind"],
                        c["market"], str(c["side"])])
        if pid in have:
            continue
        entries.append({
            "id": pid, "league": c["league"], "kind": c["kind"], "matchup": c["matchup"],
            "gameId": c["gameId"], "market": c["market"], "side": c["side"],
            "sideIsHome": c["sideIsHome"], "line": c["line"], "priceAmer": c["priceAmer"],
            "units": round(units, 2), "edge": c["edge"], "start": c["start"],
            "logged_at": now.isoformat(timespec="seconds"), "result": None,
        })
        have.add(pid)
        added += 1

    graded = grade(entries)

    settled = [e for e in entries if e.get("result") and not e["result"]["push"]]
    w = sum(1 for e in settled if e["result"]["won"])
    staked = sum(e["units"] for e in settled)
    won = sum((e["units"] * payout(e["priceAmer"])) if e["result"]["won"] else -e["units"]
              for e in settled)
    store["summary"] = {
        "n": len(entries), "graded": sum(1 for e in entries if e.get("result")),
        "pushes": sum(1 for e in entries if e.get("result") and e["result"]["push"]),
        "w": w, "l": len(settled) - w,
        "staked": round(staked, 2), "won": round(won, 2),
        "roi": round(won / staked, 4) if staked > 0 else None,
        "pending": sum(1 for e in entries if not e.get("result")),
    }
    entries.sort(key=lambda e: e.get("start") or "")
    store["entries"] = entries[-600:]
    store["updated_at"] = now.isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(store, open(OUT, "w"), separators=(",", ":"))
    s = store["summary"]
    print(f"[{OB.summary(store)}] card: +{added} advised, {graded} newly graded | {s['w']}-{s['l']} "
          f"{s['won']:+.2f}u on {s['staked']:.1f}u staked | {s['pending']} pending -> {OUT}")


if __name__ == "__main__":
    main()
