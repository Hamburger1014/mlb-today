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

Sources, all already produced by earlier steps of the same job:
  Kalshi MLB   data/kalshi_mlb.json
  DK MLB       ESPN scoreboard (one call — it carries structured moneylines now,
               which it did not in June, so no per-game summary fan-out)
  WNBA both    data/wnba_predictions.json (model prob + vegas + spread + kalshi)
"""
import json, math, os, sys, urllib.request
from datetime import datetime, timedelta, timezone

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
            req = urllib.request.Request(url, headers={"User-Agent": "card-log/1.0",
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

def kalshi_mlb():
    path = os.path.join(ROOT, "data", "kalshi_mlb.json")
    if not os.path.exists(path):
        return {}, None
    try:
        snap = json.load(open(path))
    except Exception:
        return {}, None
    age = None
    try:
        t = datetime.fromisoformat(snap["fetched_at"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - t).total_seconds() / 60.0
    except Exception:
        pass
    by_event = {}
    for series, markets in (snap.get("markets") or {}).items():
        if "GAME" not in series.upper():
            continue
        for m in markets or []:
            by_event.setdefault(m.get("event_ticker"), []).append(m)

    def mid(m):
        try:
            b, a = float(m["yes_bid_dollars"]), float(m["yes_ask_dollars"])
            if 0 <= b <= a <= 1 and a > 0:
                return (b + a) / 2
        except (TypeError, ValueError, KeyError):
            pass
        return None

    out = {}
    for _, mkts in by_event.items():
        sides = {}
        for m in mkts:
            suf = (m.get("ticker") or "").rsplit("-", 1)[-1].upper()
            v = mid(m)
            if v is not None:
                sides[suf] = v
        if len(sides) != 2:
            continue
        (t1, p1), (t2, p2) = sorted(sides.items())
        tot = p1 + p2
        if tot > 0:
            out["|".join(sorted([t1, t2]))] = {t1: p1 / tot, t2: p2 / tot}
    return out, age


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

def candidates():
    out = []

    def add(league, gid, id_src, home, away, start, side_home, p_fair, p_price,
            kind, note, price_amer=None, market="ML", line=None):
        if p_fair is None or p_price is None or not (0.02 < p_price < 0.98):
            return
        raw = p_fair - p_price
        shrink = CARD_CROSS_SHRINK if kind == "CROSS" else CARD_MODEL_SHRINK
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

    # ── cross-book: MLB (Kalshi vs DraftKings) ──
    kal, age = kalshi_mlb()
    fresh = age is None or age <= CARD_MAX_STALE_MIN
    if not fresh:
        print(f"  cross-book suppressed: Kalshi snapshot {age:.0f} min old")
    if fresh and kal:
        for key, dk in dk_mlb().items():
            k = kal.get(norm_pair(dk["home"], dk["away"]))
            if not k:
                continue
            kh = k.get(ESPN_TO_STATS.get(dk["home"], dk["home"]))
            if kh is None:
                continue
            bk = dk["fairHome"]
            for side_home in (True, False):
                kp = kh if side_home else 1 - kh
                bp = bk if side_home else 1 - bk
                amer = (dk["homeML"] if side_home else dk["awayML"])
                if bp > kp:
                    add("MLB", dk["gameId"], "espn", dk["home"], dk["away"], dk["start"],
                        side_home, bp, kp, "CROSS", "Kalshi cheap vs book")
                elif kp > bp:
                    add("MLB", dk["gameId"], "espn", dk["home"], dk["away"], dk["start"],
                        side_home, kp, bp, "CROSS", "Book cheap vs Kalshi", amer)

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
            v, kg, sp = e.get("vegas"), (e.get("kalshi") or {}).get("game"), e.get("spread")
            gid, home, away = e["id"], e["home"], e["away"]
            start = e.get("gameDate")
            dv = devig_power([ml_to_raw(v["homeML"]), ml_to_raw(v["awayML"])]) if v else None
            bk = dv[0] if dv else None
            # cross-book
            if fresh and bk is not None and kg is not None:
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
            # model — spread (home side only, as the page offers)
            mu = (e.get("model") or {}).get("mu")
            if mu is not None and sp and sp.get("homeOdds") is not None:
                sd = FIT["spreadSd"]
                thr = -sp["homeLine"]
                p_cover = 1 - norm_cdf(thr, mu, sd)
                dvs = devig_power([ml_to_raw(sp["homeOdds"]), ml_to_raw(sp["awayOdds"])])
                if dvs:
                    fair = sigmoid(W_WNBA_SP * logit(p_cover) + (1 - W_WNBA_SP) * logit(dvs[0]))
                    if fair * payout(sp["homeOdds"]) - (1 - fair) >= 0.04:
                        add("WNBA", gid, "espn", home, away, start, True, fair, dvs[0],
                            "MODEL", "spread model (w=0.40)", sp["homeOdds"],
                            market="SPREAD", line=sp["homeLine"])

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
    paths = {"MLB": "baseball/mlb", "WNBA": "basketball/wnba"}
    for (league, ymd), group in by_day.items():
        path = paths.get(league)
        if not path:
            continue
        try:
            d = get(f"https://site.api.espn.com/apis/site/v2/sports/{path}"
                    f"/scoreboard?dates={ymd}&limit=200")
        except Exception as e:
            print(f"  ! grade {league} {ymd}: {e}")
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
                m = (hs - as_) + (e.get("line") or 0)
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
    for c in candidates():
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
    print(f"card: +{added} advised, {graded} newly graded | {s['w']}-{s['l']} "
          f"{s['won']:+.2f}u on {s['staked']:.1f}u staked | {s['pending']} pending -> {OUT}")


if __name__ == "__main__":
    main()
