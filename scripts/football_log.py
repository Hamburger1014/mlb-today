#!/usr/bin/env python3
"""Closing-line capture for NFL and college football.

The football views are market-only — there is no model, so there is no
model-CLV to compute. What there IS, and what this feeds, is BET-level closing
line value: for a wager actually recorded in the Fair Value Board's ledger, did
the price taken beat the price the market closed at?

That comparison needs a closing price, and nothing was storing one: the board
fetches ESPN's DraftKings lines in the browser and throws them away on reload.
This job snapshots them on the same schedule as everything else and overwrites
`closing` while a game is still pregame, so the last write before kickoff is the
close. The page joins its own localStorage ledger against this file — the server
never needs to know what anyone bet.

Output: data/football_lines.json
"""
import importlib.util
import unicodedata
import json, math, os, urllib.request
from datetime import datetime, timedelta, timezone

_ob = importlib.util.spec_from_file_location(
    "odds_budget", os.path.join(os.path.dirname(os.path.abspath(__file__)), "odds_budget.py"))
OB = importlib.util.module_from_spec(_ob); _ob.loader.exec_module(OB)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT  = os.path.join(ROOT, "data", "football_lines.json")
BASE = "https://site.api.espn.com/apis/site/v2/sports/"
LEAGUES = {"NFL": "football/nfl", "CFB": "football/college-football"}
ODDS_PROXY = "https://mlb-kalshi.gabrielhiginio2005.workers.dev"
ODDS_SPORT = {"NFL": "nfl", "CFB": "ncaaf"}
MODEL_FILE = {"NFL": "nfl_model.json", "CFB": "cfb_model.json"}

# Paper picks are recorded at TWO fixed horizons, each written once. A pick two
# weeks out and a pick on Friday are not the same measurement — the early number
# is soft and moves for reasons unrelated to the model — so a single mixed sample
# would measure the schedule rather than the picks.
#
# Recording both also tests the Walters thesis directly: he made his money on
# early numbers, betting a line before the market had finished forming an
# opinion. If this model has information, EARLY should show more closing-line
# value than LATE. If early shows less, the model is following the market rather
# than leading it, and that is worth knowing before any money is involved.
#
#   early  120h  -> Monday for a Saturday slate
#   late    36h  -> Friday for a Saturday slate
PICK_WINDOWS = {"early": 120.0, "late": 36.0}

# The football models are loaded from their exported coefficients and used to
# record a PREGAME prediction per game. Nothing did this before: the NFL and CFB
# models predict on the page and no logger touched them, so neither could ever be
# measured against the closing price and both would have sat at weight 0 for a
# whole season by default rather than by evidence. That is the same trap the MLB
# and WNBA models were rescued from by mlb_log.py and wnba_log.py.
_MODELS = {}


def load_model(league):
    """The exported fit. Its keys are exactly what predict_points() expects."""
    if league in _MODELS:
        return _MODELS[league]
    f = MODEL_FILE.get(league)
    m = None
    if f:
        try:
            m = json.load(open(os.path.join(ROOT, "data", f)))
        except Exception as e:
            print(f"  ! model {league}: {e}")
    _MODELS[league] = m
    return m


def model_predict(league, home, away):
    """Predicted margin and P(home) from the shipped coefficients.

    Imports predict_points from nfl_model rather than re-deriving it, so this
    cannot drift from the fitter the way a fourth transcription would.
    """
    m = load_model(league)
    if not m or home not in (m.get("off") or {}) or away not in (m.get("off") or {}):
        return None            # unseen team: say nothing rather than guess
    spec = importlib.util.spec_from_file_location("nfl_model",
                                                  os.path.join(HERE, "nfl_model.py"))
    NM = _MODELS.get("_nm")
    if NM is None:
        NM = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(NM)
        _MODELS["_nm"] = NM
    ph, pa = NM.predict_points(m, home, away)
    margin = ph - pa
    sc = m.get("marginScale") or 10.0
    return {"margin": round(margin, 2), "homePts": round(ph, 1), "awayPts": round(pa, 1),
            "pHome": round(1.0 / (1.0 + math.exp(-margin / sc)), 4),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def get(url, tries=3):
    last = None
    for _ in range(tries):
        try:
            # A browser UA, deliberately. ESPN 403s a custom agent, and when
            # mlb_log.py sent "mlb-log/1.0" its closing captures stopped without
            # any error surfacing — the same silent failure this file was one
            # ESPN policy change away from.
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            last = e
    raise last


def num(v):
    """ESPN sends odds as strings like '-110', '+240', and totals as 'o47.5'."""
    if v is None:
        return None
    try:
        return float(str(v).lstrip("ouOU+"))
    except ValueError:
        return None


def scoreboard(path, back_days=3, fwd_days=10):
    """Always query an EXPLICIT date window. Never trust the default scoreboard.

    ESPN's default is WEEK-SCOPED and it rolls forward early. On 2026-08-27 the
    college default already showed Week 1 (Sep 4-7) and returned exactly ONE of
    the seven games kicking off Saturday Aug 29 — college calls late-August games
    "Week 0", and the default had moved past them two days before they kicked.

    The old code only fell back to a date range when NO event carried odds. These
    carried odds — the wrong week's — so the fallback never fired and six of seven
    games were invisible, with no error. A closing line is not backfillable, so
    that would have silently lost most of opening Saturday.

    The window reaches BACKWARD as well: finals are graded off this same feed, so
    a forward-only window would drop yesterday's games before they were graded.
    """
    now = datetime.now(timezone.utc)
    d0 = (now - timedelta(days=back_days)).strftime("%Y%m%d")
    d1 = (now + timedelta(days=fwd_days)).strftime("%Y%m%d")
    evs = get(f"{BASE}{path}/scoreboard?limit=300&dates={d0}-{d1}").get("events", []) or []
    if not evs:                       # deep offseason: fall back to whatever it has
        evs = get(f"{BASE}{path}/scoreboard?limit=200").get("events", []) or []
    return evs


def _hours_to(start_iso):
    """Hours from now until kickoff, or None if the timestamp is unusable."""
    try:
        t = datetime.fromisoformat((start_iso or "").replace("Z", "+00:00"))
    except Exception:
        return None
    return (t - datetime.now(timezone.utc)).total_seconds() / 3600.0


def norm_name(s):
    """Join key for matching ESPN team names against the odds feed.

    Accents must be STRIPPED, not merely lowercased. str.isalnum() is true for
    "e", so the old version kept it and ESPN's "San Jose State Spartans" (with
    the accent) never matched the feed's plain spelling — one CFB game silently
    lost its field spread every time that team played, with no error anywhere.
    NFD splits a letter from its combining mark; dropping the marks leaves ASCII.
    """
    d = unicodedata.normalize("NFD", (s or "").lower())
    return "".join(ch for ch in d if ch.isalnum() and not unicodedata.combining(ch))


def field_spreads(league, store=None, starts=()):
    """Every OTHER book's spread number for each game, plus DraftKings' own.

    ESPN relays DraftKings alone, so the log has never recorded what the rest of
    the market thinks the number is. That comparison is the whole football
    question — roughly 9% of NFL games land on exactly 3, so holding +3 while
    the field is at +2.5 is worth far more than any price edge on this board —
    and it is NOT backfillable. Every week this does not run is a week gone.

    DraftKings is kept OUT of the field median it gets judged against; including
    a book in its own reference drags the gap toward zero.
    """
    out = {}
    sport = ODDS_SPORT.get(league)
    if not sport:
        return out
    if store is not None and not OB.spend_ok(store, sport, starts):
        return out          # inside the budget window; keep yesterday's numbers
    try:
        d = OB.fetch_json(f"{ODDS_PROXY}/?odds={sport}&markets=spreads&regions=us,eu", store)
    except Exception as e:
        print(f"  ! field spreads ({sport}): {e}")
        return out

    saw_spreads = False
    for g in d if isinstance(d, list) else []:
        dk = None
        pin = None
        others = []
        for bk in g.get("bookmakers", []) or []:
            mk = next((m for m in bk.get("markets", []) if m.get("key") == "spreads"), None)
            if not mk:
                continue
            ho = next((o for o in mk.get("outcomes", []) if o.get("name") == g.get("home_team")), None)
            if not ho or ho.get("point") is None:
                continue
            saw_spreads = True
            if bk.get("key") == "draftkings":
                dk = ho["point"]
            elif bk.get("key") == "pinnacle":
                pin = ho["point"]        # sharper than any median of the rest
                others.append(ho["point"])
            else:
                others.append(ho["point"])
        if not others:
            continue
        others.sort()
        n = len(others)
        med = others[n // 2] if n % 2 else (others[n // 2 - 1] + others[n // 2]) / 2
        key = norm_name(g.get("home_team")) + "|" + norm_name(g.get("away_team"))
        # Prefer Pinnacle's NUMBER when it posts one. A median of soft books is a
        # weaker statement about where the line belongs, and on 2026-08-26 the
        # median's best apparent edge halved when re-measured against Pinnacle.
        out.setdefault(key, []).append({
            "fieldSp": pin if pin is not None else med, "fieldN": n, "dkSp": dk,
            "pinSp": pin, "commence": g.get("commence_time"),
        })

    if not saw_spreads:
        # The Worker ignores an unrecognised `markets` param and serves h2h, which
        # parses fine and yields no points at all. Refuse to record silence as
        # data: say so, and leave the field columns absent rather than empty.
        print(f"  ! {sport}: no spreads in the feed — is the Worker deployed with "
              f"the `markets` whitelist? (`npx wrangler deploy`)")
    return out


def pick_by_time(entries, start_iso):
    """The occurrence closest in time to the game being logged.

    Teams meet more than once and this feed runs weeks ahead, so a team-pair key
    collides across dates. Anything more than 12h away is a DIFFERENT game, not
    a line move, and must not be recorded as one.
    """
    if not entries:
        return None
    try:
        want = datetime.fromisoformat((start_iso or "").replace("Z", "+00:00"))
    except ValueError:
        return entries[0] if len(entries) == 1 else None
    best, bd = None, None
    for e in entries:
        try:
            t = datetime.fromisoformat((e.get("commence") or "").replace("Z", "+00:00"))
        except ValueError:
            continue
        d = abs((t - want).total_seconds())
        if bd is None or d < bd:
            best, bd = e, d
    return best if bd is not None and bd <= 12 * 3600 else None


def snapshot(ev):
    c = (ev.get("competitions") or [{}])[0]
    o = (c.get("odds") or [None])[0]
    if not o:
        return None
    ml, ps, tt = o.get("moneyline") or {}, o.get("pointSpread") or {}, o.get("total") or {}

    def side(blk, key):
        return ((blk.get(key) or {}).get("close") or (blk.get(key) or {}).get("open") or {})

    return {
        "book": (o.get("provider") or {}).get("name"),
        "mlHome": num(side(ml, "home").get("odds")),
        "mlAway": num(side(ml, "away").get("odds")),
        "spHomeLine": num(side(ps, "home").get("line")),
        "spHomeOdds": num(side(ps, "home").get("odds")),
        "spAwayOdds": num(side(ps, "away").get("odds")),
        "totalLine": num(side(tt, "over").get("line")),
        "overOdds": num(side(tt, "over").get("odds")),
        "underOdds": num(side(tt, "under").get("odds")),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    store = {"games": {}}
    if os.path.exists(OUT):
        try:
            store = json.load(open(OUT))
        except Exception:
            pass
    games = store.get("games", {})

    seen = closed = added = matched = predicted = picked = 0
    for league, path in LEAGUES.items():
        try:
            evs = scoreboard(path)
        except Exception as e:
            print(f"  ! {league}: {e}")
            continue
        fld = field_spreads(league, store,
                            [e.get("date") for e in evs])
        for ev in evs:
            c = (ev.get("competitions") or [{}])[0]
            state = ((c.get("status") or {}).get("type") or {}).get("name", "")
            home = next((t for t in (c.get("competitors") or []) if t.get("homeAway") == "home"), None)
            away = next((t for t in (c.get("competitors") or []) if t.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            snap = snapshot(ev)
            if not snap:
                continue
            # What the rest of the market has this game at, recorded alongside
            # DraftKings' number so the two can be compared later. Matched on
            # full team names (ESPN gives abbreviations to the store, but the
            # odds feed only knows "Cincinnati Bengals").
            hit = pick_by_time(fld.get(
                norm_name((home.get("team") or {}).get("displayName")) + "|" +
                norm_name((away.get("team") or {}).get("displayName"))), ev.get("date"))
            # Look the previous entry up by id. `g` is not assigned until a
            # few lines below, so reading it here would silently hand this game
            # the PREVIOUS game's field spread on every iteration after the
            # first.
            _prev = (games.get(str(ev["id"])) or {}).get("closing") or {}
            if hit and hit.get("fieldSp") is not None:
                snap["fieldSp"] = hit["fieldSp"]
                snap["fieldN"] = hit["fieldN"]
                snap["dkSpOdds"] = hit.get("dkSp")   # cross-check against ESPN's
                matched += 1
            elif _prev.get("fieldSp") is not None:
                # Throttled run: no fresh field number. `closing` is rebuilt from
                # scratch each pass, so without carrying the last one forward the
                # budget fix would ERASE the data it exists to protect — a field
                # spread is not backfillable.
                snap["fieldSp"] = _prev["fieldSp"]
                snap["fieldN"] = _prev.get("fieldN")
                snap["dkSpOdds"] = _prev.get("dkSpOdds")
                snap["fieldStale"] = True
            seen += 1
            gid = str(ev["id"])
            g = games.get(gid)
            if g is None:
                g = games[gid] = {
                    "league": league,
                    "home": (home.get("team") or {}).get("abbreviation"),
                    "away": (away.get("team") or {}).get("abbreviation"),
                    "start": ev.get("date"),
                    "open": snap,
                }
                added += 1
            # Overwrite while still pregame; the last write before kickoff is
            # the close. Once the game starts, stop touching it.
            if state == "STATUS_SCHEDULED":
                g["closing"] = snap
                closed += 1
                # Recorded ONCE, the first time this game is seen pregame, and
                # never overwritten. A prediction that gets refreshed as kickoff
                # approaches is not a forward prediction — it quietly absorbs
                # whatever the market learned in between, which is exactly the
                # lookahead that makes a backtest lie.
                # NEVER on preseason. The page already refuses to predict it —
                # the ratings are fit on regular-season football, where both
                # sides play their starters — and a logger that disagrees with
                # the page would fill the forward record with predictions the
                # model itself disowns, then measure them. ESPN's season type 1
                # is preseason.
                _st = (ev.get("season") or {}).get("type")
                if "model" not in g and _st != 1:
                    mp = model_predict(league, g["home"], g["away"])
                    if mp:
                        g["model"] = mp
                        predicted += 1

                # ── PAPER PICK against the spread, for CLV only ───────────
                # CFB has the strongest skill test of the four models (11.04 SE
                # vs a constant) and has NEVER been tested against a price. This
                # accrues that evidence without money at risk: record what the
                # model WOULD take and at what number, then let football_clv.py
                # grade it against the close.
                #
                # Written ONCE, inside a fixed window before kickoff, so the
                # sample is homogeneous. A pick made two weeks out and one made
                # on Friday are not the same measurement — the early line is soft
                # and moves for reasons that have nothing to do with the model.
                # PICK_WINDOW_H = 36 puts a Saturday slate on Friday.
                if (_st != 1 and g.get("model")
                        and snap.get("spHomeLine") is not None):
                    _hrs = _hours_to(g.get("start"))
                    _picks = g.setdefault("picks", {})
                    for _tag, _win in PICK_WINDOWS.items():
                        if _tag in _picks or _hrs is None or not (0 < _hrs <= _win):
                            continue
                        # Market's implied home margin is the negated home line.
                        _mkt = -snap["spHomeLine"]
                        _edge = g["model"]["margin"] - _mkt
                        _home = _edge > 0        # model likes home vs the number
                        _picks[_tag] = {
                            "side": "home" if _home else "away",
                            # THIS SIDE'S line, so CLV is `taken - close` on the
                            # same quantity for both sides. Away is the negation.
                            "line": snap["spHomeLine"] if _home else -snap["spHomeLine"],
                            "odds": snap.get("spHomeOdds") if _home else snap.get("spAwayOdds"),
                            "homeLine": snap["spHomeLine"],
                            "modelMargin": g["model"]["margin"],
                            "mktMargin": _mkt,
                            "edgePts": round(_edge, 2),
                            # The CONTROL: the side the market itself favours at
                            # pick time. Raw CLV is not a valid test on its own —
                            # measured on MLB, backing the favourite alone earned
                            # +0.493pp and made a model look like it had +0.519pp
                            # of skill. The paired difference is the real number.
                            "ctrlSide": "home" if snap["spHomeLine"] < 0 else "away",
                            "hoursOut": round(_hrs, 1),
                            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        }
                        picked += 1
            if state == "STATUS_FINAL":
                try:
                    g["final"] = {"hs": int(float(home.get("score"))),
                                  "as": int(float(away.get("score")))}
                except (TypeError, ValueError):
                    pass

    # Drop anything that kicked off more than a month ago so the file cannot
    # grow without bound across a season.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    games = {k: v for k, v in games.items() if (v.get("start") or "") >= cutoff}

    store["games"] = games
    store["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(store, open(OUT, "w"), separators=(",", ":"))
    graded = sum(1 for g in games.values() if g.get("model") and g.get("final"))
    print(f"{seen} games with odds ({added} new), closing refreshed on {closed}, "
          f"field spread on {matched}, model logged on {predicted} new, "
          f"paper picks {picked} new "
          f"({graded} now gradable), {len(games)} retained | {OB.summary(store)} -> {OUT}")


if __name__ == "__main__":
    main()
