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
import json, os, urllib.request
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT  = os.path.join(ROOT, "data", "football_lines.json")
BASE = "https://site.api.espn.com/apis/site/v2/sports/"
LEAGUES = {"NFL": "football/nfl", "CFB": "football/college-football"}
ODDS_PROXY = "https://mlb-kalshi.gabrielhiginio2005.workers.dev"
ODDS_SPORT = {"NFL": "nfl", "CFB": "ncaaf"}


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


def scoreboard(path):
    """Current slate; out of season the default is empty, so look 30 days out."""
    evs = get(f"{BASE}{path}/scoreboard?limit=200").get("events", []) or []
    if not any((e.get("competitions") or [{}])[0].get("odds") for e in evs):
        d0 = datetime.now(timezone.utc).strftime("%Y%m%d")
        d1 = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y%m%d")
        evs = get(f"{BASE}{path}/scoreboard?limit=200&dates={d0}-{d1}").get("events", []) or []
    return evs


def norm_name(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def field_spreads(league):
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
    try:
        d = get(f"{ODDS_PROXY}/?odds={sport}&markets=spreads")
    except Exception as e:
        print(f"  ! field spreads ({sport}): {e}")
        return out

    saw_spreads = False
    for g in d if isinstance(d, list) else []:
        dk = None
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
            else:
                others.append(ho["point"])
        if not others:
            continue
        others.sort()
        n = len(others)
        med = others[n // 2] if n % 2 else (others[n // 2 - 1] + others[n // 2]) / 2
        key = norm_name(g.get("home_team")) + "|" + norm_name(g.get("away_team"))
        out.setdefault(key, []).append({
            "fieldSp": med, "fieldN": n, "dkSp": dk, "commence": g.get("commence_time"),
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

    seen = closed = added = matched = 0
    for league, path in LEAGUES.items():
        try:
            evs = scoreboard(path)
        except Exception as e:
            print(f"  ! {league}: {e}")
            continue
        fld = field_spreads(league)
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
            if hit and hit.get("fieldSp") is not None:
                snap["fieldSp"] = hit["fieldSp"]
                snap["fieldN"] = hit["fieldN"]
                snap["dkSpOdds"] = hit.get("dkSp")   # cross-check against ESPN's
                matched += 1
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
    print(f"{seen} games with odds ({added} new), closing refreshed on {closed}, "
          f"field spread on {matched}, {len(games)} retained -> {OUT}")


if __name__ == "__main__":
    main()
