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


def get(url, tries=3):
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "football-log/1.0",
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

    seen = closed = added = 0
    for league, path in LEAGUES.items():
        try:
            evs = scoreboard(path)
        except Exception as e:
            print(f"  ! {league}: {e}")
            continue
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
          f"{len(games)} retained -> {OUT}")


if __name__ == "__main__":
    main()
