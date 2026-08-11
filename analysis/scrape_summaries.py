"""One pass over ESPN summaries: per-player boxscores AND closing odds.

Replaces scrape_boxscores.py (same wnba_boxscores.json output format) and adds
wnba_odds.json. ESPN's `pickcenter` block only survives for the CURRENT season,
so odds coverage is 2026-only; boxscores go back to 2022.

Resumable: re-run to fill gaps. Games already in BOTH caches are skipped.
"""
import json, os, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
games = json.load(open(os.path.join(HERE, "wnba_games.json")))
BOX = os.path.join(HERE, "wnba_boxscores.json")
ODDS = os.path.join(HERE, "wnba_odds.json")

box = json.load(open(BOX)) if os.path.exists(BOX) else {}
odds = json.load(open(ODDS)) if os.path.exists(ODDS) else {}


def fetch(eid):
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={eid}"
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def parse_box(d):
    teams = {}
    for t in d.get("boxscore", {}).get("players", []):
        abbr = t["team"]["abbreviation"]
        players = []
        for grp in t.get("statistics", [])[:1]:
            labels = grp.get("labels", [])
            i_min = labels.index("MIN") if "MIN" in labels else 0
            i_pts = labels.index("PTS") if "PTS" in labels else 1
            for ath in grp.get("athletes", []):
                st = ath.get("stats") or []
                dnp = bool(ath.get("didNotPlay")) or not st
                try:
                    mins = float(st[i_min]) if not dnp else 0.0
                except Exception:
                    mins = 0.0
                try:
                    pts = float(st[i_pts]) if not dnp else 0.0
                except Exception:
                    pts = 0.0
                players.append([ath["athlete"]["id"], round(mins), round(pts)])
        teams[abbr] = players
    return teams


def parse_odds(d):
    """Closing line from pickcenter. Prefer DraftKings, else first provider with a ML."""
    pcs = d.get("pickcenter") or []
    best = None
    for pc in pcs:
        h = pc.get("homeTeamOdds") or {}
        a = pc.get("awayTeamOdds") or {}
        if h.get("moneyLine") is None or a.get("moneyLine") is None:
            continue
        rec = {
            "provider": (pc.get("provider") or {}).get("name"),
            "homeML": float(h["moneyLine"]),
            "awayML": float(a["moneyLine"]),
            "spread": pc.get("spread"),
            "homeSpreadOdds": h.get("spreadOdds"),
            "awaySpreadOdds": a.get("spreadOdds"),
            "overUnder": pc.get("overUnder"),
            "overOdds": pc.get("overOdds"),
            "underOdds": pc.get("underOdds"),
        }
        if rec["provider"] == "DraftKings":
            return rec
        if best is None:
            best = rec
    return best


todo = [g for g in games if g["id"] not in box or g["id"] not in odds]
print(f"{len(games)} games, {len(todo)} to fetch", flush=True)

errors = 0
for n, g in enumerate(todo):
    eid = g["id"]
    try:
        d = fetch(eid)
        box[eid] = parse_box(d)
        o = parse_odds(d)
        odds[eid] = o if o else {}
    except Exception:
        errors += 1
        box.setdefault(eid, {})
        odds.setdefault(eid, {})
    if n % 100 == 0:
        json.dump(box, open(BOX, "w"))
        json.dump(odds, open(ODDS, "w"))
        print(f"{n}/{len(todo)} errors={errors}", flush=True)
    time.sleep(0.15)

json.dump(box, open(BOX, "w"))
json.dump(odds, open(ODDS, "w"))
have = sum(1 for v in odds.values() if v)
print(f"FINISHED boxscores={len(box)} odds_with_ml={have}/{len(odds)} errors={errors}")
