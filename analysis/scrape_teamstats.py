"""Scrape team-level box stats needed for possession estimates.

Possessions = FGA - OREB + TO + 0.44*FTA   (Dean Oliver)
Stored per game per team: FGA, FTA, OREB, DREB, TO, PTS, plus 3PA for later.
"""
import json, os, time, urllib.request

HERE = os.path.dirname(__file__)
games = json.load(open(os.path.join(HERE, "wnba_games.json")))
OUT = os.path.join(HERE, "wnba_teamstats.json")
done = json.load(open(OUT)) if os.path.exists(OUT) else {}

def split2(v, idx):
    """'31-75' -> idx 0 = made, 1 = attempted"""
    try: return float(str(v).split("-")[idx])
    except Exception: return None

def fetch(eid):
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={eid}"
    req = urllib.request.Request(url, headers={"Accept":"application/json","User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

errors = 0
for n, g in enumerate(games):
    eid = g["id"]
    if eid in done: continue
    try:
        d = fetch(eid)
        teams = {}
        for t in d.get("boxscore", {}).get("teams", []):
            abbr = t["team"]["abbreviation"]
            s = {x.get("name"): x.get("displayValue") for x in t.get("statistics", [])}
            fga = split2(s.get("fieldGoalsMade-fieldGoalsAttempted"), 1)
            fta = split2(s.get("freeThrowsMade-freeThrowsAttempted"), 1)
            tpa = split2(s.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"), 1)
            def num(k):
                try: return float(s.get(k))
                except Exception: return None
            oreb = num("offensiveRebounds"); dreb = num("defensiveRebounds")
            # totalTurnovers includes team turnovers; that's the right figure
            to = num("totalTurnovers")
            if to is None: to = num("turnovers")
            if None in (fga, fta, oreb, to): continue
            teams[abbr] = {"fga":fga, "fta":fta, "tpa":tpa, "oreb":oreb, "dreb":dreb, "to":to}
        done[eid] = teams
    except Exception as e:
        errors += 1; done[eid] = {}
    if n % 100 == 0:
        json.dump(done, open(OUT,"w")); print(f"{n}/{len(games)} errors={errors}", flush=True)
    time.sleep(0.2)

json.dump(done, open(OUT,"w"))
ok = sum(1 for v in done.values() if len(v)==2)
print(f"FINISHED {len(done)} games, {ok} with both teams, errors={errors}")
