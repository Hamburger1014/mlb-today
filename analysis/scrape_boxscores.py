"""Scrape per-player boxscores for every training game (who played, points)."""
import json, os, time, urllib.request

HERE = os.path.dirname(__file__)
games = json.load(open(os.path.join(HERE, "wnba_games.json")))
OUT = os.path.join(HERE, "wnba_boxscores.json")

done = {}
if os.path.exists(OUT):
    done = json.load(open(OUT))

def fetch(eid):
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={eid}"
    req = urllib.request.Request(url, headers={"Accept":"application/json","User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

errors = 0
for n, g in enumerate(games):
    eid = g["id"]
    if eid in done:
        continue
    try:
        d = fetch(eid)
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
                    try: mins = float(st[i_min]) if not dnp else 0.0
                    except Exception: mins = 0.0
                    try: pts = float(st[i_pts]) if not dnp else 0.0
                    except Exception: pts = 0.0
                    players.append([ath["athlete"]["id"], round(mins), round(pts)])
            teams[abbr] = players
        done[eid] = teams
    except Exception as e:
        errors += 1
        done[eid] = {}
    if n % 100 == 0:
        json.dump(done, open(OUT, "w"))
        print(f"{n}/{len(games)} done, errors={errors}", flush=True)
    time.sleep(0.2)

json.dump(done, open(OUT, "w"))
print(f"FINISHED {len(done)}/{len(games)} errors={errors}")
