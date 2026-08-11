"""Scrape ESPN WNBA scoreboards 2022-2026 with quarter linescores."""
import json, time, urllib.request, os

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"
OUT = os.path.join(os.path.dirname(__file__), "wnba_games.json")

RANGES = []
for year, m0, m1 in [(2022,5,10),(2023,5,10),(2024,5,10),(2025,5,10),(2026,4,10)]:
    for m in range(m0, m1+1):
        last = 31 if m in (5,7,8,10) else (30 if m in (4,6,9) else 31)
        RANGES.append(f"{year}{m:02d}01-{year}{m:02d}{last}")

def fetch(rng):
    url = f"{BASE}?dates={rng}&limit=300"
    req = urllib.request.Request(url, headers={"Accept":"application/json","User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

games = []
seen = set()
for rng in RANGES:
    try:
        d = fetch(rng)
    except Exception as e:
        print(f"{rng}: ERROR {e}")
        continue
    n = 0
    for ev in d.get("events", []):
        if ev["id"] in seen:
            continue
        seen.add(ev["id"])
        comp = ev.get("competitions",[{}])[0]
        st = comp.get("status",{}).get("type",{}).get("name","")
        if "FINAL" not in st:
            continue
        home = away = None
        for c in comp.get("competitors", []):
            rec = {
                "abbr": c.get("team",{}).get("abbreviation"),
                "score": int(c.get("score") or 0),
                "lines": [int(float(ls.get("value") or 0)) for ls in c.get("linescores",[])],
            }
            if c.get("homeAway") == "home": home = rec
            else: away = rec
        if not home or not away or len(home["lines"]) < 4 or len(away["lines"]) < 4:
            continue
        games.append({
            "id": ev["id"], "date": ev["date"],
            "seasonYear": ev.get("season",{}).get("year"),
            "seasonType": ev.get("season",{}).get("type"),
            "home": home, "away": away,
        })
        n += 1
    print(f"{rng}: +{n} finals (total {len(games)})")
    time.sleep(0.3)

with open(OUT, "w") as f:
    json.dump(games, f)
print("saved", len(games), "games ->", OUT)
