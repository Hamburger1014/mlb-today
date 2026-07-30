"""Snapshot Kalshi WNBA market data to data/kalshi_wnba.json.

Run by GitHub Actions on a schedule so the GitHub Pages site (which
cannot call Kalshi directly due to CORS) always has fresh-ish prices.
"""
import json, os, urllib.request
from datetime import datetime, timezone

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = ["KXWNBAGAME", "KXWNBA1QWINNER", "KXWNBA2QWINNER", "KXWNBA3QWINNER", "KXWNBA4QWINNER"]
KEEP = ["ticker","event_ticker","yes_sub_title","no_sub_title",
        "yes_bid_dollars","yes_ask_dollars","last_price_dollars","close_time"]

def fetch(series):
    url = f"{BASE}/markets?series_ticker={series}&status=open&limit=200"
    req = urllib.request.Request(url, headers={"Accept":"application/json","User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r).get("markets", [])

out = {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "markets": {}}
total = 0
for s in SERIES:
    try:
        mkts = [{k: m.get(k) for k in KEEP} for m in fetch(s)]
    except Exception as e:
        print(f"{s}: ERROR {e}")
        mkts = []
    out["markets"][s] = mkts
    total += len(mkts)
    print(f"{s}: {len(mkts)} markets")

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
path = os.path.join(os.path.dirname(__file__), "..", "data", "kalshi_wnba.json")
with open(path, "w") as f:
    json.dump(out, f, separators=(",",":"))
print(f"total {total} markets -> {path}")
