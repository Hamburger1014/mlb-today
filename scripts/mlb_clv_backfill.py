#!/usr/bin/env python3
"""Backfill true closing-line value for the MLB full-game model.

CLV has been the metric this project treats as primary, but the MLB sample was
n=30 and meaningless. Kalshi publishes free per-market candlesticks that survive
settlement, so every prediction already logged can have its real price path
reconstructed now instead of accruing for months.

  price path   series/KXMLBGAME/markets/{ticker}/candlesticks  (bid+ask OHLC)
  model        replayed point-in-time, same as mlb_fullgame_test.py
  CLV          implied prob of the MODEL'S SIDE at log time vs at the close

This supersedes reconstructing closes from the git history of
data/kalshi_mlb.json: candlesticks give the actual last pre-tip quote rather
than whichever 10-minute snapshot happened to land nearest.

PRE-REGISTERED, written before the first run so the result cannot be
rationalised afterwards:
    mean CLV >= +0.5pp at 2 sigma  -> the model anticipates the market. This
                                      contradicts the log-loss result and
                                      reopens the modelling question.
    within +/- 0.5pp               -> confirms no edge. Close permanently.
    <= -0.5pp                      -> anti-informative; the flagging rule was
                                      selecting the model's own errors.

THE PRE-REGISTRATION WAS INCOMPLETE, and the first run proved it. Raw CLV came
back +0.52pp at z=3.09 - apparently decisive evidence the model anticipates the
market. It is not. Kalshi prices drift toward the favourite between the open and
the close, and the model agrees with the market's favourite in 86% of games, so
it collects that drift without contributing anything. Backing whatever side the
market ALREADY favoured at log time earns +0.49pp on the same games.

So raw CLV is not a test of a model. It is only a test relative to a control
that shares the model's directional bias. Both are computed below, plus the
paired difference, which is the number that actually answers the question.
"""
import json, math, os, sys, time, urllib.request
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mlb_fullgame_test import model_prob, standings_asof, team_abbrs   # one model copy

KAL = "https://api.elections.kalshi.com/trade-api/v2"
CACHE = os.path.join(ROOT, "analysis", "kalshi_candles_cache.json")


def get(url, tries=3, pause=0.25):
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "clv-backfill/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                time.sleep(pause)
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(0.6)
    raise last


def mid(bar):
    """Mid of the closing bid/ask in a candle. Kalshi quotes both sides."""
    try:
        b = float(bar["yes_bid"]["close_dollars"])
        a = float(bar["yes_ask"]["close_dollars"])
        if 0 <= b <= a <= 1 and a > 0:
            return (b + a) / 2
    except (KeyError, TypeError, ValueError):
        pass
    return None


def series_of(ticker):
    return ticker.split("-", 1)[0]


def candles(ticker, t0, t1):
    u = (f"{KAL}/series/{series_of(ticker)}/markets/{ticker}/candlesticks"
         f"?start_ts={int(t0)}&end_ts={int(t1)}&period_interval=60")
    try:
        return get(u).get("candlesticks", []) or []
    except Exception:
        return []


def price_at(bars, ts, before=True):
    """Last quote at/before ts (or first at/after). None if no quote exists."""
    cand = [b for b in bars if (b.get("end_period_ts", 0) <= ts) == before]
    if not cand:
        return None
    cand.sort(key=lambda b: b.get("end_period_ts", 0), reverse=before)
    for b in cand:
        m = mid(b)
        if m is not None:
            return m
    return None


def main():
    arch = json.load(open(os.path.join(ROOT, "analysis",
                                       "mlb_f5_predictions.archived.json")))["entries"]
    ids = team_abbrs()
    cache = {}
    if os.path.exists(CACHE):
        try:
            cache = json.load(open(CACHE))
        except Exception:
            pass

    # point-in-time standings, one call per distinct prior day
    stand = {}
    for d in sorted({e["date"] for e in arch}):
        prior = (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            stand[d] = standings_asof(prior, ids)
        except Exception as ex:
            print(f"  ! standings {prior}: {ex}")

    rows, skipped = [], {"noticker": 0, "nomarket": 0, "noquote": 0, "nostand": 0, "noresult": 0}
    for i, e in enumerate(arch):
        tick = ((e.get("kalshi") or {}).get("ticker") or "")
        res = e.get("result") or {}
        if not tick.startswith("KXMLBF5-"):
            skipped["noticker"] += 1; continue
        st = stand.get(e["date"]) or {}
        h, a = st.get(e["home"]), st.get(e["away"])
        if not h or not a:
            skipped["nostand"] += 1; continue
        ev = "KXMLBGAME-" + tick.split("-", 1)[1]

        if ev in cache:
            mk = cache[ev]
        else:
            try:
                ms = get(f"{KAL}/markets?event_ticker={ev}").get("markets", [])
            except Exception:
                ms = []
            mk = {m["ticker"]: m.get("result") for m in ms}
            cache[ev] = mk
            if i % 20 == 0:
                json.dump(cache, open(CACHE, "w"))
        if len(mk) != 2:
            skipped["nomarket"] += 1; continue

        # the market whose suffix is the HOME abbreviation pays on a home win
        home_tk = next((t for t in mk if t.rsplit("-", 1)[-1].upper() == e["home"].upper()), None)
        away_tk = next((t for t in mk if t != home_tk), None)
        if not home_tk:
            skipped["nomarket"] += 1; continue

        tip = int(datetime.strptime(e["gameDate"], "%Y-%m-%dT%H:%M:%SZ")
                  .replace(tzinfo=timezone.utc).timestamp())
        try:
            logged = int(datetime.fromisoformat(e["logged_at"]).timestamp())
        except Exception:
            logged = tip - 4 * 3600

        ck = f"c:{home_tk}"
        if ck in cache:
            bh, ba = cache[ck]["h"], cache[ck]["a"]
        else:
            bh = candles(home_tk, logged - 3600, tip + 600)
            ba = candles(away_tk, logged - 3600, tip + 600)
            cache[ck] = {"h": bh, "a": ba}

        p0h, p1h = price_at(bh, logged, False), price_at(bh, tip, True)
        p0a, p1a = price_at(ba, logged, False), price_at(ba, tip, True)
        if None in (p0h, p1h, p0a, p1a):
            skipped["noquote"] += 1; continue
        # de-vig each end so open and close are measured on the same basis
        s0, s1 = p0h + p0a, p1h + p1a
        if s0 <= 0 or s1 <= 0:
            skipped["noquote"] += 1; continue
        open_home, close_home = p0h / s0, p1h / s1

        pm = model_prob(h, a, (e.get("starters") or {}).get("homeKBB"),
                        (e.get("starters") or {}).get("awayKBB"))
        if pm is None:
            skipped["nostand"] += 1; continue
        side_home = pm >= 0.5
        p_open = open_home if side_home else 1 - open_home
        p_close = close_home if side_home else 1 - close_home

        # outcome, straight from the finalized market
        won = None
        r = mk.get(home_tk)
        if r in ("yes", "no"):
            won = (r == "yes") if side_home else (r == "no")
        if won is None:
            skipped["noresult"] += 1

        # control: the side the market itself favoured at log time
        fav_home = open_home >= 0.5
        f_open = open_home if fav_home else 1 - open_home
        f_close = close_home if fav_home else 1 - close_home
        rows.append({"date": e["date"], "game": f"{e['away']}@{e['home']}",
                     "model": pm, "sideHome": side_home,
                     "open": p_open, "close": p_close, "clv": p_close - p_open,
                     "clvFav": f_close - f_open, "won": won})

    json.dump(cache, open(CACHE, "w"))

    if not rows:
        print("no usable rows;", skipped); return
    clv = [r["clv"] for r in rows]
    # CONTROL: back whatever side the market already favoured when we logged.
    # It shares the model's directional bias without containing any model, so
    # the paired difference isolates what the model actually contributes.
    ctrl = [r["clvFav"] for r in rows]
    diff = [a - b for a, b in zip(clv, ctrl)]
    n = len(clv)
    mean = sum(clv) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in clv) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    movers = [x for x in clv if abs(x) > 1e-9]
    beat = sum(1 for x in movers if x > 0)
    def stats(v):
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        e = sd / math.sqrt(len(v)) if v else 0.0
        return m, e, (m / e if e else 0.0)

    c_mean, c_se, c_z = stats(ctrl)
    d_mean, d_se, d_z = stats(diff)
    agree = sum(1 for r in rows if r["sideHome"] == (r["open"] >= 0.5 if r["sideHome"] else r["open"] < 0.5))
    same = sum(1 for r in rows if abs(r["clv"] - r["clvFav"]) < 1e-12)
    graded = [r for r in rows if r["won"] is not None]
    acc = sum(1 for r in graded if r["won"]) / len(graded) if graded else None

    print("\n" + "=" * 66)
    print("MLB FULL-GAME MODEL — TRUE CLV FROM KALSHI CANDLESTICKS")
    print("=" * 66)
    print(f"  games              {n}   (skipped {skipped})")
    print(f"  mean CLV           {mean*100:+.3f} pp")
    print(f"  std error          {se*100:.3f} pp     -> 95% CI "
          f"[{(mean-1.96*se)*100:+.3f}, {(mean+1.96*se)*100:+.3f}] pp")
    print(f"  z vs zero          {mean/se if se else 0:+.2f}")
    print(f"  moved our way      {beat}/{len(movers)} ({beat/len(movers)*100:.0f}%)" if movers else "  no movement")
    if acc is not None:
        print(f"  straight-up        {sum(1 for r in graded if r['won'])}/{len(graded)} ({acc*100:.1f}%)")
    print(f"\n  CONTROL (back the market's own favourite at log time)")
    print(f"  mean CLV           {c_mean*100:+.3f} pp   se {c_se*100:.3f}   z {c_z:+.2f}")
    print(f"  identical pick     {same}/{n} games — the model and the market agree")
    print(f"\n  INCREMENTAL (paired: model minus control, same games)")
    print(f"  mean               {d_mean*100:+.4f} pp")
    print(f"  std error          {d_se*100:.4f} pp     -> 95% CI "
          f"[{(d_mean-1.96*d_se)*100:+.3f}, {(d_mean+1.96*d_se)*100:+.3f}] pp")
    print(f"  z                  {d_z:+.2f}")

    print("\n  VERDICT")
    lo, hi = d_mean - 1.96 * d_se, d_mean + 1.96 * d_se
    if lo > 0:
        print("  >>> The model beats the control at 2 sigma. It anticipates market")
        print("      moves beyond simply agreeing with the favourite. REOPEN.")
    elif hi < 0:
        print("  >>> The model is WORSE than backing the market's own favourite.")
        print("      Actively anti-informative. Close, and do not bet it.")
    else:
        print("  >>> The model adds nothing over backing the market's favourite.")
        print("      Raw CLV looks positive only because prices drift toward the")
        print("      favourite and the model picks the favourite. Confirms the")
        print("      log-loss finding: no edge over the price. FILE CLOSED.")
    out = os.path.join(ROOT, "..", "mlb_clv_backfill.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"n={n} mean={mean*100:+.3f}pp se={se*100:.3f}pp z={mean/se if se else 0:+.2f}\n")
        for r in sorted(rows, key=lambda r: r["clv"]):
            f.write(f"{r['date']} {r['game']:<10} model {r['model']:.3f} "
                    f"open {r['open']:.3f} close {r['close']:.3f} clv {r['clv']*100:+.2f}pp\n")
    print(f"\n  per-game detail -> {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
