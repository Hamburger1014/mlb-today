# analysis/

Offline research pipeline. Nothing here runs in production — the site never
fetches from this directory, and the scheduled workflow never executes it.

## Canonical vs. copies — read this before editing

The production loggers and their parity gates live in **`scripts/`**, not here:

| Production (edit these) | Do not recreate here |
|---|---|
| `scripts/wnba_log.py` | WNBA prediction logger + grader |
| `scripts/mlb_f5_log.py` | MLB F5 logger + grader |
| `scripts/verify_wnba_parity.py` | gates the build on page↔logger drift |
| `scripts/verify_f5_parity.py` | same, for the F5 model |

The working folder this was lifted from (`Coinbase/wnba_model/`) also contained
copies of those four files, and three of them had already drifted out of date —
including a `wnba_log.py` from before the quarter-price fix. They were
deliberately left out so there is exactly one copy of each in the repo.

## What is here

**Scrapers**
- `scrape_wnba.py` → `wnba_games.json` — ESPN scoreboards 2022-2026 with quarter linescores.
- `scrape_summaries.py` → `wnba_boxscores.json` + `wnba_odds.json` — one pass over
  ESPN summaries pulling per-player boxscores *and* closing odds. Supersedes
  `scrape_boxscores.py`, which made the same call and discarded the odds.
  Resumable; ~1,377 requests, ~12 min.
- `scrape_teamstats.py` — team season aggregates.

**Model fitting**
- `train_wnba.py` / `train_wnba2.py` / `train_wnba3.py` — v1→v3 walk-forward fits.
  v2 is what shipped (`wnba_fit_v2.json`); v3's experiments failed their kill criteria.
- `train_live.py` → `wnba_fit_live.json` — in-game re-prediction at quarter boundaries.
- `train_pace.py` — pace/efficiency model. Failed both kill criteria; kept as a record.

**Measurement** (the part that decided things)
- `wnba_backfill.py` — replays the WNBA model point-in-time against DraftKings
  closing lines and runs the two-logit market test. Coefficients are fit on
  pre-2026 seasons and frozen, because `wnba_fit_v2.json` was fit including 2026.
- `audit_f5.py` — found the Poisson overdispersion that inflated F5 tie bets.
- `calibrate_f5.py` — fit the negative-binomial replacement.
- `check_*.py` — one-off validation scripts.

The companion tests live in `scripts/` because they are shared:
`market_edge_test.py` (de-vig + log-opinion-pool fit) and `mlb_fullgame_test.py`.

## Data caches

`wnba_games.json`, `wnba_boxscores.json`, `wnba_odds.json` (~1 MB total) are
committed so `wnba_backfill.py` runs immediately instead of re-scraping for 12
minutes. They are regenerable — delete and re-run the scrapers.

Note: `wnba_odds.json` is **2026 only**. ESPN's `pickcenter` block does not
survive past the current season, so 2022-25 return no odds at all. That caps any
ESPN-based odds backfill at the season in progress.

## Results

Written up in `docs/` — `market_edge_results.txt`, `wnba_backfill_results.txt`,
`mlb_fullgame_results.txt`.

The strategy write-up those results feed into is kept **out of this repo on
purpose** — this repo is public because GitHub Pages serves the site from it,
and the playbook is not something to publish. It lives locally at
`Coinbase/market_edge_playbook.md` and is git-ignored here so it cannot be
re-added by accident.
