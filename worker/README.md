# mlb-kalshi Worker

Cloudflare Worker that proxies The Odds API so the API key stays server-side and
never ships in `index.html`. The name is historical — it began as a Kalshi proxy,
and Kalshi was removed on 2026-08-11.

Deployed at `https://mlb-kalshi.gabrielhiginio2005.workers.dev`.

This directory is the **source of record**. The live copy was previously edited
in `C:\Users\gabri\mlb-kalshi-worker` with no version control at all, so a disk
loss would have taken the only copy of the deployed infrastructure with it.
Edit here, copy to that directory, then deploy.

## Endpoints

    /?odds=<sport>&markets=<market>   The Odds API passthrough
    /?path=/events|/markets|...       Kalshi public data (dormant)

`sport` is whitelisted to mlb, wnba, nfl, ncaaf, nba. `markets` is whitelisted to
h2h, spreads, totals and defaults to h2h. Both whitelists exist to protect the
free tier: cost is (markets x regions) credits per call against 500/month, so an
unbounded passthrough lets a single request ask for every market at once.

Responses are cached 300s server-side, keyed by sport AND market so spreads does
not evict h2h.

## Secrets (never in this repo)

    npx wrangler secret put ODDS_API_KEY

## Deploy

    npx wrangler deploy

## Gotcha that cost real time

An unrecognised `markets` value used to be ignored, and the Worker served h2h
with a 200. That parses cleanly and yields no points, so a caller cannot tell the
difference from a status code. `scripts/football_log.py` now checks that spreads
were actually present rather than trusting the response. The Worker itself now
400s an unlisted market instead of silently substituting.

## Cron: it also runs the logger

The prediction logger lives in GitHub Actions and GitHub does not reliably run
scheduled workflows. Measured 2026-08-14: the workflow asks for every 10 minutes
and fired **zero times in 52 minutes**; across 2026-08-05..10 it fired zero times
in six days while the WNBA played on four of them. Nothing ever failed — runs
simply never started, so every step stayed green while the log stopped growing.

Cloudflare cron triggers do fire. This Worker's `scheduled()` handler does one
thing: ask GitHub to run the workflow via `workflow_dispatch`, which is an
explicit API call rather than best-effort scheduling.

    [triggers]
    crons = ["*/10 14-23 * * *", "*/10 0-5 * * *"]

The GitHub schedule is deliberately left in place as a second path. The logger
skips games it has already recorded, so a duplicate run is a no-op.

**It does NOT port the logger.** The model exists in `index.html`,
`wnba_today.html` and `scripts/wnba_log.py`, and the repo carries three parity
verifiers because those copies drift. A fourth copy in JavaScript would be a new
instance of the exact bug those verifiers exist to catch. Only delivery moved,
because delivery is what was measured broken.

### The token (set once, by hand)

Needs a fine-grained personal access token scoped to this one repository with
**Actions: Read and write**, nothing else. Create it at
GitHub → Settings → Developer settings → Personal access tokens → Fine-grained.

    npx wrangler secret put GH_DISPATCH_TOKEN

Until it is set the handler no-ops and says so rather than throwing every ten
minutes:

    GET /?cron=run  ->  503 {"ok":false,"why":"GH_DISPATCH_TOKEN not set"}

Once set, that same URL returns `{"ok":true,...}` and is the way to test the
wiring without waiting for a scheduled firing.
