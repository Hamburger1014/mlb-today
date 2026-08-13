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
