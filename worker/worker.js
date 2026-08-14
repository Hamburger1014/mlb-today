/**
 * Cloudflare Worker — CORS proxy for the MLB Today app.
 *  • Kalshi PUBLIC market data: /?path=/events|/markets|/series|/exchange
 *  • The Odds API (sharp sportsbook lines): /?odds=mlb   (key from Worker SECRET)
 *
 * Caching + stale-on-error: cache responses (Cache API) keyed by a normalized
 * path so we hit Kalshi at most ~once per FRESH_S per request, and on a 429/5xx
 * serve the last good copy (up to STALE_S old) so the app keeps showing odds.
 *
 * Kalshi AUTH (added 2026-07-08): if the KALSHI_KEY_ID + KALSHI_PRIVATE_KEY
 * secrets are set, Kalshi requests are RSA-PSS signed. Authenticated requests use
 * a per-key rate limit instead of the anonymous shared-IP throttle that 429s this
 * Worker's Cloudflare egress IP. Secrets live server-side ONLY (never in the page).
 *   Set them once (from the mlb-kalshi-worker folder):
 *     npx wrangler secret put KALSHI_KEY_ID          → paste the API Key ID
 *     Get-Content key.pem -Raw | npx wrangler secret put KALSHI_PRIVATE_KEY
 *   No redeploy needed after — secrets are read live. If unset, falls back to
 *   unauthenticated public access (current behaviour).
 *
 * Security: GET-only; the Kalshi key can only READ market data (it never touches
 * the account for these public endpoints); Odds side issues one read-only request.
 */
const KALSHI = 'https://api.elections.kalshi.com/trade-api/v2';
// The Odds API. Sport is chosen per-request now rather than hardcoded to MLB,
// so the same Worker can serve every league the site covers. Whitelisted so a
// caller cannot point the key at arbitrary upstream paths.
const ODDS_BASE = 'https://api.the-odds-api.com/v4/sports';
const ODDS_SPORTS = {
  mlb:   'baseball_mlb',
  wnba:  'basketball_wnba',
  nfl:   'americanfootball_nfl',
  ncaaf: 'americanfootball_ncaaf',
  nba:   'basketball_nba',
};
const UA = 'mlb-today/1.0 (+https://github.com/Hamburger1014/mlb-today)';
const FRESH_S = 30;    // serve cached without re-fetching upstream for this long
const STALE_S = 300;   // keep a copy this long to serve on an upstream error

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

// ── Kalshi RSA-PSS request signing (only used if the secrets are configured) ──
let _keyCache = { pem: null, key: null };
function _pemToBytes(pem) {
  const b64 = pem.replace(/-----[^-]+-----/g, '').replace(/\s+/g, '');
  const bin = atob(b64);
  const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}
// Wrap a PKCS#1 (-----BEGIN RSA PRIVATE KEY-----) body in a PKCS#8 envelope so
// Web Crypto importKey('pkcs8') accepts it. PKCS#8 keys are used as-is.
function _pkcs1ToDer(pkcs1) {
  const encLen = (n) => (n < 128 ? [n] : n < 256 ? [0x81, n] : [0x82, (n >> 8) & 0xff, n & 0xff]);
  const seq = (d) => [0x30, ...encLen(d.length), ...d];
  const oct = (d) => [0x04, ...encLen(d.length), ...d];
  const algId = [0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01, 0x05, 0x00];
  const inner = seq([0x02, 0x01, 0x00, ...algId, ...oct([...pkcs1])]);
  return new Uint8Array(inner).buffer;
}
async function _importKey(pem) {
  if (_keyCache.key && _keyCache.pem === pem) return _keyCache.key;
  const bytes = _pemToBytes(pem.trim());
  const buf = pem.includes('BEGIN RSA PRIVATE KEY') ? _pkcs1ToDer(bytes) : bytes.buffer;
  const key = await crypto.subtle.importKey('pkcs8', buf, { name: 'RSA-PSS', hash: 'SHA-256' }, false, ['sign']);
  _keyCache = { pem, key };
  return key;
}
// Returns Kalshi auth headers, or null if no key configured / signing failed
// (so a bad key degrades to unauthenticated rather than breaking the proxy).
async function kalshiAuthHeaders(env, method, pathNoQuery) {
  const kid = env && env.KALSHI_KEY_ID;
  const pem = env && env.KALSHI_PRIVATE_KEY;
  if (!kid || !pem) return null;
  try {
    const ts = Date.now().toString();               // Kalshi wants milliseconds
    const key = await _importKey(pem);
    const sig = await crypto.subtle.sign(
      { name: 'RSA-PSS', saltLength: 32 },
      key,
      new TextEncoder().encode(ts + method + pathNoQuery)   // e.g. "1712.." + "GET" + "/trade-api/v2/markets"
    );
    const b64 = btoa(String.fromCharCode(...new Uint8Array(sig)));
    return { 'KALSHI-ACCESS-KEY': kid, 'KALSHI-ACCESS-TIMESTAMP': ts, 'KALSHI-ACCESS-SIGNATURE': b64 };
  } catch (e) {
    return null;
  }
}

function withCors(resp, state) {
  const h = new Headers(resp.headers);
  for (const k in CORS) h.set(k, CORS[k]);
  h.set('x-proxy-cache', state);
  return new Response(resp.body, { status: resp.status, headers: h });
}

// Cache-aware proxy: serves fresh from cache, refetches when stale, and falls
// back to the last good copy when the upstream errors. extraHeaders (e.g. Kalshi
// auth) are applied to the upstream fetch only, not to the cache key.
async function cachedProxy(keyStr, upstreamUrl, ctx, freshS, extraHeaders) {
  const cache = caches.default;
  const cacheKey = new Request('https://cache.local/' + encodeURIComponent(keyStr), { method: 'GET' });
  const hit = await cache.match(cacheKey);
  const ageOf = (r) => { const t = r && r.headers.get('x-fetched-at'); return t ? (Date.now() - (+t)) / 1000 : Infinity; };

  if (hit && ageOf(hit) < freshS) return withCors(hit, 'fresh');

  let up = null;
  try { up = await fetch(upstreamUrl, { headers: { Accept: 'application/json', 'User-Agent': UA, ...(extraHeaders || {}) } }); }
  catch (e) { up = null; }

  if (up && up.ok) {
    const body = await up.text();
    const resp = new Response(body, {
      status: 200,
      headers: { 'Content-Type': 'application/json', 'x-fetched-at': String(Date.now()), 'Cache-Control': 'public, max-age=' + STALE_S },
    });
    ctx.waitUntil(cache.put(cacheKey, resp.clone()));
    return withCors(resp, 'miss');
  }

  if (hit) return withCors(hit, 'stale');
  const status = up ? up.status : 502;
  const body = up ? await up.text() : '{"error":"upstream unreachable"}';
  return withCors(new Response(body, { status, headers: { 'Content-Type': 'application/json' } }), 'error');
}

// ── SCHEDULED: kick the GitHub Actions logger ────────────────────────────────
//
// WHY THIS EXISTS. The logging job is scheduled on GitHub cron and GitHub does
// not run it. Measured 2026-08-14: the workflow asks for every 10 minutes and
// fired ZERO times in 52 minutes; over 2026-08-05..10 it fired zero times in six
// days while the WNBA played on four of them. Nothing failed — the runs simply
// never started, so every step was green and the log just stopped growing.
//
// Cloudflare cron triggers actually fire. This handler does nothing except ask
// GitHub to run the workflow, via workflow_dispatch, which is an explicit API
// call rather than best-effort scheduling.
//
// WHAT THIS DELIBERATELY DOES NOT DO: port the logger. The model lives in
// index.html, wnba_today.html and scripts/wnba_log.py, and this repo carries
// three parity verifiers because those copies drift. A fourth copy in JavaScript
// would be a new source of exactly the bug the verifiers exist to catch. The
// problem measured was delivery, not logic, so only delivery moved.
const GH_OWNER = 'Hamburger1014';
const GH_REPO  = 'mlb-today';
const GH_FILE  = 'kalshi-snapshot.yml';

async function dispatchWorkflow(env) {
  // No token means this is not configured yet. Say so and stop — a cron that
  // throws every ten minutes is noise, not a signal.
  const tok = env && env.GH_DISPATCH_TOKEN;
  if (!tok) return { ok: false, why: 'GH_DISPATCH_TOKEN not set' };
  const url = `https://api.github.com/repos/${GH_OWNER}/${GH_REPO}/actions/workflows/${GH_FILE}/dispatches`;
  const r = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${tok}`,
      'Accept': 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      // GitHub rejects API calls with no User-Agent.
      'User-Agent': 'mlb-today-cron',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ ref: 'main' }),
  });
  // 204 No Content is success for this endpoint.
  return { ok: r.status === 204, status: r.status, why: r.status === 204 ? 'dispatched' : await r.text() };
}

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(dispatchWorkflow(env).then(res => {
      console.log('cron dispatch', JSON.stringify(res));
    }));
  },

  async fetch(request, env, ctx) {
    if (request.method === 'OPTIONS') return new Response(null, { headers: CORS });
    if (request.method !== 'GET') return new Response('GET only', { status: 405, headers: CORS });

    const url = new URL(request.url);

    // Manual trigger for the same dispatch the cron performs, so the wiring can
    // be verified without waiting up to ten minutes for a scheduled firing.
    if (url.searchParams.get('cron') === 'run') {
      const res = await dispatchWorkflow(env);
      return new Response(JSON.stringify(res), {
        status: res.ok ? 200 : 503,
        headers: { ...CORS, 'content-type': 'application/json' },
      });
    }

    // ── Odds API proxy (key stays server-side) ──
    const oddsSport = url.searchParams.get('odds');
    if (oddsSport) {
      const sportKey = ODDS_SPORTS[oddsSport];
      if (!sportKey) return new Response('unknown sport', { status: 400, headers: CORS });
      const key = env && env.ODDS_API_KEY;
      if (!key) return new Response('odds key not configured', { status: 503, headers: CORS });
      // Cost is (markets x regions) credits per call, so ONE market in ONE
      // region is 1 credit whichever market it is. Whitelisted rather than
      // passed through: an unbounded `markets` param lets a caller ask for
      // every market at once and drain the free tier (500/month) in a few
      // requests.
      const mkt = url.searchParams.get('markets') || 'h2h';
      if (!/^(h2h|spreads|totals)$/.test(mkt))
        return new Response('unknown market', { status: 400, headers: CORS });
      // Cached 300s server-side, keyed by sport AND market, so a page refresh
      // does not spend a credit and spreads does not evict h2h.
      const u = `${ODDS_BASE}/${sportKey}/odds/?apiKey=${key}`
              + `&regions=us&markets=${mkt}&oddsFormat=decimal`;
      return cachedProxy('odds:' + oddsSport + ':' + mkt, u, ctx, 300);
    }

    // ── Kalshi public market-data proxy (signed if secrets present) ──
    const path = url.searchParams.get('path') || '';
    if (!/^\/(events|markets|series|exchange)/.test(path))
      return new Response('blocked path', { status: 403, headers: CORS });
    const upstreamUrl = KALSHI + path;
    // Normalize the cache key: collapse the per-second min_close_ts so near-identical polls share one entry.
    const keyPath = path.replace(/([?&])min_close_ts=\d+/, '$1min_close_ts=BUCKET');
    const auth = await kalshiAuthHeaders(env, 'GET', new URL(upstreamUrl).pathname);
    return cachedProxy('kalshi:' + keyPath, upstreamUrl, ctx, FRESH_S, auth);
  },
};
