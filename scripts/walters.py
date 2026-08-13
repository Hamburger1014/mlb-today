#!/usr/bin/env python3
"""Billy Walters' one transferable edge: the value of the NUMBER.

Walters calls getting the best number "the single most important factor" in
betting. Most of what actually made him rich is NOT available here and it is
worth being blunt about which parts:

  - Runner networks placing simultaneously at dozens of books, so the line did
    not move before the money was down. Gabriel bets one book.
  - Deliberately betting the wrong side to move a line, then taking the number
    he wanted on the other side. Needs capital and many outs.
  - Injury and weather information ahead of the market.
  - Line shopping across Circa/Pinnacle/MGM — his own stated #1 factor, and the
    one that most directly does not survive a single-book account.

What DOES transfer is Michael Kent's actual loop, which is arithmetic:
form a number, compare it to the posted number, bet only when the gap is big
enough to survive the vig. The moneyline test already showed there is no gap
in PRICE (0 of 76 games cleared the bar). Football pays in the LINE instead,
because scoring is lumpy: 15.39% of NFL games are decided by exactly 3. Holding
+3 while the field is at +2.5 is worth an order of magnitude more than any
price discrepancy this board has ever found.

This module converts a line difference into a probability edge on the SAME
scale the rest of the site uses, so a spread candidate can be judged against
the same bar as everything else rather than by eyeball.

Frequencies: boydsbets.com NFL key-number study (n=8,700 games).
"""
import math

# Share of games decided by each EXACT absolute margin. Only the numbers with
# real mass are listed; everything else falls out of the normal below.
# 3 is more than one game in seven, which is why it dominates every half-point
# decision in football.
NFL_KEY_MASS = {3: 0.1539, 7: 0.0874, 6: 0.0584, 10: 0.0581, 4: 0.0518, 14: 0.0492}

# SD of an NFL result around the closing spread. Stable across eras near 13.2;
# college is wider because the talent gap between teams is wider.
NFL_SIGMA = 13.2
CFB_SIGMA = 16.0

# Walters values home field at ~2.5 points across 1974-2022 and under 1 point in
# recent seasons — NOT the conventional 3. Kept here so any future power rating
# starts from his number rather than the folklore one.
HOME_FIELD = 1.0
QB_OUT     = 7.0   # "a quarterback is worth about a touchdown"
KEY_OUT    = 2.75  # top non-QB, 2.5-3


_BUMP_CACHE = {}


def _bumps(sigma):
    """How much the real world exceeds a normal at each key number.

    A pure normal badly understates 3 and 7 — football does not score in
    continuous units. Solved by fixed point rather than one division: adding mass
    at the key numbers changes the normalising total, which drags every key mass
    back below target. Iterating to convergence lands them on the empirical
    figures instead of ~13% of a nominal 15.4%.
    """
    if sigma in _BUMP_CACHE:
        return _BUMP_CACHE[sigma]
    b = {k: 1.0 for k in NFL_KEY_MASS}
    for _ in range(200):
        pmf = _raw_pmf(0.0, sigma, b)
        tot = sum(pmf.values())
        worst = 0.0
        for k, emp in NFL_KEY_MASS.items():
            got = (pmf[k] + pmf[-k]) / tot
            worst = max(worst, abs(got - emp))
            b[k] *= emp / got
        if worst < 1e-9:
            break
    _BUMP_CACHE[sigma] = b
    return b


def _raw_pmf(mu, sigma, b, lo=-70, hi=70):
    out = {}
    for m in range(lo, hi + 1):
        d = math.exp(-0.5 * ((m - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        out[m] = d * b.get(abs(m), 1.0)
    return out


def margin_pmf(mu, sigma=NFL_SIGMA, lo=-70, hi=70):
    """P(home margin == m) for every integer m, centred on mu.

    Normal shape, multiplied by the key-number bumps, then renormalised. This is
    an approximation in one specific way worth stating: the bumps are calibrated
    at a pick'em and applied at every mu, so the spike at 3 is slightly overstated
    for games with a big spread, where a 3-point margin is far from the centre.
    It is right where it matters — comparing two lines that straddle 3.
    """
    pmf = _raw_pmf(mu, sigma, _bumps(sigma), lo, hi)
    tot = sum(pmf.values())
    return {m: p / tot for m, p in pmf.items()}


def cover_prob(pmf, line, side="home"):
    """(win, push, lose) for a spread bet. `line` is the HOME spread: -3 means
    home laying 3. Home covers when margin > -line, pushes when margin == -line."""
    need = -line
    win = sum(p for m, p in pmf.items() if m > need)
    push = sum(p for m, p in pmf.items() if abs(m - need) < 1e-9)
    lose = sum(p for m, p in pmf.items() if m < need)
    if side == "home":
        return win, push, lose
    return lose, push, win


def american_to_prob(a):
    """Vigged implied probability of an American price — what you actually pay."""
    if a is None:
        return None
    return (-a) / ((-a) + 100.0) if a < 0 else 100.0 / (a + 100.0)


def devig_two(a1, a2):
    """Power de-vig of a two-way market, matching the rest of the site."""
    p1, p2 = american_to_prob(a1), american_to_prob(a2)
    if p1 is None or p2 is None:
        return None
    lo, hi = 0.5, 3.0
    for _ in range(80):
        k = (lo + hi) / 2
        if p1 ** k + p2 ** k > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    return p1 ** k, p2 ** k


def calibrate_mu(field_line, field_home_odds, field_away_odds, sigma=NFL_SIGMA):
    """Centre the distribution so it reproduces the FIELD's own de-vigged price.

    The field is treated as the truth, exactly as everywhere else on this site:
    its line plus its price is the market's honest opinion, and the only question
    is whether DraftKings is offering something better than that opinion. Solving
    for mu rather than assuming mu == -line matters when the field prices a
    spread off-centre (a -3 at -120/+100 is not a 50/50 bet).
    """
    dv = devig_two(field_home_odds, field_away_odds)
    target = dv[0] if dv else 0.5
    lo, hi = -60.0, 60.0
    for _ in range(60):
        mid = (lo + hi) / 2
        w, pu, _ = cover_prob(margin_pmf(mid, sigma), field_line)
        # Push mass belongs to neither side; split it so the solve targets the
        # same quantity the de-vigged two-way price represents.
        if w + pu / 2 < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def number_edge(dk_line, dk_home_odds, dk_away_odds,
                field_line, field_home_odds, field_away_odds, sigma=NFL_SIGMA):
    """Edge in probability points for each side of DraftKings' number.

    Returns {'home': pp, 'away': pp}, positive meaning DraftKings is offering
    better than the field. Fair value comes from the field's line and price; the
    cost is DraftKings' VIGGED price, because on a sportsbook the vig IS the
    cost — there is no separate fee to add.
    """
    mu = calibrate_mu(field_line, field_home_odds, field_away_odds, sigma)
    pmf = margin_pmf(mu, sigma)
    out = {}
    for side, price in (("home", dk_home_odds), ("away", dk_away_odds)):
        w, pu, _ = cover_prob(pmf, dk_line, side)
        paid = american_to_prob(price)
        if paid is None:
            out[side] = None
            continue
        # A push refunds the stake, so it is neither won nor lost — judge the bet
        # on the probability of winning GIVEN it resolves. The American price is
        # already that same conditional break-even number (-110 means 52.38% of
        # decided bets), so it must NOT be rescaled by the push mass too; doing
        # that double-counts and reported a flat -6.24pp on a bet priced exactly
        # at the field's own number, where the honest answer is half the vig.
        live = 1.0 - pu
        out[side] = (w / live - paid) if live > 1e-9 else 0.0
    return out


def walters_units(edge_pp, bankroll_cap=0.03):
    """His sizing rule: half-unit steps from 0.5 to 3 units, never past 3% on one
    game. Deliberately blunter than Kelly — Walters spread across many games and
    refused to let one result matter, which is the same reason this site caps
    correlated exposure."""
    if edge_pp is None or edge_pp < 0.02:
        return 0.0
    u = 0.5 + 2.5 * min(1.0, (edge_pp - 0.02) / 0.06)
    return min(round(u * 2) / 2, bankroll_cap * 100)
