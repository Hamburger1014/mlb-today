# WNBA: genuine skill at picking games — which is NOT permission to bet more

Measured 2026-08-26. Reproduce with `python scripts/wnba_skill_test.py`.

Fourth and last of the skill tests, and the one that matters most: WNBA is the
only sport carrying betting weight (`WNBASP` w=0.40, `WNBAML` w=0.15).

## Read this first

**Beating a constant is not beating a price.** This test says the model picks
games well. [`market_test_results`](../analysis/README.md) says it does not beat
DraftKings: β_model +0.35 (p=0.29) *with* a lookahead injury feature, +0.07
(p=0.87) without. Both are true and they are not in tension — the market is also
good at picking games.

The card weights are set by the **market** test, not this one. Nothing here
justifies raising them.

## Two sources of lookahead had to be removed first

Both would have flattered the model, and the second is subtle:

1. **The shipped coefficients already saw the test games.** `wnba_log.FIT` is
   `analysis/wnba_fit_v2.json`, trained on 1,246 games **including 2026**. So
   the margin coefficients are refit **walk-forward** here — least squares on
   prior games only, refit once per season.
2. **`priors2025` seeds every team with 2025 ratings regardless of date.** Live
   that is correct: it is last season's prior, worth `priorW0=4` games and
   halved after 25. In a 2022–2024 replay it is the future. Ratings are seeded
   at league average instead, and the test burns in two full seasons.

## Result

Test seasons 2024+, ratings built only from games strictly before each slate.

|                              |    acc |  Brier | logloss |    res |    rel |
|------------------------------|-------:|-------:|--------:|-------:|-------:|
| always home (prior rate)      | 54.86% | 0.2478 |  0.6887 | 0.0000 | 0.0001 |
| home-edge only                | 54.86% | 0.2478 |  0.6888 | 0.0001 | 0.0003 |
| **model, refit walk-forward** | **65.31%** | 0.2217 | **0.6360** | 0.0176 | 0.0016 |
| model, shipped FIT (in-sample) | 64.70% | 0.2217 | 0.6358 | 0.0176 | 0.0012 |

**Model vs always-home:** gain **+0.05271/game**, 95% CI **[+0.02714, +0.07810]**,
P(better) **1.000**. Accuracy **+10.46pp**, ~**5.96 SE**.

Stable across every window tried, which is the strongest thing in this file:

| test window | games | model acc | base | gain/game | 95% CI |
|---|---:|---:|---:|---:|---|
| 2024+ | 813 | 65.31% | 54.86% | +0.05271 | [+0.02714, +0.07810] |
| 2025+ | 550 | 65.27% | 55.09% | +0.04992 | [+0.02020, +0.07802] |
| 2026 only | 239 | 65.27% | 53.56% | +0.05098 | [+0.00440, +0.09564] |

65.3% three times on non-nested-in-effect samples is not a fluke.

**The shipped coefficients are not overfit.** An honest walk-forward refit lands
in the same place (65.31% vs 64.70%, log-loss 0.6360 vs 0.6358). That is the
good news hiding in the lookahead problem: v2 having seen 2026 inflated its
*reported* metrics, but the coefficients themselves are sound.

As in **both** football models, **home-edge alone does not beat the base rate**
(−0.00004/game, CI spans zero). Across all three sports with skill, the signal
lives entirely in the team ratings, never in knowing who is at home.

## What this test cannot see

There is no historical point-in-time injury feed, so `injMiss` is dropped
(miss = 0 both sides). `market_test_results` records that the injury feature is
where most of the WNBA signal appeared — **and that its backtest version was
lookahead**, reading the game's own boxscore to learn who actually played.

So this is a **floor** on the live model, not an estimate of it. The resolution
here (0.0176) sits below the 0.0300 recorded from the live path, and the
difference is roughly what the injury feature is worth.

## All four sports, same protocol

```
CFB    +0.12329 logloss/game   72.38% vs 63.25%   11.04 SE   n=3660
WNBA   +0.05271               65.31% vs 54.86%    5.96 SE   n= 813
NFL    +0.04565               63.80% vs 53.91%    8.13 SE   n=1688
MLB    +0.00296               54.42% vs 52.30%    1.57 SE   n=1369   <- nothing
```

Three of the four genuinely separate games. Only MLB is flat. But the ranking
that governs betting is still the market test, where only WNBA has any positive
signal at all — and that one is unproven, not proven.
