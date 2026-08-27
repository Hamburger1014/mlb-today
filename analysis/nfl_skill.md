# The NFL model has real skill — and its margin scale was 30% too wide

Measured 2026-08-26. Reproduce with `python scripts/nfl_skill_test.py`.

Companion to [`mlb_no_skill.md`](mlb_no_skill.md), same protocol, opposite result.

## Does it beat a constant?

Walk-forward by date over 2020–2025: for each slate, fit ratings on every game
strictly before it, predict that slate. **1,688 out-of-sample games.**

|                              |    acc |  Brier | logloss |    res |    rel |
|------------------------------|-------:|-------:|--------:|-------:|-------:|
| always home (prior rate)      | 53.91% | 0.2488 |  0.6907 | 0.0002 | 0.0003 |
| home-field only               | 53.67% | 0.2484 |  0.6900 | 0.0001 | 0.0004 |
| **model, scale fitted OOS**   | 63.80% | 0.2267 |  0.6450 | 0.0173 | 0.0031 |
| model, scale fitted in-sample | 63.80% | 0.2258 |  0.6434 | 0.0161 | 0.0020 |
| model, OLD scale 16.14        | 63.80% | 0.2297 |  0.6518 | 0.0153 | 0.0052 |

**Model vs always-home:** log-loss gain **+0.04565/game**, 95% CI
**[+0.03514, +0.05620]**, P(better) **1.000**. Accuracy **+9.89pp**, ~**8.13 SE**.
Mean absolute margin error 10.26 pts.

Yes, decisively. This is not the MLB situation. Resolution 0.0173 against
uncertainty ≈0.249 — an order of magnitude above MLB's 0.0019.

Note "home-field only" is *not* better than the base rate (+0.00067/game, CI
spans zero). The skill is in the team ratings, not in knowing who is at home.

## The bug this surfaced

`marginScale` shipped at **16.14**. The converged value is **12.40**, and the
difference is significant: **+0.00674 log-loss/game, 95% CI [+0.00341, +0.01002]**,
P(better) 1.000, with calibration error nearly halved (reliability 0.0031 vs
0.0052).

The cause is in how the scale was fitted. `nfl_model.py` correctly refuses to fit
it in-sample — that mistake gives 7.38 and makes the model wildly overconfident,
and the code carries a comment about it. But the out-of-sample fix holds out the
most recent training season:

```python
pre     = seasons < TEST_SEASON        # 2023, 2024
_hold   = max(pre)                     # 2024
_fit_on = [g for g in pre if g < _hold] # 2023 ONLY — one season, 286 games
```

So the scale was calibrated against ratings built from **a single season**, then
applied to a model built from **three**. Three-season ratings separate teams
better, and better-separated margins deserve a *narrower* scale — but they got
the wide one fitted for the weak ratings. The model was systematically
**underconfident**: it said 65% where it should have said 71% on a 10-point
favourite.

Scale against `_fit_on` depth, everything else held fixed:

| `_fit_on` | games | fitted scale |
|-----------|------:|-------------:|
| 1 season  |   286 |    **16.14** |
| 4 seasons |  1126 |        12.28 |
| 6 seasons |  1662 |    **12.40** |

Converged by four seasons.

**The fix is `SEASONS`, not the scale constant.** Widening the list so `_fit_on`
is never starved makes the correct value fall out on its own. Deeper history does
*nothing* else — all-history versus trailing-3-seasons ratings differ by
+0.00004 log-loss/game, CI [−0.00013, +0.00021] — because `HALF_LIFE_DAYS = 220`
already decays a three-season-old game to ~3% weight. The seasons are there to
feed the scale fit, not the ratings. `nfl_model.py` now warns if `_fit_on` drops
below three seasons.

Rebuilt on the model's own held-out 2025 season: **61.75% → 63.4%** straight up,
Brier **0.2293 → 0.2270**. `hfa` is unchanged at +2.20, which is the tell that
this was a confidence bug and not a ratings bug.

## A second bug, found by widening the fetch: the Pro Bowl

ESPN reports the Pro Bowl in seasontype 3 as an ordinary final, with **AFC** and
**NFC** as the franchises. It was being ingested as a real game. Seven of them
sat in the training set:

```
2019-01-27 AFC 26 @ NFC  7   total  33
2024-02-04 NFC 64 @ AFC 59   total 123
2025-02-02 AFC 63 @ NFC 76   total 139
2026-02-04 NFC 66 @ AFC 52   total 118
```

Since 2023 it is a flag-football exhibition, and the seven average **44.9 points
a side against 23.0 for real games**. Two compounding harms:

1. They registered `AFC` and `NFC` as teams that never play again.
2. **The Pro Bowl is played in February, so it is the most RECENT game in the
   set.** Recency weighting therefore gave the single worst row the single
   largest weight — and because `latest` is the max date, it also anchored the
   decay clock for every other game.

The effect on projected scores was not small:

| | implied average total |
|---|---:|
| old model | 48.3 |
| **corrected** | **45.8** |
| actual 2025 | 46.0 |

`mu` fell 23.02 → 21.80. The model had been projecting every game about 2.3
points too high, which matters directly for totals and the quarter tables.
`fetch_games` now drops any game involving `NON_TEAMS`.

This bug predates the change above — the old 3-season fetch carried three Pro
Bowls. Widening `SEASONS` is what made it visible.

## Open: CFB has the same shape, unmeasured

`cfb_model.py` uses the identical `_fit_on` pattern with the same 3-season list,
so its scale is also fitted against one season of ratings. **But CFB's single
season is 911 games to the NFL's 286**, so the starvation is far milder and may
well be immaterial. Settling it needs 2018–2022 CFB fetched (~300 requests);
until then the shipped 11.709 is unverified, not condemned.
