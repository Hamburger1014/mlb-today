# CFB: the strongest model of the four, and its scale checks out clean

Measured 2026-08-26. Reproduce with `python scripts/cfb_skill_test.py`.

Third in the series — [`mlb_no_skill.md`](mlb_no_skill.md) (no skill),
[`nfl_skill.md`](nfl_skill.md) (skill, plus a scale 30% too wide), and this one,
which is where the NFL bug was **suspected but turned out not to bite**.

## Does it beat a constant?

Walk-forward by date over 2022–2025, ratings fit on every prior game only.
**3,660 out-of-sample games.**

|                              |    acc |  Brier | logloss |    res |    rel |
|------------------------------|-------:|-------:|--------:|-------:|-------:|
| always home (prior rate)      | 63.25% | 0.2326 |  0.6579 | 0.0000 | 0.0000 |
| home-field only               | 63.25% | 0.2326 |  0.6579 | 0.0002 | 0.0003 |
| **model, scale fitted OOS**   | 72.38% | 0.1799 |  0.5346 | 0.0290 | 0.0017 |
| model, scale fitted in-sample | 72.38% | 0.1789 |  0.5316 | 0.0294 | 0.0011 |
| model, shipped scale 11.709   | 72.38% | 0.1798 |  0.5351 | 0.0303 | 0.0026 |

**Model vs always-home:** log-loss gain **+0.12329/game**, 95% CI
**[+0.11323, +0.13342]**, P(better) **1.000**. Accuracy **+9.13pp**, ~**11.04 SE**.
Mean absolute margin error 13.81 pts.

The strongest of the four by a wide margin — the log-loss gain is nearly 3x
NFL's +0.04565 and the model separates games at resolution 0.0290 against
uncertainty ≈0.233.

Two caveats that keep this honest. College home advantage is huge (**63.25%**
base rate, versus 53.9% in the NFL), so the constant it beats is already a
strong one, and the *percentage-point* gain over base is about the same as NFL's.
And beating a constant is not beating a price — the market test still governs
what gets bet.

As in the NFL, **home-field only does not beat the base rate** (+0.00006/game,
CI spans zero). The skill lives in the team ratings.

## The NFL scale bug: present in shape, immaterial in degree

`cfb_model.py` uses the identical `_fit_on` pattern that cost the NFL 30% of its
confidence — hold out the most recent training season, and a 3-season list
leaves exactly one season to build the holdout ratings from.

**The difference is season size.** An NFL season is 286 games; a CFB season is
~911. College builds its holdout ratings on 3x the data and does not starve:

| `_fit_on` | games | fitted scale |
|-----------|------:|-------------:|
| 1 season (shipped) |  911 | **11.71** |
| 3 seasons | 2694 | 11.17 |
| 5 seasons | 4466 | 11.30 |
| 6 seasons | 5034 | 11.32 |

Compare NFL, where one season gave **16.14** against a converged **12.40**.

And the residual gap does not matter. Scale fitted OOS (11.99) versus the
shipped 11.709: **+0.00043 log-loss/game, 95% CI [−0.00081, +0.00164]**,
P(better) 0.755 — indistinguishable. The NFL equivalent was +0.00674 with a CI
clear of zero.

**Conclusion: no change. `SEASONS` stays at 3 and `marginScale` stays 11.709.**
`cfb_model.py` gained a warning that fires if `_fit_on` ever drops below 800
games, so a future narrowing cannot reintroduce the starvation silently. The
shipped `data/cfb_model.json` is byte-identical after the change.

## No Pro Bowl equivalent

The NFL fetch was ingesting the Pro Bowl as a real game with AFC/NFC as
franchises. College was checked for the same thing and is clean: the 32 rare
team labels are genuine FCS opponents (Austin Peay, Villanova, Yale) playing
real games, which is exactly what `pool_thin_teams` exists to fold into `OTHER`.
The highest-scoring game in the set, LSU 72 @ Texas A&M 74, is the real 2018
seven-overtime game.

One known and accepted imprecision: `hs`/`as` are final scores including
overtime, while the quarter table reads only the first four periods. That
slightly inflates `mu` in both football models. Pre-existing, small, and shared
with the NFL — not chased here.
