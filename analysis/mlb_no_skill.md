# The MLB full-game model has no demonstrable skill

> **CORRECTED 2026-08-31.** The first version of this measured a model that was
> not the one shipping. `mlb_skill_test.py` withheld a starter's K-BB% below a
> hard 100-batters-faced cutoff, while `index.html` applies a 40-BF floor and
> then a linear ramp to full weight at 250 BF. That is not a small difference:
> the live model rates a starter on **403 of 1,674 games** where the test
> refused to. Every number below is the corrected one. **The verdict did not
> change** - the interval still spans zero - but the model is better than the
> first version of this document said, and the numbers here superseded it.

Measured 2026-08-26. Reproduce with `python scripts/mlb_skill_test.py`.

## What was already known

`scripts/mlb_fullgame_test.py` tested the model against the Kalshi closing price
on 134 replayed games and found nothing: β_model +0.16 (SE 0.84, z=0.19), model
log-loss 0.6824 against 0.6728 for the price. That set the card weight to
`w:0.00` in `MARKET_EVIDENCE.MLBGAME`.

That test is capped at 134 games by how far the Kalshi snapshots in git history
go back, and it asks a hard question — beat a sharp market. A model can fail it
and still be useful.

## The weaker question, with 10x the power

Can the model beat **a constant**: "the home team wins 52.3% of the time"?

Walk-forward over the 2026 season to 2026-08-13. For each slate, fit on every
prior game only, predict that slate. 1,674 replayed games, 1,369 of them out of
sample after a 300-game burn-in. Inputs are point-in-time (standings as of the
day *before*; each starter's K-BB% from starts strictly before, ≥100 batters
faced), so there is no lookahead.

|                          |    acc |  Brier | logloss |    res |    rel |
|--------------------------|-------:|-------:|--------:|-------:|-------:|
| always home (prior rate) | 52.30% | 0.2497 |  0.6926 | 0.0000 | 0.0000 |
| **shipped coefficients** | 54.64% | 0.2475 |  0.6881 | 0.0014 | 0.0007 |
| refit walk-forward       | 53.32% | 0.2484 |  0.6900 | 0.0005 | 0.0001 |
| pythagorean only         | 54.13% | 0.2483 |  0.6898 | 0.0004 | 0.0002 |

**Shipped vs always-home, 1,369 games:** log-loss gain **+0.00452/game**, 95% CI
**[−0.00279, +0.01197]**, P(model better) 0.883. Accuracy **+2.34pp**, ~1.73 SE.

The interval spans zero on both metrics. After a full season, the model cannot
be shown to beat a constant.

## Two things this rules out

**It is not a bad fit.** Refitting the coefficients walk-forward is *worse* than
the shipped ones at every regularisation tried — 0.6923, 0.6920, 0.6915, 0.6912,
0.6906 for L2 of 1e-4 through 10, against 0.6897 shipped. The shipped
coefficients were fit on more data than one season and generalise better than
anything refit here. This is the ceiling of the feature set, not a tuning error.

**It is not the extra features.** Pythagorean differential alone gets 0.6898,
statistically indistinguishable from the full five-feature model's 0.6897. Win%,
run differential and K-BB% add nothing measurable on top of it — they are three
restatements of "the home team is better."

## A trap worth recording

A single 70/30 split said the opposite: on a 521-game test window, an
unregularised refit wanted a pythagorean coefficient of **5.92** against the
shipped **0.33**, and beat it. That looked like a shipped model 18x
under-scaled.

It did not survive walk-forward. One window with 521 games and five collinear
features is enough noise to produce a large, confident, wrong coefficient. The
walk-forward with 1,369 test games reverses the ranking. **Prefer walk-forward
to a single split whenever the features are collinear.**

## What this changes

Nothing operationally — `w` was already 0.00 and MLB Best Bets already surfaces
no plays. It changes the *reason*, and how much it would take to reopen the
question. MLB is not a market where a better fit or a shrink parameter helps;
it needs genuinely different inputs (bullpen state, park, lineup, rest) before
it is worth re-testing.

For scale: resolution across the four models, against uncertainty ≈0.249 —
WNBA 0.0300, CFB 0.0292, NFL 0.0048, **MLB 0.0019**.


## The lesson from the correction

The bug was not in the model, it was in the harness: a backtest that did not
replicate what ships. It moved P(better) from 0.735 to 0.883 - not enough to
change the verdict here, but easily enough to change one somewhere else, and it
had been quietly understating this model in a published document.

**Any replay must reproduce the live feature construction exactly, and that is
worth asserting rather than assuming.** The four parity verifiers in `scripts/`
exist for the coefficients; nothing was checking the FEATURE BUILD around them.
`kbb_asof` now carries the live constants (`KBB_LG 0.14138`, `KBB_WBF 250`,
`BF_FLOOR 40`) with a note saying they must match `index.html:_kbbQuality`.

Separately worth recording: refitting the coefficients on the corrected features
and using Bayesian shrinkage instead of the live linear ramp pushes P(better)
above 0.97. That is a DIFFERENT model, not the one shipping, and its accuracy
edge is still only ~1.5 SE. It is a reason to re-measure if the live feature
build ever changes - not a result.
