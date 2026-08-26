# The MLB full-game model has no demonstrable skill

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
| **shipped coefficients** | 54.42% | 0.2483 |  0.6897 | 0.0013 | 0.0009 |
| refit walk-forward       | 52.74% | 0.2490 |  0.6912 | 0.0002 | 0.0002 |
| pythagorean only         | 54.13% | 0.2483 |  0.6898 | 0.0004 | 0.0002 |

**Shipped vs always-home, 1,369 games:** log-loss gain **+0.00296/game**, 95% CI
**[−0.00460, +0.01055]**, P(model better) 0.780. Accuracy **+2.12pp**, ~1.57 SE.

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
