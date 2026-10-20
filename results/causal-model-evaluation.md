# Corrected model — chronological evaluation

Retrospective diagnostics on previously inspected history; prospective confirmation is required.

Decisive Decision/KO-TKO/Submission outcomes, scheduled 3 or 5 rounds; draws/NC/DQ not priced.

All feature construction is event-frozen. Base-model fitting, blending, calibration, conditional methods, duration, and count models are repeated before each test period.

| Test period | Fights | Winner accuracy | Winner log-loss | Uncalibrated log-loss | Elo log-loss | Joint accuracy | Joint log-loss | Duration MAE (minutes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024-03-16–2025-01-11 | 507 | 64.9% | 0.6310 | 0.6310 | 0.6874 | 38.5% | 1.5627 | 4.72 |
| 2025-01-18–2025-11-08 | 500 | 66.2% | 0.6220 | 0.6227 | 0.6826 | 37.6% | 1.5519 | 4.81 |
| 2025-11-15–2026-10-17 | 501 | 64.9% | 0.6272 | 0.6263 | 0.6838 | 32.1% | 1.6017 | 5.19 |

Across 1508 test fights: winner accuracy **65.3%**, log-loss **0.6268**, joint winner/method accuracy **36.1%**.

## Calibration and production contract

The production base weights are fitted before a recent calibration window. Separate event groups fit ensemble weights, fit calibration, and select calibration by log-loss. The base weights are not subsequently refitted on calibration outcomes.

Production calibration: **none**.

The direct six-class classifier is a separately evaluated challenger. Its test scores do not select the deployed model. Production uses the calibrated winner ensemble and correctly conditioned methods.

Joint accuracy scores the most likely of six compound outcomes. Its fighter can differ from the overall winner pick because the other fighter's win probability is spread across several methods.

## Market data

No timestamped historical bookmaker odds supplied.

No historical return, closing-line value, or fitted market blending weight is claimed. When prices are supplied before historical forecasts, the append-only market tools can evaluate the declared paper strategy and compare paired no-vig bookmaker baselines.

## Interpretation

These periods were already available during development. Treat these scores as retrospective diagnostics, and retain prospective forecasts before future fights for an independent confirmation. The original published benchmark uses a different dataset, feature construction, and split; its headline figures are not a controlled comparison with this run.

Count models are initial exposure mixtures; dependence between finish timing and strike/takedown rates beyond pre-fight features is not modeled. Duration uses 30-second intervals, reconciled to joint cause probabilities.
