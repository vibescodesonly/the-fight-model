# A Leakage-Safe, Time-Aware Ensemble for UFC Fight Outcome and Method-of-Victory Prediction

A machine-learning system that predicts **(1) the winner** of a UFC bout and
**(2) the method of victory** (Decision / KO-TKO / Submission), built and
evaluated on strictly chronological data with no information leakage.

The original model and GUI are in [`UFC_ML_Model.py`](UFC_ML_Model.py).
The default training path now uses shared chronological features, calibrated
winner probabilities, coherent joint outcomes, and initial duration/count
models. The historical tuning workflow remains available with `--legacy`.

## Corrected forecasting and market research

The current implementation adds:

- One event-frozen feature state for historical and live matchups. Finish
  priors, rating normalization, style thresholds, and altitude fallbacks use
  completed past events. All fights sharing a date are snapshotted before any
  results from that date update the state.
- Age from the recorded birth date, with a labelled dated-age estimate when
  birth date is unavailable; corrected inactivity/activity calculations and
  gender context from the explicit gender field.
- Separate event groups for fitting ensemble weights, fitting calibration,
  and selecting calibration by log-loss. Calibration includes an identity
  candidate and preserves corner symmetry. Base weights remain frozen before
  the calibration window rather than being refitted afterward.
- Six joint winner/method outcomes from correctly winner-conditioned method
  probabilities. A direct six-class classifier is evaluated as a challenger.
- A 30-second competing-risk timing model reconciled to joint outcome
  probabilities, plus Poisson/negative-binomial duration-mixture baselines
  for significant strikes and takedowns.
- Append-only SQLite forecast, schedule, quote, result and model records,
  with CSV ingestion, pre-fight moneyline paper backtests, paired no-vig
  bookmaker benchmarks, closing-line value, and event-block return intervals
  when enough completed events are available.

No historical bookmaker prices have been supplied. Market blending starts
**unfitted** and retains the independent probability when prices are entered.
The blend weight can be learned and validated after sufficient timestamped
historical forecasts and quotes are available. No historical ROI is claimed.

Train and generate reports:

```bash
python scripts/research.py train --model .ufc_model_cache/causal-production.pkl
python scripts/research.py predict results/ufc333-card.json /tmp/ufc333-causal.json \
  --model .ufc_model_cache/causal-production.pkl \
  --prediction-date 2026-10-19 \
  --database .ufc_model_cache/research.sqlite
```

Training evaluates three complete advancing-window folds. Every fold fits its
own imputation, feature pruning, base models, blend, calibration, methods,
duration and count models using earlier events only. The default winner lineup
uses the original default LightGBM, XGBoost, CatBoost, HistGBM, ExtraTrees and
logistic-regression families; it does not reuse future-tuned hyperparameters.
`TrainingConfig` in `research_pipeline.py` exposes event-window and model-size
settings. Reduced settings are for smoke experiments, not the default full run.

The runner writes a trained model, checksum/package/source metadata, and an
evaluation JSON beside the model. Forecasting writes JSON, Markdown and Excel,
including explanations, coherent outcomes, duration and count projections.
`prediction_date` is the reference date for age/inactivity features. It defaults
to the current date in America/New_York in this runner; pass it explicitly for
repeatable forecasts. Model histories must precede that date.

Card JSON uses canonical dataset names and weight-class labels, explicit
gender, rounds, and title status. In particular, the dataset uses Women's
Strawweight and otherwise unprefixed class labels with gender stored separately.
Use the card runner for explicit title/date context; the GUI keeps its original
basic text-entry format. Unknown fighters are skipped unless the card explicitly
sets `allow_debutant_prior: true`.

The SQLite archive can retain date-only events, but exact timezone-aware starts
are required for market evaluation. Supply `--event-start` when known or import
the schedule with real odds. List the registered fight IDs/corner order before
mapping provider data:

```bash
python scripts/research.py list-fights --database .ufc_model_cache/research.sqlite
python scripts/research.py import-odds YOUR_ODDS.csv --database .ufc_model_cache/research.sqlite
python scripts/research.py import-results YOUR_RESULTS.csv --database .ufc_model_cache/research.sqlite
python scripts/research.py backtest --database .ufc_model_cache/research.sqlite \
  --model-id MODEL_ID_FROM_METADATA --output /tmp/moneyline-backtest.json
```

CSV headers are provided in `examples/odds-template.csv` and
`examples/results-template.csv`; they contain no invented prices or results.
Times must include UTC offsets. Supply one price format or exactly consistent
decimal/American prices. Moneyline `a`/`b` always refers to the registered corner
order. Unknown fights can be registered by including event/fighter identity and
schedule fields in the odds CSV. Local fighter IDs are retained; source IDs can
be supplied through the API. The modeling histories remain name-keyed.

The initial backtester uses one-unit moneyline stakes, at most one qualifying
bet per fight, only offers observed before a forecast, a configurable quote-age
limit and EV threshold, and the same bookmaker for entry/closing comparisons.
Draws and no-contests void two-way moneylines. Other market quotes can be stored,
but method/prop settlement simulation requires market-specific rules and is not
part of this initial moneyline backtester. Closing prices never influence entry
selection. Retrospective imports are explicitly labelled, and a model cutoff
must precede both its simulated forecast and event dates.

After historical forecasts and quotes are available, learn a market layer:

```bash
python scripts/research.py fit-market --database .ufc_model_cache/research.sqlite \
  --model .ufc_model_cache/causal-production.pkl --bookmaker "BOOKMAKER_NAME" \
  --output .ufc_model_cache/market-production.pkl
```

At least 100 historical samples and separate later validation events are
required. A combined forecast is retained only when it improves validation
log-loss over the independent model. It is also compared with the market alone.

Model/source/data/configuration/package fingerprints invalidate stale artifacts.
Pre-correction binary models must be retrained. Historical test periods were
already inspected during development: current evaluation is retrospective, and
prospective forecasts should be retained before future fights. Six-outcome
probabilities condition on decisive Decision/KO-TKO/Submission results; draws,
no-contests and DQ are not explicitly priced. Initial count models do not model
rate/finish-time dependence beyond pre-bout features, and interval timing is an
approximation. Feature contributions and component disagreement retain the
limitations described in the explanation section below.

## Current dataset and UFC 333 forecasts

The October 18, 2026 update contains **9,389 completed fights through October
17, 2026**. It adds 192 fights from 15 UFC events while preserving all 9,197
original records and the 320-column schema. Source events, checksums, missing
elevation coverage, and validation are recorded in
[`results/data-update-2026-10-18.json`](results/data-update-2026-10-18.json).

The corrected **v2.0.0-causal.1** forecasts for all 11 requested UFC 333 bouts
are saved as [matchup explanations](results/ufc333-causal-2026-10-19.md),
[full JSON](results/ufc333-causal-2026-10-19.json), and
[Excel](results/ufc333-causal-2026-10-19.xlsx), using the
[card inputs](results/ufc333-card.json) and an October 19 prediction date.
The October 18 [table](results/ufc333-predictions-2026-10-18.md) and
[JSON](results/ufc333-predictions-2026-10-18.json) are retained as historical
outputs from the previous model.

The completed corrected run scored **65.3% winner accuracy across 1,508
chronological test fights**, with log-loss **0.6268**. These are retrospective
diagnostics on previously inspected history, not independent confirmation or
a controlled improvement over the original benchmark. The production
calibration selection retained identity. Read the
[evaluation report](results/causal-model-evaluation.md) and
[machine-readable evaluation](results/causal-model-evaluation.json).

The [version manifest](results/model-v2-manifest.json) records model, source,
dataset, package and artifact checksums. The complete 221 MB trained checkpoint
is preserved losslessly as [models/causal-production.pkl.xz](models/causal-production.pkl.xz)
(58 MB compressed), with [metadata](models/causal-production.meta.json).
Decompression was checked against the original checkpoint SHA-256. The
**v2.0.0-causal.1** research release starts as a draft visible to repository
maintainers until published. Its checkpoint is included directly in the Git
snapshot; it does not require access to release assets. Install this version's
pinned requirements, restore the checkpoint, and run:

```bash
mkdir -p .ufc_model_cache
xz --decompress --stdout models/causal-production.pkl.xz > .ufc_model_cache/causal-production.pkl
cp models/causal-production.meta.json .ufc_model_cache/
python scripts/research.py predict results/ufc333-card.json /tmp/ufc333-causal.json \
  --model .ufc_model_cache/causal-production.pkl \
  --data pure_fight_data_with_event_and_camp_altitudes.csv \
  --prediction-date 2026-10-19
```

Only load trusted checkpoints: pickle loading executes Python code. The runner
validates the checkpoint and source checksums before loading, then validates
the dataset. The explicit `--data` path supports restoring a checkpoint in a
different checkout without relaxing its dataset checksum.

## Original published benchmark

From the original published run (dataset through June 27, 2026; 9,197 fights, of
which 7,873 rows from the selected 2010+ training era):

| Metric (500 most-recent-fight holdout) | Value |
|---|---|
| Winner accuracy | **69.0%** |
| Winner log-loss | 0.6102 |
| Brier score | 0.2107 |
| Expected calibration error (ECE) | 0.0644 |
| Method accuracy (given winner pick correct) | **52.2%** (majority baseline: 41.7%) |
| Method macro-F1 (given winner pick correct) | 49.0% |

Walk-forward diagnostics (six expanding-window folds retraining a single
gradient-boosted model per fold) show accuracy improving monotonically toward
the present — 59.5% on the earliest fold to 67.2% on the most recent —
confirming that skill is not an artifact of one lucky holdout window.

These numbers use the original dataset at commit
[`ca597d0`](https://github.com/vibescodesonly/the-fight-model/tree/ca597d04edf5974d14ab6e594ae61164c9017f61).
Training on the updated dataset evaluates a later holdout window (see
[Reproducing results](#reproducing-results)).

## Original ensemble design

The following describes the historical tuning architecture. The corrected
default workflow and its evaluation contract are described above; use the
original commit for the paper's exact implementation and dataset.

### 1. Leak-safe chronological feature engineering

Every fight row is constructed exclusively from each fighter's **pre-fight**
state; running state is updated only *after* a row is emitted, so no feature
can contain information from its own fight. Features include:

- **Career and recency statistics** — striking/grappling rates, Bayesian-shrunk
  accuracies, round-by-round pacing and cardio profiles, finish-round
  distributions, durability and submission composites, and
  opponent-baseline-adjusted (`oba_*`) outputs that credit production against
  each opponent's own pre-fight statistical norms.
- **Glicko-2 ratings**, optionally updated with a continuous
  **margin-of-victory score** (finish timing + box-score dominance) so dominant
  wins move ratings more than split decisions — without ever flipping the true
  result.
- **Elo and divisional Elo**, with divisional rank snapshots and rating-trajectory
  slopes.
- **Phase Glicko ratings** — four ratings per fighter (striking-offense,
  striking-defense, grappling-offense, grappling-defense) updated bipartite
  within every fight, so "elite wrestler with weak takedown defense" is
  representable and matchups cross offense against the opponent's defense.
- **Style-matchup records** — each fighter's historical win rate against
  opponents bucketed by style (striker / wrestler / submission hunter).
- **Altitude and acclimatization** — venue elevation combined with each
  fighter's training-camp elevation (shock of fighting above camp, benefit of
  descending from a high camp).
- **Context finish priors** — time-decayed finish rates for fights of the same
  weight class / gender / rounds / title status.

A **feature-routing table** (`FEATURE_ROUTING`) declares which of the three
models (winner, finish-vs-decision, KO-vs-sub) is allowed to see each
engineered feature, so outcome-conditional method signals never add noise to
the winner model and vice versa.

### 2. Corner-swap-symmetric winner ensemble

UFC data has a strong red-corner convention (the promoted fighter is red). The
winner model is made exactly **corner-symmetric**: training data is augmented
with every fight seen from both corners (antisymmetric features negated,
probability features complemented, raw red/blue pairs exchanged), and at
inference each prediction is the average of the forward and corner-swapped
passes. Base models — LightGBM, XGBoost, CatBoost (default + Optuna-tuned
variants), HistGradientBoosting, RandomForest, and ExtraTrees — are combined
by a simplex-weighted blend fit on time-series out-of-fold predictions, with
Platt/isotonic calibration fit on early validation data and accepted only if
it helps on late validation data.

Feature selection is two-stage and leak-safe (fit on training rows only):
correlation pruning (|r| > 0.95, keeping the more target-relevant member of
each pair) followed by Meinshausen–Bühlmann stability selection over 15
bootstrap runs.

### 3. Two-stage method model

Method prediction is conditioned on the predicted winner: the feature matrix
is re-oriented so the picked winner occupies the positive corner, then a
**Stage 1** classifier (Finish vs Decision) and **Stage 2** classifier (KO vs
Submission, trained on finishes only) — each a subsample-bagged
HistGradientBoosting ensemble with RandomForest/ExtraTrees side models — are
combined with direct multiclass, one-vs-rest, and linear heads plus history-,
group-, and submission-signal priors. The ~20 blend hyperparameters are tuned
by Optuna against a chunked walk-forward objective, and a
**champion/challenger gate** persists the best configuration across runs so
retraining noise cannot silently degrade the deployed blend.

### 4. Evaluation protocol

- Chronological split by **fixed fight counts**: the 500 most-recent fights are
  an untouched holdout; the 600 before them are the validation window; the
  rest is training. Fixed counts keep the evaluation windows identical across
  training-era candidates, making era selection a like-for-like comparison.
- **Strict future mode** (default ON) forbids every holdout-informed choice:
  the combiner, calibrator, and decision threshold are selected on validation
  data only.
- Trained stages are **cached content-addressed** on the SHA-256 of the
  dataset contents plus every configuration value that affects the result, so
  a cache hit is only possible when retraining would reproduce the identical
  model.

## Repository contents

| Path | Description |
|---|---|
| `UFC_ML_Model.py` | The complete model: features, ratings, training, evaluation, GUI. |
| `pure_fight_data_with_event_and_camp_altitudes.csv` | Fight dataset (see [Data](#data)). |
| `Stat Definitions.txt` | Data dictionary: a definition for every dataset column. |
| `results/` | Original benchmark artifacts, dated data-update audit, and UFC 333 forecasts. |
| `requirements.txt` | Pinned Python dependencies. |
| `temporal_features.py` | Shared event-frozen historical/live feature state. |
| `research_pipeline.py` | Default chronological fitting and complete rolling evaluation. |
| `probability_calibration.py` | Calibration selection and fitted market blending. |
| `outcome_models.py` | Conditional/joint outcomes, timing, and count distributions. |
| `market_store.py` | Append-only research database and moneyline backtests. |
| `matchup_reports.py` | Grouped model explanations and readable reports. |
| `scripts/research.py` | Training, forecasting, archiving, CSV ingestion and evaluation CLI. |
| `CITATION.cff` | Citation metadata. |
| `.ufc_model_cache/` | Generated: cached training stages (safe to delete; forces retrain). |
| `UFC_Predictions.xlsx` | Generated: prediction export from the GUI. |

## Requirements

- Python ≥ 3.10 (developed on 3.13); Tkinter (bundled with standard CPython
  installers) is needed only for the GUI.
- `pip install -r requirements.txt`

LightGBM, XGBoost, CatBoost, and Optuna are technically optional — the script
degrades gracefully if any is missing — but the published results use all of
them.

## Matchup explanation reports

The GUI now includes an **Explanations** worksheet in its prediction export.
For Python callers, enable reports explicitly:

```python
prediction = pipeline.predict_matchup(
    "Alexander Volkanovski", "Movsar Evloev",
    weight_class="Featherweight", gender="Men", rounds=5,
    is_title=True, prediction_date="2026-10-18",
    include_explanation=True,
)
report = prediction["explanation"]
```

Each report contains fighter statistical comparisons, recorded-history and
missing-input flags, component-model probabilities, model disagreement, fair
odds, and grouped contributions to the raw winner-model probability. If both
moneylines are supplied, it also reports market probabilities and model-implied
expected returns. Market blending is kept separate from statistical attribution.

Contributions are exact Shapley values **for the specified feature groups and
reference**: the reference averages the matchup's original and corner-swapped
feature vectors. Every combination of up to ten groups is evaluated, and the
contributions add to the difference between reference and model probabilities.
Synthetic combinations of correlated inputs may be unlike real fights; these
are explanations of model behavior, not causal effects. Disagreement between
correlated models is not a statistical confidence interval. Method-of-victory
contributions are not attributed.

Generate JSON, Markdown, and Excel reports for a card from a trusted serialized
pipeline and its metadata:

```bash
python scripts/predict_card_reports.py results/ufc333-card.json /tmp/ufc333-reports.json \
  --model /path/to/trained-pipeline.pkl \
  --metadata /path/to/trained-pipeline-meta.json \
  --prediction-date 2026-10-18
```

Metadata must include `source_data_sha256`, the SHA-256 of training-source bytes
followed by CSV bytes, and `ufc_environment`, the explicitly set `UFC_*` feature
switches listed in the runner. For a retained model trained before the reporting
changes, provide its attested original source with `--training-source`. The runner
verifies the fingerprint and rejects changes to training definitions or settings;
presentation and matchup API changes are permitted. It records training,
inference, report-source, and dataset fingerprints separately. Model pickle files
must come from a trusted source.

The corrected pipeline supplies live finish priors and fixes age/date/gender
handling. The legacy refit preserves stage-specific feature selections. Reports
still expose missing inputs and sparse histories. Reporting itself does not
establish improved accuracy. A past prediction date requires an appropriately
earlier model/history snapshot; changing a date cannot remove future knowledge.

Validation:

```bash
python -m unittest discover -s tests -v
```

## Data

One row per fight (9,389 UFC fights, 1993–October 17, 2026), with:

- event metadata: date, location, venue elevation, weight class, gender,
  title-bout flag, scheduled rounds;
- outcome: winner corner, method string, finish round, fight time;
- per-corner box score: significant/total strikes (landed + attempted),
  knockdowns, takedowns, submission attempts, reversals, control time, and
  strike distribution by target (head/body/leg) and position
  (distance/clinch/ground) — as fight totals and per round (`r_rd1_*`, …);
- fighter attributes: height, reach, weight, stance, age at event, and
  training-camp elevation.

Every column is defined in [`Stat Definitions.txt`](Stat%20Definitions.txt).

Statistics were collected from [ufcstats.com](http://ufcstats.com) (the UFC's
official statistics provider) and enriched with venue and training-camp
elevations via geocoding. The current CSV includes the October 18 update;
the original benchmark dataset is available at the commit linked above.
The CSV's SHA-256 participates in the cache keys, so modifying it invalidates
the corresponding cached training stages.

For the 192 added rows, unverified training-camp elevations remain blank;
27 rows also lack a verified venue elevation. The model uses its existing
fallbacks for missing elevation data. Weight-class labels follow the original
dataset's conventions. The updated CSV SHA-256 is
`79cbca8c3f429f108f25360b30e664fce125d79cd76b4856b157a29d06d29bf1`.

## Reproducing results

```bash
pip install -r requirements.txt
python UFC_ML_Model.py --train-only
```

This runs the corrected default training/evaluation workflow on the current
dataset. `python UFC_ML_Model.py --train-only --legacy` selects the historical
tuning workflow with corrected features. To reproduce the original
published benchmark, use the original commit linked above with its dataset.
The roughly 1 GB serialized production model is retained in the cloud
environment; generated model binaries and training caches are excluded from
GitHub. A fresh checkout rebuilds trained models using the command above.

The script prints a structured report: split contract, feature pruning,
Optuna tuning, combiner selection, calibration, holdout evaluation, method
evaluation (conditioned on winner pick), and walk-forward diagnostics.

Notes:

- **Determinism.** All RNGs are seeded (`RANDOM_SEED = 42`); repeated runs on
  the same data and configuration reproduce the same results and hit the
  stage caches. Minor cross-platform variation is possible from multithreaded
  tree libraries.
- **Runtime.** The legacy cold run performs the full Optuna search (80 trials × 3
  libraries for the winner stage, 80 × 2 + 400 blend trials for the method
  stage) — expect several hours on a desktop CPU. Subsequent runs replay the
  cached stages in minutes. Default corrected training uses fixed default
  model families and evaluates the complete pipeline across chronological folds;
  unchanged runs load the source/data/configuration-addressed cache.
- **GUI.** `python UFC_ML_Model.py` (no flags) opens the prediction GUI: one
  matchup per line as `Red,Blue,Weight Class,Gender,Rounds[,Elevation ft or
  Location]`, with optional American moneyline odds after each name
  (auto-detected). Predictions are exported to `UFC_Predictions.xlsx`.

### Configuration switches (controlled A/B tests)

Documented ablations are exposed as environment variables; defaults reproduce
the published configuration.

| Variable | Default | Effect |
|---|---|---|
| `UFC_MOV_ENABLED` | `1` | Margin-of-victory-adjusted ratings (`0` = hard win/loss only). |
| `UFC_MOV_MODE` | `full` | MOV recipe: `full` (finish timing + box score) or `buckets` (low-overfit baseline). |
| `UFC_PHASE_RATINGS` | `1` | Four-dimensional phase Glicko ratings on/off. |
| `UFC_OBA_ENABLED` | `1` | Opponent-baseline-adjusted (`oba_*`) features on/off. |
| `UFC_CORNER_CORRECTION` | `0` | One-parameter red-corner logit correction (tested; defaults off). |
| `UFC_WINNER_ALL_FEATURES` | `0` | Winner model bypasses feature routing (tested; defaults off). |
| `UFC_COMBINER_ROBUST` | `1` | Restrict combiner to simplex blends (`0` re-allows stacker meta-learners). |

Key in-file constants: `FORCED_START_YEAR` (training-era pin; `None`
re-enables automatic era selection), `VAL_FIGHTS` / `TEST_FIGHTS` (split
sizes), `OPTUNA_TRIALS`, `METHOD_TUNING_TRIALS`, and `STRICT_KEEP_MODELS`
(the active base-model lineup). Each is documented where it is defined.

## Limitations

- Fighter identity is name-based; distinct fighters sharing a name would be
  merged.
- Debutants have no history: the model falls back to prior features, and the
  GUI skips matchups involving debutants by default.
- The method model's rarest class (Submission) remains the hardest
  (F1 ≈ 0.33 on the holdout); results are reported per class.
- Betting-value utilities are provided for research completeness; nothing in
  this repository constitutes betting advice.

## Reproducibility statement

All results in the accompanying paper were produced at the pinned dependency
versions in `requirements.txt`, from the original 9,197-fight dataset at commit
`ca597d0`, with all random seeds fixed. The dated update and forecast artifacts
identify the current dataset separately. Code was developed by the author with assistance
from Anthropic's Claude (AI pair programming); all modeling decisions were
validated empirically through the ablations and audits described above.

## Citation

If you use this code or dataset, please cite it (see `CITATION.cff`).

## License

The code is released under the [MIT License](LICENSE).

The dataset consists of factual fight statistics compiled from
[ufcstats.com](http://ufcstats.com), enriched with geocoded elevations, and is
included solely to make the published results reproducible. UFC® is a
trademark of Zuffa, LLC; this project is not affiliated with or endorsed by
the UFC.
