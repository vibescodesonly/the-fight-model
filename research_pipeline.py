"""Versioned event-causal training and complete rolling-origin evaluation."""

from dataclasses import dataclass, asdict
import gc
import hashlib
import json
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import cloudpickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score

from probability_calibration import (event_boundary, last_event_window, probability_metrics,
                                     select_binary_calibrator, MarketBlender)
from outcome_models import ConditionalMethods, DirectJointChallenger, DurationModel, CountModels, JOINT_LABELS
from temporal_features import load_frame


VERSION = "causal-research-v1"
MODEL_FILES = ("UFC_ML_Model.py", "temporal_features.py", "probability_calibration.py",
               "outcome_models.py", "research_pipeline.py", "requirements.txt")


@dataclass
class TrainingConfig:
    start_year: int = 2010
    validation_fights: int = 600
    test_fights: int = 500
    walk_forward_folds: int = 3
    minimum_training_fights: int = 1000
    tree_limit: int | None = None
    outcome_iterations: int = 300
    duration_iterations: int = 240
    count_iterations: int = 220


def source_fingerprint(root):
    root = Path(root)
    digest = hashlib.sha256()
    for name in MODEL_FILES:
        digest.update(name.encode()+b"\0"+(root/name).read_bytes())
    return digest.hexdigest()


def runtime_contract(core):
    packages = {}
    for name in ("numpy","pandas","scikit-learn","scipy","lightgbm","xgboost","catboost"):
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = None
    flags = {name:getattr(core,name) for name in ("MOV_RATINGS_ENABLED","MOV_MODE","PHASE_RATINGS_ENABLED",
        "OBA_FEATURES_ENABLED","CORNER_CORRECTION_ENABLED","WINNER_SEES_ALL_FEATURES","WINNER_COMBINER_ROBUST")}
    return {"packages":packages,"flags":flags}


def fit_winner(x, y, meta, core, config, log=print):
    split = last_event_window(meta.event_date, config.validation_fights)
    if split < config.minimum_training_fights:
        raise ValueError("Insufficient whole-event training history before calibration.")
    train, validation = x.iloc[:split].reset_index(drop=True), x.iloc[split:].reset_index(drop=True)
    train_y, val_y = y.iloc[:split].reset_index(drop=True), y.iloc[split:].reset_index(drop=True)
    columns = [c for c in train if core._feature_allowed(c, "winner")]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(train[columns])
    imp = pd.DataFrame(imputer.transform(train[columns]), columns=columns)
    selected = core._correlation_prune(imp, train_y, .95)
    columns = core._complete_swap_pairs(selected, columns)
    # Refit preprocessing to the selected schema, on training rows only.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(train[columns])
    train = pd.DataFrame(imputer.transform(train[columns]), columns=columns)
    augmented, labels = core._augment_swap(train, train_y)
    scaler = StandardScaler().fit(augmented)
    scaled = pd.DataFrame(scaler.transform(augmented), columns=columns)
    families = {"LightGBM", "XGBoost", "CatBoost", "ExtraTrees", "HistGBM", "LogReg"}
    specifications = [(name, factory) for name, factory in core._make_model_specs() if name in families]
    if not specifications:
        raise RuntimeError("No winner models available.")
    models = {}
    for name, factory in specifications:
        model = factory()
        parameters = model.get_params()
        updates = {}
        for key in ("n_jobs", "thread_count"):
            if key in parameters or (name == "CatBoost" and key == "thread_count"):
                updates[key] = 4
        if config.tree_limit is not None:
            for key in ("n_estimators", "iterations", "max_iter"):
                if key in parameters:
                    updates[key] = min(int(parameters[key]), config.tree_limit)
        if updates:
            model.set_params(**updates)
        log(f"Fitting {name}: {len(train)} training rows, {len(columns)} features")
        models[name] = core._fit_model(name, model, scaled if name in core.NEEDS_SCALE else augmented,
                                     labels, sample_weight=core._augment_weights(core._time_weights(len(train))))
    order = list(models)
    combiner = {"kind":"weighted", "model_order":order, "weights":{name:1/len(order) for name in order}}
    ensemble = core.SuperEnsembleModel(models, imputer, scaler, columns, combiner)
    _, components = ensemble.predict_proba_rows(validation, return_components=True)
    dates = meta.event_date.iloc[split:].reset_index(drop=True)
    first = event_boundary(dates, int(len(dates)/3))
    second = event_boundary(dates, int(len(dates)*2/3))
    if min(first, second-first, len(dates)-second) < 5:
        raise ValueError("Calibration needs three distinct groups of earlier events.")
    values = components.iloc[:first].to_numpy()
    target = val_y.iloc[:first].to_numpy()
    equal = np.full(len(order), 1/len(order))
    solution = minimize(lambda w: log_loss(target, np.clip(values@w, 1e-6, 1-1e-6), labels=[0, 1])
                         + .02*np.sum((w-equal)**2), equal, method="SLSQP",
                         bounds=[(0., 1.)]*len(order), constraints={"type":"eq", "fun":lambda w:w.sum()-1})
    if not solution.success:
        log(f"Blend optimization did not converge ({solution.message}); using equal weights")
    weights = solution.x if solution.success else equal
    combiner["weights"] = dict(zip(order, map(float, weights)))
    raw = core._combine_probs(components, combiner)
    calibrator, calibration = select_binary_calibrator(raw[first:second], val_y.iloc[first:second],
                                                       raw[second:], val_y.iloc[second:])
    ensemble.calibrator = calibrator
    ensemble.market_blender = MarketBlender()
    ensemble.research_version = VERSION
    ensemble.calibration_report = calibration
    ensemble.training_contract = {
        "weight_training_rows":split, "weight_training_end":meta.event_date.iloc[split-1].date().isoformat(),
        "blend_fit_rows":first, "calibration_fit_rows":second-first,
        "calibration_selection_rows":len(dates)-second,
        "calibration_start":dates.iloc[0].date().isoformat(),
        "last_supervised_date":meta.event_date.iloc[-1].date().isoformat(),
        "production_refit":"Base weights are frozen before calibration; no post-calibration refit changes their distribution.",
        "model_lineup":"Fixed original default families; no reuse of hyperparameters tuned on future folds.",
    }
    return ensemble, split


class FittedSystem:
    def __init__(self, core, config):
        self.core, self.config = core, config

    def fit(self, x, y, meta, log=print, winner_only=False):
        self.winner, split = fit_winner(x, y, meta, self.core, self.config, log)
        if winner_only:
            return self
        train_x, cal_x = x.iloc[:split].reset_index(drop=True), x.iloc[split:].reset_index(drop=True)
        train_meta, cal_meta = meta.iloc[:split].reset_index(drop=True), meta.iloc[split:].reset_index(drop=True)
        log("Fitting correctly winner-conditioned method probabilities")
        self.methods = ConditionalMethods(self.core, self.config.outcome_iterations).fit(train_x, train_meta, cal_x, cal_meta)
        self.winner.conditional_methods = self.methods
        self.winner.method_feat_cols = self.methods.input_columns
        self.winner.method_bundle = {"kind":"conditional_three_class", "method_columns":self.methods.columns}
        log("Fitting direct six-class outcome challenger")
        self.direct_joint = DirectJointChallenger(self.core,self.config.outcome_iterations).fit(train_x,train_meta,cal_x,cal_meta)
        log("Fitting competing-risk duration model on 30-second intervals")
        self.duration = DurationModel(self.core, self.config.duration_iterations).fit(train_x, train_meta)
        log("Fitting duration-exposure count models")
        self.counts = CountModels(self.core, self.config.count_iterations).fit(train_x, train_meta, cal_x, cal_meta)
        return self

    def predict(self, x, meta):
        p = self.winner.predict_proba_rows(x)
        joint = self.methods.joint(x, p)
        duration = self.duration.predict(x, meta, joint)
        props = self.counts.predict(x, duration)
        return p, joint, duration, props

    def evaluate(self, x, y, meta):
        p, joint, durations, props = self.predict(x, meta)
        metrics = probability_metrics(y, p)
        raw_calibrator = self.winner.calibrator
        self.winner.calibrator = None
        raw = self.winner.predict_proba_rows(x)
        self.winner.calibrator = raw_calibrator
        metrics["uncalibrated"] = probability_metrics(y, raw)
        metrics["elo_baseline"] = probability_metrics(y, x.elo_win_prob)
        target = meta.method.map(self.core._normalize_method_label).map({m:i for i,m in enumerate(self.core.METHOD_LABELS)}).to_numpy()
        target = target + np.where(meta.winner.to_numpy() == "Red", 0, 3)
        metrics["joint_log_loss"] = float(log_loss(target, joint, labels=np.arange(6)))
        metrics["joint_accuracy"] = float(np.mean(joint.argmax(axis=1) == target))
        metrics["joint_brier"] = float(np.mean(np.sum((joint-np.eye(6)[target])**2, axis=1)))
        direct = self.direct_joint.predict(x)
        metrics["direct_joint_challenger"] = {
            "joint_log_loss":float(log_loss(target,direct,labels=np.arange(6))),
            "joint_accuracy":float(np.mean(direct.argmax(axis=1) == target)),
            "winner":probability_metrics(y,direct[:,:3].sum(axis=1)),
            "deployment":"benchmark only; holdout results do not select the deployed model"}
        metrics["joint_class_support"] = {label:int(np.sum(target == i)) for i,label in enumerate(JOINT_LABELS)}
        metrics["duration_mae_seconds"] = float(np.mean(np.abs(
            np.array([d["expected_seconds"] for d in durations])-meta.total_fight_time_sec)))
        by_threshold = {}
        for threshold in (.5, 1.5, 2.5, 3.5, 4.5):
            ids = [i for i,d in enumerate(durations) if str(threshold) in d["over_rounds_probabilities"]]
            if ids:
                q = np.array([durations[i]["over_rounds_probabilities"][str(threshold)] for i in ids])
                truth = meta.total_fight_time_sec.iloc[ids].to_numpy() > threshold*300
                by_threshold[str(threshold)] = {"rows":len(ids), "brier":float(np.mean((q-truth)**2))}
        metrics["duration_thresholds"] = by_threshold
        metrics["count_scores"] = self.counts.log_probabilities(x, meta, durations)
        for side,prefix in (("a","r"), ("b","b")):
            for metric in self.counts.metrics:
                truth = meta[f"{prefix}_{metric}"].to_numpy()
                pred = np.array([r[f"{side}_{metric}"]["expected_count"] for r in props])
                metrics["count_scores"][f"{side}_{metric}_mae"] = float(np.nanmean(np.abs(pred-truth)))
        return metrics


def train_pipeline(pipeline, core, winner_only=False):
    config = getattr(pipeline, "research_config", TrainingConfig())
    if isinstance(config, dict):
        config = TrainingConfig(**config)
    root = Path(core.__file__).parent
    source_key = source_fingerprint(root)
    data_sha = hashlib.sha256(Path(pipeline.csv_path).read_bytes()).hexdigest()
    key = hashlib.sha256((source_key+data_sha+json.dumps(runtime_contract(core),sort_keys=True)+json.dumps(asdict(config), sort_keys=True)
                         +str(winner_only)).encode()).hexdigest()[:24]
    cache = Path(core.CACHE_DIR)/f"research_{key}.pkl"
    if cache.exists():
        pipeline._log(f"Loading verified {VERSION} cache {key}")
        with cache.open("rb") as stream:
            saved = cloudpickle.load(stream)
        csv_path = pipeline.csv_path
        pipeline.__dict__.update(saved)
        pipeline.csv_path = csv_path
        return pipeline.model
    bundle = load_frame(pipeline.csv_path, core, pipeline._log)
    meta = bundle["meta"]
    normalized = meta.method.map(core._normalize_method_label)
    valid = (meta.event_date >= pd.Timestamp(f"{config.start_year}-01-01")) & normalized.isin(core.METHOD_LABELS)
    # Nonstandard early bouts can inform histories; scheduled 3/5-round bouts
    # are the supported duration/count forecast population.
    valid &= meta.total_rounds.isin([3, 5]) & (meta.total_fight_time_sec > 0)
    x = bundle["X"].loc[valid].reset_index(drop=True)
    meta = meta.loc[valid].reset_index(drop=True)
    y = (meta.winner == "Red").astype(int)
    cutoff = last_event_window(meta.event_date, config.test_fights)
    if cutoff < config.minimum_training_fights+config.validation_fights:
        raise ValueError("Not enough data for the configured chronological evaluation.")
    report = {"version":VERSION, "config":asdict(config), "source_sha256":source_key,
        "runtime":runtime_contract(core),
        "dataset_sha256":data_sha, "data_cutoff":bundle["state"].cutoff.date().isoformat(),
        "outcome_population":"Decisive Decision/KO-TKO/Submission outcomes, scheduled 3 or 5 rounds; draws/NC/DQ not priced.",
        "evaluation_status":"Retrospective diagnostics on previously inspected history; prospective confirmation is required.",
        "feature_contract":"All events on a date share frozen pre-date state; appending later events cannot change earlier features.",
        "folds":[], "market_evaluation":{"status":"unavailable", "reason":"No timestamped historical bookmaker odds supplied."}}
    if not winner_only:
        boundaries = [len(meta)]
        for _ in range(config.walk_forward_folds):
            boundaries.insert(0, last_event_window(meta.event_date.iloc[:boundaries[0]], config.test_fights))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start < config.minimum_training_fights+config.validation_fights:
                continue
            pipeline._log(f"Full-pipeline fold: train/calibrate before {meta.event_date.iloc[start].date()}, test {end-start} fights")
            system = FittedSystem(core, config).fit(x.iloc[:start], y.iloc[:start], meta.iloc[:start], pipeline._log)
            score = system.evaluate(x.iloc[start:end].reset_index(drop=True),
                y.iloc[start:end].reset_index(drop=True), meta.iloc[start:end].reset_index(drop=True))
            score.update(test_start=meta.event_date.iloc[start].date().isoformat(),
                         test_end=meta.event_date.iloc[end-1].date().isoformat(),
                         training_contract=system.winner.training_contract,
                         calibration=system.winner.calibration_report)
            report["folds"].append(score)
            pipeline._log(f"Fold winner accuracy {score['accuracy']:.1%}; log-loss {score['log_loss']:.4f}; joint log-loss {score['joint_log_loss']:.4f}")
            del system
            gc.collect()
    pipeline._log("Fitting production system with a separate recent calibration window")
    system = FittedSystem(core, config).fit(x, y, meta, pipeline._log, winner_only=winner_only)
    bundle["state"].attach(pipeline)
    pipeline.model = system.winner
    pipeline.research_system = system
    pipeline.research_report = report
    pipeline.research_version = VERSION
    pipeline.model_fingerprint = key
    pipeline.training_source_sha256 = source_key
    pipeline.dataset_sha256 = data_sha
    report["production"] = {"training_contract":system.winner.training_contract,
        "calibration":system.winner.calibration_report, "ensemble_weights":system.winner.combiner["weights"],
        "market_blender":"unfitted; independent probabilities retained when prices are supplied"}
    if not winner_only:
        pipeline.duration_model, pipeline.count_models = system.duration, system.counts
        pipeline.method_model = system.methods
        pipeline.method_feat_cols = system.methods.input_columns
        report["production"]["method_calibration"] = system.methods.calibration_report
        report["production"]["count_dispersion"] = system.counts.dispersion_report
        if report["folds"]:
            last = report["folds"][-1]
            pipeline.benchmarks = core.BenchmarkScores(super_raw_logloss=last["uncalibrated"]["log_loss"],
                super_cal_logloss=last["log_loss"], super_brier=last["brier"],
                super_acc=last["accuracy"], super_ece=last["ece"],
                ml_baseline_logloss=None, old_baseline_logloss=last["elo_baseline"]["log_loss"])
            pipeline.method_metrics = {"joint_accuracy":last["joint_accuracy"], "joint_log_loss":last["joint_log_loss"]}
    if source_fingerprint(root) != source_key:
        raise RuntimeError("Model source changed during training; do not publish this run.")
    if hashlib.sha256(Path(pipeline.csv_path).read_bytes()).hexdigest() != data_sha:
        raise RuntimeError("Dataset changed during training; do not publish this run.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    saved = {key:value for key,value in pipeline.__dict__.items() if key not in ("log", "progress_cb")}
    with cache.with_suffix(".tmp").open("wb") as stream:
        cloudpickle.dump(saved, stream)
    cache.with_suffix(".tmp").replace(cache)
    (cache.parent/f"research_{key}_evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    pipeline._log(f"Saved {VERSION} model {key}")
    return pipeline.model
