"""Conditional methods, competing-risk duration, and exposure-mixture counts."""

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

from probability_calibration import TemperatureCalibration, event_boundary


METHODS = ("Decision", "KO/TKO", "Submission")
JOINT_LABELS = ("a_decision", "a_ko_tko", "a_submission", "b_decision", "b_ko_tko", "b_submission")
CAUSE_JOINT = (1, 4, 2, 5)  # A KO, B KO, A submission, B submission.


def fixed_probabilities(model, x, classes):
    if isinstance(x,np.ndarray) and hasattr(model,"feature_names_in_"):
        x = pd.DataFrame(x,columns=model.feature_names_in_)
    predicted = model.predict_proba(x)
    result = np.full((len(x), classes), 1e-8)
    for i, label in enumerate(model.classes_):
        result[:, int(label)] = predicted[:, i]
    return result/result.sum(axis=1, keepdims=True)


def classifier(core, iterations=300, min_leaf=35):
    if core.lgb is not None:
        return core.lgb.LGBMClassifier(n_estimators=iterations, learning_rate=.04,
            max_depth=5, num_leaves=23, min_child_samples=min_leaf, reg_lambda=3.,
            colsample_bytree=.8, n_jobs=4, random_state=core.RANDOM_SEED,
            verbosity=-1, deterministic=True, force_col_wise=True)
    return HistGradientBoostingClassifier(max_iter=iterations, learning_rate=.04,
        max_leaf_nodes=23, min_samples_leaf=min_leaf, l2_regularization=3.,
        early_stopping=False, random_state=core.RANDOM_SEED)


class ConditionalMethods:
    def __init__(self, core, iterations=300):
        self.core, self.iterations = core, iterations
        self.calibration = TemperatureCalibration()

    def matrix(self, x, winner_a):
        orientation = np.full(len(x), int(winner_a)) if np.isscalar(winner_a) else np.asarray(winner_a)
        return self.core._augment_method_features(self.core._oriented_method_matrix(x, orientation))

    def fit(self, x, meta, calibration_x, calibration_meta):
        y = meta.method.map(self.core._normalize_method_label).map({label:i for i,label in enumerate(METHODS)}).astype(int)
        oriented = self.matrix(x, (meta.winner == "Red").astype(int))
        eligible = [name for name in oriented.columns if
                    self.core._feature_allowed(name, "stage1") or self.core._feature_allowed(name, "stage2")]
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(oriented[eligible])
        imp = pd.DataFrame(self.imputer.transform(oriented[eligible]), columns=eligible)
        self.columns = self.core._correlation_prune(imp, y, .95)
        self.input_columns = eligible
        self.model = classifier(self.core, self.iterations)
        if len(np.unique(y)) < 2:
            self.model = DummyClassifier(strategy="prior")
        self.model.fit(imp[self.columns], y)
        split = event_boundary(calibration_meta.event_date, int(len(calibration_meta)*.65))
        if split >= 20 and len(calibration_meta)-split >= 20:
            raw = self.predict(calibration_x, (calibration_meta.winner == "Red").astype(int), calibrated=False)
            target = calibration_meta.method.map(self.core._normalize_method_label).map({label:i for i,label in enumerate(METHODS)}).astype(int).to_numpy()
            self.calibration_report = self.calibration.fit_select(raw[:split], target[:split], raw[split:], target[split:])
        else:
            self.calibration_report = {"temperature":1., "reason":"insufficient separate calibration events"}
        return self

    def predict(self, x, winner_a, calibrated=True):
        oriented = self.matrix(x.reset_index(drop=True), winner_a)
        values = pd.DataFrame(self.imputer.transform(oriented.reindex(columns=self.input_columns)),
                              columns=self.input_columns)[self.columns]
        probabilities = fixed_probabilities(self.model, values, 3)
        return self.calibration.predict(probabilities) if calibrated else probabilities

    def joint(self, x, winner_probabilities):
        p = np.asarray(winner_probabilities).reshape(-1, 1)
        return np.column_stack([p*self.predict(x, True), (1-p)*self.predict(x, False)])


class DirectJointChallenger:
    """Six-class benchmark, evaluated separately from deployed marginals."""
    def __init__(self, core, iterations=300):
        self.core,self.iterations = core,iterations
        self.calibration = TemperatureCalibration()

    def fit(self,x,meta,calibration_x,calibration_meta):
        target = meta.method.map(self.core._normalize_method_label).map({m:i for i,m in enumerate(METHODS)}).to_numpy()
        target += np.where(meta.winner.to_numpy() == "Red",0,3)
        self.columns = list(x.columns)
        rows = pd.concat([x,self.core._swap_features(x)],ignore_index=True)
        y = np.r_[target,(target+3)%6]
        self.imputer = SimpleImputer(strategy="median",keep_empty_features=True)
        values = self.imputer.fit_transform(rows)
        self.model = classifier(self.core,self.iterations)
        self.model.fit(values,y)
        split = event_boundary(calibration_meta.event_date,int(len(calibration_meta)*.65))
        if split >= 20 and len(calibration_meta)-split >= 20:
            raw = self.predict(calibration_x,calibrated=False)
            labels = calibration_meta.method.map(self.core._normalize_method_label).map({m:i for i,m in enumerate(METHODS)}).to_numpy()
            labels += np.where(calibration_meta.winner.to_numpy() == "Red",0,3)
            self.calibration_report = self.calibration.fit_select(raw[:split],labels[:split],raw[split:],labels[split:])
        return self

    def predict(self,x,calibrated=True):
        a = fixed_probabilities(self.model,self.imputer.transform(x[self.columns]),6)
        b = fixed_probabilities(self.model,self.imputer.transform(self.core._swap_features(x)[self.columns]),6)
        probabilities = (a+b[:,[3,4,5,0,1,2]])/2
        return self.calibration.predict(probabilities) if calibrated else probabilities


def duration_distribution(raw_hazards, joint, seconds=900, interval=30):
    """Reconcile cause-specific timings to calibrated six-outcome marginals."""
    hazards = np.clip(np.asarray(raw_hazards, dtype=float), 1e-12, 1)
    hazards /= hazards.sum(axis=1, keepdims=True)
    survival = np.r_[1., np.cumprod(hazards[:-1, 0])]
    raw_mass = survival[:, None]*hazards[:, 1:]
    timing = raw_mass/np.maximum(raw_mass.sum(axis=0, keepdims=True), 1e-300)
    masses = timing*np.asarray(joint)[list(CAUSE_JOINT)]
    left = np.arange(len(hazards))*interval
    right = np.minimum(left+interval, seconds)
    middle = (left+right)/2
    finish_mass = masses.sum(axis=1)
    distance = float(joint[0]+joint[3])
    support = np.r_[middle, seconds]
    weights = np.r_[finish_mass, distance]
    weights /= weights.sum()
    totals = {}
    for rounds in (.5, 1.5, 2.5, 3.5, 4.5):
        threshold = rounds*300
        if threshold < seconds:
            fraction_before = np.clip((threshold-left)/np.maximum(right-left, 1), 0, 1)
            totals[str(rounds)] = float(1-np.sum(finish_mass*fraction_before))
    # The conditional hazard representation can be reconstructed from these
    # reconciled masses; survival and winner/method markets remain consistent.
    return {"interval_seconds":interval, "scheduled_seconds":int(seconds),
            "expected_seconds":float(np.dot(support, weights)),
            "goes_distance_probability":distance, "over_rounds_probabilities":totals,
            "finish_probability_by_interval":finish_mass.tolist(),
            "cause_probability_by_interval":masses.tolist(),
            "duration_support_seconds":support.tolist(), "duration_support_weights":weights.tolist(),
            "interpretation":"Finish times are uniform within intervals; finish-cause masses are reconciled to joint outcome probabilities."}


class DurationModel:
    def __init__(self, core, iterations=240, interval=30):
        self.core, self.iterations, self.interval = core, iterations, interval

    def _grid(self, x, meta, observed):
        counts = np.ceil((meta.total_fight_time_sec.clip(lower=1) if observed
                          else meta.total_rounds*300)/self.interval).astype(int).to_numpy()
        owners = np.repeat(np.arange(len(x)), counts)
        elapsed = np.concatenate([np.arange(n)*self.interval for n in counts])
        result = x.reset_index(drop=True).iloc[owners].reset_index(drop=True).copy()
        result["hazard_elapsed_minutes"] = elapsed/60
        result["hazard_round"] = elapsed//300+1
        result["hazard_remaining_minutes"] = meta.total_rounds.to_numpy()[owners]*5-elapsed/60
        target = np.zeros(len(result), dtype=int)
        if observed:
            endpoints = np.cumsum(counts)-1
            normalized = meta.method.map(self.core._normalize_method_label)
            for i, end in enumerate(endpoints):
                method = normalized.iloc[i]
                if method != "Decision":
                    target[end] = (1 if meta.winner.iloc[i] == "Red" else 2) + (2 if method == "Submission" else 0)
        return result, target, counts

    def fit(self, x, meta):
        # Hazard model inputs are pre-bout features plus elapsed time, not
        # in-fight strike/control totals or the eventual observed duration.
        mirrored = self.core._swap_features(x)
        swapped_meta = meta.copy()
        swapped_meta["winner"] = swapped_meta.winner.map({"Red":"Blue", "Blue":"Red"})
        all_x = pd.concat([x, mirrored], ignore_index=True)
        all_meta = pd.concat([meta, swapped_meta], ignore_index=True)
        rows, target, _ = self._grid(all_x, all_meta, observed=True)
        self.columns = list(rows.columns)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        values = self.imputer.fit_transform(rows).astype(np.float32)
        self.model = classifier(self.core, self.iterations, min_leaf=100)
        if len(np.unique(target)) < 2:
            self.model = DummyClassifier(strategy="prior")
        self.model.fit(values, target)
        self.training_period_rows = len(rows)
        return self

    def predict(self, x, meta, joint):
        rows, _, counts = self._grid(x, meta, observed=False)
        values = self.imputer.transform(rows[self.columns]).astype(np.float32)
        p = fixed_probabilities(self.model, values, 5)
        # Symmetrize the cause predictions as well as the winner model.
        reverse_rows, _, _ = self._grid(self.core._swap_features(x), meta, observed=False)
        rp = fixed_probabilities(self.model, self.imputer.transform(reverse_rows[self.columns]).astype(np.float32), 5)
        p = (p+rp[:, [0, 2, 1, 4, 3]])/2
        results, start = [], 0
        for i, count in enumerate(counts):
            results.append(duration_distribution(p[start:start+count], joint[i],
                           int(meta.total_rounds.iloc[i])*300, self.interval))
            start += count
        return results


class CountModels:
    metrics = ("td", "sig_str")

    def __init__(self, core, iterations=220):
        self.core, self.iterations = core, iterations
        self.models, self.dispersions, self.dispersion_report = {}, {}, {}

    def fit(self, x, meta, calibration_x, calibration_meta):
        self.columns = list(x.columns)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        rows = pd.concat([x, self.core._swap_features(x)], ignore_index=True)
        values = self.imputer.fit_transform(rows)
        exposure = np.tile(meta.total_fight_time_sec.clip(lower=1).to_numpy()/60, 2)
        cal_rows = pd.concat([calibration_x, self.core._swap_features(calibration_x)], ignore_index=True)
        cal_values = self.imputer.transform(cal_rows)
        cal_exposure = np.tile(calibration_meta.total_fight_time_sec.clip(lower=1).to_numpy()/60, 2)
        for metric in self.metrics:
            target = np.r_[meta[f"r_{metric}"].to_numpy(), meta[f"b_{metric}"].to_numpy()].astype(float)
            valid = np.isfinite(target) & (target >= 0)
            model = HistGradientBoostingRegressor(loss="poisson", max_iter=self.iterations,
                max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=3.,
                learning_rate=.04, early_stopping=False, random_state=self.core.RANDOM_SEED)
            if not np.any(target[valid] > 0):
                model = DummyRegressor(strategy="constant", constant=0.)
            model.fit(values[valid], target[valid]/exposure[valid], sample_weight=exposure[valid])
            self.models[metric] = model
            counts = np.r_[calibration_meta[f"r_{metric}"], calibration_meta[f"b_{metric}"]].astype(float)
            means = np.maximum(model.predict(cal_values)*cal_exposure, 1e-8)
            valid = np.isfinite(counts) & (counts >= 0)
            boundary = event_boundary(calibration_meta.event_date, int(len(calibration_meta)*.65))
            early = valid & np.tile(np.arange(len(calibration_meta)) < boundary, 2)
            late = valid & ~np.tile(np.arange(len(calibration_meta)) < boundary, 2)
            if early.sum() < 10 or late.sum() < 10:
                early, late = valid, valid
                candidate = 0.
            else:
                candidate = float(np.clip(np.sum((counts[early]-means[early])**2-counts[early]) /
                                          max(np.sum(means[early]**2), 1e-8), 0, 100))
            p_score = float(-np.mean(poisson.logpmf(counts[late], means[late])))
            nb_score = (float(-np.mean(nbinom.logpmf(counts[late], 1/candidate,
                                                   1/(1+candidate*means[late])))) if candidate > 1e-8 else p_score)
            self.dispersions[metric] = candidate if nb_score < p_score-1e-4 else 0.
            self.dispersion_report[metric] = {"poisson_nll":p_score, "negative_binomial_nll":nb_score,
                                             "selected_dispersion":self.dispersions[metric]}
        return self

    def predict(self, x, durations, thresholds=None):
        thresholds = thresholds or {"td":[.5, 1.5, 2.5, 3.5, 4.5], "sig_str":[24.5, 49.5, 74.5, 99.5, 124.5]}
        result = [{"interpretation":"Exposure-mixture baseline; rate/finish-time dependence beyond pre-bout features is not modeled."} for _ in range(len(x))]
        for side, rows in (("a", x), ("b", self.core._swap_features(x))):
            values = self.imputer.transform(rows.reindex(columns=self.columns))
            for metric, model in self.models.items():
                rates = np.maximum(model.predict(values), 1e-8)
                alpha = self.dispersions[metric]
                for i, duration in enumerate(durations):
                    exposure = np.asarray(duration["duration_support_seconds"])/60
                    weights = np.asarray(duration["duration_support_weights"])
                    means = rates[i]*exposure
                    over = {}
                    for threshold in thresholds[metric]:
                        k = np.floor(threshold)
                        sf = nbinom.sf(k, 1/alpha, 1/(1+alpha*means)) if alpha > 1e-8 else poisson.sf(k, means)
                        over[str(threshold)] = float(np.dot(weights, sf))
                    result[i][f"{side}_{metric}"] = {
                        "expected_count":float(np.dot(weights, means)), "rate_per_minute":float(rates[i]),
                        "distribution":"negative_binomial_duration_mixture" if alpha > 1e-8 else "poisson_duration_mixture",
                        "dispersion":alpha, "over_probabilities":over}
        return result

    def log_probabilities(self, x, meta, durations):
        scores = {}
        for side, prefix, rows in (("a", "r", x), ("b", "b", self.core._swap_features(x))):
            values = self.imputer.transform(rows.reindex(columns=self.columns))
            for metric, model in self.models.items():
                rates = np.maximum(model.predict(values), 1e-8)
                alpha, logps = self.dispersions[metric], []
                for i, duration in enumerate(durations):
                    count = meta[f"{prefix}_{metric}"].iloc[i]
                    if not np.isfinite(count) or count < 0:
                        continue
                    means = rates[i]*np.asarray(duration["duration_support_seconds"])/60
                    pmf = (nbinom.pmf(count, 1/alpha, 1/(1+alpha*means)) if alpha > 1e-8
                           else poisson.pmf(count, means))
                    logps.append(np.log(max(np.dot(duration["duration_support_weights"], pmf), 1e-300)))
                scores[f"{side}_{metric}_nll"] = float(-np.mean(logps)) if logps else None
        return scores
