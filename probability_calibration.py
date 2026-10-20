"""Chronological calibration and market blending; no holdout fitting."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss


def event_boundary(dates, target_start):
    dates = np.asarray(pd.to_datetime(dates))
    if not len(dates):
        return 0
    target_start = min(max(int(target_start), 0), len(dates)-1)
    return int(np.searchsorted(dates, dates[target_start], side="left"))


def last_event_window(dates, target_rows):
    return event_boundary(dates, max(0, len(dates)-int(target_rows)))


def probability_metrics(y, probabilities):
    y, p = np.asarray(y, dtype=int), np.clip(np.asarray(probabilities), 1e-6, 1-1e-6)
    bins = np.minimum((p*10).astype(int), 9)
    ece = sum(np.mean(bins == b)*abs(np.mean(y[bins == b])-np.mean(p[bins == b]))
              for b in range(10) if np.any(bins == b))
    return dict(rows=int(len(y)), accuracy=float(np.mean((p >= .5) == y)),
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                brier=float(brier_score_loss(y, p)), ece=float(ece))


class SymmetricPlatt:
    def __init__(self, slope=1.):
        self.slope = float(slope)

    def fit(self, probabilities, outcomes):
        x = logit(np.clip(probabilities, 1e-6, 1-1e-6))
        y = np.asarray(outcomes, dtype=int)
        fit = minimize_scalar(lambda slope: log_loss(y, expit(slope*x), labels=[0, 1]),
                              bounds=(.1, 5.), method="bounded")
        self.slope = float(fit.x)
        return self

    def predict_proba(self, x):
        p = expit(self.slope*logit(np.clip(np.asarray(x).reshape(-1), 1e-6, 1-1e-6)))
        return np.column_stack([1-p, p])


class SymmetricIsotonic:
    def fit(self, probabilities, outcomes):
        p, y = np.asarray(probabilities), np.asarray(outcomes)
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1-1e-6)
        self.model.fit(np.r_[p, 1-p], np.r_[y, 1-y])
        return self

    def predict_proba(self, x):
        p = np.asarray(x).reshape(-1)
        calibrated = (self.model.predict(p) + 1-self.model.predict(1-p))/2
        return np.column_stack([1-calibrated, calibrated])


def select_binary_calibrator(p_fit, y_fit, p_select, y_select):
    candidates = [("none", None, probability_metrics(y_select, p_select)["log_loss"])]
    if len(p_fit) >= 30 and len(np.unique(y_fit)) == 2:
        for name, estimator in (("symmetric_platt", SymmetricPlatt()),
                                 ("symmetric_isotonic", SymmetricIsotonic())):
            estimator.fit(p_fit, y_fit)
            pred = estimator.predict_proba(np.asarray(p_select).reshape(-1, 1))[:, 1]
            candidates.append((name, estimator, probability_metrics(y_select, pred)["log_loss"]))
    best = min(candidates, key=lambda row: row[2])
    if best[2] > candidates[0][2]-1e-4:
        best = candidates[0]
    return best[1], {"selected": best[0], "selection_metric": "log_loss",
                     "fit_rows":len(p_fit), "selection_rows":len(p_select),
                     "candidates":{name:score for name, _, score in candidates}}


class TemperatureCalibration:
    def __init__(self):
        self.temperature = 1.

    def predict(self, probabilities):
        return softmax(np.log(np.clip(probabilities, 1e-8, 1))/self.temperature, axis=1)

    def fit_select(self, fit_p, fit_y, select_p, select_y):
        classes = np.arange(fit_p.shape[1])
        fit = minimize_scalar(lambda t: log_loss(fit_y, softmax(np.log(np.clip(fit_p, 1e-8, 1))/t, axis=1),
                                                   labels=classes), bounds=(.5, 4.), method="bounded")
        candidate = float(fit.x)
        original = log_loss(select_y, select_p, labels=classes)
        adjusted = log_loss(select_y, softmax(np.log(np.clip(select_p, 1e-8, 1))/candidate, axis=1), labels=classes)
        self.temperature = candidate if adjusted < original-1e-4 else 1.
        return {"temperature":self.temperature, "identity_log_loss":float(original),
                "candidate_log_loss":float(adjusted), "fit_rows":len(fit_y), "selection_rows":len(select_y)}


class MarketBlender:
    """Fits only explicit historical samples; new instances do not blend."""
    def __init__(self):
        self.market_weight = None
        self.training_cutoff = None
        self.validation = None

    @property
    def is_fitted(self):
        return self.market_weight is not None

    def predict(self, independent, market):
        if not self.is_fitted:
            return np.asarray(independent)
        a = self.market_weight
        return expit((1-a)*logit(np.clip(independent, 1e-6, 1-1e-6))
                     + a*logit(np.clip(market, 1e-6, 1-1e-6)))

    def fit(self, samples, minimum_rows=100):
        required = {"event_date", "generated_at", "observed_at", "event_start", "probability", "market_probability", "outcome"}
        if not required <= set(samples):
            raise ValueError("Market blending requires timestamped historical forecast/odds samples.")
        frame = samples.sort_values("event_date", kind="stable").copy()
        for column in ("generated_at", "observed_at", "event_start"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        if (frame.observed_at > frame.generated_at).any() or (frame.generated_at >= frame.event_start).any():
            raise ValueError("Odds and forecasts must have been available before the event.")
        if len(frame) < minimum_rows:
            raise ValueError(f"At least {minimum_rows} historical market samples are required.")
        if not frame.outcome.isin([0, 1]).all() or not frame[["probability", "market_probability"]].map(lambda x: np.isfinite(x) and 0<x<1).all().all():
            raise ValueError("Invalid market sample probabilities or outcomes.")
        split = event_boundary(frame.event_date, int(len(frame)*.7))
        if split < 20 or len(frame)-split < 20:
            raise ValueError("Need separate calibration and later validation events.")
        train, validation = frame.iloc[:split], frame.iloc[split:]
        x, m = logit(train.probability), logit(train.market_probability)
        fit = minimize_scalar(lambda a: log_loss(train.outcome, expit((1-a)*x+a*m), labels=[0, 1]),
                              bounds=(0., 1.), method="bounded")
        self.market_weight = float(fit.x)
        combined = self.predict(validation.probability, validation.market_probability)
        scores = {"independent":probability_metrics(validation.outcome, validation.probability),
                  "market":probability_metrics(validation.outcome, validation.market_probability),
                  "combined":probability_metrics(validation.outcome, combined)}
        if scores["combined"]["log_loss"] > scores["independent"]["log_loss"]-1e-4:
            self.market_weight = 0.
        self.training_cutoff = pd.Timestamp(frame.event_date.max()).isoformat()
        self.validation = scores
        return self
