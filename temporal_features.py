"""One event-frozen feature state for historical and live UFC predictions."""

from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


class Moments:
    def __init__(self):
        self.values = defaultdict(lambda: [0, 0.0, 0.0])

    def add(self, key, value):
        if np.isfinite(value):
            n, mean, m2 = self.values[key]
            n += 1
            delta = value - mean
            mean += delta / n
            self.values[key] = [n, mean, m2 + delta * (value - mean)]

    def scale(self, key):
        n, mean, m2 = self.values.get(key, [0, 0.0, 0.0])
        std = np.sqrt(max(0.0, m2) / n) if n > 1 else 0.
        return mean, float(std) if std > 1e-6 else 1.0


class ContextPriors:
    def __init__(self):
        self.groups = {}
        self.finishes = 0
        self.total = 0

    @staticmethod
    def key(wc, gender, rounds, title):
        gender = str(gender).lower()
        if gender in ("female", "women"):
            wc = str(wc).replace("Women's ", "")
            gender = "women"
        return str(wc), gender, int(rounds), int(bool(title))

    def predict(self, wc, gender, rounds, title, when):
        key = self.key(wc, gender, rounds, title)
        fs, weight, count_finish, count, last = self.groups.get(key, (0., 0., 0, 0, when))
        decay = 2.0 ** (-max(0., (pd.Timestamp(when) - last).days) / 730.)
        prior = (self.finishes + 20 * .55) / (self.total + 20)
        return {"ctx_finish_prior_2y": (fs * decay + 20 * prior) / (weight * decay + 20),
                "ref_finish_prior": (count_finish + 20 * prior) / (count + 20)}

    def observe(self, row, method):
        if method not in ("Decision", "KO/TKO", "Submission"):
            return
        when = row["event_date"]
        key = self.key(row.get("weight_class", ""), row.get("gender", ""),
                       row.get("total_rounds", 3), row.get("is_title_bout", 0))
        fs, weight, fc, count, last = self.groups.get(key, (0., 0., 0, 0, when))
        decay = 2.0 ** (-max(0., (when - last).days) / 730.)
        finish = int(method != "Decision")
        self.groups[key] = fs * decay + finish, weight * decay + 1, fc + finish, count + 1, when
        self.finishes += finish
        self.total += 1


class TemporalState:
    feature_version = "event-causal-v1"

    def __init__(self, core):
        self.core = core
        self.histories = defaultdict(list)
        self.glicko = {}
        self.opponents = defaultdict(list)
        self.elo = {}
        self.div_elo = {}
        self.last_dates = {}
        self.elo_history = defaultdict(list)
        self.phase = ({}, {}, {}, {})
        self.style = core._StyleMatchupTracker({d: (.33, .66, 1.) for d, _ in core._STYLE_DIMS})
        self.mov_moments, self.phase_moments = Moments(), Moments()
        self.context = ContextPriors()
        self.location_altitude = {}
        self.camps = {}
        self.venue_observations, self.camp_observations = [], []
        self.cutoff = None
        self._alt_median, self._camp_median = 0., 0.

    @property
    def alt_median(self):
        return self._alt_median

    @property
    def camp_median(self):
        return self._camp_median

    def profile(self, name):
        if not self.histories.get(name):
            return {}
        result = {}
        for record in reversed(self.histories[name]):
            for key in ("height", "reach", "ape_index", "weight", "age", "stance", "date_of_birth"):
                value = record.get(key)
                if key not in result and value is not None and pd.notna(value) and value != "":
                    result[key] = value
        return result

    def snapshot(self, a, b, when, weight_class="", gender="", rounds=3, is_title=False,
                 location=None, event_elevation=None, profiles=None, camps=None):
        c = self.core
        when = pd.Timestamp(when).normalize()
        if self.cutoff is not None and when <= self.cutoff:
            raise ValueError("Prediction must follow the stored event history; use an earlier snapshot.")
        default = (c.MU_0, c.PHI_0, c.SIGMA_0)
        incoming = profiles or ({}, {})
        profiles = []
        for name, supplied in zip((a, b), incoming):
            profile = self.profile(name)
            profile.update({key:value for key,value in supplied.items()
                            if value is not None and pd.notna(value) and value != ""})
            profiles.append(profile)
        af = c.compute_fighter_features(self.histories.get(a, []), self.glicko.get(a, default),
                                       self.opponents.get(a, []), when, profiles[0])
        bf = c.compute_fighter_features(self.histories.get(b, []), self.glicko.get(b, default),
                                       self.opponents.get(b, []), when, profiles[1])
        base = c.compute_matchup_features(af, bf, int(is_title), rounds, weight_class, gender)
        base.update(self.style.matchup_features(a, b))
        explicit_alt = c._num_or(event_elevation, float("nan"))
        alt = (explicit_alt if np.isfinite(explicit_alt) else
               self.location_altitude.get(location, self.alt_median))
        ca, cb = camps if camps is not None else (self.camps.get(a), self.camps.get(b))
        ca, cb = (c._num_or(value, float("nan")) for value in (ca, cb))
        ca = ca if np.isfinite(ca) else self.camps.get(a, float("nan"))
        cb = cb if np.isfinite(cb) else self.camps.get(b, float("nan"))
        known = np.isfinite(ca) and np.isfinite(cb)
        base.update(c._altitude_feature_row(c._num_or(alt, self.alt_median),
            float(np.clip(ca, -500, 9000)) if np.isfinite(ca) else self.camp_median,
            float(np.clip(cb, -500, 9000)) if np.isfinite(cb) else self.camp_median, float(known)))
        base.update(self.context.predict(weight_class, gender, rounds, is_title, when))
        ra, rb = self.elo.get(a, c.ELO_BASE), self.elo.get(b, c.ELO_BASE)
        division = c._normalize_division(weight_class, gender)
        da, db = self.div_elo.get((a, division), c.ELO_BASE), self.div_elo.get((b, division), c.ELO_BASE)
        p = 1. / (1. + 10 ** (-(ra-rb)/400))
        dp = 1. / (1. + 10 ** (-(da-db)/400))
        sa, sb = c._elo_slope(self.elo_history.get(a, [])), c._elo_slope(self.elo_history.get(b, []))
        elo = dict(elo_r=ra, elo_b=rb, d_elo=ra-rb, elo_win_prob=p, d_elo_win_prob=p-.5,
                   abs_elo_gap=abs(ra-rb), elo_sum=ra+rb, div_elo_r=da, div_elo_b=db,
                   d_div_elo=da-db, div_elo_win_prob=dp, d_div_elo_win_prob=dp-.5,
                   abs_div_elo_gap=abs(da-db), elo_divergence=p-dp, elo_agreement=1-abs(p-dp),
                   r_elo_slope_5=sa, b_elo_slope_5=sb, d_elo_slope_5=sa-sb)
        elo.update(c._div_rank_feature_row(a, b, division, self.div_elo, self.last_dates, when))
        phase = c._phase_glicko_features(*[rating for mapping in self.phase
                                           for rating in (mapping.get(a, default), mapping.get(b, default))])
        full = dict(base, **elo)
        if c.PHASE_RATINGS_ENABLED:
            full.update(phase)
        full = c._augment_matchup_features(pd.DataFrame([full])).iloc[0].to_dict()
        return full, af, bf, base, elo, phase

    def observe_event(self, event, snapshots):
        """Called only after EVERY pre-event snapshot has been emitted."""
        c = self.core
        default = (c.MU_0, c.PHI_0, c.SIGMA_0)
        mov_scales = {key: self.mov_moments.scale(key)[1] for key in ("kd", "sig_pm", "ctrl_frac", "td")}
        phase_scales = {key: self.phase_moments.scale(key) for key in
                        ("strike_vol", "strike_acc", "kd_pm", "grap_td", "grap_ctrl", "grap_gnp", "grap_sub", "rev")}
        for row, snapshot in zip(event, snapshots):
            _, af, bf, _, _, _ = snapshot
            a, b, winner = row["r_name"], row["b_name"], row["winner"]
            ga, gb = self.glicko.get(a, default), self.glicko.get(b, default)
            result_a, result_b = (("W", "L") if winner == "Red" else
                                  ("L", "W") if winner == "Blue" else ("D", "D"))
            score_a, score_b = (c._mov_performance_score(row, mov_scales) if c.MOV_RATINGS_ENABLED
                                else (1., 0.) if winner == "Red" else (0., 1.) if winner == "Blue" else (.5, .5))
            for name, prefix, opp_prefix, result, rating, opp_rating, opponent in (
                (a, "r", "b", result_a, ga, gb, bf), (b, "b", "r", result_b, gb, ga, af)):
                record = c.extract_fight_record(row, prefix, opp_prefix, result, opp_rating[0])
                record["self_glicko"] = rating[0]
                for key, feature in (("opp_off_sig_str_pm", "sig_str_pm"),
                    ("opp_def_sig_str_pm", "def_sig_str_pm"), ("opp_off_td_p15", "td_p15"),
                    ("opp_td_def_p15", "def_td_p15"), ("opp_off_ctrl_pct", "ctrl_pct"),
                    ("opp_def_ctrl_pct", "def_ctrl_pct")):
                    record[key] = opponent.get(feature, float("nan"))
                self.histories[name].append(record)
                self.opponents[name].append(opp_rating[0])
            self.glicko[a] = c.glicko2_update(ga, [(gb[0], gb[1], score_a)])
            self.glicko[b] = c.glicko2_update(gb, [(ga[0], ga[1], score_b)])
            ra, rb = self.elo.get(a, c.ELO_BASE), self.elo.get(b, c.ELO_BASE)
            p = 1. / (1. + 10 ** (-(ra-rb)/400))
            self.elo[a], self.elo[b] = ra+c.ELO_K*(score_a-p), rb+c.ELO_K*(score_b-(1-p))
            division = c._normalize_division(row.get("weight_class", ""), row.get("gender", ""))
            da, db = self.div_elo.get((a, division), c.ELO_BASE), self.div_elo.get((b, division), c.ELO_BASE)
            p = 1. / (1. + 10 ** (-(da-db)/400))
            self.div_elo[(a, division)] = da+c.ELO_K*(score_a-p)
            self.div_elo[(b, division)] = db+c.ELO_K*(score_b-(1-p))
            for name in (a, b):
                self.last_dates[name] = row["event_date"]
                self.elo_history[name].append(self.elo[name])
            if c.PHASE_RATINGS_ENABLED and winner in ("Red", "Blue"):
                so, sd, go, gd = self.phase
                rsa, rsb, rga, rgb, sea, seb, gea, geb = c._phase_success(row, phase_scales)
                old = [mapping.get(name, default) for mapping in self.phase for name in (a, b)]
                for engaged, offense, defense, oname, dname, rating, opponent, score in (
                    (sea, so, sd, a, b, old[0], old[3], rsa),
                    (seb, so, sd, b, a, old[1], old[2], rsb),
                    (gea, go, gd, a, b, old[4], old[7], rga),
                    (geb, go, gd, b, a, old[5], old[6], rgb)):
                    if engaged:
                        offense[oname] = c.glicko2_update(rating, [(opponent[0], opponent[1], score)])
                        defense[dname] = c.glicko2_update(opponent, [(rating[0], rating[1], 1-score)])
            self.style.update_after_fight(a, b, winner, row)
            self.context.observe(row, c._normalize_method_label(row["method"]))
            altitude = c._num_or(row.get("event_altitude"), float("nan"))
            if np.isfinite(altitude):
                self.venue_observations.append(altitude)
                self.location_altitude[row.get("event_location", "")] = altitude
            for name, prefix in ((a, "r"), (b, "b")):
                value = c._num_or(row.get(f"{prefix}_training_altitude_ft"), float("nan"))
                if np.isfinite(value):
                    value = float(np.clip(value, -500, 9000))
                    self.camps[name] = value
                    self.camp_observations.append(value)
            self._observe_scales(row)
        self.cutoff = event[-1]["event_date"]
        self._alt_median = float(np.median(self.venue_observations)) if self.venue_observations else 0.
        self._camp_median = float(np.median(self.camp_observations)) if self.camp_observations else 0.
        # Bucket boundaries are learned only from the completed past.
        for dim, _ in c._STYLE_DIMS:
            values = [self.style._stat(name, dim) for name, history in self.histories.items() if len(history) >= 3]
            if len(values) >= 10:
                self.style.thresholds[dim] = tuple(np.quantile(values, [.25, .5, .75]))

    def _observe_scales(self, row):
        c = self.core
        f = lambda key: c._num_or(row.get(key), 0.)
        elapsed = max(f("total_fight_time_sec"), 1.)
        for key, value in dict(kd=f("r_kd")-f("b_kd"),
            sig_pm=(f("r_sig_str")-f("b_sig_str"))/(elapsed/60),
            ctrl_frac=(f("r_ctrl_sec")-f("b_ctrl_sec"))/elapsed,
            td=f("r_td")-f("b_td")).items():
            self.mov_moments.add(key, value)
        standing, ground = c._phase_time_split(elapsed, f("r_ctrl_sec"), f("b_ctrl_sec"))
        for p in ("r", "b"):
            for key, value in dict(
                strike_vol=f(f"{p}_sig_str")*(f(f"{p}_distance")+f(f"{p}_clinch"))/(standing/60),
                strike_acc=f(f"{p}_sig_str_acc"), kd_pm=f(f"{p}_kd")/(standing/60),
                grap_td=f(f"{p}_td")/max(elapsed/900, 1/30),
                grap_ctrl=f(f"{p}_ctrl_sec")/elapsed,
                grap_gnp=f(f"{p}_sig_str")*f(f"{p}_ground")/(ground/60),
                grap_sub=f(f"{p}_sub_att")/max(elapsed/900, 1/30), rev=f(f"{p}_rev")).items():
                self.phase_moments.add(key, value)

    def attach(self, pipeline):
        pipeline._temporal_state = self
        pipeline._feature_version = self.feature_version
        pipeline.fighter_history, pipeline.glicko_ratings = self.histories, self.glicko
        pipeline.opp_glicko_list = self.opponents
        pipeline.elo_ratings, pipeline.div_elo_ratings = self.elo, self.div_elo
        pipeline.last_fight_date, pipeline.elo_history = self.last_dates, self.elo_history
        (pipeline.strike_off_ratings, pipeline.strike_def_ratings,
         pipeline.grapple_off_ratings, pipeline.grapple_def_ratings) = self.phase
        pipeline.style_tracker = self.style
        pipeline.location_altitude, pipeline.alt_median = self.location_altitude, self.alt_median
        pipeline.fighter_train_alt, pipeline.train_alt_median = self.camps, self.camp_median


def build_frame(csv_path, core, progress=None, frame=None):
    data = pd.read_csv(csv_path) if frame is None else frame.copy()
    data["event_date"] = pd.to_datetime(data["event_date"], format="%m/%d/%Y")
    data = data.sort_values("event_date", kind="stable").reset_index(drop=True)
    state = TemporalState(core)
    rows, metadata, bases, elos, phases = [], [], [], [], []
    for date, event in data.groupby("event_date", sort=True):
        records = event.to_dict("records")
        snapshots = [state.snapshot(row["r_name"], row["b_name"], date,
            row.get("weight_class", ""), row.get("gender", ""), int(row.get("total_rounds", 3)),
            bool(row.get("is_title_bout", 0)), row.get("event_location"), row.get("event_altitude"),
            (core._extract_profile_from_row(row, "r"), core._extract_profile_from_row(row, "b")),
            (row.get("r_training_altitude_ft"), row.get("b_training_altitude_ft"))) for row in records]
        for row, snapshot in zip(records, snapshots):
            if row["winner"] in ("Red", "Blue"):
                full, _, _, base, elo, phase = snapshot
                rows.append(full); metadata.append(row); bases.append(base); elos.append(elo); phases.append(phase)
        state.observe_event(records, snapshots)
        if progress and len(rows) % 500 < len(event):
            progress(f"Causal features: {len(rows)} decisive rows through {date.date()}")
    meta = pd.DataFrame(metadata)
    return {"X": pd.DataFrame(rows), "meta": meta, "state": state,
            "y": (meta["winner"] == "Red").astype(int),
            "base": pd.DataFrame(bases), "elo": pd.DataFrame(elos), "phase": pd.DataFrame(phases)}


_MEMORY_CACHE = {}


def load_frame(csv_path, core, progress=None):
    source = Path(__file__).read_bytes() + Path(core.__file__).read_bytes()
    flags = [core.MOV_RATINGS_ENABLED, core.MOV_MODE, core.PHASE_RATINGS_ENABLED,
             core.OBA_FEATURES_ENABLED, core.CORNER_CORRECTION_ENABLED,
             core.WINNER_SEES_ALL_FEATURES, core.WINNER_COMBINER_ROBUST]
    key = hashlib.sha256(Path(csv_path).read_bytes() + source + repr(flags).encode()).hexdigest()
    if key in _MEMORY_CACHE:
        return _MEMORY_CACHE[key]
    import cloudpickle
    path = Path(core.CACHE_DIR) / f"causal_features_{key[:20]}.pkl"
    if path.exists():
        with path.open("rb") as stream:
            bundle = cloudpickle.load(stream)
    else:
        bundle = build_frame(csv_path, core, progress)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            cloudpickle.dump(bundle, stream)
        temporary.replace(path)
    _MEMORY_CACHE.clear()
    _MEMORY_CACHE[key] = bundle
    return bundle
