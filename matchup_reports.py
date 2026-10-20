"""Local winner-model explanations and readable matchup reports.

Uses exact grouped Shapley values against a corner-neutral feature reference.
No attribution dependency, training, or changes to the learned probabilities.
"""

import math
import re
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_LABELS = {
    "ratings": "Ratings and opposition",
    "striking": "Striking output and defense",
    "wrestling": "Wrestling and control",
    "finishing": "Finishing and durability",
    "cardio": "Pace and cardio",
    "form": "Recent form and activity",
    "experience": "Experience and results",
    "physical": "Age, size and stance",
    "interactions": "Matchup interactions",
    "other": "Other inputs",
}

COMPARISONS = (
    ("num_fights", "Recorded fights", "count"),
    ("sig_str_pm", "Significant strikes landed/min", "rate"),
    ("def_sig_str_pm", "Significant strikes absorbed/min", "rate"),
    ("td_p15", "Takedowns landed/15 min", "rate"),
    ("td_defense_rate", "Takedown defense", "percent"),
    ("ctrl_pct", "Control-time share", "percent"),
    ("sub_att_p15", "Submission attempts/15 min", "rate"),
    ("last3_win_rate", "Smoothed last-three win rate", "percent"),
    ("cardio_ratio", "Historical cardio ratio", "rate"),
    ("glicko_mu", "Glicko rating", "rate"),
    ("age", "Age at prediction date", "rate"),
    ("reach", "Reach (inches)", "rate"),
    ("days_inactive", "Inactivity feature (model input, days)", "count"),
)


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _feature_group(name):
    """Keep raw corner pairs together; expose grouping in the JSON report."""
    if re.search(r"_x_|interaction|synergy|hybrid|poly2|vs_defense|vs_tdd|"
                 r"striker_grappler|tdd_vs_td|wr_vs_", name):
        return "interactions"
    if re.search(r"glicko|elo|div_rank|opp_quality|quality_win|best_win|worst_loss", name):
        return "ratings"
    if re.search(r"(^|_)age($|_)|reach|height|(^|_)stance($|_)|orthodox|southpaw|switch|ape_index|d_weight", name):
        return "physical"
    if re.search(r"finish|(^|_)ko|sub_|_sub|durab|damage|(^|_)kd|decision|dec_win|resistance", name):
        return "finishing"
    if re.search(r"cardio|late_|rd[1-5]|round|intensity|pace|time_min", name):
        return "cardio"
    if re.search(r"(^|_)td|tdd|ctrl|control|grappl|rev|ground|wrestl", name):
        return "wrestling"
    if re.search(r"str_|strik|head|leg|body|distance|clinch|acc|defense|exchange|diversity", name):
        return "striking"
    if re.search(r"recent|last[135]|form|streak|momentum|inactive|activity|short_notice|"
                 r"recovery|fights_last_year|days_since|fights_per_year", name):
        return "form"
    if re.search(r"fight|experience|win_rate|loss|consistency|stability|exp_", name):
        return "experience"
    return "other"


def _predict_rows(ensemble, rows, core):
    """The same forward/reverse, blend, calibration and clipping as inference."""
    if hasattr(ensemble, "predict_proba_rows"):
        return ensemble.predict_proba_rows(rows, return_components=True)
    prepared, scaled = ensemble._prepare(rows)
    forward = ensemble._predict_all(prepared, scaled)
    swapped, swapped_scaled = ensemble._prepare(core._swap_features(rows))
    reverse = ensemble._predict_all(swapped, swapped_scaled)
    components = pd.DataFrame({
        name: (np.asarray(forward[name]) + 1.0 - np.asarray(reverse[name])) / 2.0
        for name in ensemble.model_order
    }, columns=ensemble.model_order)
    probs = np.asarray(core._combine_probs(components, ensemble.combiner), dtype=float)
    if ensemble.calibrator is not None:
        probs = ensemble.calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
    if ensemble.corner_correction:
        probs = core._apply_corner_correction(probs, ensemble.corner_correction)
    return np.clip(probs, 1e-6, 1 - 1e-6), components


def explain_winner(ensemble, feature_row, core):
    """Exact Shapley allocation over <=10 groups, for the stated reference.

    The reference averages the original and corner-swapped feature vectors.
    It preserves symmetric context while removing directional differences.
    Coalition rows are synthetic model inputs, not physically simulated fights.
    """
    actual = feature_row.reindex(columns=ensemble.feat_cols).astype(float)
    if len(actual) != 1:
        raise ValueError("An explanation requires exactly one matchup row.")
    reference = (actual + core._swap_features(actual)) / 2.0
    positions = {}
    for index, name in enumerate(actual.columns):
        positions.setdefault(_feature_group(name), []).append(index)
    # Include unchanged inputs in the audit, but do not evaluate dummy groups.
    active = {
        group: indices for group, indices in positions.items()
        if not np.allclose(actual.iloc[:, indices], reference.iloc[:, indices],
                           rtol=0, atol=0, equal_nan=True)
    }
    groups = list(active)
    count = len(groups)
    coalitions = np.repeat(reference.to_numpy(), 1 << count, axis=0)
    values = actual.to_numpy()[0]
    masks = np.arange(1 << count)
    for bit, group in enumerate(groups):
        selected = np.flatnonzero((masks & (1 << bit)) != 0)
        indices = active[group]
        coalitions[np.ix_(selected, indices)] = values[indices]
    rows = pd.DataFrame(coalitions, columns=actual.columns)
    probabilities, components = _predict_rows(ensemble, rows, core)
    contributions = {group: 0.0 for group in positions}
    for bit, group in enumerate(groups):
        for mask in range(1 << count):
            if mask & (1 << bit):
                continue
            size = mask.bit_count()
            weight = 1.0 / (count * math.comb(count - 1, size))
            contributions[group] += weight * (
                probabilities[mask | (1 << bit)] - probabilities[mask]
            )
    factors = [{
        "group": group, "label": GROUP_LABELS[group],
        "contribution_probability_a": float(contributions[group]),
        "feature_names": [actual.columns[i] for i in indices],
    } for group, indices in positions.items()]
    factors.sort(key=lambda item: abs(item["contribution_probability_a"]), reverse=True)
    baseline, probability = float(probabilities[0]), float(probabilities[-1])
    residual = probability - baseline - sum(contributions.values())
    if abs(residual) > 1e-10:
        raise RuntimeError("Explanation contributions did not reconstruct the prediction.")
    return {
        "method": "exact_grouped_shapley",
        "reference": "Mean of the actual and corner-swapped feature vectors",
        "scope": "Raw winner model, including any fitted calibration and corner correction; excludes market blending",
        "baseline_probability_a": baseline,
        "model_probability_a": probability,
        "reconstruction_residual": float(residual),
        "evaluated_coalitions": int(len(rows)),
        "factors": factors,
        "limitations": (
            "Contributions depend on the chosen groups and reference. Synthetic combinations "
            "of correlated features may be unlike real fights. They explain model behavior, "
            "not causal advantages or calibrated uncertainty. Method-of-victory factors are not attributed."
        ),
    }, {name: float(components.iloc[-1][name]) for name in ensemble.model_order}


def _votes(ensemble, probabilities, name_a, name_b):
    weighted = ensemble.combiner["kind"] == "weighted"
    weights = ensemble.combiner.get("weights", {}) if weighted else {}
    rows = []
    for name, probability in probabilities.items():
        weight = float(weights.get(name, 0.0)) if weighted else None
        active = weight > 1e-12 if weighted else True
        rows.append({
            "model": name, "probability_a": probability,
            "pick": name_a if probability >= 0.5 else name_b,
            "ensemble_weight": weight, "active": active,
        })
    active_rows = [row for row in rows if row["active"]]
    if not active_rows:
        raise ValueError("No active ensemble members for disagreement reporting.")
    w = np.array([r["ensemble_weight"] if weighted else 1.0 for r in active_rows])
    w /= w.sum()
    p = np.array([r["probability_a"] for r in active_rows])
    mean = float(np.sum(w * p))
    return rows, {
        "active_models": len(active_rows),
        "minimum_probability_a": float(p.min()),
        "maximum_probability_a": float(p.max()),
        "weighted_standard_deviation": float(np.sqrt(np.sum(w * (p - mean) ** 2))),
        "weight_picking_a": float(np.sum(w[p >= 0.5])),
        "weighting": "ensemble weights" if weighted else "equal weights for descriptive comparison; combiner is nonlinear",
        "interpretation": "Spread between correlated component models before final calibration; not a confidence interval.",
    }


def _coverage(pipeline, name, features, date):
    history = pipeline.fighter_history.get(name, [])
    last_date = max((pd.Timestamp(h["date"]) for h in history), default=None)
    missing = [key for key in ("height", "reach", "age", "weight")
               if _number(features.get(key)) is None]
    flags = []
    if not history:
        flags.append("No recorded fighter history; prediction uses generic debutant priors.")
    elif len(history) < 3:
        flags.append(f"Only {len(history)} recorded fight(s); limited evidence.")
    if missing:
        flags.append("Missing profile fields: " + ", ".join(missing) + ".")
    if last_date is not None:
        profile = pipeline._temporal_state.profile(name) if getattr(pipeline, "_temporal_state", None) else history[-1]
        if pd.isna(pd.to_datetime(profile.get("date_of_birth"), errors="coerce")):
            flags.append("Birth date unavailable; age advances a dated observation and is approximate.")
        actual_days = int((date - last_date).days)
        model_days = _number(features.get("days_inactive"))
        if model_days is not None and not math.isclose(model_days, actual_days, abs_tol=1):
            flags.append(f"Inactivity input is {model_days:.0f} days; recorded dates imply "
                         f"{actual_days} days. Date handling needs review.")
    camp = _number(pipeline.fighter_train_alt.get(name))
    if camp is None:
        flags.append("Training-camp elevation unavailable; model uses its fallback.")
    return {
        "name": name, "recorded_fights": len(history),
        "last_fight_date": last_date.date().isoformat() if last_date is not None else None,
        "days_since_last_fight": int((date - last_date).days) if last_date is not None else None,
        "missing_profile_fields": missing, "camp_elevation_ft": camp, "flags": flags,
    }


def _fair_odds(probability):
    return {
        "decimal": 1.0 / probability,
        "american": (-100.0 * probability / (1 - probability) if probability > 0.5
                     else 100.0 * (1 - probability) / probability),
    }


def build_matchup_report(pipeline, feature_row, prediction, a_features, b_features,
                         prediction_date, rounds, is_title, location, event_elevation, core):
    ensemble = pipeline.model
    attribution, components = explain_winner(ensemble, feature_row, core)
    if not math.isclose(attribution["model_probability_a"], prediction["prob_a_model"], abs_tol=1e-10):
        raise RuntimeError("Explanation and prediction probabilities differ.")
    name_a, name_b = prediction["name_a"], prediction["name_b"]
    votes, disagreement = _votes(ensemble, components, name_a, name_b)
    missing = [name for name in ensemble.feat_cols
               if name not in feature_row or _number(feature_row.iloc[0][name]) is None]
    method_inputs = set(ensemble.method_feat_cols)
    absent_priors = [key for key in ("ctx_finish_prior_2y", "ref_finish_prior")
                    if key in method_inputs and key not in feature_row]
    flags = []
    if missing:
        flags.append(f"{len(missing)} of {len(ensemble.feat_cols)} winner inputs use imputation.")
    if absent_priors:
        flags.append("Live context finish priors are absent and imputed by the method model.")
    if "stage1" in ensemble.method_bundle and "stage1_cols" not in ensemble.method_bundle:
        flags.append("Production method refit uses a different feature selection from evaluation.")
    if ensemble.calibrator is None:
        flags.append("The model-selection procedure retained no probability calibrator.")
    if prediction.get("market_blending_status", "").startswith("unfitted") and prediction.get("market_prob_a") is not None:
        flags.append("Market blending lacks historical calibration; the independent probability is retained.")
    if str(prediction["gender"]).lower() in ("women", "female") and feature_row.iloc[0].get("gender_flag") == 0:
        flags.append("Gender context is inferred from an unprefixed weight-class label; feature handling needs review.")
    known_location = _number(pipeline.location_altitude.get(location)) if location else None
    altitude = (_number(event_elevation) if event_elevation is not None else
                known_location if known_location is not None else _number(pipeline.alt_median))
    altitude_source = ("explicit" if event_elevation is not None else
                       "location lookup" if known_location is not None else "dataset fallback")
    if altitude_source == "dataset fallback":
        flags.append("Venue elevation uses the dataset fallback.")
    for factor in attribution["factors"]:
        contribution = factor["contribution_probability_a"]
        factor["favors"] = name_a if contribution > 1e-10 else name_b if contribution < -1e-10 else None
    break_even_a = core._american_odds_to_prob(prediction["red_odds"])
    break_even_b = core._american_odds_to_prob(prediction["blue_odds"])
    market_a = prediction["market_prob_a"]
    return {
        "schema_version": 1,
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prediction_date": prediction_date.date().isoformat(),
        "fighters": [name_a, name_b],
        "bout_context": {"rounds": int(rounds), "is_title": is_title,
                         "weight_class": prediction["weight_class"], "gender": prediction["gender"],
                         "location": location, "event_elevation_ft": altitude,
                         "event_elevation_source": altitude_source},
        "comparison": [{"feature": key, "label": label, "format": fmt,
                        "a": _number(a_features.get(key)), "b": _number(b_features.get(key))}
                       for key, label, fmt in COMPARISONS],
        "coverage": [_coverage(pipeline, name_a, a_features, prediction_date),
                     _coverage(pipeline, name_b, b_features, prediction_date)],
        "winner_inputs": {"selected": len(ensemble.feat_cols), "imputed": missing},
        "model_votes": votes, "disagreement": disagreement,
        "winner_attribution": attribution,
        "fair_odds_model_only": {name_a: _fair_odds(prediction["prob_a_model"]),
                                  name_b: _fair_odds(prediction["prob_b_model"])},
        "market": {"odds_blended": prediction["odds_blended"],
                   "american_odds_a": prediction["red_odds"],
                   "american_odds_b": prediction["blue_odds"],
                   "no_vig_probability_a": market_a,
                   "break_even_probability_a": break_even_a,
                   "break_even_probability_b": break_even_b,
                   "model_edge_vs_no_vig_a": (prediction["prob_a_model"] - market_a
                                               if market_a is not None else None),
                   "expected_return_a": (prediction["prob_a_model"] / break_even_a - 1
                                          if break_even_a else None),
                   "expected_return_b": (prediction["prob_b_model"] / break_even_b - 1
                                          if break_even_b else None),
                   "final_probability_a": prediction["prob_a"],
                   "market_blend_change_a": prediction["prob_a"] - prediction["prob_a_model"]},
        "flags": flags,
        "method_interpretation": ("Method distribution is conditional on the selected winner; joint compound outcomes are reported separately. Draw/NC/DQ are not priced."
                                  if prediction.get("joint_outcome_probs") else
                                  "Method distribution is oriented around the selected winner; it is not a joint winner-and-method price."),
    }


def report_summary(report):
    """Compact text for the workbook, also useful in other clients."""
    attribution = report["winner_attribution"]
    lines = []
    for name in report["fighters"]:
        factors = [factor for factor in attribution["factors"] if factor["favors"] == name][:3]
        text = "; ".join(f"{factor['label']} ({abs(factor['contribution_probability_a'])*100:.1f} pp)"
                         for factor in factors)
        lines.append(f"Factors favoring {name}: {text or 'No material positive group contributions.'}")
    d = report["disagreement"]
    lines.append(f"Active-model range for {report['fighters'][0]}: "
                 f"{d['minimum_probability_a']:.1%}–{d['maximum_probability_a']:.1%}; "
                 f"spread {d['weighted_standard_deviation']*100:.1f} pp. Not a confidence interval.")
    for name,odds in report["fair_odds_model_only"].items():
        lines.append(f"Model fair odds for {name}: {odds['american']:+.0f} (decimal {odds['decimal']:.2f}).")
    market = report["market"]
    if market["no_vig_probability_a"] is not None:
        lines.append(f"Market no-vig first-fighter probability: {market['no_vig_probability_a']:.1%}; "
                     f"model-implied returns at supplied prices: first {market['expected_return_a']:+.1%}, "
                     f"second {market['expected_return_b']:+.1%}.")
    for coverage in report["coverage"]:
        lines.append(f"{coverage['name']}: {coverage['recorded_fights']} recorded fights. "
                     + " ".join(coverage["flags"]))
    lines.extend(report["flags"])
    lines.append(attribution["limitations"])
    return "\n".join(lines)


def _display(value, fmt="rate"):
    if value is None:
        return "Unavailable"
    return f"{value:.1%}" if fmt == "percent" else f"{value:.0f}" if fmt == "count" else f"{value:.2f}"


def render_card_markdown(card):
    lines = [f"# {card.get('event', 'Fight card')} — matchup explanations", "",
             f"Prediction date: {card['prediction_date']}. Data cutoff: {card['data_cutoff']}.", ""]
    for bout in card["bouts"]:
        lines += [f"## {' vs. '.join(bout['fighters'])}", ""]
        if "prediction" not in bout:
            lines += [bout["status"], ""]
            continue
        prediction = bout["prediction"]
        report = prediction["explanation"]
        a, b = report["fighters"]
        context = report["bout_context"]
        lines += [f"{context['rounds']} rounds; title bout: {'yes' if context['is_title'] else 'no'}; "
                  f"location: {context['location'] or 'unspecified'}. "
                  f"Venue elevation: {_display(context['event_elevation_ft'])} ft ({context['event_elevation_source']}).", ""]
        lines += [f"Model: **{a} {prediction['prob_a_model']:.1%} / {b} {prediction['prob_b_model']:.1%}**.", ""]
        odds = report["fair_odds_model_only"]
        lines += [f"Model fair odds: {a} {odds[a]['american']:+.0f} "
                  f"(decimal {odds[a]['decimal']:.2f}); "
                  f"{b} {odds[b]['american']:+.0f} (decimal {odds[b]['decimal']:.2f}).", ""]
        if prediction["market_prob_a"] is not None:
            if prediction["odds_blended"]:
                lines += [f"After odds blending: {a} {prediction['prob_a']:.1%} / {b} {prediction['prob_b']:.1%}.", ""]
            else:
                lines += ["Historical market calibration is unavailable; the independent prediction is retained.", ""]
            market = report["market"]
            lines += [f"At supplied prices, model-implied expected return per unit staked: "
                      f"{a} {market['expected_return_a']:+.1%}; {b} {market['expected_return_b']:+.1%}. "
                      "These estimates depend on probability accuracy; no historical profitability is established.", ""]
        lines += ["Method distribution around the selected winner: " + ", ".join(
            f"{key} {value:.1%}" for key, value in prediction["method_probs"].items()) + ".", "",
            *(["| Joint outcome | Probability |", "|---|---:|",
               *[f"| {key} | {value:.1%} |" for key,value in prediction["joint_outcome_probs"].items()], "",
               prediction["outcome_population"], ""] if prediction.get("joint_outcome_probs") else []),
            f"| Statistical comparison | {a} | {b} |", "|---|---:|---:|"]
        for row in report["comparison"]:
            lines.append(f"| {row['label']} | {_display(row['a'], row['format'])} | {_display(row['b'], row['format'])} |")
        lines += ["", "| Model factor group | Effect on first fighter's probability | Favors |", "|---|---:|---|"]
        for factor in report["winner_attribution"]["factors"]:
            lines.append(f"| {factor['label']} | {factor['contribution_probability_a']*100:+.2f} pp | {factor['favors'] or 'Neither'} |")
        attr = report["winner_attribution"]
        lines += ["", f"Reference {attr['baseline_probability_a']:.2%} + group contributions = "
                  f"model {attr['model_probability_a']:.2%}.", "", attr["limitations"], "",
                  f"| Component model | {a} win probability | Blend weight |", "|---|---:|---:|"]
        for vote in report["model_votes"]:
            if vote["active"]:
                weight = "Nonlinear blend" if vote["ensemble_weight"] is None else f"{vote['ensemble_weight']:.1%}"
                lines.append(f"| {vote['model']} | {vote['probability_a']:.1%} | {weight} |")
        d = report["disagreement"]
        lines += ["", f"Active-model range: {d['minimum_probability_a']:.1%}–{d['maximum_probability_a']:.1%}; "
                  f"weighted spread {d['weighted_standard_deviation']*100:.1f} pp. **Not a confidence interval.**", ""]
        for coverage in report["coverage"]:
            lines += [f"**{coverage['name']}**: {coverage['recorded_fights']} recorded fights; "
                      f"last recorded bout {coverage['last_fight_date'] or 'unavailable'}.", ""]
            lines += [f"- {flag}" for flag in coverage["flags"]] + [""]
        lines += [f"- {flag}" for flag in report["flags"]] + ["", report["method_interpretation"], ""]
        if prediction.get("duration_forecast"):
            duration, props = prediction["duration_forecast"], prediction["statistical_props"]
            lines += [f"Expected duration **{duration['expected_seconds']/60:.1f} minutes**; "
                      f"goes distance **{duration['goes_distance_probability']:.1%}**.", "",
                      "| Total rounds | Probability of over |", "|---|---:|"]
            lines += [f"| {threshold} | {probability:.1%} |" for threshold,probability in duration["over_rounds_probabilities"].items()]
            lines += ["", "| Statistical projection | Expected count |", "|---|---:|"]
            for side,name in (("a",a),("b",b)):
                for metric,label in (("td","takedowns"),("sig_str","significant strikes")):
                    lines.append(f"| {name} {label} | {props[f'{side}_{metric}']['expected_count']:.1f} |")
            lines += ["", duration["interpretation"], props["interpretation"], ""]
    return "\n".join(lines)


def render_evaluation_markdown(report):
    folds = report["folds"]
    lines = ["# Corrected model — chronological evaluation", "",
        report["evaluation_status"], "", report["outcome_population"], "",
        "All feature construction is event-frozen. Base-model fitting, blending, "
        "calibration, conditional methods, duration, and count models are repeated before each test period.", "",
        "| Test period | Fights | Winner accuracy | Winner log-loss | Uncalibrated log-loss | Elo log-loss | Joint accuracy | Joint log-loss | Duration MAE (minutes) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for fold in folds:
        lines.append(f"| {fold['test_start']}–{fold['test_end']} | {fold['rows']} | {fold['accuracy']:.1%} | "
            f"{fold['log_loss']:.4f} | {fold['uncalibrated']['log_loss']:.4f} | {fold['elo_baseline']['log_loss']:.4f} | "
            f"{fold['joint_accuracy']:.1%} | {fold['joint_log_loss']:.4f} | {fold['duration_mae_seconds']/60:.2f} |")
    if folds:
        total = sum(fold["rows"] for fold in folds)
        mean = lambda key:sum(fold[key]*fold["rows"] for fold in folds)/total
        lines += ["", f"Across {total} test fights: winner accuracy **{mean('accuracy'):.1%}**, "
            f"log-loss **{mean('log_loss'):.4f}**, joint winner/method accuracy **{mean('joint_accuracy'):.1%}**.", ""]
    lines += ["## Calibration and production contract", "",
        "The production base weights are fitted before a recent calibration window. "
        "Separate event groups fit ensemble weights, fit calibration, and select calibration by log-loss. "
        "The base weights are not subsequently refitted on calibration outcomes.", "",
        f"Production calibration: **{report['production']['calibration']['selected']}**.", "",
        "The direct six-class classifier is a separately evaluated challenger. Its test scores do not select "
        "the deployed model. Production uses the calibrated winner ensemble and correctly conditioned methods.", "",
        "Joint accuracy scores the most likely of six compound outcomes. Its fighter can differ "
        "from the overall winner pick because the other fighter's win probability is spread across several methods.", "",
        "## Market data", "", report["market_evaluation"]["reason"], "",
        "No historical return, closing-line value, or fitted market blending weight is claimed. "
        "When prices are supplied before historical forecasts, the append-only market tools can evaluate "
        "the declared paper strategy and compare paired no-vig bookmaker baselines.", "",
        "## Interpretation", "",
        "These periods were already available during development. Treat these scores as retrospective diagnostics, "
        "and retain prospective forecasts before future fights for an independent confirmation. "
        "The original published benchmark uses a different dataset, feature construction, and split; "
        "its headline figures are not a controlled comparison with this run.", "",
        "Count models are initial exposure mixtures; dependence between finish timing and strike/takedown rates "
        "beyond pre-fight features is not modeled. Duration uses 30-second intervals, reconciled to joint cause probabilities.", ""]
    return "\n".join(lines)
