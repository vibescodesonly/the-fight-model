"""Generate JSON, Markdown and Excel matchup explanations from a trusted model.

Example (after preparing a serialized pipeline and its metadata):
  python scripts/predict_card_reports.py results/ufc333-card.json /tmp/reports.json \
      --model /path/trained-pipeline.pkl --metadata /path/trained-pipeline-meta.json

For an existing model trained before the report-only changes, pass its original
source snapshot with --training-source. Training definitions must still match.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cloudpickle
import UFC_ML_Model as core
from matchup_reports import render_card_markdown


FLAGS = ("UFC_MOV_ENABLED", "UFC_MOV_MODE", "UFC_PHASE_RATINGS", "UFC_OBA_ENABLED",
         "UFC_CORNER_CORRECTION", "UFC_WINNER_ALL_FEATURES", "UFC_COMBINER_ROBUST")


def _training_ast(source):
    tree = ast.parse(source)
    # These presentation/inference entry points can change without retraining.
    # Everything else, including global settings and ALL training definitions,
    # must remain identical to the attested original training source.
    tree.body = [node for node in tree.body if not (
        isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in {"export_to_excel", "SuperModelGUI"})]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "UFCSuperModelPipeline":
            node.body = [child for child in node.body if not (
                isinstance(child, ast.FunctionDef) and child.name == "predict_matchup")]
    return ast.dump(tree, include_attributes=False)


def validate_model_metadata(metadata, training_source, data, current_source):
    fingerprint = hashlib.sha256(training_source + data).hexdigest()
    if metadata.get("source_data_sha256") != fingerprint:
        raise ValueError("Model metadata does not match the provided training source and current dataset.")
    flags = {key: os.environ[key] for key in FLAGS if key in os.environ}
    if metadata.get("ufc_environment", {}) != flags:
        raise ValueError("Model feature configuration changed; rebuild the model.")
    if metadata.get("format") == "causal-research-v1":
        from research_pipeline import source_fingerprint
        if metadata.get("source_sha256") != source_fingerprint(Path(core.__file__).parent):
            raise ValueError("Corrected model source changed; rebuild before reporting.")
        if metadata.get("dataset_sha256") != hashlib.sha256(data).hexdigest():
            raise ValueError("Corrected model dataset changed; rebuild before reporting.")
    if _training_ast(training_source) != _training_ast(current_source):
        raise ValueError("Training code or configuration changed; rebuild the model.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card", type=Path)
    parser.add_argument("output", type=Path, help="JSON output; matching .md and .xlsx files are also written")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--training-source", type=Path, default=Path(core.__file__))
    parser.add_argument("--prediction-date", default=None, help="YYYY-MM-DD; defaults to today")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("Output must have a .json suffix.")
    card = json.loads(args.card.read_text())
    metadata = json.loads(args.metadata.read_text())
    current_source = Path(core.__file__).read_bytes()
    training_source = args.training_source.read_bytes()
    data = Path(core.DATA_PATH).read_bytes()
    validate_model_metadata(metadata, training_source, data, current_source)
    if metadata.get("model_sha256") and metadata["model_sha256"] != hashlib.sha256(args.model.read_bytes()).hexdigest():
        raise ValueError("Saved model checksum does not match its metadata.")
    with args.model.open("rb") as handle:
        pipeline = cloudpickle.load(handle)
    if pipeline.model is None or not pipeline.model.method_bundle:
        raise ValueError("Both winner and method models must be available.")
    date = (core.pd.Timestamp.now().date() if args.prediction_date is None
            else core.pd.Timestamp(args.prediction_date).date())
    bouts = []
    for bout in card["bouts"]:
        a, b = bout["fighters"]
        missing = [name for name in (a, b) if pipeline.is_debutant(name)]
        row = dict(bout, missing_history=missing)
        if missing and not bout.get("allow_debutant_prior", False):
            row["status"] = "Missing fighter history; no prediction produced"
        else:
            kwargs = {key: bout[key] for key in ("weight_class", "gender", "rounds")}
            for key in ("red_odds", "blue_odds"):
                if key in bout:
                    kwargs[key] = bout[key]
            row["prediction"] = pipeline.predict_matchup(
                a, b, **kwargs, location=card.get("location"),
                event_elevation=card.get("event_elevation"),
                is_title=bout.get("is_title", False), prediction_date=date,
                include_explanation=True,
            )
            row["status"] = "Predicted using generic debutant priors" if missing else "Predicted"
            print(f"Explained {a} vs. {b}", flush=True)
        bouts.append(row)
    dates = core.pd.to_datetime(core.pd.read_csv(core.DATA_PATH, usecols=["event_date"])["event_date"],
                                format="%m/%d/%Y")
    output = {
        "event": card["event"], "event_date": card.get("event_date"),
        "location": card.get("location"), "prediction_date": date.isoformat(),
        "data_cutoff": dates.max().date().isoformat(),
        "dataset_sha256": hashlib.sha256(data).hexdigest(),
        "training_source_sha256": hashlib.sha256(training_source).hexdigest(),
        "inference_source_sha256": hashlib.sha256(current_source).hexdigest(),
        "report_source_sha256": hashlib.sha256(
            (Path(core.__file__).parent / "matchup_reports.py").read_bytes()).hexdigest(),
        "training_source_data_sha256": metadata["source_data_sha256"],
        "bouts": bouts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".md").write_text(render_card_markdown(output))
    core.export_to_excel(str(args.output.with_suffix(".xlsx")),
                         [row["prediction"] for row in bouts if "prediction" in row], [])
    print(f"Saved JSON, Markdown and Excel reports: {args.output}", flush=True)


if __name__ == "__main__":
    main()
