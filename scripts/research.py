"""Train, forecast, archive, and evaluate the event-causal UFC research model."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".ufc_model_cache"/"matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
os.environ.setdefault("MKL_NUM_THREADS","2")

import cloudpickle
import pandas as pd
import UFC_ML_Model as core
from market_store import MarketStore
from matchup_reports import render_card_markdown
from probability_calibration import MarketBlender
from research_pipeline import TrainingConfig, VERSION, source_fingerprint


FLAGS = ("UFC_MOV_ENABLED","UFC_MOV_MODE","UFC_PHASE_RATINGS","UFC_OBA_ENABLED",
         "UFC_CORNER_CORRECTION","UFC_WINNER_ALL_FEATURES","UFC_COMBINER_ROBUST")


def save_model(pipeline,path):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    current_source = source_fingerprint(ROOT)
    current_data = hashlib.sha256(Path(pipeline.csv_path).read_bytes()).hexdigest()
    if pipeline.training_source_sha256 != current_source or pipeline.dataset_sha256 != current_data:
        raise ValueError("Model source/data changed since training; rebuild before saving.")
    pipeline.log = print
    with path.with_suffix(".tmp").open("wb") as handle:
        cloudpickle.dump(pipeline,handle)
    path.with_suffix(".tmp").replace(path)
    metadata = {
        "format":VERSION,"model_id":pipeline.model_fingerprint,
        "source_sha256":current_source,
        "dataset_sha256":current_data,
        "source_data_sha256":hashlib.sha256(Path(core.__file__).read_bytes()+Path(pipeline.csv_path).read_bytes()).hexdigest(),
        "history_cutoff":pipeline._temporal_state.cutoff.date().isoformat(),
        "last_supervised_date":pipeline.model.training_contract["last_supervised_date"],
        "training_cutoff":max(pipeline._temporal_state.cutoff.date().isoformat(),
            str(pd.Timestamp(pipeline.model.market_blender.training_cutoff).date()) if pipeline.model.market_blender.is_fitted else "0001-01-01"),
        "ufc_environment":{key:os.environ[key] for key in FLAGS if key in os.environ},
        "packages":{name:version(name) for name in
                    ("numpy","pandas","scikit-learn","scipy","lightgbm","xgboost","catboost","cloudpickle")},
        "model_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata


def resolve_model_data(pipeline, metadata, data_path=None):
    """Restore a checkpoint in another checkout without relaxing data checks."""
    dataset = Path(data_path or pipeline.csv_path)
    if data_path is None and not dataset.is_file():
        dataset = Path(core.DATA_PATH)
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != metadata["dataset_sha256"]:
        raise ValueError("Model dataset changed; retrain before forecasting.")
    pipeline.csv_path = str(dataset.resolve())


def load_model(path, data_path=None):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".meta.json").read_text())
    if metadata.get("format") != VERSION or metadata["source_sha256"] != source_fingerprint(ROOT):
        raise ValueError("Model source changed; retrain the corrected pipeline.")
    if metadata["ufc_environment"] != {key:os.environ[key] for key in FLAGS if key in os.environ}:
        raise ValueError("Feature configuration changed; retrain.")
    if metadata["model_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Model artifact checksum does not match metadata.")
    with path.open("rb") as handle:
        pipeline = cloudpickle.load(handle)
    if pipeline.model_fingerprint != metadata["model_id"] or pipeline.dataset_sha256 != metadata["dataset_sha256"]:
        raise ValueError("Model identity differs from its metadata.")
    expected_cutoff = max(pipeline._temporal_state.cutoff.date().isoformat(),
        str(pd.Timestamp(pipeline.model.market_blender.training_cutoff).date()) if pipeline.model.market_blender.is_fitted else "0001-01-01")
    if metadata["training_cutoff"] != expected_cutoff:
        raise ValueError("Model training cutoff differs from its metadata.")
    for name,trained_version in metadata["packages"].items():
        if version(name) != trained_version:
            raise ValueError(f"Package version changed for {name}; restore pinned dependencies or retrain.")
    resolve_model_data(pipeline, metadata, data_path)
    return pipeline,metadata


def forecast(pipeline,card,prediction_date):
    rows = []
    for bout in card["bouts"]:
        a,b = bout["fighters"]
        missing = [name for name in (a,b) if pipeline.is_debutant(name)]
        row = dict(bout,missing_history=missing)
        if missing and not bout.get("allow_debutant_prior",False):
            row["status"] = "Missing fighter history; no prediction produced"
        else:
            row["prediction"] = pipeline.predict_matchup(a,b,
                **{key:bout[key] for key in ("weight_class","gender","rounds")},
                red_odds=bout.get("red_odds"),blue_odds=bout.get("blue_odds"),
                location=card.get("location"),event_elevation=card.get("event_elevation"),
                is_title=bout.get("is_title",False),prediction_date=prediction_date,
                include_explanation=True)
            row["status"] = "Predicted using generic debutant priors" if missing else "Predicted"
            print(f"Predicted and explained {a} vs. {b}",flush=True)
        rows.append(row)
    return {"event":card["event"],"event_date":card.get("event_date"),
        "location":card.get("location"),"prediction_date":str(pd.Timestamp(prediction_date).date()),
        "generated_at":datetime.now(timezone.utc).isoformat(),"model_id":pipeline.model_fingerprint,
        "source_sha256":pipeline.training_source_sha256,"dataset_sha256":pipeline.dataset_sha256,
        "report_source_sha256":hashlib.sha256((ROOT/"matchup_reports.py").read_bytes()).hexdigest(),
        "data_cutoff":pipeline._temporal_state.cutoff.date().isoformat(),"bouts":rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    train = commands.add_parser("train")
    train.add_argument("--data",default=core.DATA_PATH,type=Path)
    train.add_argument("--model",default=ROOT/".ufc_model_cache"/"causal-production.pkl",type=Path)
    train.add_argument("--config",type=Path,help="TrainingConfig JSON; reduced settings are smoke experiments, not the full run")
    predict = commands.add_parser("predict")
    predict.add_argument("card",type=Path); predict.add_argument("output",type=Path)
    predict.add_argument("--model",default=train.get_default("model"),type=Path)
    predict.add_argument("--data",type=Path,help="Checkpoint dataset location; must match its recorded checksum")
    predict.add_argument("--prediction-date",default=None)
    predict.add_argument("--database",type=Path)
    predict.add_argument("--event-start",help="Exact ISO timestamp with timezone; optional for forecast-only archiving")
    for name in ("import-odds","import-results","backtest","fit-market","list-fights"):
        command = commands.add_parser(name)
        command.add_argument("--database",required=True,type=Path)
        if name in ("import-odds","import-results"):
            command.add_argument("csv",type=Path)
        if name in ("backtest","fit-market"):
            command.add_argument("--output",required=True,type=Path)
        if name == "backtest":
            command.add_argument("--model-id",required=True)
            command.add_argument("--min-ev",type=float,default=.03)
        if name == "fit-market":
            command.add_argument("--model",required=True,type=Path)
            command.add_argument("--bookmaker",required=True)
    args = parser.parse_args()
    if args.command == "train":
        pipeline = core.UFCSuperModelPipeline(str(args.data))
        pipeline.research_config = TrainingConfig(**json.loads(args.config.read_text())) if args.config else TrainingConfig()
        pipeline.train()
        save_model(pipeline,args.model)
        args.model.with_suffix(".evaluation.json").write_text(json.dumps(pipeline.research_report,indent=2,allow_nan=False)+"\n")
        print(f"Model, metadata and evaluation saved: {args.model}")
        return
    if args.command == "predict":
        pipeline,metadata = load_model(args.model,args.data)
        date = args.prediction_date or pd.Timestamp.now(tz="America/New_York").date().isoformat()
        card = json.loads(args.card.read_text())
        output = forecast(pipeline,card,date)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
        args.output.with_suffix(".md").write_text(render_card_markdown(output))
        core.export_to_excel(args.output.with_suffix(".xlsx"),[r["prediction"] for r in output["bouts"] if "prediction" in r],[])
        if args.database:
            store = MarketStore(args.database)
            event_id = card.get("event_source_url") or "local:"+hashlib.sha256(f"{card['event']}|{card['event_date']}".encode()).hexdigest()[:24]
            store.register_event(event_id,card["event"],card["event_date"],args.event_start,
                                 output["generated_at"] if args.event_start else None,"card manifest")
            store.register_model(metadata["model_id"],metadata,metadata["training_cutoff"])
            for bout in output["bouts"]:
                if "prediction" not in bout:
                    continue
                a,b = bout["prediction"]["name_a"],bout["prediction"]["name_b"]
                fight_id = bout.get("fight_id") or "local:"+hashlib.sha256(f"{event_id}|{a}|{b}".encode()).hexdigest()[:24]
                store.register_fight(fight_id,event_id,a,b)
                store.append_prediction(fight_id,metadata["model_id"],bout["prediction"],output["generated_at"])
            store.close()
        print(f"Saved JSON, Markdown and Excel forecasts: {args.output}")
        return
    store = MarketStore(args.database)
    try:
        if args.command == "import-odds":
            print(json.dumps({"imported_rows":store.import_quotes(args.csv)}))
        elif args.command == "import-results":
            print(json.dumps({"imported_results":store.import_results(args.csv)}))
        elif args.command == "backtest":
            result = store.backtest(args.model_id,args.min_ev)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        elif args.command == "fit-market":
            pipeline,_ = load_model(args.model)
            samples = store.market_training_samples(pipeline.model_fingerprint,args.bookmaker)
            blender = MarketBlender().fit(samples)
            pipeline.model.market_blender = blender
            pipeline.model_fingerprint = hashlib.sha256((pipeline.model_fingerprint+samples.to_json()
                +str(blender.market_weight)).encode()).hexdigest()[:24]
            save_model(pipeline,args.output)
        elif args.command == "list-fights":
            rows = store.connection.execute("SELECT f.id,e.name event,a.name first_fighter,b.name second_fighter "
                "FROM fights f JOIN events e ON f.event_id=e.id JOIN fighters a ON f.a_id=a.id JOIN fighters b ON f.b_id=b.id")
            print(json.dumps([dict(row) for row in rows],indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
