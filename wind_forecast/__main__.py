import argparse
from pathlib import Path
import pandas as pd
from .core import read_turbines
from .weather import fetch_snapshot
from .pipeline import run_forecast


def main():
    parser = argparse.ArgumentParser(description="Nakasato ensemble wind generation forecasting")
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("forecast", help="Live Open-Meteo forecast, or reproduce a saved snapshot")
    f.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    f.add_argument("--hours", type=int, default=24)
    f.add_argument("--out", help="Output directory; default is a new timestamped runs/ folder")
    f.add_argument("--replay", help="Saved weather_snapshot directory; performs no API calls")
    f.add_argument("--plan", help="Known-at-issue operational plan CSV")
    t = sub.add_parser("train", help="Train/validate the residual ML ensemble using real SCADA")
    t.add_argument("--features", required=True, help="Archived forecast_features.csv files, glob expression")
    t.add_argument("--scada", required=True)
    t.add_argument("--out", default="models/calibrated")
    t.add_argument("--min-train-rows", type=int, default=300)
    p = sub.add_parser("predict-ml", help="Apply a locally trained model; raw weather spread stays separate")
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--out", default="runs/ml_predictions.csv")
    args = parser.parse_args()
    if args.command == "forecast":
        root = Path(args.root)
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%fZ")
        if not args.out:
            args.out = str(root/"runs"/stamp)
        snapshot = args.replay
        if not snapshot:
            snapshot = fetch_snapshot(read_turbines(root/"data/turbines.csv"), args.hours,
                                      root/"weather_archive"/stamp)
        farm, _, metadata = run_forecast(root, snapshot, args.out, args.plan)
        print(f"Saved {len(farm)} hours for {metadata['n_turbines']} turbines to {args.out}")
        print("Status: uncalibrated physical/weather ensemble. See run_metadata.json for assumptions.")
    elif args.command == "train":
        from .learning import train
        train(args.features, args.scada, args.out, args.min_train_rows)
    else:
        from .learning import predict_saved
        predict_saved(args.model, args.features, args.out)


if __name__ == "__main__":
    main()
