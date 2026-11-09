import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description="Process MapAnything/PI3 TUM results")
parser.add_argument("--stride", type=str, default="1", help="stride (downsample factor) used during eval")
parser.add_argument("--model", type=str, default="mapanything", choices=["mapanything", "pi3"], help="model used during eval")
args = parser.parse_args()

# Path to log file produced by eval_tum_mapanything.sh
log_path = Path.cwd() / f"logs/tum_{args.model}_results_s{args.stride}.txt"

# Load the CSV log
df = pd.read_csv(log_path)

# Clean: remove any existing 'Average' rows
df = df[df["Dataset"] != "Average"]

# Ensure proper data types
df["Run"] = df["Run"].astype(int)

print("=== Per-Experiment RMSE APE (Run x Dataset) ===")
for run in sorted(df["Run"].unique()):
    print(f"\n--- Run {run} ---")
    run_df = df[df["Run"] == run]
    for _, row in run_df.iterrows():
        print(f"{row['Dataset']}: {row['RMSE']:.4f}")

print("\n=== Per-Run Average RMSE APE ===")
per_run_avg = df.groupby("Run")["RMSE"].mean()
for run, val in per_run_avg.items():
    print(f"Run {run}: {val:.4f}")

print("\n=== Per-Dataset Average RMSE APE ===")
per_dataset_avg = df.groupby("Dataset")["RMSE"].mean()
for dataset, val in per_dataset_avg.items():
    print(f"{dataset}: {val:.4f}")

overall_avg = df["RMSE"].mean()
print("\n=== Overall Average RMSE APE Across All Runs ===")
print(f"Overall Average RMSE: {overall_avg:.4f}")
