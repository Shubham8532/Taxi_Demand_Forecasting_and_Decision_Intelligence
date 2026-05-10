"""Build the lightweight region stats file for Azure deployment.

Run this ONCE locally (or in CI) after training:

    python scripts/build_inference_stats.py

It reads the full ``final_data.csv`` and produces a tiny
``region_inference_stats.pkl`` (~6 KB) that the Flask app
uses instead of loading the CSV at runtime.

Requires the data file at one of these locations:
    - data/interim/final_data.csv
    - data/interim/historical_features.csv
"""

from pathlib import Path
import sys

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.inference.realtime import build_region_stats


def main():
    data_candidates = [
        project_root / "data" / "processed" / "final_data.csv",
        project_root / "data" / "interim" / "final_data.csv",
        project_root / "data" / "interim" / "historical_features.csv",
    ]

    data_path = next((p for p in data_candidates if p.exists()), None)

    if data_path is None:
        print("ERROR: No data file found. Expected one of:")
        for p in data_candidates:
            print(f"  - {p}")
        print("\nPlease place your final_data.csv in data/interim/")
        sys.exit(1)

    output_path = project_root / "models" / "region_inference_stats.pkl"

    print(f"Reading data from: {data_path}")
    print(f"Output will be saved to: {output_path}")

    result = build_region_stats(
        data_path=data_path,
        output_path=output_path,
    )

    print(f"\nDone! Stats built for {result['n_regions']} regions.")
    print(f"Global mean demand: {result['global_mean']:.2f}")
    print(f"File saved: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
