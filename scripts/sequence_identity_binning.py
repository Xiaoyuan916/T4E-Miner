"""
Assign candidate proteins to sequence-identity bins.

The input CSV must already contain pairwise sequence-identity values. This
script does not calculate sequence identity; it assigns each candidate to the
identity intervals used for candidate stratification and exports both a master
table and one CSV file per non-empty interval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BIN_ORDER = [
    "<20%",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-100%",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign candidates to predefined sequence-identity bins."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        type=Path,
        help="Input CSV containing candidate records and sequence identities.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("result/sequence_bins"),
        type=Path,
        help="Directory for the binned master table and per-bin CSV files.",
    )
    parser.add_argument(
        "--identity-column",
        default="Identity(%)",
        help="Name of the percentage identity column (default: Identity(%%)).",
    )
    parser.add_argument(
        "--master-filename",
        default="candidates_binned.csv",
        help="Filename for the complete binned table.",
    )
    return parser.parse_args()


def get_bin(identity: float) -> str:
    """Return the predefined bin label for a percentage identity value."""
    if identity >= 80:
        return "80-100%"
    if identity >= 70:
        return "70-80%"
    if identity >= 60:
        return "60-70%"
    if identity >= 50:
        return "50-60%"
    if identity >= 40:
        return "40-50%"
    if identity >= 30:
        return "30-40%"
    if identity >= 20:
        return "20-30%"
    return "<20%"


def safe_bin_name(bin_label: str) -> str:
    return (
        bin_label.replace("-", "_")
        .replace("%", "")
        .replace("<", "under_")
    )


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    if args.identity_column not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise KeyError(
            f"Missing identity column '{args.identity_column}'. "
            f"Available columns: {available}"
        )

    identities = pd.to_numeric(df[args.identity_column], errors="raise")
    invalid = identities.isna() | (identities < 0) | (identities > 100)
    if invalid.any():
        rows = ", ".join(str(i + 2) for i in df.index[invalid])
        raise ValueError(
            "Sequence identity must be between 0 and 100. "
            f"Invalid values were found on CSV row(s): {rows}"
        )

    df["Sequence_Bin"] = identities.apply(get_bin)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    master_path = args.output_dir / args.master_filename
    df.to_csv(master_path, index=False)

    counts = df["Sequence_Bin"].value_counts()
    print(f"Loaded {len(df)} candidate records from: {args.input_csv}")
    print("Sequence-identity distribution:")
    for bin_label in BIN_ORDER:
        count = int(counts.get(bin_label, 0))
        print(f"  {bin_label:>8}: {count}")
        if count:
            subset = df[df["Sequence_Bin"] == bin_label]
            subset_path = args.output_dir / f"candidates_{safe_bin_name(bin_label)}.csv"
            subset.to_csv(subset_path, index=False)

    print(f"Binned master table: {master_path}")


if __name__ == "__main__":
    main()
