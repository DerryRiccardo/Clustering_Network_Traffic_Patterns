from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_PATH = Path("cicids2017_cleaned.csv")
OUTPUT_PATH = Path("data/sample/cicids2017_sample_200000.csv")
LABEL_COLUMN = "Attack Type"
TARGET_ROWS = 200_000
RANDOM_STATE = 42


def make_stratified_sample() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_PATH)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required label column: {LABEL_COLUMN}")

    counts = df[LABEL_COLUMN].value_counts().sort_index()
    exact_allocations = counts / counts.sum() * TARGET_ROWS
    allocations = np.floor(exact_allocations).astype(int)

    remainder = int(TARGET_ROWS - allocations.sum())
    if remainder > 0:
        order = (exact_allocations - allocations).sort_values(ascending=False).index
        for label in order[:remainder]:
            allocations[label] += 1
    elif remainder < 0:
        order = (exact_allocations - allocations).sort_values(ascending=True).index
        for label in order[: abs(remainder)]:
            allocations[label] -= 1

    parts = []
    for label, sample_count in allocations.items():
        group = df[df[LABEL_COLUMN] == label]
        parts.append(group.sample(n=int(sample_count), random_state=RANDOM_STATE))

    sample = (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False)

    print(f"Source rows: {len(df):,}")
    print(f"Sample rows: {len(sample):,}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nFull distribution:")
    full_dist = pd.DataFrame(
        {
            "count": counts,
            "percent": (counts / counts.sum() * 100).round(2),
        }
    )
    print(full_dist.to_string())

    print("\nSample distribution:")
    sample_counts = sample[LABEL_COLUMN].value_counts().sort_index()
    sample_dist = pd.DataFrame(
        {
            "count": sample_counts,
            "percent": (sample_counts / sample_counts.sum() * 100).round(2),
        }
    )
    print(sample_dist.to_string())

    return sample


if __name__ == "__main__":
    make_stratified_sample()
