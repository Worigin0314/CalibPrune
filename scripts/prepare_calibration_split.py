"""Create deterministic cal500/test split index files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibprune.data.splits import default_split_path, make_split_indices, write_split_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-total", type=int, required=True)
    parser.add_argument("--n-cal", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--output-dir", default="data/splits")
    args = parser.parse_args()
    if args.n_total <= args.n_cal:
        raise ValueError("n-total must be larger than n-cal.")
    payload = make_split_indices(args.n_total, args.n_cal, args.seed)
    payload.update({
        "dataset": args.dataset,
    })
    out = default_split_path(args.dataset, args.n_cal, args.seed, output_dir=args.output_dir)
    write_split_file(out, payload)
    print(out)


if __name__ == "__main__":
    main()
