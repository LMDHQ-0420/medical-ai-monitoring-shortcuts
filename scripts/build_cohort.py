"""Build MIMIC-IV and eICU cohorts and write feature parquet files.

Usage:
    python scripts/build_cohort.py --config configs/default.yaml [--smoke-test]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.data.mimic_cohort import build_cohort as build_mimic
from src.data.eicu_cohort import build_cohort as build_eicu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Process first 500 stays only, print 10 rows"
    )
    parser.add_argument("--dataset", choices=["mimic", "eicu", "both"], default="both")
    parser.add_argument("--output-dir", default=None, help="处理后数据输出目录")
    parser.add_argument("--build-sequences", action="store_true", help="同时构建真实逐小时序列")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir or ("data/processed/smoke" if args.smoke_test else "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("mimic", "both"):
        mimic_dir = cfg["data"]["mimic_iv_dir"]
        print(f"=== Building MIMIC-IV cohort from {mimic_dir} ===")
        build_mimic(
            mimic_dir=mimic_dir,
            time_windows=cfg["time_windows"],
            output_dir=str(output_dir),
            max_stays=500 if args.smoke_test else None,
        )
        if args.smoke_test:
            import pandas as pd
            df = pd.read_parquet(output_dir / f"mimic_iv_{cfg['time_windows'][0]}h.parquet")
            print(f"\nSmoke-test sample (10 rows, {len(df.columns)} columns):")
            print(df.head(10).to_string())

    if args.dataset in ("eicu", "both"):
        eicu_dir = cfg["data"]["eicu_dir"]
        print(f"\n=== Building eICU cohort from {eicu_dir} ===")
        build_eicu(
            eicu_dir=eicu_dir,
            time_windows=cfg["time_windows"],
            output_dir=str(output_dir),
            max_stays=500 if args.smoke_test else None,
        )

    if args.build_sequences:
        if args.smoke_test or output_dir.resolve() != Path("data/processed").resolve():
            parser.error("逐小时序列仅支持正式 data/processed 队列")
        if args.dataset in ("mimic", "both"):
            from src.data.mimic_sequence import build_sequence
            build_sequence()
        if args.dataset in ("eicu", "both"):
            from src.data.eicu_sequence import build_sequence
            build_sequence()


if __name__ == "__main__":
    main()
