#!/usr/bin/env python3
"""Write Adult-template HPO Excel + notation for one or all datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.loader import load_dataset_config
from tuning.hpo_report import write_dataset


def main() -> int:
    p = argparse.ArgumentParser(description="Write Adult-structure HPO workbooks")
    p.add_argument("--dataset", default=None, help="Single dataset key")
    p.add_argument("--all", action="store_true", help="Write every dataset that has completed gens")
    args = p.parse_args()
    keys = list(load_dataset_config()) if args.all else [args.dataset]
    if not keys or keys == [None]:
        p.error("pass --dataset KEY or --all")
    rc = 0
    for key in keys:
        try:
            xlsx, txt = write_dataset(key)
            print(f"Wrote {xlsx}")
            print(f"Wrote {txt}")
        except (SystemExit, ValueError) as exc:
            print(f"skip {key}: {exc}")
            rc = 2
        except Exception as exc:
            print(f"fail {key}: {type(exc).__name__}: {exc}")
            rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
