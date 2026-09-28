#!/usr/bin/env python3
"""Write air_quality HPO Excel + notation using the Adult workbook template."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tuning.hpo_report import write_dataset


def main() -> None:
    xlsx, txt = write_dataset("air_quality")
    print(f"Wrote {xlsx}")
    print(f"Wrote {txt}")


if __name__ == "__main__":
    main()
