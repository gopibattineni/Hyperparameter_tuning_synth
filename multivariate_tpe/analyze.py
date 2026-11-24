#!/usr/bin/env python3
"""Rebuild comparison tables/figures including this experiment's results."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
raise SystemExit(runpy.run_path(str(_ROOT / "scripts" / "compare_bayesian_samplers.py"), run_name="__main__") or 0)
