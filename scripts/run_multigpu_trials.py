#!/usr/bin/env python3
"""Compatibility shim — moved to standard_tpe/scripts/."""
from __future__ import annotations
import runpy
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
raise SystemExit(runpy.run_path(str(_ROOT / "standard_tpe" / "scripts" / "run_multigpu_trials.py"), run_name="__main__") or 0)
