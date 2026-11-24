#!/usr/bin/env python3
"""Compatibility shim — Standard TPE entry points now live under standard_tpe/."""
from __future__ import annotations
import runpy
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
raise SystemExit(runpy.run_path(str(_ROOT / "standard_tpe" / "run_tuning.py"), run_name="__main__") or 0)
