#!/usr/bin/env python3
"""Alias entry point — delegates to ``run_experiments``."""

from __future__ import annotations

from run_experiments import main

if __name__ == "__main__":
    raise SystemExit(main())
