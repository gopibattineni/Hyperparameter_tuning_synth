#!/usr/bin/env python3
"""Build the O2–O5 multiseed job queue (15×10×3×7 × 4 presets = 12,600)."""

from __future__ import annotations

import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(_ROOT))

from generators.device import CPU_ONLY_GENERATORS
from multiseed_comparison import ALL_SEEDS_10, SAMPLER_NAMES
from multivariate_tpe.datasets_catalog import ALL_DATASETS, GENERATORS_NO_FOREST

PRESETS = ("O2", "O3", "O4", "O5")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=str,
        default=str(_ROOT / "results/multiseed_weights_O2_O5/logs/queue_o2_o5_gpu.tsv"),
    )
    p.add_argument(
        "--out-cpu",
        type=str,
        default=str(_ROOT / "results/multiseed_weights_O2_O5/logs/queue_cpu_only.tsv"),
    )
    p.add_argument(
        "--presets",
        nargs="*",
        default=list(PRESETS),
        help="Default O2 O3 O4 O5 (O1 already done)",
    )
    args = p.parse_args()
    out = Path(args.out)
    out_cpu = Path(args.out_cpu)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_cpu.parent.mkdir(parents=True, exist_ok=True)

    gpu_gens = [g for g in GENERATORS_NO_FOREST if g not in CPU_ONLY_GENERATORS]
    cpu_gens = [g for g in GENERATORS_NO_FOREST if g in CPU_ONLY_GENERATORS]

    def build(generators: list[str]) -> list[str]:
        lines: list[str] = []
        for preset in args.presets:
            for seed in ALL_SEEDS_10:
                for dataset in ALL_DATASETS:
                    for generator in generators:
                        for sampler in SAMPLER_NAMES:
                            lines.append(
                                f"{preset}\t{sampler}\t{seed}\t{dataset}\t{generator}\n"
                            )
        return lines

    gpu_lines = build(gpu_gens)
    cpu_lines = build(cpu_gens)
    out.write_text("".join(gpu_lines), encoding="utf-8")
    out_cpu.write_text("".join(cpu_lines), encoding="utf-8")
    expected = (
        len(args.presets)
        * len(ALL_SEEDS_10)
        * len(ALL_DATASETS)
        * len(GENERATORS_NO_FOREST)
        * len(SAMPLER_NAMES)
    )
    print(f"GPU queue {len(gpu_lines)} gens={gpu_gens} → {out}")
    print(f"CPU-only queue {len(cpu_lines)} gens={cpu_gens} → {out_cpu}")
    print(f"total {len(gpu_lines)+len(cpu_lines)} (expected {expected})")
    assert len(gpu_lines) + len(cpu_lines) == expected
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
