#!/usr/bin/env python3
"""Render OpenBench pairwise Diff_Plot NetCDF data as a KDE figure."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from openbench.core._comparison_diff_plot import plot_diff_file

LOG = logging.getLogger("plot_diff_kde")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="One or more OpenBench gridded __diff.nc files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for all figures (default: sibling comparisons/Diff_Kernel_Density_Estimate)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failures: list[tuple[Path, Exception]] = []
    for input_path in args.inputs:
        try:
            output_path = plot_diff_file(input_path, args.output_dir)
            print(f"Created {output_path}")
        except Exception as exc:
            LOG.error("Failed to plot %s: %s", input_path, exc)
            failures.append((input_path, exc))

    if failures:
        LOG.error("%d of %d input files failed", len(failures), len(args.inputs))
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
