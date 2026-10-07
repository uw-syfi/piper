# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
"""Render Figure 1 from its committed, estimator-valid aggregate source."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from generate_factor_figures import plot_figure1


def main() -> None:
    bundle = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=bundle / "data/figure1_overlap_response_by_victim_source.csv",
    )
    parser.add_argument("--output", type=Path, default=bundle / "generated")
    args = parser.parse_args()

    table = pd.read_csv(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_figure1(
        table,
        args.output / "figure1_overlap_response_by_victim.png",
        args.output / "figure1_overlap_response_by_victim.svg",
    )
    print(args.output / "figure1_overlap_response_by_victim.png")


if __name__ == "__main__":
    main()
