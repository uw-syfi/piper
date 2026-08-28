"""Attribute normal-to-isolated replay savings to logical profile units."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from src.attribution import attribute_profile_mode_savings
from src.profiling import ProfiledTrainingPlan


def _write_csv(path: Path, result) -> None:
    fields = (
        "feature_id",
        "pp_indices",
        "uids",
        "node_kinds",
        "streams",
        "ranks",
        "binding_count",
        "raw_profile_savings_us",
        "attributed_savings_us",
        "contribution_std_us",
        "contribution_ci95_us",
        "share_of_step_savings_pct",
        "positive_exposure_probability",
        "hidden_probability",
        "negative_probability",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in result.features:
            feature = item.feature
            writer.writerow(
                {
                    "feature_id": feature.feature_id,
                    "pp_indices": ";".join(map(str, feature.pp_indices)),
                    "uids": ";".join(feature.uids),
                    "node_kinds": ";".join(feature.node_kinds),
                    "streams": ";".join(feature.streams),
                    "ranks": ";".join(map(str, feature.ranks)),
                    "binding_count": feature.binding_count,
                    "raw_profile_savings_us": feature.raw_profile_savings_us,
                    "attributed_savings_us": item.attributed_savings_us,
                    "contribution_std_us": item.contribution_std_us,
                    "contribution_ci95_us": item.contribution_ci95_us,
                    "share_of_step_savings_pct": (
                        item.share_of_step_savings_pct
                    ),
                    "positive_exposure_probability": (
                        item.positive_exposure_probability
                    ),
                    "hidden_probability": item.hidden_probability,
                    "negative_probability": item.negative_probability,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--permutations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exposure-threshold-us", type=float, default=1.0)
    parser.add_argument("--no-antithetic", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    plan = ProfiledTrainingPlan.load_json(args.profile)
    result = attribute_profile_mode_savings(
        plan,
        permutations=args.permutations,
        seed=args.seed,
        antithetic=not args.no_antithetic,
        exposure_threshold_us=args.exposure_threshold_us,
        strict=not args.no_strict,
    )
    payload = result.to_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if args.csv_output:
        _write_csv(args.csv_output, result)

    print(
        f"normal={result.normal_step_time_us / 1000.0:.3f} ms, "
        f"isolated={result.isolated_step_time_us / 1000.0:.3f} ms, "
        f"savings={result.total_step_savings_us / 1000.0:.3f} ms, "
        f"residual={result.additivity_residual_us:.6f} us",
        file=sys.stderr,
    )
    for item in result.features[: max(0, args.top)]:
        print(
            f"{item.attributed_savings_us / 1000.0:9.3f} ms  "
            f"p_exposed={item.positive_exposure_probability:5.1%}  "
            f"{item.feature.feature_id}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
