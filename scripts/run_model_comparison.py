from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from km3posterior.model_comparison import ModelComparisonConfig, ModelComparisonLikelihood, ModelComparisonPriors, ModelComparisonSelection, run_model_comparison


def main() -> int:
    p = argparse.ArgumentParser(description="Bayesian model comparison for KM3-230213A (H1 single track vs H2 bundle vs H3 time-uniform).")
    p.add_argument("--event", default="upstream-KM3-230213A-data/data/event/KM3-230213A_allhits.json.gz", help="Path to KM3-230213A_allhits.json.gz")
    p.add_argument("--out", default="results-modelcomp/model_comparison.json", help="Output JSON path")
    p.add_argument("--seed", type=int, default=230213, help="Random seed for restarts")
    p.add_argument("--restarts-h2", type=int, default=8, help="Number of MAP restarts for H2 (bundle)")
    p.add_argument("--bundle-sigma-m", type=float, default=50.0, help="Prior sigma for bundle transverse offset (meters)")
    p.add_argument("--outlier-frac", type=float, default=0.01, help="Time-uniform outlier fraction")
    p.add_argument("--all-hits", action="store_true", help="Use all hits (instead of first triggered hit per PMT)")
    args = p.parse_args()

    cfg = ModelComparisonConfig(
        selection=ModelComparisonSelection(triggered_only=True, first_hit_per_pmt=not args.all_hits),
        priors=ModelComparisonPriors(bundle_sigma_offset_m=float(args.bundle_sigma_m)),
        likelihood=ModelComparisonLikelihood(outlier_frac=float(args.outlier_frac)),
        restarts_h1=1,
        restarts_h2=int(args.restarts_h2),
        seed=int(args.seed),
    )

    out_path = Path(args.out)
    res = run_model_comparison(args.event, out_path=out_path, config=cfg)
    print(out_path)

    # Print a compact human summary.
    bf = res["bayes_factors"]
    print("logB(H2/H1):", bf["logB_H2_over_H1"])
    print("logB(H1/H3):", bf["logB_H1_over_H3"])
    print("logB(H1/H0):", bf["logB_H1_over_H0"])
    if "logB_H1_over_H4" in bf:
        print("logB(H1/H4):", bf["logB_H1_over_H4"])
        h4 = res["models"]["H4_beta_track"]["map"]
        beta = h4.get("beta")
        beta_sig = h4.get("beta_sigma_laplace")
        if beta is not None:
            if beta_sig is None:
                print("H4 beta:", beta)
            else:
                print("H4 beta:", beta, "+/-", beta_sig)
    print("H2 offset_norm_m:", res["models"]["H2_bundle_two_tracks"]["map"]["offset_norm_m"])
    print("H2 w_track1:", res["models"]["H2_bundle_two_tracks"]["map"]["w_track1"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
