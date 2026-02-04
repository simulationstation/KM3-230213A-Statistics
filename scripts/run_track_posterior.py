from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from km3posterior.track_inference import sample_track_posterior_importance


def main() -> int:
    p = argparse.ArgumentParser(description="KM3-230213A timing-only Bayesian-ish track posterior (importance sampling).")
    p.add_argument(
        "--event",
        required=True,
        help="Path to KM3-230213A_allhits.json.gz",
    )
    p.add_argument(
        "--out",
        default="results",
        help="Output directory (default: results)",
    )
    p.add_argument("--proposals", type=int, default=50_000, help="Number of Gaussian proposals (default: 50000)")
    p.add_argument("--posterior", type=int, default=20_000, help="Number of posterior samples to resample (default: 20000)")
    p.add_argument("--scale", type=float, default=1.5, help="Proposal covariance scale factor (default: 1.5)")
    p.add_argument("--cores", type=int, default=16, help="Processes for logp evaluation (default: 16)")
    p.add_argument("--seed", type=int, default=230213, help="RNG seed (default: 230213)")
    args = p.parse_args()

    out = sample_track_posterior_importance(
        args.event,
        out_dir=Path(args.out),
        n_proposals=args.proposals,
        n_posterior=args.posterior,
        proposal_scale=args.scale,
        cores=args.cores,
        seed=args.seed,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
