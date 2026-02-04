from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from km3posterior.loss_inference import run_loss_posterior_from_track_map


def main() -> int:
    p = argparse.ArgumentParser(description="KM3-230213A Bayesian-ish stochastic-loss posterior (binned emission tomography).")
    p.add_argument("--event", required=True, help="Path to KM3-230213A_allhits.json.gz")
    p.add_argument("--track", required=True, help="Path to track_posterior.npz (uses MAP track)")
    p.add_argument("--out", default="results", help="Output directory (default: results)")
    p.add_argument("--chains", type=int, default=16, help="Number of MCMC chains (default: 16)")
    p.add_argument("--cores", type=int, default=16, help="Processes (default: 16)")
    p.add_argument("--steps", type=int, default=20_000, help="Steps per chain (default: 20000)")
    p.add_argument("--burn", type=int, default=8_000, help="Burn-in steps per chain (default: 8000)")
    p.add_argument("--thin", type=int, default=10, help="Thinning (default: 10)")
    p.add_argument("--seed", type=int, default=230213, help="RNG seed (default: 230213)")
    args = p.parse_args()

    out = run_loss_posterior_from_track_map(
        args.event,
        track_posterior_npz=args.track,
        out_dir=Path(args.out),
        chains=args.chains,
        cores=args.cores,
        n_steps=args.steps,
        burn_in=args.burn,
        thin=args.thin,
        seed=args.seed,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
