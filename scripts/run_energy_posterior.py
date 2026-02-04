from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from km3posterior.data import load_event_json_gz
from km3posterior.energy_inference import (
    energy_posterior_from_ntrigpmt,
    load_figure2_calibration,
    save_energy_posterior_npz,
)


def main() -> int:
    p = argparse.ArgumentParser(description="KM3-230213A muon-energy posterior from NTrigPMT calibration (Figure 2 source).")
    p.add_argument("--event", required=True, help="Path to KM3-230213A_allhits.json.gz")
    p.add_argument(
        "--figure2",
        required=True,
        help="Path to upstream data/supplementary/figuresource/figure2.csv",
    )
    p.add_argument("--out", default="results/energy_posterior.npz", help="Output NPZ path")
    p.add_argument("--seed", type=int, default=230213, help="RNG seed (default: 230213)")
    p.add_argument("--samples", type=int, default=100_000, help="Posterior sample count (default: 100000)")
    args = p.parse_args()

    evt = load_event_json_gz(args.event)
    n_trig = int(len(np.unique(evt.pmt_key[evt.triggered])))

    calib = load_figure2_calibration(args.figure2)
    post = energy_posterior_from_ntrigpmt(n_trig_pmt=n_trig, calibration=calib, seed=args.seed, n_samples=args.samples)
    out = save_energy_posterior_npz(args.out, post)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
