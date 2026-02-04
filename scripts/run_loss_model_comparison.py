from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from km3posterior.loss_model_comparison import LossModelComparisonConfig, run_loss_model_comparison


def main() -> int:
    p = argparse.ArgumentParser(description="Model comparison on the emission-point histogram (baseline vs baseline+losses).")
    p.add_argument("--event", default="upstream-KM3-230213A-data/data/event/KM3-230213A_allhits.json.gz", help="Path to KM3-230213A_allhits.json.gz")
    p.add_argument("--track-posterior", default="results-latest/track_posterior.npz", help="Path to track_posterior.npz (for MAP track)")
    p.add_argument("--out", default="results-suite/loss_model_comparison_count.json", help="Output JSON path")
    p.add_argument("--weight", choices=["count", "tot"], default="count", help="Histogram weight: count or tot")
    p.add_argument("--restarts-full", type=int, default=6, help="Restarts for full (loss) model optimization")
    p.add_argument("--seed", type=int, default=230213, help="Random seed")
    args = p.parse_args()

    cfg = LossModelComparisonConfig(weight=args.weight, restarts_full=int(args.restarts_full), seed=int(args.seed))
    out_path = Path(args.out)
    run_loss_model_comparison(args.event, track_posterior_npz=args.track_posterior, out_path=out_path, cfg=cfg)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

