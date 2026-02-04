from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from km3posterior.model_comparison import (
    ModelComparisonConfig,
    ModelComparisonLikelihood,
    ModelComparisonPriors,
    ModelComparisonSelection,
    run_model_comparison,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Run a small robustness sweep for the timing model comparison.")
    p.add_argument("--event", default="upstream-KM3-230213A-data/data/event/KM3-230213A_allhits.json.gz", help="Path to KM3-230213A_allhits.json.gz")
    p.add_argument("--out-dir", default="results-suite/timing_sweep", help="Directory to write per-run JSONs and a summary")
    p.add_argument("--seed", type=int, default=230213, help="Base seed")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []

    def add_run(
        name: str,
        *,
        first_hit_per_pmt: bool,
        outlier_frac: float,
        bundle_sigma_m: float,
        bundle_sigma_angle_deg: float,
        dir_kappa: float,
        restarts_scale: float = 1.0,
    ) -> None:
        runs.append(
            {
                "name": name,
                "selection": {"triggered_only": True, "first_hit_per_pmt": bool(first_hit_per_pmt)},
                "likelihood": {"outlier_frac": float(outlier_frac)},
                "priors": {
                    "bundle_sigma_offset_m": float(bundle_sigma_m),
                    "bundle_sigma_angle_deg": float(bundle_sigma_angle_deg),
                    "dir_kappa": float(dir_kappa),
                },
                "restarts_scale": float(restarts_scale),
            }
        )

    # Baseline + a handful of robustness checks.
    add_run("baseline_firsthit_eps001", first_hit_per_pmt=True, outlier_frac=0.01, bundle_sigma_m=50.0, bundle_sigma_angle_deg=2.0, dir_kappa=200.0)
    add_run("eps0005", first_hit_per_pmt=True, outlier_frac=0.005, bundle_sigma_m=50.0, bundle_sigma_angle_deg=2.0, dir_kappa=200.0)
    add_run("eps002", first_hit_per_pmt=True, outlier_frac=0.02, bundle_sigma_m=50.0, bundle_sigma_angle_deg=2.0, dir_kappa=200.0)
    add_run("bundle_sigma10", first_hit_per_pmt=True, outlier_frac=0.01, bundle_sigma_m=10.0, bundle_sigma_angle_deg=2.0, dir_kappa=200.0)
    add_run("bundle_sigma100", first_hit_per_pmt=True, outlier_frac=0.01, bundle_sigma_m=100.0, bundle_sigma_angle_deg=2.0, dir_kappa=200.0)
    add_run("dir_kappa0", first_hit_per_pmt=True, outlier_frac=0.01, bundle_sigma_m=50.0, bundle_sigma_angle_deg=2.0, dir_kappa=0.0)
    add_run("allhits_eps001_fast", first_hit_per_pmt=False, outlier_frac=0.01, bundle_sigma_m=50.0, bundle_sigma_angle_deg=2.0, dir_kappa=200.0, restarts_scale=0.5)

    summary = []
    for i, r in enumerate(runs):
        sel = ModelComparisonSelection(**r["selection"])
        pri = ModelComparisonPriors(**r["priors"])
        lik = ModelComparisonLikelihood(**r["likelihood"])

        scale = float(r["restarts_scale"])
        cfg = ModelComparisonConfig(
            selection=sel,
            priors=pri,
            likelihood=lik,
            restarts_h1=1,
            restarts_h2=max(2, int(round(8 * scale))),
            restarts_h5=max(2, int(round(8 * scale))),
            restarts_h6=max(3, int(round(12 * scale))),
            restarts_h7=max(2, int(round(4 * scale))),
            restarts_sign=max(1, int(round(2 * scale))),
            seed=int(args.seed) + i * 1000,
        )

        out_path = out_dir / f"{r['name']}.json"
        res = run_model_comparison(args.event, out_path=out_path, config=cfg)

        bf = res.get("bayes_factors", {})
        summary.append(
            {
                "name": r["name"],
                "n_hits": res["inputs"]["n_hits"],
                "selection": asdict(sel),
                "likelihood": asdict(lik),
                "priors": asdict(pri),
                "bayes_factors": {k: bf.get(k) for k in sorted(bf.keys()) if k.startswith("logB_")},
                "direction_sign_test": res.get("direction_sign_test", {}),
                "path": str(out_path),
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

