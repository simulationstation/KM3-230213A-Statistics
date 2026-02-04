from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description="Collect a compact summary of the experiment suite outputs.")
    p.add_argument("--suite-dir", default="results-suite", help="Directory containing suite outputs")
    p.add_argument("--out", default="results-suite/experiments_summary.json", help="Output JSON path")
    args = p.parse_args()

    suite = Path(args.suite_dir)
    out_path = Path(args.out)

    timing_full = _load_json(suite / "model_comparison_full.json")
    loss_count = _load_json(suite / "loss_model_comparison_count.json")
    loss_tot = _load_json(suite / "loss_model_comparison_tot.json")
    timing_sweep = _load_json(suite / "timing_sweep" / "summary.json")

    def pick_bf(r: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        bf = r.get("bayes_factors", {})
        return {k: bf.get(k) for k in keys if k in bf}

    out = {
        "timing_full": {
            "n_hits": timing_full["inputs"]["n_hits"],
            "bayes_factors": pick_bf(
                timing_full,
                [
                    "logB_H2_over_H1",
                    "logB_H5_over_H1",
                    "logB_H6_over_H1",
                    "logB_H7_over_H1",
                    "logB_H1_over_H3",
                    "logB_H1_over_H4",
                    "logB_H1_over_H1_dir_minus",
                ],
            ),
            "H4_beta": timing_full["models"]["H4_beta_track"]["map"]["beta"],
        },
        "loss_count": {
            "weight": loss_count["inputs"]["weight"],
            "delta_loglike": loss_count["comparisons"]["delta_loglike"],
            "delta_BIC_full_minus_base": loss_count["comparisons"]["delta_BIC_full_minus_base"],
        },
        "loss_tot": {
            "weight": loss_tot["inputs"]["weight"],
            "delta_loglike": loss_tot["comparisons"]["delta_loglike"],
            "delta_BIC_full_minus_base": loss_tot["comparisons"]["delta_BIC_full_minus_base"],
        },
        "timing_sweep": [
            {
                "name": r["name"],
                "n_hits": r["n_hits"],
                "bayes_factors": {k: r["bayes_factors"].get(k) for k in r["bayes_factors"] if k in {
                    "logB_H2_over_H1",
                    "logB_H5_over_H1",
                    "logB_H6_over_H1",
                    "logB_H7_over_H1",
                    "logB_H1_over_H4",
                    "logB_H1_over_H1_dir_minus",
                }},
            }
            for r in timing_sweep
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

