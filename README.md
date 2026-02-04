# KM3-230213A-Statistics
Goal: a reproducible, end-to-end Bayesian(-ish) posterior over:

- Track direction in detector coordinates (timing-only, hit-level likelihood)
- Muon energy at detector (calibrated from the public Figure 2 NTrigPMT distribution)
- Stochastic-loss profile along the muon path (1D emission-point "tomography")

This repo is designed to work directly off the official KM3NeT public release repo.

## Upstream data
The official open-data repository is expected at `upstream-KM3-230213A-data/` (clone it yourself or let the agent do it):

- `data/event/KM3-230213A_allhits.json.gz`
- `data/supplementary/figuresource/figure2.csv`
- `data/event/KM3-230213A_voevent.xml`

## Quick start (end-to-end)
From the repo root:

```powershell
python scripts/run_end_to_end.py --upstream upstream-KM3-230213A-data --out results --cores 16
```

For a fast smoke run:

```powershell
python scripts/run_end_to_end.py --upstream upstream-KM3-230213A-data --out results-quick --cores 16 --quick
```

## Outputs
The pipeline writes:

- `results/report.json` (human-readable summary + systematic budget)
- `results/track_posterior.npz` (timing-only posterior samples)
- `results/loss_posterior.npz` (stochastic-loss posterior samples from emission hist)
- `results/energy_posterior.npz` (energy posterior samples + metadata)
- `results/sky_posterior_stat.npz` (RA/Dec samples mapped from detector coords using VOEvent time/location)
- `results/sky_posterior_total.npz` (RA/Dec samples convolved with the ~1.5° absolute-orientation systematic)

## Notes (what this *is* / *isn't*)
- Track inference uses an exponentially-modified Gaussian timing model (prompt + late light) and marginalizes the late-light scale.
- Loss inference uses the "emission point along track" construction from the official notebook, but fits it with a Bayesian binned model (baseline + 3 loss components).
- Energy inference is an approximation from NTrigPMT (Figure 2 source); it is not a full detector-response calorimetric model.
