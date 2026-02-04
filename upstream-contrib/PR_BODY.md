### Summary

Adds an optional, self-contained “Bayesian-style” end-to-end posterior pipeline on top of the KM3-230213A public release artifacts:

- Timing-only hit-level track posterior (importance sampling) in detector coordinates
- Sky localization posterior (RA/Dec) using VOEvent time/location; includes a “stat-only” version and a version convolved with the reported ~1.5° absolute-orientation systematic
- 1D stochastic-loss “tomography” via the emission-point-along-track construction (binned Poisson model + MCMC)
- Approximate muon-energy posterior from the public Figure-2 NTrigPMT distribution

### How to run

```bash
python src/run_end_to_end_posterior.py --quick --cores 16
```

Full run:

```bash
python src/run_end_to_end_posterior.py --cores 16
```

Outputs go to `data/results/bayesian_posterior/` by default.

### Notes / scope

- Uses only publicly released files: `KM3-230213A_allhits.json.gz`, `KM3-230213A_voevent.xml`, `figure2.csv`.
- Does not attempt to replace the collaboration reconstruction; it produces a reusable posterior object and a transparent systematic budget split (stat vs abs orientation; plus a simple late-light/timing nuisance marginalization).

