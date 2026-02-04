# Upstream contribution bundle (KM3NeT open-data)

This folder contains a ready-to-apply patch against the official KM3NeT public repository:

- `KM3NeT/KM3-230213A-data` (GitHub) / the corresponding GitLab open-data mirror.

## What’s in the patch

`0001-add-end-to-end-bayesian-style-posterior-track-loss-sky.patch` adds:

- `src/run_end_to_end_posterior.py`: end-to-end pipeline (track posterior + loss posterior + energy posterior + sky localization)
- `src/km3posterior/`: self-contained analysis helpers
- Minimal `.gitignore` entries
- A short README mention of the new script

Outputs are written under `data/results/` by default so they stay out of version control.

## Applying it (for maintainers)

```bash
git clone https://github.com/KM3NeT/KM3-230213A-data.git
cd KM3-230213A-data
git checkout -b bayesian-posterior
git am < /path/to/0001-add-end-to-end-bayesian-style-posterior-track-loss-sky.patch
python src/run_end_to_end_posterior.py --quick --cores 16
```

## Notes

- No extra dependencies beyond the repo’s existing scientific stack (uses `astropy`, `numpy`, `scipy`, `pandas`).
- Sky mapping uses the VOEvent location/time and automatically chooses the sign convention by matching the VOEvent RA/Dec.

