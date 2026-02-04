from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnergyCalibration:
    """Simple calibration: NTrigPMT ~ Normal(mu(logE), sigma(logE))."""

    # Polynomial coefficients for mu(log10(E/PeV)) and sigma(log10(E/PeV)).
    mu_coef: np.ndarray  # shape (deg+1,)
    sigma_coef: np.ndarray  # shape (deg+1,)
    energies_pev: np.ndarray
    mu_points: np.ndarray
    sigma_points: np.ndarray


def load_figure2_calibration(path: str | Path) -> EnergyCalibration:
    """Fit a smooth calibration from the official Figure 2 source data.

    Expects columns: E (PeV), syst, bin_center, hist_val
    """
    path = Path(path)
    df = pd.read_csv(path)
    df = df[df["syst"] == "no_syst"].copy()

    energies = np.array(sorted(df["E"].unique()), dtype=np.float64)
    loge = np.log10(energies)

    mu = []
    sig = []
    for E in energies:
        sub = df[df["E"] == E].sort_values("bin_center")
        x = sub["bin_center"].to_numpy(dtype=np.float64)
        w = sub["hist_val"].to_numpy(dtype=np.float64)
        w = w / np.sum(w)
        m = float(np.sum(x * w))
        v = float(np.sum((x - m) ** 2 * w))
        mu.append(m)
        sig.append(math.sqrt(max(v, 1e-12)))

    mu = np.array(mu, dtype=np.float64)
    sig = np.array(sig, dtype=np.float64)

    # Degree-2 polynomials exactly interpolate 3 points in log10(E).
    mu_coef = np.polyfit(loge, mu, deg=2)
    sigma_coef = np.polyfit(loge, sig, deg=2)

    return EnergyCalibration(
        mu_coef=mu_coef,
        sigma_coef=sigma_coef,
        energies_pev=energies,
        mu_points=mu,
        sigma_points=sig,
    )


def _polyval(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.polyval(coef, x)


def energy_posterior_from_ntrigpmt(
    *,
    n_trig_pmt: int,
    calibration: EnergyCalibration,
    log10_e_min: float = 0.5,  # 3.16 PeV
    log10_e_max: float = 3.7,  # 5,000 PeV
    grid: int = 4096,
    prior: str = "log_uniform",  # or "uniform"
    seed: int = 230213,
    n_samples: int = 100_000,
) -> dict[str, Any]:
    """Approximate posterior over E (PeV) given NTrigPMT using a simple calibration."""
    n_trig_pmt = int(n_trig_pmt)
    loge = np.linspace(float(log10_e_min), float(log10_e_max), int(grid), dtype=np.float64)
    mu = _polyval(calibration.mu_coef, loge)
    sigma = np.maximum(_polyval(calibration.sigma_coef, loge), 1.0)

    # Likelihood: Normal(N | mu(E), sigma(E)).
    z = (float(n_trig_pmt) - mu) / sigma
    log_like = -0.5 * (z * z + np.log(2.0 * np.pi) + 2.0 * np.log(sigma))

    if prior == "uniform":
        log_prior = np.zeros_like(log_like)
    elif prior == "log_uniform":
        # uniform in log10(E) -> constant in this grid parameterization.
        log_prior = np.zeros_like(log_like)
    else:
        raise ValueError("prior must be 'uniform' or 'log_uniform'")

    log_post = log_like + log_prior
    log_post -= np.max(log_post)
    w = np.exp(log_post)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        raise RuntimeError("Energy posterior normalization failed.")
    w /= w_sum

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(np.arange(loge.size), size=int(n_samples), replace=True, p=w)
    loge_s = loge[idx]
    e_pev = 10.0 ** loge_s

    # Summaries.
    def q(p: float) -> float:
        return float(np.quantile(e_pev, p))

    return {
        "samples_pev": e_pev,
        "summary_pev": {
            "q16": q(0.16),
            "q50": q(0.50),
            "q84": q(0.84),
            "mean": float(np.mean(e_pev)),
        },
        "calibration": {
            "energies_pev": calibration.energies_pev.tolist(),
            "mu_points": calibration.mu_points.tolist(),
            "sigma_points": calibration.sigma_points.tolist(),
            "mu_coef": calibration.mu_coef.tolist(),
            "sigma_coef": calibration.sigma_coef.tolist(),
        },
        "config": {
            "n_trig_pmt": n_trig_pmt,
            "log10_e_min": float(log10_e_min),
            "log10_e_max": float(log10_e_max),
            "grid": int(grid),
            "prior": prior,
            "seed": int(seed),
            "n_samples": int(n_samples),
        },
    }


def save_energy_posterior_npz(out_path: str | Path, payload: dict[str, Any]) -> Path:
    """Save samples to NPZ and metadata (without samples) as JSON string."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(payload)
    samples = np.asarray(meta.pop("samples_pev"))
    np.savez_compressed(out_path, samples_pev=samples, meta=json.dumps(meta, indent=2))
    return out_path
