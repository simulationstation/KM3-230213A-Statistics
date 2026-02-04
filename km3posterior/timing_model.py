from __future__ import annotations

import numpy as np
from scipy.special import erfc, erfcx


def logpdf_emg(x: np.ndarray, *, sigma_ns: float, tau_ns: float, mu_ns: float = 0.0) -> np.ndarray:
    """Exponentially-modified Gaussian log-pdf.

    Models a prompt Gaussian timing core convolved with an exponential late-light tail.
    Parameterization uses `tau_ns` as the exponential mean.
    """
    x = np.asarray(x, dtype=np.float64)
    sigma_ns = float(sigma_ns)
    tau_ns = float(tau_ns)

    if not (sigma_ns > 0.0 and tau_ns > 0.0):
        return np.full_like(x, -np.inf, dtype=np.float64)

    lam = 1.0 / tau_ns

    x0 = x - float(mu_ns)
    z = (lam * sigma_ns * sigma_ns - x0) / (np.sqrt(2.0) * sigma_ns)
    # log(erfc(z)) computed stably:
    # - for z>0: use erfcx(z)=exp(z^2)*erfc(z) to avoid underflow in erfc(z)
    # - for z<=0: use erfc(z) directly to avoid overflow in erfcx(z)
    log_erfc = np.empty_like(z, dtype=np.float64)
    pos = z > 0
    log_erfc[pos] = np.log(erfcx(z[pos])) - z[pos] * z[pos]
    log_erfc[~pos] = np.log(erfc(z[~pos]))

    return np.log(lam / 2.0) + 0.5 * lam * (lam * sigma_ns * sigma_ns - 2.0 * x0) + log_erfc


def logpdf_emg_with_outliers(
    x: np.ndarray,
    *,
    sigma_ns: float,
    tau_ns: float,
    outlier_frac: float,
    outlier_min_ns: float,
    outlier_max_ns: float,
    mu_ns: float = 0.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    outlier_frac = float(outlier_frac)
    if not (0.0 <= outlier_frac < 1.0):
        return np.full_like(x, -np.inf, dtype=np.float64)
    if outlier_max_ns <= outlier_min_ns:
        return np.full_like(x, -np.inf, dtype=np.float64)

    log_core = logpdf_emg(x, sigma_ns=sigma_ns, tau_ns=tau_ns, mu_ns=mu_ns)
    log_uni = -np.log(float(outlier_max_ns - outlier_min_ns))
    return np.logaddexp(np.log1p(-outlier_frac) + log_core, np.log(outlier_frac) + log_uni)
