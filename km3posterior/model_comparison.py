from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .constants import C_LIGHT, V_LIGHT, WATER_INDEX
from .data import EventData, load_event_json_gz
from .geometry import cherenkov_times_ns, unit_from_theta_phi
from .timing_model import logpdf_emg


@dataclass(frozen=True)
class ModelComparisonSelection:
    triggered_only: bool = True
    first_hit_per_pmt: bool = True


@dataclass(frozen=True)
class ModelComparisonPriors:
    sigma_pos_m: float = 100.0
    sigma_t_ns: float = 2_000.0
    dir_kappa: float = 200.0

    log_sigma_mu: float = math.log(3.0)
    log_sigma_sigma: float = 0.7
    log_tau_mu: float = math.log(60.0)
    log_tau_sigma: float = 0.7

    bundle_sigma_offset_m: float = 50.0


@dataclass(frozen=True)
class ModelComparisonLikelihood:
    outlier_frac: float = 0.01


@dataclass(frozen=True)
class ModelComparisonConfig:
    selection: ModelComparisonSelection = ModelComparisonSelection()
    priors: ModelComparisonPriors = ModelComparisonPriors()
    likelihood: ModelComparisonLikelihood = ModelComparisonLikelihood()

    # Multiple restarts helps avoid local optima in the bundle model.
    restarts_h1: int = 1
    restarts_h2: int = 8

    # Random seed for restarts.
    seed: int = 230213


def _first_hit_per_pmt_indices(evt: EventData, *, triggered_only: bool) -> np.ndarray:
    idx = np.arange(evt.t_ns.size)
    if triggered_only:
        idx = idx[evt.triggered]
    order = np.argsort(evt.t_ns[idx], kind="mergesort")
    idx_sorted = idx[order]
    pmt_sorted = evt.pmt_key[idx_sorted]
    _, first = np.unique(pmt_sorted, return_index=True)
    return idx_sorted[first]


def select_hits_for_model_comparison(evt: EventData, sel: ModelComparisonSelection) -> np.ndarray:
    if sel.first_hit_per_pmt:
        return _first_hit_per_pmt_indices(evt, triggered_only=bool(sel.triggered_only))
    idx = np.arange(evt.t_ns.size)
    if sel.triggered_only:
        idx = idx[evt.triggered]
    return idx


def _log_normal_vec(x: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = float(sigma)
    if not (sigma > 0.0):
        return -math.inf
    d = int(x.size)
    dx = x - mu
    return float(-0.5 * (d * math.log(2.0 * math.pi * sigma * sigma) + float(dx @ dx) / (sigma * sigma)))


def _log_normal_scalar(x: float, mu: float, sigma: float) -> float:
    sigma = float(sigma)
    if not (sigma > 0.0):
        return -math.inf
    dx = float(x) - float(mu)
    return float(-0.5 * (math.log(2.0 * math.pi * sigma * sigma) + (dx * dx) / (sigma * sigma)))


def _log_vmf_on_sphere(u: np.ndarray, mu: np.ndarray, kappa: float) -> float:
    """vMF density on S^2 with full normalization (in unit-vector measure dΩ)."""
    u = np.asarray(u, dtype=np.float64).reshape(3)
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    u /= np.linalg.norm(u)
    mu /= np.linalg.norm(mu)
    kappa = float(kappa)
    if kappa < 1e-10:
        # Uniform on sphere.
        return -math.log(4.0 * math.pi)

    # C(kappa) = kappa / (4π sinh(kappa))
    # log(sinh(kappa)) computed stably for large kappa.
    if kappa > 30.0:
        log_sinh = kappa - math.log(2.0) + math.log1p(-math.exp(-2.0 * kappa))
    else:
        log_sinh = math.log(math.sinh(kappa))
    log_c = math.log(kappa) - math.log(4.0 * math.pi) - log_sinh
    return float(log_c + kappa * float(u @ mu))


def _log_dir_prior_theta_phi(theta: float, phi: float, *, reco_dir: np.ndarray, kappa: float) -> float:
    """vMF prior over direction, expressed in (theta,phi) coordinates (includes Jacobian sin(theta))."""
    theta = float(theta)
    if not (0.0 < theta < math.pi):
        return -math.inf
    u = unit_from_theta_phi(theta, float(phi))
    return _log_vmf_on_sphere(u, reco_dir, float(kappa)) + math.log(math.sin(theta))


def _finite_difference_hessian(f, x0: np.ndarray, step: np.ndarray) -> np.ndarray:
    x0 = np.asarray(x0, dtype=np.float64)
    step = np.asarray(step, dtype=np.float64)
    n = int(x0.size)

    f0 = float(f(x0))
    H = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        ei = np.zeros(n, dtype=np.float64)
        ei[i] = step[i]
        f_p = float(f(x0 + ei))
        f_m = float(f(x0 - ei))
        H[i, i] = (f_p - 2.0 * f0 + f_m) / (step[i] * step[i])

    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n, dtype=np.float64)
            ej = np.zeros(n, dtype=np.float64)
            ei[i] = step[i]
            ej[j] = step[j]

            f_pp = float(f(x0 + ei + ej))
            f_pm = float(f(x0 + ei - ej))
            f_mp = float(f(x0 - ei + ej))
            f_mm = float(f(x0 - ei - ej))

            H_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step[i] * step[j])
            H[i, j] = H_ij
            H[j, i] = H_ij

    return H


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _orthonormal_basis_perp(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float64).reshape(3)
    u /= np.linalg.norm(u)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64) if abs(float(u[2])) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    b1 = np.cross(ref, u)
    n1 = float(np.linalg.norm(b1))
    if n1 < 1e-12:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        b1 = np.cross(ref, u)
        n1 = float(np.linalg.norm(b1))
    b1 /= max(n1, 1e-12)
    b2 = np.cross(u, b1)
    b2 /= max(float(np.linalg.norm(b2)), 1e-12)
    return b1, b2


def _loglik_single_track(
    x: np.ndarray,
    *,
    hit_pos_m: np.ndarray,
    hit_t_ns: np.ndarray,
    t_min_ns: float,
    t_max_ns: float,
    outlier_frac: float,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    pos = x[0:3]
    theta = float(x[3])
    phi = float(x[4])
    t0 = float(x[5])
    sigma_ns = float(math.exp(float(x[6])))
    tau_ns = float(math.exp(float(x[7])))

    u = unit_from_theta_phi(theta, phi)
    t_pred, *_ = cherenkov_times_ns(track_pos_m=pos, track_dir=u, track_t_ns=t0, hit_pos_m=hit_pos_m)
    res = hit_t_ns - t_pred

    log_core = logpdf_emg(res, sigma_ns=sigma_ns, tau_ns=tau_ns, mu_ns=0.0)
    log_uni = -math.log(float(t_max_ns - t_min_ns))
    eps = float(outlier_frac)
    if not (0.0 <= eps < 1.0):
        return -math.inf
    return float(np.logaddexp(np.log1p(-eps) + log_core, math.log(eps) + log_uni).sum())


def _loglik_bundle_two_tracks(
    x: np.ndarray,
    *,
    hit_pos_m: np.ndarray,
    hit_t_ns: np.ndarray,
    t_min_ns: float,
    t_max_ns: float,
    outlier_frac: float,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    pos = x[0:3]
    theta = float(x[3])
    phi = float(x[4])
    t0 = float(x[5])
    sigma_ns = float(math.exp(float(x[6])))
    tau_ns = float(math.exp(float(x[7])))
    off_a = float(x[8])
    off_b = float(x[9])
    logit_w = float(x[10])

    w = _sigmoid(logit_w)

    u = unit_from_theta_phi(theta, phi)
    b1, b2 = _orthonormal_basis_perp(u)
    pos2 = pos + off_a * b1 + off_b * b2

    t_pred1, *_ = cherenkov_times_ns(track_pos_m=pos, track_dir=u, track_t_ns=t0, hit_pos_m=hit_pos_m)
    t_pred2, *_ = cherenkov_times_ns(track_pos_m=pos2, track_dir=u, track_t_ns=t0, hit_pos_m=hit_pos_m)
    res1 = hit_t_ns - t_pred1
    res2 = hit_t_ns - t_pred2

    log1 = logpdf_emg(res1, sigma_ns=sigma_ns, tau_ns=tau_ns, mu_ns=0.0)
    log2 = logpdf_emg(res2, sigma_ns=sigma_ns, tau_ns=tau_ns, mu_ns=0.0)
    log_mix = np.logaddexp(math.log(w) + log1, math.log1p(-w) + log2)

    log_uni = -math.log(float(t_max_ns - t_min_ns))
    eps = float(outlier_frac)
    if not (0.0 <= eps < 1.0):
        return -math.inf
    return float(np.logaddexp(np.log1p(-eps) + log_mix, math.log(eps) + log_uni).sum())


def _loglik_cascade(
    x: np.ndarray,
    *,
    hit_pos_m: np.ndarray,
    hit_t_ns: np.ndarray,
    t_min_ns: float,
    t_max_ns: float,
    outlier_frac: float,
) -> float:
    """Point-source (cascade-like) timing model.

    t_pred = t0 + |r_hit - r0| / v_g (seawater group velocity).
    """
    x = np.asarray(x, dtype=np.float64)
    pos = x[0:3]
    t0 = float(x[3])
    sigma_ns = float(math.exp(float(x[4])))
    tau_ns = float(math.exp(float(x[5])))

    d = np.linalg.norm(hit_pos_m - pos[None, :], axis=1)
    t_pred = t0 + d / float(V_LIGHT)
    res = hit_t_ns - t_pred

    log_core = logpdf_emg(res, sigma_ns=sigma_ns, tau_ns=tau_ns, mu_ns=0.0)
    log_uni = -math.log(float(t_max_ns - t_min_ns))
    eps = float(outlier_frac)
    if not (0.0 <= eps < 1.0):
        return -math.inf
    return float(np.logaddexp(np.log1p(-eps) + log_core, math.log(eps) + log_uni).sum())


def _cherenkov_times_beta_ns(
    *,
    track_pos_m: np.ndarray,
    track_dir: np.ndarray,
    track_t_ns: float,
    hit_pos_m: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generalization of the earliest-arrival model to a particle with speed beta*c.

    Uses Cherenkov angle cos(theta_c)=1/(n*beta); for beta<=1/n, returns inf predicted times.
    """
    track_pos_m = np.asarray(track_pos_m, dtype=np.float64).reshape(3)
    track_dir = np.asarray(track_dir, dtype=np.float64).reshape(3)
    track_dir = track_dir / np.linalg.norm(track_dir)
    hit_pos_m = np.asarray(hit_pos_m, dtype=np.float64)

    beta = float(beta)
    if not (beta > 0.0):
        n = hit_pos_m.shape[0]
        inf = np.full(n, np.inf, dtype=np.float64)
        return inf, inf, inf, inf

    cos_tc = 1.0 / (float(WATER_INDEX) * beta)
    if cos_tc >= 1.0:
        n = hit_pos_m.shape[0]
        inf = np.full(n, np.inf, dtype=np.float64)
        return inf, inf, inf, inf

    sin_tc = math.sqrt(max(0.0, 1.0 - cos_tc * cos_tc))
    tan_tc = sin_tc / cos_tc

    v = hit_pos_m - track_pos_m[None, :]
    l = v @ track_dir  # (N,)
    v2 = np.einsum("ij,ij->i", v, v)
    k2 = np.maximum(v2 - l * l, 0.0)

    d_closest_m = np.sqrt(k2)
    d_photon_m = d_closest_m / max(sin_tc, 1e-12)
    d_trk_m = l - d_closest_m / max(tan_tc, 1e-12)

    t_pred_ns = float(track_t_ns) + d_trk_m / (beta * float(C_LIGHT)) + d_photon_m / float(V_LIGHT)
    return t_pred_ns, d_closest_m, d_photon_m, d_trk_m


def _loglik_single_track_beta(
    x: np.ndarray,
    *,
    hit_pos_m: np.ndarray,
    hit_t_ns: np.ndarray,
    t_min_ns: float,
    t_max_ns: float,
    outlier_frac: float,
    beta_min: float,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    pos = x[0:3]
    theta = float(x[3])
    phi = float(x[4])
    t0 = float(x[5])
    sigma_ns = float(math.exp(float(x[6])))
    tau_ns = float(math.exp(float(x[7])))
    z_beta = float(x[8])

    w = _sigmoid(z_beta)
    beta = float(beta_min + (1.0 - beta_min) * w)

    u = unit_from_theta_phi(theta, phi)
    t_pred, *_ = _cherenkov_times_beta_ns(track_pos_m=pos, track_dir=u, track_t_ns=t0, hit_pos_m=hit_pos_m, beta=beta)
    res = hit_t_ns - t_pred

    log_core = logpdf_emg(res, sigma_ns=sigma_ns, tau_ns=tau_ns, mu_ns=0.0)
    log_uni = -math.log(float(t_max_ns - t_min_ns))
    eps = float(outlier_frac)
    if not (0.0 <= eps < 1.0):
        return -math.inf
    return float(np.logaddexp(np.log1p(-eps) + log_core, math.log(eps) + log_uni).sum())


def _logprior_h1(
    x: np.ndarray,
    *,
    evt: EventData,
    pri: ModelComparisonPriors,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    pos = x[0:3]
    theta = float(x[3])
    phi = float(x[4])
    t0 = float(x[5])
    log_sigma = float(x[6])
    log_tau = float(x[7])

    lp = 0.0
    lp += _log_normal_vec(pos, evt.reco_pos_m, pri.sigma_pos_m)
    lp += _log_dir_prior_theta_phi(theta, phi, reco_dir=evt.reco_dir, kappa=pri.dir_kappa)
    lp += _log_normal_scalar(t0, float(evt.reco_t_ns), pri.sigma_t_ns)
    lp += _log_normal_scalar(log_sigma, pri.log_sigma_mu, pri.log_sigma_sigma)
    lp += _log_normal_scalar(log_tau, pri.log_tau_mu, pri.log_tau_sigma)
    return float(lp)


def _logprior_h2_bundle(
    x: np.ndarray,
    *,
    evt: EventData,
    pri: ModelComparisonPriors,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    pos = x[0:3]
    theta = float(x[3])
    phi = float(x[4])
    t0 = float(x[5])
    log_sigma = float(x[6])
    log_tau = float(x[7])
    off_a = float(x[8])
    off_b = float(x[9])
    logit_w = float(x[10])

    w = _sigmoid(logit_w)

    lp = 0.0
    lp += _log_normal_vec(pos, evt.reco_pos_m, pri.sigma_pos_m)
    lp += _log_dir_prior_theta_phi(theta, phi, reco_dir=evt.reco_dir, kappa=pri.dir_kappa)
    lp += _log_normal_scalar(t0, float(evt.reco_t_ns), pri.sigma_t_ns)
    lp += _log_normal_scalar(log_sigma, pri.log_sigma_mu, pri.log_sigma_sigma)
    lp += _log_normal_scalar(log_tau, pri.log_tau_mu, pri.log_tau_sigma)

    # Bundle offset prior in the transverse plane (2D Gaussian).
    lp += _log_normal_scalar(off_a, 0.0, pri.bundle_sigma_offset_m)
    lp += _log_normal_scalar(off_b, 0.0, pri.bundle_sigma_offset_m)

    # Uniform prior on w in [0,1] transformed from logit(w): p(z)=w(1-w).
    lp += math.log(max(w, 1e-300)) + math.log(max(1.0 - w, 1e-300))
    return float(lp)


def _logprior_h3_cascade(
    x: np.ndarray,
    *,
    evt: EventData,
    pri: ModelComparisonPriors,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    pos = x[0:3]
    t0 = float(x[3])
    log_sigma = float(x[4])
    log_tau = float(x[5])

    lp = 0.0
    lp += _log_normal_vec(pos, evt.reco_pos_m, pri.sigma_pos_m)
    lp += _log_normal_scalar(t0, float(evt.reco_t_ns), pri.sigma_t_ns)
    lp += _log_normal_scalar(log_sigma, pri.log_sigma_mu, pri.log_sigma_sigma)
    lp += _log_normal_scalar(log_tau, pri.log_tau_mu, pri.log_tau_sigma)
    return float(lp)


def _logprior_h4_beta_track(
    x: np.ndarray,
    *,
    evt: EventData,
    pri: ModelComparisonPriors,
    beta_min: float,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    pos = x[0:3]
    theta = float(x[3])
    phi = float(x[4])
    t0 = float(x[5])
    log_sigma = float(x[6])
    log_tau = float(x[7])
    z_beta = float(x[8])

    lp = 0.0
    lp += _log_normal_vec(pos, evt.reco_pos_m, pri.sigma_pos_m)
    lp += _log_dir_prior_theta_phi(theta, phi, reco_dir=evt.reco_dir, kappa=pri.dir_kappa)
    lp += _log_normal_scalar(t0, float(evt.reco_t_ns), pri.sigma_t_ns)
    lp += _log_normal_scalar(log_sigma, pri.log_sigma_mu, pri.log_sigma_sigma)
    lp += _log_normal_scalar(log_tau, pri.log_tau_mu, pri.log_tau_sigma)

    # Uniform prior in beta on [beta_min, 1] using a logistic transform beta=beta_min+(1-beta_min)*sigmoid(z).
    # Adds the Jacobian |dbeta/dz| = (1-beta_min)*w*(1-w).
    w = _sigmoid(z_beta)
    jac = (1.0 - float(beta_min)) * w * (1.0 - w)
    lp += math.log(max(jac, 1e-300))
    return float(lp)


def _pack_initial_h1(evt: EventData) -> np.ndarray:
    # Start at published reco.
    # Convert reco_dir to (theta,phi).
    u = evt.reco_dir
    theta = float(math.acos(float(np.clip(u[2], -1.0, 1.0))))
    phi = float(math.atan2(float(u[1]), float(u[0])) % (2.0 * math.pi))
    return np.array(
        [
            evt.reco_pos_m[0],
            evt.reco_pos_m[1],
            evt.reco_pos_m[2],
            theta,
            phi,
            float(evt.reco_t_ns),
            math.log(3.0),
            math.log(60.0),
        ],
        dtype=np.float64,
    )


def _pack_initial_h2(evt: EventData) -> np.ndarray:
    x = _pack_initial_h1(evt)
    # Append off_a, off_b, logit_w.
    return np.concatenate([x, np.array([0.0, 0.0, 0.0], dtype=np.float64)], axis=0)


def _pack_initial_h3(evt: EventData) -> np.ndarray:
    return np.array(
        [
            evt.reco_pos_m[0],
            evt.reco_pos_m[1],
            evt.reco_pos_m[2],
            float(evt.reco_t_ns),
            math.log(3.0),
            math.log(60.0),
        ],
        dtype=np.float64,
    )


def _pack_initial_h4(evt: EventData) -> np.ndarray:
    # z_beta=+8 puts beta extremely close to 1 without sitting exactly at the boundary.
    x = _pack_initial_h1(evt)
    return np.concatenate([x, np.array([8.0], dtype=np.float64)], axis=0)


def _fit_map_generic(
    *,
    x0: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
    neg_log_joint,
    step: np.ndarray,
    restarts: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    best = None
    best_fun = float("inf")

    for r in range(int(restarts)):
        if r == 0:
            x_init = x0.copy()
        else:
            x_init = x0 + rng.normal(scale=step * 2.0, size=x0.shape[0])
            # Keep theta/phi within bounds by clipping/modulo.
            if x_init.size >= 5:
                x_init[3] = float(np.clip(x_init[3], 1e-6, math.pi - 1e-6))
                x_init[4] = float(x_init[4] % (2.0 * math.pi))

        opt = minimize(neg_log_joint, x_init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 600})
        if opt.success and float(opt.fun) < best_fun:
            best_fun = float(opt.fun)
            best = opt

    if best is None:
        raise RuntimeError("MAP optimization failed for all restarts.")

    x_map = np.asarray(best.x, dtype=np.float64)
    H = _finite_difference_hessian(neg_log_joint, x_map, step)

    # Ensure PD for logdet; add jitter if needed.
    jitter = 1e-8
    for _ in range(12):
        Hj = H + jitter * np.eye(H.shape[0])
        sign, logdet = np.linalg.slogdet(Hj)
        if sign > 0 and np.isfinite(logdet):
            break
        jitter *= 10.0
    else:
        raise RuntimeError("Failed to compute stable logdet(H) for Laplace evidence.")

    return {"map_x": x_map, "hessian": H, "hessian_jitter": jitter, "neg_log_joint_at_map": float(best_fun), "opt": best}


def laplace_log_evidence(map_x: np.ndarray, hessian_neg_log_joint: np.ndarray, *, log_joint_at_map: float) -> float:
    H = np.asarray(hessian_neg_log_joint, dtype=np.float64)
    d = int(H.shape[0])
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0:
        raise ValueError("Hessian must be positive definite for Laplace evidence.")
    return float(log_joint_at_map + 0.5 * d * math.log(2.0 * math.pi) - 0.5 * float(logdet))


def run_model_comparison(
    evt_path: str | Path,
    *,
    out_path: str | Path | None = None,
    config: ModelComparisonConfig | None = None,
) -> dict[str, Any]:
    config = config or ModelComparisonConfig()
    evt = load_event_json_gz(evt_path)
    idx = select_hits_for_model_comparison(evt, config.selection)
    hit_pos = evt.pos_m[idx]
    hit_t = evt.t_ns[idx]

    t_min = float(np.min(hit_t))
    t_max = float(np.max(hit_t))
    if not (t_max > t_min):
        raise RuntimeError("Degenerate hit time range for model comparison selection.")

    pri = config.priors
    lik = config.likelihood

    # --- H1: single track
    x0_h1 = _pack_initial_h1(evt)
    bounds_h1 = [
        (None, None),
        (None, None),
        (None, None),
        (1e-6, math.pi - 1e-6),
        (0.0, 2.0 * math.pi),
        (float(evt.reco_t_ns) - 10_000.0, float(evt.reco_t_ns) + 10_000.0),
        (math.log(0.2), math.log(30.0)),
        (math.log(1.0), math.log(2_000.0)),
    ]

    def neg_log_joint_h1(x: np.ndarray) -> float:
        lp = _logprior_h1(x, evt=evt, pri=pri)
        if not np.isfinite(lp):
            return float("inf")
        ll = _loglik_single_track(
            x,
            hit_pos_m=hit_pos,
            hit_t_ns=hit_t,
            t_min_ns=t_min,
            t_max_ns=t_max,
            outlier_frac=lik.outlier_frac,
        )
        if not np.isfinite(ll):
            return float("inf")
        return -(ll + lp)

    step_h1 = np.array([0.5, 0.5, 0.5, 2e-4, 2e-4, 1.0, 1e-2, 1e-2], dtype=np.float64)
    fit_h1 = _fit_map_generic(
        x0=x0_h1,
        bounds=bounds_h1,
        neg_log_joint=neg_log_joint_h1,
        step=step_h1,
        restarts=config.restarts_h1,
        seed=config.seed,
    )
    log_joint_h1 = -float(fit_h1["neg_log_joint_at_map"])
    logZ_h1 = laplace_log_evidence(fit_h1["map_x"], fit_h1["hessian"] + fit_h1["hessian_jitter"] * np.eye(8), log_joint_at_map=log_joint_h1)

    # --- H2: 2-track bundle (shared direction and time; transverse offset + mixture weight)
    x0_h2 = _pack_initial_h2(evt)
    bounds_h2 = bounds_h1 + [
        (-300.0, 300.0),
        (-300.0, 300.0),
        (-8.0, 8.0),
    ]

    def neg_log_joint_h2(x: np.ndarray) -> float:
        lp = _logprior_h2_bundle(x, evt=evt, pri=pri)
        if not np.isfinite(lp):
            return float("inf")
        ll = _loglik_bundle_two_tracks(
            x,
            hit_pos_m=hit_pos,
            hit_t_ns=hit_t,
            t_min_ns=t_min,
            t_max_ns=t_max,
            outlier_frac=lik.outlier_frac,
        )
        if not np.isfinite(ll):
            return float("inf")
        return -(ll + lp)

    step_h2 = np.array([0.5, 0.5, 0.5, 2e-4, 2e-4, 1.0, 1e-2, 1e-2, 1.0, 1.0, 0.1], dtype=np.float64)
    fit_h2 = _fit_map_generic(
        x0=x0_h2,
        bounds=bounds_h2,
        neg_log_joint=neg_log_joint_h2,
        step=step_h2,
        restarts=config.restarts_h2,
        seed=config.seed + 1,
    )
    log_joint_h2 = -float(fit_h2["neg_log_joint_at_map"])
    logZ_h2 = laplace_log_evidence(
        fit_h2["map_x"],
        fit_h2["hessian"] + fit_h2["hessian_jitter"] * np.eye(11),
        log_joint_at_map=log_joint_h2,
    )

    # --- H3: cascade-like point source
    x0_h3 = _pack_initial_h3(evt)
    bounds_h3 = [
        (None, None),
        (None, None),
        (None, None),
        (float(evt.reco_t_ns) - 10_000.0, float(evt.reco_t_ns) + 10_000.0),
        (math.log(0.2), math.log(30.0)),
        (math.log(1.0), math.log(2_000.0)),
    ]

    def neg_log_joint_h3(x: np.ndarray) -> float:
        lp = _logprior_h3_cascade(x, evt=evt, pri=pri)
        if not np.isfinite(lp):
            return float("inf")
        ll = _loglik_cascade(
            x,
            hit_pos_m=hit_pos,
            hit_t_ns=hit_t,
            t_min_ns=t_min,
            t_max_ns=t_max,
            outlier_frac=lik.outlier_frac,
        )
        if not np.isfinite(ll):
            return float("inf")
        return -(ll + lp)

    step_h3 = np.array([0.5, 0.5, 0.5, 1.0, 1e-2, 1e-2], dtype=np.float64)
    fit_h3 = _fit_map_generic(
        x0=x0_h3,
        bounds=bounds_h3,
        neg_log_joint=neg_log_joint_h3,
        step=step_h3,
        restarts=2,
        seed=config.seed + 2,
    )
    log_joint_h3 = -float(fit_h3["neg_log_joint_at_map"])
    logZ_h3 = laplace_log_evidence(
        fit_h3["map_x"],
        fit_h3["hessian"] + fit_h3["hessian_jitter"] * np.eye(6),
        log_joint_at_map=log_joint_h3,
    )

    # --- H0: time-uniform background (no parameters)
    logZ_h0 = float(-len(hit_t) * math.log(float(t_max - t_min)))

    # --- H4: single track with beta<1 (LLCP-like) timing
    beta_min = (1.0 / float(WATER_INDEX)) + 1e-4
    x0_h4 = _pack_initial_h4(evt)
    bounds_h4 = [
        (None, None),
        (None, None),
        (None, None),
        (1e-6, math.pi - 1e-6),
        (0.0, 2.0 * math.pi),
        (float(evt.reco_t_ns) - 10_000.0, float(evt.reco_t_ns) + 10_000.0),
        (math.log(0.2), math.log(30.0)),
        (math.log(1.0), math.log(2_000.0)),
        (None, None),  # z_beta
    ]

    def neg_log_joint_h4(x: np.ndarray) -> float:
        lp = _logprior_h4_beta_track(x, evt=evt, pri=pri, beta_min=beta_min)
        if not np.isfinite(lp):
            return float("inf")
        ll = _loglik_single_track_beta(
            x,
            hit_pos_m=hit_pos,
            hit_t_ns=hit_t,
            t_min_ns=t_min,
            t_max_ns=t_max,
            outlier_frac=lik.outlier_frac,
            beta_min=beta_min,
        )
        if not np.isfinite(ll):
            return float("inf")
        return -(ll + lp)

    step_h4 = np.array([0.5, 0.5, 0.5, 2e-4, 2e-4, 1.0, 1e-2, 1e-2, 0.05], dtype=np.float64)
    fit_h4 = _fit_map_generic(
        x0=x0_h4,
        bounds=bounds_h4,
        neg_log_joint=neg_log_joint_h4,
        step=step_h4,
        restarts=2,
        seed=config.seed + 3,
    )
    log_joint_h4 = -float(fit_h4["neg_log_joint_at_map"])
    H4 = fit_h4["hessian"] + fit_h4["hessian_jitter"] * np.eye(9)
    logZ_h4 = laplace_log_evidence(fit_h4["map_x"], H4, log_joint_at_map=log_joint_h4)

    # Summaries.
    def unpack_dir(x: np.ndarray) -> dict[str, float]:
        theta = float(x[3])
        phi = float(x[4])
        u = unit_from_theta_phi(theta, phi)
        return {
            "theta_deg": float(np.degrees(theta)),
            "phi_deg": float(np.degrees(phi) % 360.0),
            "u_x": float(u[0]),
            "u_y": float(u[1]),
            "u_z": float(u[2]),
        }

    # Bundle-specific derived params.
    x2 = np.asarray(fit_h2["map_x"], dtype=np.float64)
    w2 = _sigmoid(float(x2[10]))
    u2 = unit_from_theta_phi(float(x2[3]), float(x2[4]))
    b1, b2 = _orthonormal_basis_perp(u2)
    offset_vec = float(x2[8]) * b1 + float(x2[9]) * b2
    offset_norm_m = float(np.linalg.norm(offset_vec))

    # Beta-track derived params.
    x4 = np.asarray(fit_h4["map_x"], dtype=np.float64)
    w4 = _sigmoid(float(x4[8]))
    beta4 = float(beta_min + (1.0 - beta_min) * w4)
    beta_sigma = None
    try:
        cov4 = np.linalg.inv(H4)
        zsig = float(math.sqrt(max(0.0, float(cov4[8, 8]))))
        db_dz = (1.0 - float(beta_min)) * w4 * (1.0 - w4)
        beta_sigma = float(abs(db_dz) * zsig)
    except Exception:
        beta_sigma = None

    results = {
        "inputs": {
            "event": str(Path(evt_path)),
            "selection": asdict(config.selection),
            "n_hits": int(idx.size),
            "t_min_ns": t_min,
            "t_max_ns": t_max,
            "t_range_ns": float(t_max - t_min),
        },
        "config": {
            "priors": asdict(pri),
            "likelihood": asdict(lik),
            "restarts_h1": int(config.restarts_h1),
            "restarts_h2": int(config.restarts_h2),
            "seed": int(config.seed),
        },
        "models": {
            "H1_single_track": {
                "map": {
                    "pos_m": [float(v) for v in fit_h1["map_x"][0:3]],
                    "t0_ns": float(fit_h1["map_x"][5]),
                    "log_sigma_ns": float(fit_h1["map_x"][6]),
                    "log_tau_ns": float(fit_h1["map_x"][7]),
                    **unpack_dir(fit_h1["map_x"]),
                },
                "log_joint_at_map": log_joint_h1,
                "log_evidence_laplace": logZ_h1,
            },
            "H2_bundle_two_tracks": {
                "map": {
                    "pos1_m": [float(v) for v in x2[0:3]],
                    "t0_ns": float(x2[5]),
                    "log_sigma_ns": float(x2[6]),
                    "log_tau_ns": float(x2[7]),
                    "offset_a_m": float(x2[8]),
                    "offset_b_m": float(x2[9]),
                    "offset_norm_m": offset_norm_m,
                    "w_track1": float(w2),
                    **unpack_dir(x2),
                },
                "log_joint_at_map": log_joint_h2,
                "log_evidence_laplace": logZ_h2,
            },
            "H3_cascade_point_source": {
                "map": {
                    "pos_m": [float(v) for v in fit_h3["map_x"][0:3]],
                    "t0_ns": float(fit_h3["map_x"][3]),
                    "log_sigma_ns": float(fit_h3["map_x"][4]),
                    "log_tau_ns": float(fit_h3["map_x"][5]),
                },
                "log_joint_at_map": log_joint_h3,
                "log_evidence_laplace": logZ_h3,
            },
            "H0_time_uniform": {
                "log_evidence_exact": logZ_h0,
            },
            "H4_beta_track": {
                "map": {
                    "pos_m": [float(v) for v in x4[0:3]],
                    "t0_ns": float(x4[5]),
                    "log_sigma_ns": float(x4[6]),
                    "log_tau_ns": float(x4[7]),
                    "beta_min": float(beta_min),
                    "beta": beta4,
                    "beta_sigma_laplace": beta_sigma,
                    **unpack_dir(x4),
                },
                "log_joint_at_map": log_joint_h4,
                "log_evidence_laplace": logZ_h4,
            },
        },
        "bayes_factors": {
            "logB_H2_over_H1": float(logZ_h2 - logZ_h1),
            "B_H2_over_H1": float(math.exp(min(700.0, max(-700.0, logZ_h2 - logZ_h1)))),
            "logB_H1_over_H3": float(logZ_h1 - logZ_h3),
            "logB_H2_over_H3": float(logZ_h2 - logZ_h3),
            "logB_H1_over_H0": float(logZ_h1 - logZ_h0),
            "logB_H2_over_H0": float(logZ_h2 - logZ_h0),
            "logB_H3_over_H0": float(logZ_h3 - logZ_h0),
            "logB_H4_over_H1": float(logZ_h4 - logZ_h1),
            "logB_H1_over_H4": float(logZ_h1 - logZ_h4),
            "logB_H4_over_H0": float(logZ_h4 - logZ_h0),
        },
        "notes": [
            "Evidences for H1/H2 are Laplace approximations around the MAP (Gaussian approximation to the posterior).",
            "H2 is a simplified 'bundle' model: two parallel tracks with a transverse offset and a mixture weight; shared direction and time.",
            "H3 is a cascade-like point-source timing model (t_pred=t0+|r-r0|/v_g).",
            "H0 is a minimal null: hit times are i.i.d. uniform over the selected hit time range.",
            "H4 is a single-track timing model with beta<1 (LLCP-like); Cherenkov angle uses cos(theta_c)=1/(n*beta).",
        ],
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results
