from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .data import EventData, load_event_json_gz
from .geometry import cherenkov_times_ns, theta_phi_from_unit, unit_from_theta_phi
from .timing_model import logpdf_emg_with_outliers


@dataclass(frozen=True)
class TimingSelection:
    distance_range_m: tuple[float, float] = (0.0, 100.0)
    residual_range_ns: tuple[float, float] = (-50.0, 1000.0)
    first_hit_per_pmt: bool = True


@dataclass(frozen=True)
class TrackPrior:
    # Loose, informative priors centered on the published reconstruction.
    sigma_pos_m: float = 100.0
    sigma_t_ns: float = 2_000.0
    # von Mises-Fisher concentration around reco direction.
    dir_kappa: float = 200.0

    # Log-normal priors for timing model hyper-parameters.
    log_sigma_mu: float = math.log(3.0)
    log_sigma_sigma: float = 0.7
    log_tau_mu: float = math.log(60.0)
    log_tau_sigma: float = 0.7

    # Outlier mixture.
    outlier_frac: float = 0.01


@dataclass(frozen=True)
class TrackFitResult:
    map_x: np.ndarray  # (D,)
    hessian: np.ndarray  # (D, D)
    cov: np.ndarray  # (D, D)
    selection_indices: np.ndarray  # (M,)
    config: dict[str, Any]


def _logpost_chunk_worker(payload_json: str, chunk: np.ndarray) -> np.ndarray:
    cfg = json.loads(payload_json)
    evt_local = load_event_json_gz(cfg["evt_path"])
    idx_local = np.array(cfg["selection_indices"], dtype=np.int64)
    hit_pos = evt_local.pos_m[idx_local]
    hit_t = evt_local.t_ns[idx_local]
    prior_local = TrackPrior(**cfg["prior"])
    selection_local = TimingSelection(**cfg["selection"])

    out = np.empty(chunk.shape[0], dtype=np.float64)
    for i in range(chunk.shape[0]):
        out[i] = log_posterior_track(
            chunk[i],
            hit_pos_m=hit_pos,
            hit_t_ns=hit_t,
            reco_pos_m=evt_local.reco_pos_m,
            reco_dir=evt_local.reco_dir,
            reco_t_ns=evt_local.reco_t_ns,
            prior=prior_local,
            residual_range_ns=selection_local.residual_range_ns,
        )
    return out


def _wrap_angles(theta: float, phi: float) -> tuple[float, float]:
    # Keep theta in [0, pi] by reflection, phi in [0, 2pi) by modulo.
    phi = float(phi) % (2.0 * math.pi)
    theta = float(theta)
    if theta < 0.0:
        theta = -theta
    if theta > math.pi:
        theta = 2.0 * math.pi - theta
    # If still out (rare), fold again.
    theta = float(np.clip(theta, 0.0, math.pi))
    return theta, phi


def select_hits_for_timing(evt: EventData, sel: TimingSelection) -> np.ndarray:
    """Select a stable subset of hits for timing-only direction inference."""
    t_pred, d_closest_m, *_ = cherenkov_times_ns(
        track_pos_m=evt.reco_pos_m,
        track_dir=evt.reco_dir,
        track_t_ns=evt.reco_t_ns,
        hit_pos_m=evt.pos_m,
    )
    residual_ns = evt.t_ns - t_pred

    d0, d1 = sel.distance_range_m
    r0, r1 = sel.residual_range_ns

    mask = (d_closest_m >= d0) & (d_closest_m < d1) & (residual_ns >= r0) & (residual_ns < r1)
    idx = np.nonzero(mask)[0]

    if not sel.first_hit_per_pmt:
        return idx

    # For "first hit per PMT", do it after filtering, in time order.
    t_idx = evt.t_ns[idx]
    order = np.argsort(t_idx, kind="mergesort")
    idx_sorted = idx[order]
    pmt_sorted = evt.pmt_key[idx_sorted]
    _, first = np.unique(pmt_sorted, return_index=True)
    return idx_sorted[first]


def pack_initial_x(evt: EventData) -> np.ndarray:
    theta0, phi0 = theta_phi_from_unit(evt.reco_dir)
    return np.array(
        [
            evt.reco_pos_m[0],
            evt.reco_pos_m[1],
            evt.reco_pos_m[2],
            theta0,
            phi0,
            evt.reco_t_ns,
            math.log(3.0),  # log sigma_ns
            math.log(60.0),  # log tau_ns
        ],
        dtype=np.float64,
    )


def log_posterior_track(
    x: np.ndarray,
    *,
    hit_pos_m: np.ndarray,
    hit_t_ns: np.ndarray,
    reco_pos_m: np.ndarray,
    reco_dir: np.ndarray,
    reco_t_ns: float,
    prior: TrackPrior,
    residual_range_ns: tuple[float, float],
) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (8,):
        raise ValueError("Expected x shape (8,)")

    pos = x[0:3]
    theta, phi = _wrap_angles(x[3], x[4])
    t0 = float(x[5])
    log_sigma = float(x[6])
    log_tau = float(x[7])

    sigma_ns = math.exp(log_sigma)
    tau_ns = math.exp(log_tau)

    track_dir = unit_from_theta_phi(theta, phi)

    t_pred, *_ = cherenkov_times_ns(track_pos_m=pos, track_dir=track_dir, track_t_ns=t0, hit_pos_m=hit_pos_m)
    residual = hit_t_ns - t_pred

    rmin, rmax = residual_range_ns
    log_like = float(
        logpdf_emg_with_outliers(
            residual,
            sigma_ns=sigma_ns,
            tau_ns=tau_ns,
            outlier_frac=prior.outlier_frac,
            outlier_min_ns=rmin,
            outlier_max_ns=rmax,
            mu_ns=0.0,
        ).sum()
    )

    # Priors (drop constants).
    log_prior = 0.0

    dp = (pos - reco_pos_m) / float(prior.sigma_pos_m)
    log_prior += -0.5 * float(dp @ dp)

    dt = (t0 - float(reco_t_ns)) / float(prior.sigma_t_ns)
    log_prior += -0.5 * dt * dt

    log_prior += float(prior.dir_kappa) * float(track_dir @ reco_dir)

    log_prior += -0.5 * ((log_sigma - prior.log_sigma_mu) / prior.log_sigma_sigma) ** 2
    log_prior += -0.5 * ((log_tau - prior.log_tau_mu) / prior.log_tau_sigma) ** 2

    return log_like + log_prior


def _finite_difference_hessian(f, x0: np.ndarray, step: np.ndarray) -> np.ndarray:
    x0 = np.asarray(x0, dtype=np.float64)
    step = np.asarray(step, dtype=np.float64)
    n = x0.size

    f0 = float(f(x0))
    H = np.zeros((n, n), dtype=np.float64)

    # Diagonal terms.
    for i in range(n):
        ei = np.zeros(n, dtype=np.float64)
        ei[i] = step[i]
        f_p = float(f(x0 + ei))
        f_m = float(f(x0 - ei))
        H[i, i] = (f_p - 2.0 * f0 + f_m) / (step[i] * step[i])

    # Off-diagonal terms.
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


def fit_track_map_and_cov(
    evt: EventData,
    *,
    sel: TimingSelection,
    prior: TrackPrior,
    step_scale: float = 1.0,
) -> TrackFitResult:
    idx = select_hits_for_timing(evt, sel)

    hit_pos_m = evt.pos_m[idx]
    hit_t_ns = evt.t_ns[idx]

    x0 = pack_initial_x(evt)

    bounds = [
        (None, None),  # x
        (None, None),  # y
        (None, None),  # z
        (0.0, math.pi),  # theta
        (0.0, 2.0 * math.pi),  # phi
        (None, None),  # t0
        (math.log(0.2), math.log(30.0)),  # log sigma
        (math.log(1.0), math.log(2_000.0)),  # log tau
    ]

    def neg_log_post(x: np.ndarray) -> float:
        return -log_posterior_track(
            x,
            hit_pos_m=hit_pos_m,
            hit_t_ns=hit_t_ns,
            reco_pos_m=evt.reco_pos_m,
            reco_dir=evt.reco_dir,
            reco_t_ns=evt.reco_t_ns,
            prior=prior,
            residual_range_ns=sel.residual_range_ns,
        )

    opt = minimize(neg_log_post, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    if not opt.success:
        raise RuntimeError(f"MAP fit failed: {opt.message}")

    x_map = np.asarray(opt.x, dtype=np.float64)

    # Finite-difference Hessian around MAP (on neg log posterior).
    # Use scale-aware step sizes (tuned for this dataset).
    base_step = np.array([0.5, 0.5, 0.5, 2e-4, 2e-4, 1.0, 1e-2, 1e-2], dtype=np.float64) * float(step_scale)

    H = _finite_difference_hessian(neg_log_post, x_map, base_step)

    # Regularize if needed.
    jitter = 1e-8
    for _ in range(8):
        try:
            cov = np.linalg.inv(H + jitter * np.eye(H.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        raise RuntimeError("Failed to invert Hessian (covariance).")

    cov = 0.5 * (cov + cov.T)
    # Ensure PSD for proposal sampling.
    evals, evecs = np.linalg.eigh(cov)
    floor = float(np.max(evals)) * 1e-12
    evals = np.maximum(evals, floor)
    cov = (evecs * evals[None, :]) @ evecs.T

    cfg = {"selection": asdict(sel), "prior": asdict(prior)}
    return TrackFitResult(map_x=x_map, hessian=H, cov=cov, selection_indices=idx, config=cfg)


def _mvn_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    d = mean.size

    xc = x - mean[None, :]
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, xc.T)
    maha = np.einsum("ij,ij->j", sol, sol)
    logdet = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + maha)


def sample_track_posterior_importance(
    evt_path: str | Path,
    *,
    out_dir: str | Path,
    n_proposals: int = 50_000,
    n_posterior: int = 20_000,
    proposal_scale: float = 1.5,
    cores: int | None = None,
    seed: int = 230213,
    selection: TimingSelection | None = None,
    prior: TrackPrior | None = None,
) -> Path:
    evt_path = Path(evt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = selection or TimingSelection()
    prior = prior or TrackPrior()

    evt = load_event_json_gz(evt_path)
    fit = fit_track_map_and_cov(evt, sel=selection, prior=prior)

    rng = np.random.default_rng(int(seed))

    mean = fit.map_x
    cov = fit.cov * float(proposal_scale) ** 2

    # Draw proposals; keep angular params inside bounds via reflection/modulo (small-angle approx).
    props = rng.multivariate_normal(mean=mean, cov=cov, size=int(n_proposals)).astype(np.float64)
    for i in range(props.shape[0]):
        theta, phi = _wrap_angles(props[i, 3], props[i, 4])
        props[i, 3] = theta
        props[i, 4] = phi

    # Data subset used for timing likelihood.
    idx = fit.selection_indices
    hit_pos_m = evt.pos_m[idx]
    hit_t_ns = evt.t_ns[idx]

    def logpost(x: np.ndarray) -> float:
        return log_posterior_track(
            x,
            hit_pos_m=hit_pos_m,
            hit_t_ns=hit_t_ns,
            reco_pos_m=evt.reco_pos_m,
            reco_dir=evt.reco_dir,
            reco_t_ns=evt.reco_t_ns,
            prior=prior,
            residual_range_ns=selection.residual_range_ns,
        )

    # Parallel log-posterior evaluations (processes each load minimal state).
    cores = int(cores or os.cpu_count() or 1)
    cores = max(1, min(cores, 32))

    if cores == 1:
        logp = np.array([logpost(x) for x in props], dtype=np.float64)
    else:
        from concurrent.futures import ProcessPoolExecutor

        # Avoid pickling huge arrays: send only the proposal chunk, re-load event in each process.
        payload = {
            "evt_path": str(evt_path),
            "selection": asdict(selection),
            "prior": asdict(prior),
            "selection_indices": idx.tolist(),
        }

        chunks = np.array_split(props, cores)
        payload_json = json.dumps(payload)
        with ProcessPoolExecutor(max_workers=cores) as ex:
            futures = [ex.submit(_logpost_chunk_worker, payload_json, ch) for ch in chunks if len(ch)]
            logp = np.concatenate([f.result() for f in futures], axis=0)

    logq = _mvn_logpdf(props, mean=mean, cov=cov)
    logw = logp - logq
    logw -= np.max(logw)
    w = np.exp(logw)
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        raise RuntimeError("Importance weights collapsed (non-finite).")
    w /= w_sum

    ess = 1.0 / float(np.sum(w * w))

    idx_resample = rng.choice(np.arange(props.shape[0]), size=int(n_posterior), replace=True, p=w)
    post = props[idx_resample]

    out_path = out_dir / "track_posterior.npz"
    np.savez_compressed(
        out_path,
        samples=post,
        map_x=fit.map_x,
        cov=fit.cov,
        hessian=fit.hessian,
        selection_indices=fit.selection_indices,
        ess=ess,
        config=json.dumps(
            {
                "event_path": str(evt_path),
                "n_proposals": int(n_proposals),
                "n_posterior": int(n_posterior),
                "proposal_scale": float(proposal_scale),
                "cores": int(cores),
                "seed": int(seed),
                "timing_selection": asdict(selection),
                "track_prior": asdict(prior),
            },
            indent=2,
        ),
    )
    return out_path


def direction_unit_vectors(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    theta = samples[:, 3]
    phi = samples[:, 4]
    st = np.sin(theta)
    u = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u
