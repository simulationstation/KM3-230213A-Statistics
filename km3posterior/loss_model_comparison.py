from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf

from .data import EventData, load_event_json_gz
from .geometry import cherenkov_times_ns, emission_point_solutions_m, unit_from_theta_phi
from .loss_inference import LossModelConfig, LossSelection


@dataclass(frozen=True)
class LossModelComparisonConfig:
    selection: LossSelection = LossSelection()
    model: LossModelConfig = LossModelConfig()
    weight: Literal["count", "tot"] = "count"

    # Optimize with a few restarts for the full model.
    restarts_full: int = 6
    seed: int = 230213


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


def _laplace_log_evidence(log_joint_at_map: float, hessian_neg_log_joint: np.ndarray) -> float:
    H = np.asarray(hessian_neg_log_joint, dtype=np.float64)
    d = int(H.shape[0])
    sign, logdet = np.linalg.slogdet(H)
    if sign <= 0:
        raise ValueError("Hessian must be positive definite for Laplace evidence.")
    return float(log_joint_at_map + 0.5 * d * math.log(2.0 * math.pi) - 0.5 * float(logdet))


def _sigmoid(x: float) -> float:
    x = float(x)
    if x >= 0.0:
        ex = math.exp(-x)
        return 1.0 / (1.0 + ex)
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _decode_ordered_mus(u: np.ndarray, *, s_min: float, s_max: float) -> np.ndarray:
    """Map 3 unconstrained reals -> ordered (mu1,mu2,mu3) in [s_min,s_max]."""
    u = np.asarray(u, dtype=np.float64)
    if u.shape != (3,):
        raise ValueError("Expected u shape (3,)")

    span_total = float(s_max - s_min)
    mu1 = s_min + span_total * _sigmoid(float(u[0]))
    span = (s_max - mu1) * _sigmoid(float(u[1]))
    frac = _sigmoid(float(u[2]))
    mu2 = mu1 + span * frac
    mu3 = mu1 + span
    return np.array([mu1, mu2, mu3], dtype=np.float64)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _lambda_full_model(
    *,
    edges: np.ndarray,
    centers: np.ndarray,
    x: np.ndarray,
    sel: LossSelection,
    cfg: LossModelConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (lambda_per_bin, decoded_params) for the baseline+3-loss model."""
    x = np.asarray(x, dtype=np.float64)
    S = int(cfg.n_base_segments)
    if x.shape != (S + 9,):
        raise ValueError("Unexpected parameter vector size for full loss model.")

    log_base_seg = x[:S]
    u_mu = x[S : S + 3]
    log_area = x[S + 3 : S + 6]
    log_width = x[S + 6 : S + 9]

    mu_m = _decode_ordered_mus(u_mu, s_min=sel.s_range_m[0], s_max=sel.s_range_m[1])
    area = np.exp(np.clip(log_area, -50.0, 50.0))
    width_m = np.exp(np.clip(log_width, -50.0, 50.0))

    smin, smax = sel.s_range_m
    seg_len = float(smax - smin) / float(S)
    seg_idx = np.clip(((centers - smin) / max(1e-9, seg_len)).astype(int), 0, S - 1)
    base = np.exp(log_base_seg[seg_idx])
    lam = base.astype(np.float64).copy()

    for k in range(3):
        mu = float(mu_m[k])
        w = float(width_m[k])
        if not (w > 0.0):
            continue
        z0 = (edges[:-1] - mu) / w
        z1 = (edges[1:] - mu) / w
        p = _normal_cdf(z1) - _normal_cdf(z0)
        lam += float(area[k]) * p

    lam = np.maximum(lam, 1e-12)
    return lam, {"mu_m": mu_m, "area": area, "width_m": width_m, "base_seg": np.exp(log_base_seg)}


def emission_values_along_track(
    evt: EventData,
    *,
    track_pos_m: np.ndarray,
    track_dir: np.ndarray,
    track_t_ns: float,
    sel: LossSelection,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (s_values, tot_values, meta) for hits in a residual window with valid emission solutions."""
    rmin, rmax = sel.residual_range_ns
    smin, smax = sel.s_range_m

    t_pred, *_ = cherenkov_times_ns(
        track_pos_m=track_pos_m,
        track_dir=track_dir,
        track_t_ns=track_t_ns,
        hit_pos_m=evt.pos_m,
    )
    residual = evt.t_ns - t_pred
    keep = (residual >= rmin) & (residual < rmax)

    s1, s2, D = emission_point_solutions_m(
        track_pos_m=track_pos_m,
        track_dir=track_dir,
        track_t_ns=track_t_ns,
        hit_pos_m=evt.pos_m[keep],
        hit_t_ns=evt.t_ns[keep],
    )
    valid = (D >= 0.0) & np.isfinite(s2) & (s2 >= smin) & (s2 <= smax)

    s = s2[valid].astype(np.float64)
    tot = evt.tot[keep][valid].astype(np.float64)
    meta = {"n_hits_in_window": int(np.sum(keep)), "n_hits_binned": int(s.size)}
    return s, tot, meta


def histogram_along_s(
    s: np.ndarray,
    *,
    sel: LossSelection,
    weight: Literal["count", "tot"] = "count",
    tot: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    smin, smax = sel.s_range_m
    edges = np.linspace(float(smin), float(smax), int(sel.n_bins) + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if weight == "count":
        y, _ = np.histogram(s, bins=edges)
        return edges, centers, y.astype(np.float64)

    if tot is None:
        raise ValueError("tot must be provided when weight='tot'")
    y, _ = np.histogram(s, bins=edges, weights=np.asarray(tot, dtype=np.float64))
    return edges, centers, y.astype(np.float64)


def _loglike_poisson(y: np.ndarray, lam: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    lam = np.maximum(lam, 1e-12)
    return float(np.sum(y * np.log(lam) - lam))


def _logposterior_baseline(
    x: np.ndarray,
    *,
    centers: np.ndarray,
    y: np.ndarray,
    sel: LossSelection,
    cfg: LossModelConfig,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    S = int(cfg.n_base_segments)
    if x.shape != (S,):
        raise ValueError("Unexpected parameter vector size for baseline model.")

    smin, smax = sel.s_range_m
    seg_len = float(smax - smin) / float(S)
    seg_idx = np.clip(((centers - smin) / max(1e-9, seg_len)).astype(int), 0, S - 1)
    lam = np.exp(x[seg_idx])

    ll = _loglike_poisson(y, lam)

    # Gaussian priors (unnormalized, consistent with LossModelConfig usage in the MCMC model).
    lp = float(np.sum(-0.5 * ((x - cfg.log_base_mu) / cfg.log_base_sigma) ** 2))
    return ll + lp


def _logposterior_full(
    x: np.ndarray,
    *,
    edges: np.ndarray,
    centers: np.ndarray,
    y: np.ndarray,
    sel: LossSelection,
    cfg: LossModelConfig,
) -> float:
    # Re-implement the posterior used by the MCMC model (constants dropped).
    x = np.asarray(x, dtype=np.float64)
    S = int(cfg.n_base_segments)
    if x.shape != (S + 9,):
        raise ValueError("Unexpected parameter vector size for full model.")

    log_base_seg = x[:S]
    log_area = x[S + 3 : S + 6]
    log_width = x[S + 6 : S + 9]

    if np.any(log_width < cfg.log_width_min) or np.any(log_width > cfg.log_width_max):
        return -math.inf

    lam, _ = _lambda_full_model(edges=edges, centers=centers, x=x, sel=sel, cfg=cfg)
    ll = _loglike_poisson(y, lam)

    lp = 0.0
    lp += float(np.sum(-0.5 * ((log_base_seg - cfg.log_base_mu) / cfg.log_base_sigma) ** 2))
    lp += float(np.sum(-0.5 * ((log_area - cfg.log_area_mu) / cfg.log_area_sigma) ** 2))
    lp += float(np.sum(-0.5 * ((log_width - cfg.log_width_mu) / cfg.log_width_sigma) ** 2))
    return float(ll + lp)


def _fit_map_and_laplace(
    *,
    x0: np.ndarray,
    bounds: list[tuple[float | None, float | None]] | None,
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

        opt = minimize(neg_log_joint, x_init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 800})
        if opt.success and float(opt.fun) < best_fun:
            best_fun = float(opt.fun)
            best = opt

    if best is None:
        raise RuntimeError("MAP optimization failed for all restarts.")

    x_map = np.asarray(best.x, dtype=np.float64)
    H = _finite_difference_hessian(neg_log_joint, x_map, step)

    jitter = 1e-8
    for _ in range(12):
        Hj = H + jitter * np.eye(H.shape[0])
        sign, logdet = np.linalg.slogdet(Hj)
        if sign > 0 and np.isfinite(logdet):
            break
        jitter *= 10.0
    else:
        raise RuntimeError("Failed to compute stable logdet(H) for Laplace evidence.")

    log_joint = -float(best_fun)
    logZ = _laplace_log_evidence(log_joint, H + jitter * np.eye(H.shape[0]))
    return {"map_x": x_map, "log_joint_at_map": log_joint, "logZ_laplace": logZ, "hessian_jitter": jitter}


def run_loss_model_comparison(
    evt_path: str | Path,
    *,
    track_posterior_npz: str | Path,
    out_path: str | Path | None = None,
    cfg: LossModelComparisonConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or LossModelComparisonConfig()
    evt = load_event_json_gz(evt_path)

    with np.load(Path(track_posterior_npz), allow_pickle=False) as f:
        map_x = np.asarray(f["map_x"], dtype=np.float64)

    track_pos_m = map_x[0:3]
    track_dir = unit_from_theta_phi(float(map_x[3]), float(map_x[4]))
    track_t_ns = float(map_x[5])

    s, tot, meta_hits = emission_values_along_track(
        evt,
        track_pos_m=track_pos_m,
        track_dir=track_dir,
        track_t_ns=track_t_ns,
        sel=cfg.selection,
    )

    edges, centers, y = histogram_along_s(s, sel=cfg.selection, weight=cfg.weight, tot=tot)

    model = cfg.model
    S = int(model.n_base_segments)

    # Baseline-only model (S params).
    x0_base = np.full(S, math.log(max(1.0, float(np.median(y)))), dtype=np.float64)
    step_base = np.full(S, 0.03, dtype=np.float64)

    def neg_base(x: np.ndarray) -> float:
        lp = _logposterior_baseline(x, centers=centers, y=y, sel=cfg.selection, cfg=model)
        return float("inf") if not np.isfinite(lp) else -float(lp)

    fit_base = _fit_map_and_laplace(x0=x0_base, bounds=None, neg_log_joint=neg_base, step=step_base, restarts=1, seed=cfg.seed)
    lam_base = np.exp(fit_base["map_x"])[np.clip(((centers - cfg.selection.s_range_m[0]) / ((cfg.selection.s_range_m[1] - cfg.selection.s_range_m[0]) / S)).astype(int), 0, S - 1)]
    loglike_base = _loglike_poisson(y, lam_base)

    # Full (baseline + 3 losses) model (S+9 params).
    x0_full = np.zeros(S + 9, dtype=np.float64)
    x0_full[:S] = math.log(max(1.0, float(np.median(y))))
    # crude init for u_mu based on top bins
    order = np.argsort(y)[::-1]
    chosen: list[float] = []
    min_sep_m = 25.0
    for i in order:
        sv = float(centers[i])
        if all(abs(sv - c) >= min_sep_m for c in chosen):
            chosen.append(sv)
        if len(chosen) == 3:
            break
    if len(chosen) != 3:
        chosen = [float(centers[i]) for i in order[:3]]
    peak_s = np.sort(np.array(chosen, dtype=np.float64))

    def inv_sigmoid(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        return math.log(p / (1.0 - p))

    smin, smax = cfg.selection.s_range_m
    p1 = (peak_s[0] - smin) / (smax - smin)
    u0 = inv_sigmoid(p1)
    span_guess = max(10.0, float(peak_s[-1] - peak_s[0]))
    p_span = span_guess / max(1e-6, (smax - (smin + (smax - smin) * p1)))
    u1 = inv_sigmoid(np.clip(p_span, 1e-3, 0.999))
    p_frac = (peak_s[1] - peak_s[0]) / max(1e-6, span_guess)
    u2 = inv_sigmoid(np.clip(p_frac, 1e-3, 0.999))

    x0_full[S : S + 3] = np.array([u0, u1, u2], dtype=np.float64)
    x0_full[S + 3 : S + 6] = model.log_area_mu
    x0_full[S + 6 : S + 9] = model.log_width_mu

    step_full = np.array([0.03] * S + [0.06, 0.06, 0.06] + [0.10] * 3 + [0.07] * 3, dtype=np.float64)

    bounds_full: list[tuple[float | None, float | None]] = [(None, None)] * (S + 9)
    # Keep the optimizer away from extreme values that can cause numerical overflows.
    for k in range(3):
        bounds_full[S + k] = (-20.0, 20.0)  # u_mu
        bounds_full[S + 3 + k] = (-10.0, 20.0)  # log_area
    for k in range(3):
        bounds_full[S + 6 + k] = (float(model.log_width_min), float(model.log_width_max))

    def neg_full(x: np.ndarray) -> float:
        lp = _logposterior_full(x, edges=edges, centers=centers, y=y, sel=cfg.selection, cfg=model)
        return float("inf") if not np.isfinite(lp) else -float(lp)

    fit_full = _fit_map_and_laplace(
        x0=x0_full,
        bounds=bounds_full,
        neg_log_joint=neg_full,
        step=step_full,
        restarts=int(cfg.restarts_full),
        seed=cfg.seed + 1,
    )
    lam_full, decoded = _lambda_full_model(edges=edges, centers=centers, x=fit_full["map_x"], sel=cfg.selection, cfg=model)
    loglike_full = _loglike_poisson(y, lam_full)

    n = int(y.size)
    k_base = int(S)
    k_full = int(S + 9)
    bic_base = float(k_base * math.log(max(1, n)) - 2.0 * loglike_base)
    bic_full = float(k_full * math.log(max(1, n)) - 2.0 * loglike_full)

    out = {
        "inputs": {
            "event": str(Path(evt_path)),
            "track_posterior_npz": str(Path(track_posterior_npz)),
            "track_map_x": map_x.tolist(),
            "weight": cfg.weight,
            "hist_meta": meta_hits,
        },
        "selection": asdict(cfg.selection),
        "model_config": asdict(model),
        "baseline_only": {
            "k": k_base,
            "loglike_at_map": loglike_base,
            "logZ_laplace": float(fit_base["logZ_laplace"]),
        },
        "baseline_plus_losses": {
            "k": k_full,
            "loglike_at_map": loglike_full,
            "logZ_laplace": float(fit_full["logZ_laplace"]),
            "decoded_map": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in decoded.items()},
        },
        "comparisons": {
            "delta_loglike": float(loglike_full - loglike_base),
            "delta_logZ_laplace": float(fit_full["logZ_laplace"] - fit_base["logZ_laplace"]),
            "delta_BIC_full_minus_base": float(bic_full - bic_base),
        },
        "artifacts": {
            "edges": edges.tolist(),
            "centers": centers.tolist(),
            "y": y.tolist(),
            "lambda_base_map": lam_base.tolist(),
            "lambda_full_map": lam_full.tolist(),
        },
        "notes": [
            "This is a brightness proxy test on the 1D emission-point histogram along the reconstructed track.",
            "Evidences are Laplace approximations around the MAP; priors/constants for the loss-location parameters are approximate (treat as a model-comparison heuristic).",
            "BIC is also reported as a prior-insensitive sanity check.",
        ],
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    return out
