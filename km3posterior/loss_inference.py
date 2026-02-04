from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.special import erf

from .data import EventData, load_event_json_gz
from .geometry import cherenkov_times_ns, emission_point_solutions_m, unit_from_theta_phi


@dataclass(frozen=True)
class LossSelection:
    residual_range_ns: tuple[float, float] = (-50.0, 1000.0)
    s_range_m: tuple[float, float] = (0.0, 600.0)
    n_bins: int = 300


@dataclass(frozen=True)
class LossModelConfig:
    n_losses: int = 3
    n_base_segments: int = 6

    # Priors (log-space Gaussians; drop constants).
    log_base_mu: float = math.log(45.0)  # baseline counts per bin
    log_base_sigma: float = 0.6

    log_area_mu: float = math.log(300.0)  # total excess counts per loss
    log_area_sigma: float = 1.0

    log_width_mu: float = math.log(6.0)  # meters
    log_width_sigma: float = 0.6
    log_width_min: float = math.log(1.0)
    log_width_max: float = math.log(80.0)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _decode_ordered_mus(u: np.ndarray, *, s_min: float, s_max: float) -> np.ndarray:
    """Map 3 unconstrained reals -> ordered (mu1,mu2,mu3) in [s_min,s_max]."""
    if u.shape != (3,):
        raise ValueError("Expected 3 unconstrained mu parameters.")

    span_total = float(s_max - s_min)
    mu1 = s_min + span_total * _sigmoid(float(u[0]))
    span = (s_max - mu1) * _sigmoid(float(u[1]))
    frac = _sigmoid(float(u[2]))
    mu2 = mu1 + span * frac
    mu3 = mu1 + span
    return np.array([mu1, mu2, mu3], dtype=np.float64)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    # Phi(x) using erf.
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def emission_histogram(
    evt: EventData,
    *,
    track_pos_m: np.ndarray,
    track_dir: np.ndarray,
    track_t_ns: float,
    sel: LossSelection,
    weight: Literal["count"] = "count",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute emission-along-track histogram (bins + centers + counts).

    Uses the same logic as the official notebook (Hit distributions.ipynb):
    - keep hits in a residual window w.r.t. the Cherenkov earliest-time model
    - solve for emission point along the track using hit time + position
    """
    rmin, rmax = sel.residual_range_ns
    smin, smax = sel.s_range_m

    edges = np.linspace(smin, smax, int(sel.n_bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

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
    s = s2[valid]

    if weight != "count":
        raise ValueError("Only weight='count' is supported in the current loss model.")

    y, _ = np.histogram(s, bins=edges)
    meta = {"n_hits_in_window": int(np.sum(keep)), "n_hits_binned": int(s.size)}
    return edges, centers, y.astype(np.int64), meta


def _lambda_per_bin(
    *,
    edges: np.ndarray,
    base: np.ndarray,
    mu_m: np.ndarray,
    area: np.ndarray,
    width_m: np.ndarray,
) -> np.ndarray:
    """Expected counts per bin.

    Model: y_bin ~ Poisson(base + sum_k area_k * Normal(bin | mu_k, width_k)).
    Here "Normal(bin | ...)" is the probability mass of a Gaussian inside the bin.
    """
    edges = np.asarray(edges, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    K = mu_m.size
    n_bins = edges.size - 1
    if base.shape != (n_bins,):
        raise ValueError("Expected base shape (n_bins,)")
    lam = base.copy()

    for k in range(K):
        mu = float(mu_m[k])
        w = float(width_m[k])
        if w <= 0.0:
            return np.full(n_bins, 1e-12, dtype=np.float64)
        z0 = (edges[:-1] - mu) / w
        z1 = (edges[1:] - mu) / w
        p = _normal_cdf(z1) - _normal_cdf(z0)
        lam += float(area[k]) * p

    return np.maximum(lam, 1e-12)


def _log_posterior_loss(
    x: np.ndarray,
    *,
    edges: np.ndarray,
    y: np.ndarray,
    sel: LossSelection,
    cfg: LossModelConfig,
) -> float:
    """Log posterior in an unconstrained parameterization.

    Parameter vector (K=3):
      [ log_base_seg0..log_base_seg{S-1},
        u_mu1,u_mu_span,u_mu_frac,
        log_area1..3,
        log_width1..3 ]
    """
    x = np.asarray(x, dtype=np.float64)
    K = int(cfg.n_losses)
    if K != 3:
        raise ValueError("Currently only n_losses=3 is implemented.")
    S = int(cfg.n_base_segments)
    if x.shape != (S + 3 + K + K,):
        raise ValueError("Unexpected parameter vector size for loss model.")

    log_base_seg = x[:S]
    u_mu = x[S : S + 3]
    log_area = x[S + 3 : S + 6]
    log_width = x[S + 6 : S + 9]

    # Hard bounds for widths (keeps inference well-posed).
    if np.any(log_width < cfg.log_width_min) or np.any(log_width > cfg.log_width_max):
        return -np.inf

    mu_m = _decode_ordered_mus(u_mu, s_min=sel.s_range_m[0], s_max=sel.s_range_m[1])
    area = np.exp(log_area)
    width_m = np.exp(log_width)

    centers = 0.5 * (edges[:-1] + edges[1:])
    smin, smax = sel.s_range_m
    seg_len = float(smax - smin) / float(S)
    seg_idx = np.clip(((centers - smin) / max(1e-9, seg_len)).astype(int), 0, S - 1)
    base = np.exp(log_base_seg[seg_idx])

    lam = _lambda_per_bin(edges=edges, base=base, mu_m=mu_m, area=area, width_m=width_m)
    y = np.asarray(y, dtype=np.float64)

    log_like = float(np.sum(y * np.log(lam) - lam))  # drop log(y!)

    log_prior = 0.0
    log_prior += float(np.sum(-0.5 * ((log_base_seg - cfg.log_base_mu) / cfg.log_base_sigma) ** 2))
    log_prior += float(np.sum(-0.5 * ((log_area - cfg.log_area_mu) / cfg.log_area_sigma) ** 2))
    log_prior += float(np.sum(-0.5 * ((log_width - cfg.log_width_mu) / cfg.log_width_sigma) ** 2))

    return log_like + log_prior


def _metropolis_chain(
    *,
    edges: np.ndarray,
    y: np.ndarray,
    sel: LossSelection,
    cfg: LossModelConfig,
    x0: np.ndarray,
    step: np.ndarray,
    n_steps: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x0, dtype=np.float64).copy()
    step = np.asarray(step, dtype=np.float64)

    logp = float(_log_posterior_loss(x, edges=edges, y=y, sel=sel, cfg=cfg))
    if not np.isfinite(logp):
        raise RuntimeError("Initial loss chain state has non-finite log posterior.")

    keep = []
    accepted = 0
    proposed = 0
    window_acc = 0
    window_prop = 0
    step_scale = 1.0
    adapt_every = 250

    for t in range(int(n_steps)):
        x_prop = x + (step * step_scale) * rng.normal(size=x.shape[0])
        logp_prop = float(_log_posterior_loss(x_prop, edges=edges, y=y, sel=sel, cfg=cfg))
        proposed += 1
        window_prop += 1

        if np.isfinite(logp_prop) and math.log(rng.random()) < (logp_prop - logp):
            x = x_prop
            logp = logp_prop
            accepted += 1
            window_acc += 1

        if t < burn_in and (t + 1) % adapt_every == 0:
            acc = window_acc / max(1, window_prop)
            # Coarse global scaling to get into a reasonable acceptance regime.
            if acc < 0.12:
                step_scale *= 0.7
            elif acc > 0.35:
                step_scale *= 1.25
            step_scale = float(np.clip(step_scale, 1e-3, 1e3))
            window_acc = 0
            window_prop = 0

        if t >= burn_in and ((t - burn_in) % thin == 0):
            keep.append(x.copy())

    samples = np.stack(keep, axis=0) if keep else np.empty((0, x.size), dtype=np.float64)
    return {"samples": samples, "accept_rate": accepted / max(1, proposed), "step_scale": step_scale}


def _loss_chain_worker(payload_json: str, chain_id: int) -> dict[str, Any]:
    cfg = json.loads(payload_json)
    edges = np.asarray(cfg["edges"], dtype=np.float64)
    y = np.asarray(cfg["y"], dtype=np.float64)
    sel = LossSelection(**cfg["selection"])
    model = LossModelConfig(**cfg["model"])
    x0 = np.asarray(cfg["x0"], dtype=np.float64)
    step = np.asarray(cfg["step"], dtype=np.float64)
    n_steps = int(cfg["n_steps"])
    burn_in = int(cfg["burn_in"])
    thin = int(cfg["thin"])
    seed = int(cfg["seed"]) + int(chain_id) * 10_000
    # Chain-specific jitter on init helps avoid identical trajectories.
    rng = np.random.default_rng(seed)
    x0 = x0 + rng.normal(scale=0.02, size=x0.shape[0])

    return _metropolis_chain(
        edges=edges,
        y=y,
        sel=sel,
        cfg=model,
        x0=x0,
        step=step,
        n_steps=n_steps,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
    )


def _initial_x0_from_peaks(centers: np.ndarray, y: np.ndarray, sel: LossSelection, *, n_base_segments: int) -> np.ndarray:
    # Rough init around the strongest *separated* peaks (helps mixing).
    # The raw top-3 bins can easily pick two adjacent bins from the same loss peak.
    order = np.argsort(y)[::-1]
    chosen: list[float] = []
    min_sep_m = 25.0
    for i in order:
        s = float(centers[i])
        if all(abs(s - c) >= min_sep_m for c in chosen):
            chosen.append(s)
        if len(chosen) == 3:
            break
    if len(chosen) != 3:
        chosen = [float(centers[i]) for i in order[:3]]
    peak_s = np.sort(np.array(chosen, dtype=np.float64))

    def inv_sigmoid(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        return math.log(p / (1.0 - p))

    smin, smax = sel.s_range_m
    p1 = (peak_s[0] - smin) / (smax - smin)
    u0 = inv_sigmoid(p1)

    span_guess = max(10.0, float(peak_s[-1] - peak_s[0]))
    p_span = span_guess / max(1e-6, (smax - (smin + (smax - smin) * p1)))
    u1 = inv_sigmoid(np.clip(p_span, 1e-3, 0.999))

    p_frac = (peak_s[1] - peak_s[0]) / max(1e-6, span_guess)
    u2 = inv_sigmoid(np.clip(p_frac, 1e-3, 0.999))

    S = int(n_base_segments)
    x0 = np.zeros(S + 3 + 3 + 3, dtype=np.float64)
    x0[:S] = math.log(max(1.0, float(np.median(y))))
    x0[S : S + 3] = np.array([u0, u1, u2], dtype=np.float64)
    x0[S + 3 : S + 6] = math.log(300.0)
    x0[S + 6 : S + 9] = math.log(8.0)
    return x0


def fit_loss_posterior_mcmc(
    *,
    edges: np.ndarray,
    centers: np.ndarray,
    y: np.ndarray,
    sel: LossSelection,
    cfg: LossModelConfig,
    chains: int = 16,
    cores: int | None = None,
    n_steps: int = 20_000,
    burn_in: int = 8_000,
    thin: int = 10,
    seed: int = 230213,
) -> dict[str, Any]:
    x0 = _initial_x0_from_peaks(centers, y, sel, n_base_segments=int(cfg.n_base_segments))

    # Step sizes in the unconstrained parameterization (tuned to mix OK by default).
    S = int(cfg.n_base_segments)
    step = np.array([0.03] * S + [0.06, 0.06, 0.06] + [0.10] * 3 + [0.07] * 3, dtype=np.float64)

    cores = int(cores or os.cpu_count() or 1)
    cores = max(1, min(int(cores), int(chains)))

    payload = {
        "edges": edges.tolist(),
        "y": y.tolist(),
        "selection": asdict(sel),
        "model": asdict(cfg),
        "x0": x0.tolist(),
        "step": step.tolist(),
        "n_steps": int(n_steps),
        "burn_in": int(burn_in),
        "thin": int(thin),
        "seed": int(seed),
    }
    payload_json = json.dumps(payload)

    if cores == 1:
        chains_out = [_loss_chain_worker(payload_json, i) for i in range(int(chains))]
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=cores) as ex:
            futures = [ex.submit(_loss_chain_worker, payload_json, i) for i in range(int(chains))]
            chains_out = [f.result() for f in futures]

    samples = np.concatenate([c["samples"] for c in chains_out], axis=0)
    accept = [float(c["accept_rate"]) for c in chains_out]
    step_scale = [float(c.get("step_scale", 1.0)) for c in chains_out]

    # Decode to physical space.
    base_seg = np.exp(samples[:, :S])
    mu = np.stack(
        [_decode_ordered_mus(samples[i, S : S + 3], s_min=sel.s_range_m[0], s_max=sel.s_range_m[1]) for i in range(samples.shape[0])],
        axis=0,
    )
    area = np.exp(samples[:, S + 3 : S + 6])
    width = np.exp(samples[:, S + 6 : S + 9])

    return {
        "samples_unconstrained": samples,
        "samples": {"base_seg": base_seg, "mu_m": mu, "area": area, "width_m": width},
        "accept_rate": accept,
        "step_scale": step_scale,
        "config": {"selection": asdict(sel), "model": asdict(cfg), "chains": int(chains), "cores": int(cores), "n_steps": int(n_steps), "burn_in": int(burn_in), "thin": int(thin), "seed": int(seed)},
    }


def run_loss_posterior_from_track_map(
    evt_path: str | Path,
    *,
    track_posterior_npz: str | Path,
    out_dir: str | Path,
    loss_selection: LossSelection | None = None,
    model: LossModelConfig | None = None,
    chains: int = 16,
    cores: int | None = None,
    n_steps: int = 20_000,
    burn_in: int = 8_000,
    thin: int = 10,
    seed: int = 230213,
) -> Path:
    evt = load_event_json_gz(evt_path)
    track_posterior_npz = Path(track_posterior_npz)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(track_posterior_npz, allow_pickle=False) as f:
        map_x = f["map_x"]

    track_pos_m = map_x[0:3]
    track_dir = unit_from_theta_phi(float(map_x[3]), float(map_x[4]))
    track_t_ns = float(map_x[5])

    loss_selection = loss_selection or LossSelection()
    model = model or LossModelConfig()

    edges, centers, y, meta = emission_histogram(
        evt,
        track_pos_m=track_pos_m,
        track_dir=track_dir,
        track_t_ns=track_t_ns,
        sel=loss_selection,
        weight="count",
    )

    post = fit_loss_posterior_mcmc(
        edges=edges,
        centers=centers,
        y=y,
        sel=loss_selection,
        cfg=model,
        chains=chains,
        cores=cores,
        n_steps=n_steps,
        burn_in=burn_in,
        thin=thin,
        seed=seed,
    )

    out_path = out_dir / "loss_posterior.npz"
    np.savez_compressed(
        out_path,
        edges=edges,
        centers=centers,
        y=y.astype(np.int64),
        base_seg=post["samples"]["base_seg"],
        mu_m=post["samples"]["mu_m"],
        area=post["samples"]["area"],
        width_m=post["samples"]["width_m"],
        accept_rate=np.array(post["accept_rate"], dtype=np.float64),
        step_scale=np.array(post["step_scale"], dtype=np.float64),
        meta=json.dumps(
            {
                "event_path": str(evt_path),
                "track_posterior": str(track_posterior_npz),
                "hist_meta": meta,
                **post["config"],
            },
            indent=2,
        ),
    )
    return out_path
