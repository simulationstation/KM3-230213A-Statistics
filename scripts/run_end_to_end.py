from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from km3posterior.data import load_event_json_gz
from km3posterior.energy_inference import energy_posterior_from_ntrigpmt, load_figure2_calibration, save_energy_posterior_npz
from km3posterior.geometry import cherenkov_times_ns, emission_point_solutions_m, unit_from_theta_phi
from km3posterior.loss_inference import run_loss_posterior_from_track_map
from km3posterior.sky import (
    angular_distance_deg,
    credible_region_from_samples_equirect,
    enu_unit_to_icrs_unit,
    parse_voevent,
    pick_direction_sign_for_voevent,
    radec_deg_to_unit,
    smear_isotropic_orientation,
    sky_histogram_equirect,
    summarize_sky_posterior,
    unit_to_radec_deg,
    voevent_astropy_context,
)
from km3posterior.track_inference import TrackPrior, direction_unit_vectors, sample_track_posterior_importance


def _angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    c = np.einsum("...i,...i->...", u, v)
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def _quantiles(x: np.ndarray, ps=(0.16, 0.5, 0.84)) -> dict[str, float]:
    return {f"q{int(p*100):02d}": float(np.quantile(x, p)) for p in ps}


def main() -> int:
    p = argparse.ArgumentParser(description="End-to-end KM3-230213A posterior pipeline (track + losses + energy + systematics report).")
    p.add_argument("--upstream", default="upstream-KM3-230213A-data", help="Path to official KM3-230213A-data repo")
    p.add_argument("--out", default="results", help="Output directory (default: results)")
    p.add_argument("--cores", type=int, default=16, help="CPU cores / processes (default: 16)")
    p.add_argument("--seed", type=int, default=230213, help="Base RNG seed (default: 230213)")
    p.add_argument("--quick", action="store_true", help="Reduce compute for a fast smoke run")
    p.add_argument("--model-comparison", action="store_true", help="Run H1/H2/H3 model comparison and write model_comparison.json")
    p.add_argument("--model-comp-bundle-sigma-m", type=float, default=50.0, help="H2 prior sigma for bundle transverse offset (meters)")
    p.add_argument("--model-comp-restarts-h2", type=int, default=8, help="Number of MAP restarts for H2 (bundle)")
    p.add_argument("--model-comp-outlier-frac", type=float, default=0.01, help="Time-uniform outlier fraction for H1/H2")
    args = p.parse_args()

    upstream = Path(args.upstream)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    event_path = upstream / "data" / "event" / "KM3-230213A_allhits.json.gz"
    voevent_path = upstream / "data" / "event" / "KM3-230213A_voevent.xml"
    figure2_path = upstream / "data" / "supplementary" / "figuresource" / "figure2.csv"

    if not event_path.exists():
        raise SystemExit(f"Missing event file: {event_path}")
    if not figure2_path.exists():
        raise SystemExit(f"Missing figure2.csv: {figure2_path}")

    # --- Track posterior (baseline)
    track_npz = sample_track_posterior_importance(
        event_path,
        out_dir=out_dir,
        n_proposals=10_000 if args.quick else 80_000,
        n_posterior=5_000 if args.quick else 30_000,
        proposal_scale=2.0,
        cores=args.cores,
        seed=args.seed,
    )

    # --- Track posterior (fixed scattering hyper-params -> isolates geometry/timing-only)
    fixed = TrackPrior(log_sigma_sigma=1e-4, log_tau_sigma=1e-4, outlier_frac=0.005)
    track_fixed_npz = sample_track_posterior_importance(
        event_path,
        out_dir=out_dir / "fixed_scattering",
        n_proposals=10_000 if args.quick else 80_000,
        n_posterior=5_000 if args.quick else 30_000,
        proposal_scale=2.0,
        cores=args.cores,
        seed=args.seed + 1,
        prior=fixed,
    )

    # --- Loss posterior (conditional on track MAP)
    loss_npz = run_loss_posterior_from_track_map(
        event_path,
        track_posterior_npz=track_npz,
        out_dir=out_dir,
        chains=4 if args.quick else 16,
        cores=args.cores,
        n_steps=4_000 if args.quick else 20_000,
        burn_in=1_500 if args.quick else 8_000,
        thin=5 if args.quick else 10,
        seed=args.seed + 2,
    )

    # --- Energy posterior (from NTrigPMT calibration)
    evt = load_event_json_gz(event_path)
    n_trig = int(len(np.unique(evt.pmt_key[evt.triggered])))
    calib = load_figure2_calibration(figure2_path)
    energy_post = energy_posterior_from_ntrigpmt(
        n_trig_pmt=n_trig,
        calibration=calib,
        seed=args.seed + 3,
        n_samples=20_000 if args.quick else 200_000,
    )
    energy_npz = save_energy_posterior_npz(out_dir / "energy_posterior.npz", energy_post)

    # --- Summaries + systematic budget
    def load_track(npz_path: Path) -> dict[str, Any]:
        with np.load(npz_path, allow_pickle=False) as f:
            out: dict[str, Any] = {"samples": f["samples"], "map_x": f["map_x"]}
            if "ess" in f:
                out["ess"] = float(f["ess"])
            if "selection_indices" in f:
                out["n_selected_hits"] = int(f["selection_indices"].size)
            return out

    trk = load_track(track_npz)
    trk_fix = load_track(track_fixed_npz)

    u = direction_unit_vectors(trk["samples"])
    u0 = np.mean(u, axis=0)
    u0 /= np.linalg.norm(u0)
    ang = _angle_deg(u, u0)

    theta_deg = float(np.degrees(np.arccos(np.clip(u0[2], -1.0, 1.0))))
    phi_deg = float(np.degrees(np.arctan2(u0[1], u0[0]) % (2.0 * np.pi)))

    u_fix = direction_unit_vectors(trk_fix["samples"])
    u_fix0 = np.mean(u_fix, axis=0)
    u_fix0 /= np.linalg.norm(u_fix0)
    ang_fix = _angle_deg(u_fix, u_fix0)

    # Published absolute orientation systematic (VOEvent uses 68% radius ~1.5 deg).
    vo_info = parse_voevent(voevent_path) if voevent_path.exists() else None
    vo_ref = {} if vo_info is None else asdict(vo_info)
    orient_r68 = float((vo_info.r68_deg if vo_info and vo_info.r68_deg is not None else 1.5))

    r68_stat = float(np.quantile(ang, 0.68))
    r68_fixed = float(np.quantile(ang_fix, 0.68))
    r68_total = math.sqrt(r68_stat**2 + orient_r68**2)

    # Loss summary
    with np.load(loss_npz, allow_pickle=False) as f:
        loss_mu = f["mu_m"]
        loss_width = f["width_m"]
        loss_area = f["area"]
    loss_mu_mean = loss_mu.mean(axis=0)
    loss_width_mean = loss_width.mean(axis=0)

    # Saturation sensitivity (uses TOT as a proxy; not a full charge model).
    map_x = trk["map_x"]
    map_pos = map_x[0:3]
    map_dir = unit_from_theta_phi(float(map_x[3]), float(map_x[4]))
    map_t0 = float(map_x[5])
    t_pred, *_ = cherenkov_times_ns(track_pos_m=map_pos, track_dir=map_dir, track_t_ns=map_t0, hit_pos_m=evt.pos_m)
    residual = evt.t_ns - t_pred
    keep = (residual >= -50.0) & (residual < 1000.0)
    s1, s2, D = emission_point_solutions_m(
        track_pos_m=map_pos,
        track_dir=map_dir,
        track_t_ns=map_t0,
        hit_pos_m=evt.pos_m[keep],
        hit_t_ns=evt.t_ns[keep],
    )
    valid = (D >= 0.0) & np.isfinite(s2) & (s2 >= 0.0) & (s2 <= 600.0)
    s = s2[valid]
    tot = evt.tot[keep][valid].astype(np.float64)
    sat = tot >= 255.0

    def tot_sums(sat_factor: float) -> list[float]:
        t_corr = tot.copy()
        t_corr[sat] *= float(sat_factor)
        out = []
        for k in range(3):
            lo = float(loss_mu_mean[k] - 2.0 * loss_width_mean[k])
            hi = float(loss_mu_mean[k] + 2.0 * loss_width_mean[k])
            m = (s >= lo) & (s <= hi)
            out.append(float(np.sum(t_corr[m])))
        return out

    tot_sat_1 = tot_sums(1.0)
    tot_sat_2 = tot_sums(2.0)

    # Sky mapping + localization contours (RA/Dec)
    sky = None
    sky_stat_path = None
    sky_total_path = None
    if vo_info is not None:
        loc, t = voevent_astropy_context(vo_info)
        sign_pick = pick_direction_sign_for_voevent(enu_unit_mean=u0, vo=vo_info)
        sign = int(sign_pick["sign"])

        u_icrs_stat = enu_unit_to_icrs_unit((sign * u).astype(np.float64), location=loc, obstime=t)
        ra_stat, dec_stat = unit_to_radec_deg(u_icrs_stat)

        sky_stat_path = out_dir / "sky_posterior_stat.npz"
        np.savez_compressed(sky_stat_path, ra_deg=ra_stat, dec_deg=dec_stat, u_icrs=u_icrs_stat, meta=json.dumps(sign_pick, indent=2))
        sky_stat = summarize_sky_posterior(u_icrs_stat)
        sky_stat["posterior_npz"] = str(sky_stat_path)
        if vo_info.ra_deg is not None and vo_info.dec_deg is not None:
            ref = radec_deg_to_unit(vo_info.ra_deg, vo_info.dec_deg)
            center = radec_deg_to_unit(sky_stat["center_ra_deg"], sky_stat["center_dec_deg"])
            sky_stat["delta_to_voevent_deg"] = float(angular_distance_deg(center, ref))

        sky_stat_map = sky_histogram_equirect(ra_stat, dec_stat, ra_bins=720, dec_bins=360)
        sky_stat_map_path = out_dir / "sky_map_stat.npz"
        np.savez_compressed(
            sky_stat_map_path,
            ra_edges_deg=sky_stat_map["ra_edges_deg"],
            dec_edges_deg=sky_stat_map["dec_edges_deg"],
            p=sky_stat_map["p"],
            area_deg2=sky_stat_map["area_deg2"],
        )
        sky_stat["map_npz"] = str(sky_stat_map_path)
        sky_stat["credible_regions_sparse"] = credible_region_from_samples_equirect(ra_stat, dec_stat, d_ra_deg=0.05, d_dec_deg=0.05)

        rng = np.random.default_rng(int(args.seed) + 4)
        u_icrs_total = smear_isotropic_orientation(u_icrs_stat, r68_deg=orient_r68, rng=rng)
        ra_tot, dec_tot = unit_to_radec_deg(u_icrs_total)

        sky_total_path = out_dir / "sky_posterior_total.npz"
        np.savez_compressed(
            sky_total_path,
            ra_deg=ra_tot,
            dec_deg=dec_tot,
            u_icrs=u_icrs_total,
            meta=json.dumps({**sign_pick, "orientation_r68_deg": orient_r68}, indent=2),
        )
        sky_total = summarize_sky_posterior(u_icrs_total)
        sky_total["posterior_npz"] = str(sky_total_path)
        sky_total["orientation_r68_deg"] = orient_r68
        if vo_info.ra_deg is not None and vo_info.dec_deg is not None:
            ref = radec_deg_to_unit(vo_info.ra_deg, vo_info.dec_deg)
            center = radec_deg_to_unit(sky_total["center_ra_deg"], sky_total["center_dec_deg"])
            sky_total["delta_to_voevent_deg"] = float(angular_distance_deg(center, ref))

        sky_total_map = sky_histogram_equirect(ra_tot, dec_tot, ra_bins=720, dec_bins=360)
        sky_total_map_path = out_dir / "sky_map_total.npz"
        np.savez_compressed(
            sky_total_map_path,
            ra_edges_deg=sky_total_map["ra_edges_deg"],
            dec_edges_deg=sky_total_map["dec_edges_deg"],
            p=sky_total_map["p"],
            area_deg2=sky_total_map["area_deg2"],
        )
        sky_total["map_npz"] = str(sky_total_map_path)
        sky_total["credible_regions_sparse"] = credible_region_from_samples_equirect(ra_tot, dec_tot, d_ra_deg=0.05, d_dec_deg=0.05)

        sky = {"sign_choice": sign_pick, "stat_only": sky_stat, "with_orientation": sky_total}

    report = {
        "inputs": {
            "event": str(event_path),
            "figure2": str(figure2_path),
            "voevent": str(voevent_path) if voevent_path.exists() else None,
            "n_trig_pmt": n_trig,
        },
        "track": {
            "direction_unit_mean": u0.tolist(),
            "direction_theta_deg": theta_deg,
            "direction_phi_deg": phi_deg,
            "direction_r68_stat_deg": r68_stat,
            "direction_r90_stat_deg": float(np.quantile(ang, 0.90)),
            "n_selected_hits": int(trk.get("n_selected_hits", -1)),
            "importance_sampling_ess": None if trk.get("ess") is None else float(trk["ess"]),
            "scattering_hyperparams": {
                "log_sigma_ns": _quantiles(trk["samples"][:, 6]),
                "log_tau_ns": _quantiles(trk["samples"][:, 7]),
            },
        },
        "losses": {
            "mu_m": {"mean": loss_mu_mean.tolist(), "q16_q50_q84_each": [_quantiles(loss_mu[:, i]) for i in range(3)]},
            "width_m": {"mean": loss_width.mean(axis=0).tolist(), "q16_q50_q84_each": [_quantiles(loss_width[:, i]) for i in range(3)]},
            "area_hits": {"mean": loss_area.mean(axis=0).tolist(), "q16_q50_q84_each": [_quantiles(loss_area[:, i]) for i in range(3)]},
        },
        "energy": json.loads(str(np.load(energy_npz, allow_pickle=False)["meta"]))["summary_pev"],
        "systematics_budget": {
            "scattering_model_deg": {
                "r68_fixed_scattering_deg": r68_fixed,
                "r68_marginalized_deg": r68_stat,
                "delta_deg": r68_stat - r68_fixed,
            },
            "orientation_deg": {"r68_abs_orientation_deg": orient_r68, "r68_total_deg_quadrature": r68_total},
            "saturation_tot_proxy": {
                "note": "Uses TOT as a proxy; saturated hits (tot>=255) scaled by sat_factor.",
                "sat_factor_1": tot_sat_1,
                "sat_factor_2": tot_sat_2,
                "ratio_2_over_1": [float(b / a) if a else None for a, b in zip(tot_sat_1, tot_sat_2)],
            },
            "voevent_reference": vo_ref,
        },
        "sky": sky,
        "model_comparison": None,
        "artifacts": {
            "track_posterior": str(track_npz),
            "track_posterior_fixed_scattering": str(track_fixed_npz),
            "loss_posterior": str(loss_npz),
            "energy_posterior": str(energy_npz),
            "sky_posterior_stat": None if sky_stat_path is None else str(sky_stat_path),
            "sky_posterior_total": None if sky_total_path is None else str(sky_total_path),
            "sky_map_stat": None if vo_info is None else str(out_dir / "sky_map_stat.npz"),
            "sky_map_total": None if vo_info is None else str(out_dir / "sky_map_total.npz"),
            "model_comparison": None,
        },
    }

    if args.model_comparison:
        from km3posterior.model_comparison import ModelComparisonConfig, ModelComparisonLikelihood, ModelComparisonPriors, run_model_comparison

        model_comp_path = out_dir / "model_comparison.json"
        cfg = ModelComparisonConfig(
            priors=ModelComparisonPriors(bundle_sigma_offset_m=float(args.model_comp_bundle_sigma_m)),
            likelihood=ModelComparisonLikelihood(outlier_frac=float(args.model_comp_outlier_frac)),
            restarts_h1=1,
            restarts_h2=int(args.model_comp_restarts_h2),
            seed=int(args.seed) + 10,
        )
        model_comp = run_model_comparison(event_path, out_path=model_comp_path, config=cfg)
        report["model_comparison"] = {
            "bayes_factors": model_comp.get("bayes_factors", {}),
            "inputs": model_comp.get("inputs", {}),
            "config": model_comp.get("config", {}),
        }
        report["artifacts"]["model_comparison"] = str(model_comp_path)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(out_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
