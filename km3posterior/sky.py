from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VoeventInfo:
    iso_time_utc: str
    lon_deg: float
    lat_deg: float
    height_m: float
    ra_deg: float | None
    dec_deg: float | None
    r68_deg: float | None
    reported_energy_pev: float | None


def parse_voevent(path: str | Path) -> VoeventInfo:
    path = Path(path)
    root = ET.fromstring(path.read_text(encoding="utf-8"))

    def find_text(xpath: str) -> str | None:
        el = root.find(xpath)
        return None if el is None else el.text

    def find_float(xpath: str) -> float | None:
        t = find_text(xpath)
        return None if t is None else float(t)

    # Note: the public VOEvent file has a namespaced root tag, but un-namespaced child tags.
    iso_time = find_text(".//ObservationLocation//TimeInstant/ISOTime")
    if iso_time is None:
        raise ValueError("VOEvent is missing ISOTime.")

    lon = find_float(".//ObservatoryLocation//Position3D//Value3/C1")
    lat = find_float(".//ObservatoryLocation//Position3D//Value3/C2")
    height = find_float(".//ObservatoryLocation//Position3D//Value3/C3")
    if lon is None or lat is None or height is None:
        raise ValueError("VOEvent is missing observatory geodetic coordinates.")

    ra = find_float(".//Position2D/Value2/C1")
    dec = find_float(".//Position2D/Value2/C2")
    r68 = find_float(".//Position2D/Error2Radius68")
    energy_pev = None
    for param in root.iter("Param"):
        if param.attrib.get("name") == "Energy":
            try:
                energy_pev = float(param.attrib.get("value"))
            except Exception:
                energy_pev = None

    return VoeventInfo(
        iso_time_utc=iso_time,
        lon_deg=float(lon),
        lat_deg=float(lat),
        height_m=float(height),
        ra_deg=None if ra is None else float(ra),
        dec_deg=None if dec is None else float(dec),
        r68_deg=None if r68 is None else float(r68),
        reported_energy_pev=None if energy_pev is None else float(energy_pev),
    )


def voevent_astropy_context(vo: VoeventInfo):
    import astropy.units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy.utils import iers

    iers.conf.auto_download = False

    loc = EarthLocation.from_geodetic(vo.lon_deg * u.deg, vo.lat_deg * u.deg, vo.height_m * u.m)
    t = Time(vo.iso_time_utc, scale="utc")
    return loc, t


def unit_to_radec_deg(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float64)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    ra = np.degrees(np.arctan2(u[..., 1], u[..., 0]) % (2.0 * np.pi))
    dec = np.degrees(np.arcsin(np.clip(u[..., 2], -1.0, 1.0)))
    return ra, dec


def radec_deg_to_unit(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = math.radians(float(ra_deg))
    dec = math.radians(float(dec_deg))
    c = math.cos(dec)
    return np.array([c * math.cos(ra), c * math.sin(ra), math.sin(dec)], dtype=np.float64)


def angular_distance_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    c = np.einsum("...i,...i->...", u, v)
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def enu_unit_to_icrs_unit(enu_u: np.ndarray, *, location, obstime) -> np.ndarray:
    """Convert ENU unit vectors to ICRS Cartesian unit vectors.

    Convention: ENU with azimuth measured east of north.
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, ICRS, SkyCoord
    from astropy.utils import iers

    iers.conf.auto_download = False

    enu_u = np.asarray(enu_u, dtype=np.float64)
    enu_u = enu_u / np.linalg.norm(enu_u, axis=-1, keepdims=True)
    x = enu_u[..., 0]
    y = enu_u[..., 1]
    z = enu_u[..., 2]

    az = np.arctan2(x, y)
    alt = np.arcsin(np.clip(z, -1.0, 1.0))

    altaz_frame = AltAz(obstime=obstime, location=location)
    c_altaz = SkyCoord(az=az * u.rad, alt=alt * u.rad, frame=altaz_frame)
    c_icrs = c_altaz.transform_to(ICRS())
    xyz = c_icrs.cartesian.xyz.to_value(u.one).T  # (N,3)
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
    return xyz


def pick_direction_sign_for_voevent(
    *,
    enu_unit_mean: np.ndarray,
    vo: VoeventInfo,
) -> dict[str, Any]:
    """Pick +1/-1 for detector->sky conversion to best match VOEvent RA/Dec (if present)."""
    loc, t = voevent_astropy_context(vo)

    u = np.asarray(enu_unit_mean, dtype=np.float64).reshape(3)
    u /= np.linalg.norm(u)

    icrs_p = enu_unit_to_icrs_unit(u[None, :], location=loc, obstime=t)[0]
    icrs_m = enu_unit_to_icrs_unit((-u)[None, :], location=loc, obstime=t)[0]

    ra_p, dec_p = unit_to_radec_deg(icrs_p)
    ra_m, dec_m = unit_to_radec_deg(icrs_m)

    if vo.ra_deg is None or vo.dec_deg is None:
        return {
            "sign": -1,
            "reason": "VOEvent RA/Dec missing; defaulting to -1 (matches official package sign convention in practice).",
            "candidate_plus": {"ra_deg": float(ra_p), "dec_deg": float(dec_p)},
            "candidate_minus": {"ra_deg": float(ra_m), "dec_deg": float(dec_m)},
        }

    ref = radec_deg_to_unit(vo.ra_deg, vo.dec_deg)
    d_p = float(angular_distance_deg(icrs_p, ref))
    d_m = float(angular_distance_deg(icrs_m, ref))
    sign = -1 if d_m < d_p else 1
    return {
        "sign": int(sign),
        "reason": "Chose the sign whose mapped mean direction is closest to VOEvent RA/Dec.",
        "dist_deg_plus": d_p,
        "dist_deg_minus": d_m,
        "candidate_plus": {"ra_deg": float(ra_p), "dec_deg": float(dec_p)},
        "candidate_minus": {"ra_deg": float(ra_m), "dec_deg": float(dec_m)},
        "voevent_ref": {"ra_deg": float(vo.ra_deg), "dec_deg": float(vo.dec_deg)},
    }


def smear_isotropic_orientation(u: np.ndarray, *, r68_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Apply an isotropic orientation uncertainty with a Rayleigh radial model.

    r68_deg is the 68% containment radius.
    """
    u = np.asarray(u, dtype=np.float64)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)

    r68 = float(r68_deg)
    # Rayleigh CDF: 1-exp(-r^2/(2*sigma^2)).
    sigma = math.radians(r68) / math.sqrt(-2.0 * math.log(1.0 - 0.68))
    r = rng.rayleigh(scale=sigma, size=u.shape[0]).astype(np.float64)

    a = rng.normal(size=u.shape).astype(np.float64)
    a -= np.einsum("ij,ij->i", a, u)[:, None] * u
    an = np.linalg.norm(a, axis=1, keepdims=True)
    an = np.maximum(an, 1e-12)
    a /= an

    out = u * np.cos(r)[:, None] + a * np.sin(r)[:, None]
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def summarize_sky_posterior(u_icrs: np.ndarray) -> dict[str, Any]:
    u = np.asarray(u_icrs, dtype=np.float64)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)

    mean_u = u.mean(axis=0)
    mean_u /= np.linalg.norm(mean_u)

    ang = angular_distance_deg(u, mean_u)
    ra, dec = unit_to_radec_deg(mean_u)
    return {
        "center_ra_deg": float(ra),
        "center_dec_deg": float(dec),
        "r50_deg": float(np.quantile(ang, 0.50)),
        "r68_deg": float(np.quantile(ang, 0.68)),
        "r90_deg": float(np.quantile(ang, 0.90)),
        "r99_deg": float(np.quantile(ang, 0.99)),
        "n_samples": int(u.shape[0]),
        "note": "Containment radii are computed around the posterior mean direction on the sphere.",
    }


def sky_histogram_equirect(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    ra_bins: int = 720,
    dec_bins: int = 360,
) -> dict[str, Any]:
    """Create a simple RA/Dec probability map on an equirectangular grid.

    Returns a dict with:
      - ra_edges_deg, dec_edges_deg
      - p (dec_bin, ra_bin), normalized to sum to 1
      - area_deg2 (dec_bin, ra_bin)
    """
    ra = np.asarray(ra_deg, dtype=np.float64) % 360.0
    dec = np.asarray(dec_deg, dtype=np.float64)

    ra_edges = np.linspace(0.0, 360.0, int(ra_bins) + 1, dtype=np.float64)
    dec_edges = np.linspace(-90.0, 90.0, int(dec_bins) + 1, dtype=np.float64)

    # histogram2d uses x/y terminology; we use dec as x and ra as y to get [dec, ra] layout.
    H, _, _ = np.histogram2d(dec, ra, bins=[dec_edges, ra_edges])
    total = float(H.sum())
    p = H / total if total > 0 else H

    ra_edges_rad = np.deg2rad(ra_edges)
    dec_edges_rad = np.deg2rad(dec_edges)
    d_ra = ra_edges_rad[1:] - ra_edges_rad[:-1]  # (ra_bins,)
    d_sin_dec = np.sin(dec_edges_rad[1:]) - np.sin(dec_edges_rad[:-1])  # (dec_bins,)
    area_sr = d_sin_dec[:, None] * d_ra[None, :]
    area_deg2 = area_sr * (180.0 / np.pi) ** 2

    return {
        "ra_edges_deg": ra_edges,
        "dec_edges_deg": dec_edges,
        "p": p.astype(np.float64),
        "area_deg2": area_deg2.astype(np.float64),
        "n_samples": int(ra.size),
    }


def credible_region_from_map(
    p: np.ndarray,
    area_deg2: np.ndarray,
    *,
    levels: tuple[float, ...] = (0.50, 0.68, 0.90, 0.99),
) -> list[dict[str, float]]:
    """Compute HPD-like regions from a discretized probability map.

    For each level L, finds the smallest set of pixels (by descending p) whose summed p >= L.
    Returns a list of dicts with p_threshold and area_deg2.
    """
    p = np.asarray(p, dtype=np.float64)
    area = np.asarray(area_deg2, dtype=np.float64)
    if p.shape != area.shape:
        raise ValueError("p and area_deg2 must have the same shape.")

    p_flat = p.ravel()
    area_flat = area.ravel()

    order = np.argsort(p_flat)[::-1]
    p_sorted = p_flat[order]
    area_sorted = area_flat[order]

    cdf = np.cumsum(p_sorted)
    out: list[dict[str, float]] = []
    for L in levels:
        L = float(L)
        if not (0.0 < L < 1.0):
            continue
        idx = int(np.searchsorted(cdf, L, side="left"))
        idx = min(idx, p_sorted.size - 1)
        thresh = float(p_sorted[idx])
        area_sum = float(np.sum(area_sorted[: idx + 1]))
        out.append({"level": L, "p_threshold": thresh, "area_deg2": area_sum})
    return out


def credible_region_from_samples_equirect(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    *,
    d_ra_deg: float = 0.05,
    d_dec_deg: float = 0.05,
    levels: tuple[float, ...] = (0.50, 0.68, 0.90, 0.99),
) -> dict[str, Any]:
    """HPD-like regions computed from samples binned on a fine equirectangular grid.

    Uses a sparse binning (only stores non-empty pixels), so it stays fast even for very fine grids.
    """
    d_ra_deg = float(d_ra_deg)
    d_dec_deg = float(d_dec_deg)
    if not (d_ra_deg > 0 and d_dec_deg > 0):
        raise ValueError("d_ra_deg and d_dec_deg must be positive.")
    if abs((360.0 / d_ra_deg) - round(360.0 / d_ra_deg)) > 1e-9:
        raise ValueError("d_ra_deg must divide 360 exactly for simple binning.")
    if abs((180.0 / d_dec_deg) - round(180.0 / d_dec_deg)) > 1e-9:
        raise ValueError("d_dec_deg must divide 180 exactly for simple binning.")

    ra = (np.asarray(ra_deg, dtype=np.float64) % 360.0).copy()
    dec = np.asarray(dec_deg, dtype=np.float64).copy()

    n = int(ra.size)
    if n == 0:
        return {"binning": {"d_ra_deg": d_ra_deg, "d_dec_deg": d_dec_deg}, "regions": []}

    # Clip dec to the open interval [-90, 90) so binning behaves.
    dec = np.clip(dec, -90.0, np.nextafter(90.0, -np.inf))

    n_ra = int(round(360.0 / d_ra_deg))
    n_dec = int(round(180.0 / d_dec_deg))

    ra_bin = np.floor(ra / d_ra_deg).astype(np.int64)
    dec_bin = np.floor((dec + 90.0) / d_dec_deg).astype(np.int64)
    ra_bin = np.clip(ra_bin, 0, n_ra - 1)
    dec_bin = np.clip(dec_bin, 0, n_dec - 1)

    flat = dec_bin * n_ra + ra_bin
    uniq, counts = np.unique(flat, return_counts=True)

    # Pixel areas depend on dec only.
    dec_i = (uniq // n_ra).astype(np.int64)
    dec_lo = -90.0 + dec_i.astype(np.float64) * d_dec_deg
    dec_hi = dec_lo + d_dec_deg
    d_ra_rad = np.deg2rad(d_ra_deg)
    area_deg2 = (np.sin(np.deg2rad(dec_hi)) - np.sin(np.deg2rad(dec_lo))) * d_ra_rad * (180.0 / np.pi) ** 2

    p = counts.astype(np.float64) / float(n)
    order = np.argsort(p)[::-1]
    p_sorted = p[order]
    area_sorted = area_deg2[order]
    cdf = np.cumsum(p_sorted)

    regions: list[dict[str, float]] = []
    for L in levels:
        L = float(L)
        if not (0.0 < L < 1.0):
            continue
        idx = int(np.searchsorted(cdf, L, side="left"))
        idx = min(idx, p_sorted.size - 1)
        thresh = float(p_sorted[idx])
        area_sum = float(np.sum(area_sorted[: idx + 1]))
        regions.append({"level": L, "p_threshold": thresh, "area_deg2": area_sum})

    return {
        "binning": {"d_ra_deg": d_ra_deg, "d_dec_deg": d_dec_deg, "n_ra": n_ra, "n_dec": n_dec},
        "regions": regions,
    }
