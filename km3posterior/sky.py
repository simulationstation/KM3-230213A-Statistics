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
