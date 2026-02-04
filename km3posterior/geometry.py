from __future__ import annotations

import numpy as np

from .constants import C_LIGHT, SIN_CHERENKOV_ANGLE, TAN_CHERENKOV_ANGLE, V_LIGHT


def unit_from_theta_phi(theta: float, phi: float) -> np.ndarray:
    """Return unit vector for polar angle theta and azimuth phi (radians).

    Convention: theta is the polar angle from +z (0..pi), phi in xy-plane from +x (0..2pi).
    """
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], dtype=np.float64)


def theta_phi_from_unit(u: np.ndarray) -> tuple[float, float]:
    u = np.asarray(u, dtype=np.float64)
    u = u / np.linalg.norm(u)
    theta = float(np.arccos(np.clip(u[2], -1.0, 1.0)))
    phi = float(np.arctan2(u[1], u[0]) % (2 * np.pi))
    return theta, phi


def cherenkov_times_ns(
    *,
    track_pos_m: np.ndarray,
    track_dir: np.ndarray,
    track_t_ns: float,
    hit_pos_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Cherenkov 'earliest' arrival time model.

    Matches the logic in the KM3NeT open-data `Cherenkov` helper.
    """
    track_pos_m = np.asarray(track_pos_m, dtype=np.float64).reshape(3)
    track_dir = np.asarray(track_dir, dtype=np.float64).reshape(3)
    track_dir = track_dir / np.linalg.norm(track_dir)
    hit_pos_m = np.asarray(hit_pos_m, dtype=np.float64)

    v = hit_pos_m - track_pos_m[None, :]
    l = v @ track_dir  # (N,)
    v2 = np.einsum("ij,ij->i", v, v)
    k2 = np.maximum(v2 - l * l, 0.0)

    d_closest_m = np.sqrt(k2)
    d_photon_m = d_closest_m / SIN_CHERENKOV_ANGLE
    d_trk_m = l - d_closest_m / TAN_CHERENKOV_ANGLE

    t_pred_ns = track_t_ns + d_trk_m / C_LIGHT + d_photon_m / V_LIGHT
    return t_pred_ns, d_closest_m, d_photon_m, d_trk_m


def emission_point_solutions_m(
    *,
    track_pos_m: np.ndarray,
    track_dir: np.ndarray,
    track_t_ns: float,
    hit_pos_m: np.ndarray,
    hit_t_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve for emission points along the track (meters).

    This is the vectorized form of `src/cherenkov.py:emission_points` in the official package.
    Returns (s1, s2, discriminant), where s2 is typically the physical solution.
    """
    track_pos_m = np.asarray(track_pos_m, dtype=np.float64).reshape(3)
    track_dir = np.asarray(track_dir, dtype=np.float64).reshape(3)
    track_dir = track_dir / np.linalg.norm(track_dir)
    hit_pos_m = np.asarray(hit_pos_m, dtype=np.float64)
    hit_t_ns = np.asarray(hit_t_ns, dtype=np.float64)

    q = hit_pos_m - track_pos_m[None, :]
    T = hit_t_ns - float(track_t_ns)

    v2 = V_LIGHT * V_LIGHT
    c = C_LIGHT

    A = v2 / (c * c) - 1.0  # note A < 0
    B = 2.0 * (q @ track_dir) - 2.0 * T * v2 / c
    C = T * T * v2 - np.einsum("ij,ij->i", q, q)

    D = B * B - 4.0 * A * C
    sqrtD = np.sqrt(np.maximum(D, 0.0))

    s1 = (-B + sqrtD) / (2.0 * A)
    s2 = (-B - sqrtD) / (2.0 * A)
    return s1, s2, D

