from __future__ import annotations

from math import acos, sin, tan

# Reference values at 460 nm (copied from the KM3NeT open-data repo).
WATER_INDEX = 1.3499
DNDL = 0.0298

# Speed of light in vacuum (m/ns).
C_LIGHT = 299_792_458 * 1e-9

# Group velocity of light in seawater at 460 nm (m/ns).
V_LIGHT = C_LIGHT / (WATER_INDEX + DNDL)

COS_CHERENKOV_ANGLE = 1.0 / WATER_INDEX
CHERENKOV_ANGLE = acos(COS_CHERENKOV_ANGLE)

SIN_CHERENKOV_ANGLE = sin(CHERENKOV_ANGLE)
TAN_CHERENKOV_ANGLE = tan(CHERENKOV_ANGLE)

