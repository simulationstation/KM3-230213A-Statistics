from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EventData:
    pos_m: np.ndarray  # (N, 3) float64
    t_ns: np.ndarray  # (N,) float64
    tot: np.ndarray  # (N,) int16
    dom_id: np.ndarray  # (N,) int64
    channel_id: np.ndarray  # (N,) int16
    triggered: np.ndarray  # (N,) bool

    reco_pos_m: np.ndarray  # (3,) float64
    reco_dir: np.ndarray  # (3,) float64 (unit-ish)
    reco_t_ns: float

    utc_timestamp: str | None = None

    @property
    def pmt_key(self) -> np.ndarray:
        # channel_id fits in 6 bits (0..30); store (dom_id, channel_id) in one int64.
        return (self.dom_id.astype(np.int64) << 6) | self.channel_id.astype(np.int64)


def load_event_json_gz(path: str | Path, *, track_time_shift_ns: float = 5.0) -> EventData:
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)

    hits = raw["hits"]
    pos_m = np.array([(h["pos_x"], h["pos_y"], h["pos_z"]) for h in hits], dtype=np.float64)
    t_ns = np.array([h["t"] for h in hits], dtype=np.float64)
    tot = np.array([h["tot"] for h in hits], dtype=np.int16)
    dom_id = np.array([h["dom_id"] for h in hits], dtype=np.int64)
    channel_id = np.array([h["channel_id"] for h in hits], dtype=np.int16)
    triggered = np.array([bool(h.get("triggered", False)) for h in hits], dtype=bool)

    reco = raw["reconstructed_track"]
    reco_pos_m = np.array([reco["pos_x"], reco["pos_y"], reco["pos_z"]], dtype=np.float64)
    reco_dir = np.array([reco["dir_x"], reco["dir_y"], reco["dir_z"]], dtype=np.float64)
    reco_dir = reco_dir / np.linalg.norm(reco_dir)
    reco_t_ns = float(reco["t"]) + float(track_time_shift_ns)

    utc_timestamp = raw.get("utc_timestamp")

    return EventData(
        pos_m=pos_m,
        t_ns=t_ns,
        tot=tot,
        dom_id=dom_id,
        channel_id=channel_id,
        triggered=triggered,
        reco_pos_m=reco_pos_m,
        reco_dir=reco_dir,
        reco_t_ns=reco_t_ns,
        utc_timestamp=utc_timestamp,
    )

